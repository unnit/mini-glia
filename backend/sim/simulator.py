"""
Toy distributed LLM-inference simulator.

The point of this simulator is NOT physical realism. It is to faithfully
reproduce ONE dynamic from the Glia paper (arXiv:2510.27176): the KV-cache
memory / restart interaction that a batch scheduler can either fall victim to
or defend against.

The paper's key discovered insight (§3.3): vLLM allocates KV-cache blocks
incrementally during decode. A request whose prompt looks small at admission
can, over many decode steps, need far more blocks. When a GPU runs out of
free blocks, the youngest request is EVICTED and RESTARTED, losing all its
decode progress. ~26% of requests restarted in their workload. The fix Glia
found: order by prefill length (shortest-first, approximating SJF) AND reserve
KV-cache headroom so admissions don't trigger a cascade of preemptions.

This simulator models exactly that, so a scheduler that (a) reserves headroom
and (b) prioritises shorter prefills will measurably win -- which means an
agent reasoning over the telemetry can *discover* that, rather than us hard-
coding the answer.

Design surface exposed to the agent:  schedule(pending, gpus, cfg) -> dict
mapping gpu_id -> ordered list of request_ids to (re)admit this step.
Everything else (arrivals, decode growth, eviction, metrics) is fixed physics.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable


# --------------------------------------------------------------------------
# Workload + system configuration
# --------------------------------------------------------------------------

@dataclass
class SimConfig:
    """All knobs. Defaults produce a workload that restarts badly under FCFS,
    so there is real headroom for a scheduler to improve things."""
    num_gpus: int = 4
    blocks_per_gpu: int = 56          # KV-cache capacity, in blocks, per GPU
    tokens_per_block: int = 16        # vLLM-like: 16 tokens per KV block

    num_requests: int = 200
    seed: int = 0

    # Arrival process: bursty. `arrival_gap_mean` is the MEAN inter-arrival
    # gap in time-steps; log-normal with `burst_sigma` gives burstiness.
    # Tuned so the offered load is high (GPUs stay busy, queues form) but the
    # system can still drain within the horizon.
    arrival_gap_mean: float = 2.6
    burst_sigma: float = 1.0

    # Prompt (prefill) length distribution, in tokens. Heavy-tailed:
    # most prompts short, a few very long. The spread matters: it gives
    # shortest-prefill-first ordering something to work with.
    prefill_mean: float = 90.0
    prefill_tail_frac: float = 0.05   # 5% of prompts are inflated
    prefill_tail_mult: float = 8.0

    # Decode length distribution, in tokens. THIS is the hidden variable the
    # scheduler cannot see at admission time -- the crux of the problem.
    # Sized so that a GPU packed to max_batch will overflow its KV cache
    # (48*16=768 tokens) once requests decode, forcing restarts -- the exact
    # dynamic headroom defends against.
    decode_mean: float = 160.0
    decode_tail_frac: float = 0.10
    decode_tail_mult: float = 5.0

    # Throughput model: tokens processed per GPU per time-step (shared across
    # the active batch on that GPU).
    gpu_token_budget: float = 95.0

    # Head-of-line blocking: prefill is compute-heavy (like real vLLM/Sarathi).
    # A request still in its prefill phase consumes prefill_compute_weight x the
    # share of a decoding request. So a LONG prefill starves the decodes sharing
    # its GPU -- admitting long prefills first blocks everything behind them.
    # This is what makes shortest-prefill-first genuinely help (idea composition).
    prefill_compute_weight: float = 12.0

    max_batch: int = 16               # generous batch -> memory, not batch, binds
    sim_horizon: int = 6000           # hard cap on time-steps


@dataclass
class Request:
    rid: int
    arrival: int
    prefill_tokens: int
    decode_tokens: int                # hidden from scheduler
    # dynamic state
    admitted_at: int | None = None
    gpu: int | None = None
    prefill_done: int = 0
    decode_done: int = 0
    finished_at: int | None = None
    restarts: int = 0
    blocks_held: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prefill_tokens + self.decode_tokens

    @property
    def tokens_processed(self) -> int:
        return self.prefill_done + self.decode_done

    @property
    def is_done(self) -> bool:
        return self.finished_at is not None

    def blocks_needed_now(self, tokens_per_block: int) -> int:
        """Blocks required to hold current progress (KV grows with position)."""
        return max(1, math.ceil(self.tokens_processed / tokens_per_block))


@dataclass
class GpuState:
    gpu_id: int
    blocks_total: int
    active: list[int] = field(default_factory=list)   # request ids
    blocks_used: int = 0

    def free_blocks(self) -> int:
        return self.blocks_total - self.blocks_used

    def utilization(self) -> float:
        return self.blocks_used / self.blocks_total if self.blocks_total else 0.0


# --------------------------------------------------------------------------
# The scheduler view: read-only snapshots handed to the agent's code.
# We deliberately DO NOT expose decode_tokens -- the whole difficulty is that
# future decode length is unknown at scheduling time.
# --------------------------------------------------------------------------

@dataclass
class PendingView:
    rid: int
    prefill_tokens: int
    waiting_since: int
    restarts: int


@dataclass
class GpuView:
    gpu_id: int
    blocks_total: int
    blocks_used: int
    free_blocks: int
    num_active: int
    max_batch: int

    @property
    def utilization(self) -> float:
        return self.blocks_used / self.blocks_total if self.blocks_total else 0.0


# Scheduler signature: (pending, gpus, now, params) -> {gpu_id: [rid, ...]}
Scheduler = Callable[[list[PendingView], list[GpuView], int, dict], dict]


# --------------------------------------------------------------------------
# Simulator core
# --------------------------------------------------------------------------

class Simulator:
    def __init__(self, cfg: SimConfig, trace: bool = False):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.requests: dict[int, Request] = {}
        self.gpus: list[GpuState] = [
            GpuState(gpu_id=i, blocks_total=cfg.blocks_per_gpu)
            for i in range(cfg.num_gpus)
        ]
        self.pending: list[int] = []       # request ids waiting for admission
        self.arrivals_by_step: dict[int, list[int]] = {}
        self.now = 0
        self._build_workload()

        # telemetry accumulators
        self.total_restarts = 0
        self.wasted_token_steps = 0

        # --- explainability tracing (opt-in; off during agent eval runs) ---
        # When on, we record a per-step snapshot of every GPU's KV usage and a
        # log of causal events (admissions, evictions/restarts). This is the
        # raw material for the "why" trace and the KV-usage-over-time chart.
        self.trace_enabled = trace
        self.trace_steps: list[dict] = []   # downsampled GPU memory snapshots
        self.events: list[dict] = []        # causal events with explanations
        # target ~120 snapshots total regardless of horizon, so the trace stays
        # small enough to ship in a replay file and chart cleanly.
        self._sample_every = max(1, cfg.sim_horizon // 120)
        self._restart_events_full = 0       # true count (events list is capped)

    # ---- workload generation -------------------------------------------
    def _sample_lognormal(self, mean: float, sigma: float) -> float:
        # parameterise so E[X] ~= mean
        mu = math.log(max(mean, 1e-6)) - 0.5 * sigma * sigma
        return math.exp(self.rng.gauss(mu, sigma))

    def _sample_len(self, mean, tail_frac, tail_mult) -> int:
        base = self._sample_lognormal(mean, 0.6)
        if self.rng.random() < tail_frac:
            base *= tail_mult
        return max(1, int(base))

    def _build_workload(self):
        cfg = self.cfg
        t = 0
        for rid in range(cfg.num_requests):
            gap = self._sample_lognormal(cfg.arrival_gap_mean, cfg.burst_sigma)
            t += max(0, int(round(gap)))
            req = Request(
                rid=rid,
                arrival=t,
                prefill_tokens=self._sample_len(
                    cfg.prefill_mean, cfg.prefill_tail_frac, cfg.prefill_tail_mult),
                decode_tokens=self._sample_len(
                    cfg.decode_mean, cfg.decode_tail_frac, cfg.decode_tail_mult),
            )
            self.requests[rid] = req
            self.arrivals_by_step.setdefault(t, []).append(rid)

    # ---- per-step physics ----------------------------------------------
    def _release_arrivals(self):
        for rid in self.arrivals_by_step.get(self.now, []):
            self.pending.append(rid)

    def _build_views(self):
        pending_views = [
            PendingView(
                rid=r.rid,
                prefill_tokens=r.prefill_tokens,
                waiting_since=r.arrival if r.admitted_at is None else r.admitted_at,
                restarts=r.restarts,
            )
            for r in (self.requests[i] for i in self.pending)
        ]
        gpu_views = [
            GpuView(
                gpu_id=g.gpu_id,
                blocks_total=g.blocks_total,
                blocks_used=g.blocks_used,
                free_blocks=g.free_blocks(),
                num_active=len(g.active),
                max_batch=self.cfg.max_batch,
            )
            for g in self.gpus
        ]
        return pending_views, gpu_views

    def _admit(self, decisions: dict):
        """Apply scheduler decisions: move chosen pending reqs onto GPUs.
        The scheduler proposes; physics disposes -- we still enforce hard
        constraints (batch size, at least one block free). A scheduler that
        crams a GPU to 100% is ALLOWED to; it will simply pay in restarts
        later. That is the lesson we want discoverable."""
        pending_set = set(self.pending)
        for gpu in self.gpus:
            chosen = decisions.get(gpu.gpu_id, []) or []
            for rid in chosen:
                if rid not in pending_set:
                    continue
                if len(gpu.active) >= self.cfg.max_batch:
                    break
                if gpu.free_blocks() < 1:
                    break
                req = self.requests[rid]
                # one block to seed the prompt's first slice
                req.admitted_at = self.now
                req.gpu = gpu.gpu_id
                req.blocks_held = 1
                gpu.blocks_used += 1
                gpu.active.append(rid)
                pending_set.discard(rid)
                self.pending.remove(rid)

    def _run_batches(self):
        """Advance decode/prefill on every GPU, grow KV, evict on OOM.

        Compute is allocated by WEIGHT, not evenly: a request still in its
        prefill phase claims `prefill_compute_weight` shares, a decoding request
        claims 1 share. So a long prefill soaks up most of the GPU's budget for
        many steps, stalling the decodes sharing that GPU. This head-of-line
        blocking is why admitting shorter prefills first reduces mean latency --
        the effect the agent composes with headroom."""
        cfg = self.cfg
        for gpu in self.gpus:
            if not gpu.active:
                continue
            # compute each active request's weight this step
            weights = {}
            for rid in gpu.active:
                req = self.requests[rid]
                in_prefill = req.prefill_done < req.prefill_tokens
                weights[rid] = cfg.prefill_compute_weight if in_prefill else 1.0
            total_weight = sum(weights.values()) or 1.0

            # process in a stable order. NOTE: _evict_youngest can remove an
            # arbitrary request from gpu.active mid-iteration. We guard the
            # completion removal below against that race so no scheduler can
            # crash the sim, without altering the eviction dynamics.
            for rid in list(gpu.active):
                req = self.requests[rid]
                # this request's slice of the GPU budget, by weight
                work = cfg.gpu_token_budget * (weights.get(rid, 1.0) / total_weight)
                # prefill first, then decode
                if req.prefill_done < req.prefill_tokens:
                    step = min(work, req.prefill_tokens - req.prefill_done)
                    req.prefill_done += step
                    work -= step
                if work > 0 and req.decode_done < req.decode_tokens:
                    req.decode_done += min(work, req.decode_tokens - req.decode_done)

                # KV cache grows with tokens processed
                need = req.blocks_needed_now(cfg.tokens_per_block)
                if need > req.blocks_held:
                    delta = need - req.blocks_held
                    if gpu.free_blocks() >= delta:
                        gpu.blocks_used += delta
                        req.blocks_held = need
                    else:
                        # OOM: evict the YOUNGEST request on this GPU (vLLM behaviour)
                        self._evict_youngest(gpu)
                        # the current req may itself have been evicted; stop
                        if rid not in gpu.active:
                            continue

                # completion (guard remove against a concurrent eviction)
                if (req.prefill_done >= req.prefill_tokens
                        and req.decode_done >= req.decode_tokens
                        and not req.is_done):
                    req.finished_at = self.now
                    if rid in gpu.active:
                        gpu.active.remove(rid)
                        gpu.blocks_used -= req.blocks_held
                        req.blocks_held = 0

    def _evict_youngest(self, gpu: GpuState):
        if not gpu.active:
            return
        # youngest = most recently admitted
        youngest = max(gpu.active, key=lambda r: self.requests[r].admitted_at)
        req = self.requests[youngest]
        # count wasted work
        wasted = req.tokens_processed
        self.wasted_token_steps += wasted

        # explainability: log WHY this eviction happened, before we mutate state
        if self.trace_enabled:
            self._restart_events_full += 1
            self._restarts_since_snapshot = getattr(self, "_restarts_since_snapshot", 0) + 1
            # keep only the first ~40 full explanations (representative sample);
            # the running counter above stays exact for the frequency chart.
            if len(self.events) < 40:
                self.events.append({
                    "step": self.now,
                    "type": "restart",
                    "gpu": gpu.gpu_id,
                    "request": youngest,
                    "blocks_used": gpu.blocks_used,
                    "blocks_total": gpu.blocks_total,
                    "wasted_tokens": int(wasted),
                    "restart_number": req.restarts + 1,
                    "explanation": (
                        f"GPU {gpu.gpu_id} hit {gpu.blocks_used}/{gpu.blocks_total} "
                        f"KV blocks (full). Evicted youngest request #{youngest}, "
                        f"losing {int(wasted)} tokens of progress -> it must restart."
                    ),
                })

        gpu.active.remove(youngest)
        gpu.blocks_used -= req.blocks_held
        # reset progress -- the restart penalty
        req.prefill_done = 0
        req.decode_done = 0
        req.blocks_held = 0
        req.admitted_at = None
        req.gpu = None
        req.restarts += 1
        self.total_restarts += 1
        self.pending.append(youngest)   # goes back to the queue

    # ---- driver ---------------------------------------------------------
    def run(self, scheduler: Scheduler, params: dict | None = None) -> dict:
        params = params or {}
        while self.now < self.cfg.sim_horizon:
            self._release_arrivals()
            if self.pending:
                pending_views, gpu_views = self._build_views()
                try:
                    decisions = scheduler(pending_views, gpu_views, self.now, params)
                except Exception as e:  # a broken scheduler shouldn't crash the sim
                    return {"error": f"scheduler raised: {type(e).__name__}: {e}"}
                if not isinstance(decisions, dict):
                    decisions = {}
                self._admit(decisions)
            self._run_batches()

            # explainability: snapshot per-GPU KV usage this step
            if self.trace_enabled and (self.now % self._sample_every == 0):
                restarts_delta = getattr(self, "_restarts_since_snapshot", 0)
                self._restarts_since_snapshot = 0
                self.trace_steps.append({
                    "step": self.now,
                    "queue_depth": len(self.pending),
                    "restarts_delta": restarts_delta,
                    "gpus": [
                        {"gpu": g.gpu_id,
                         "used": g.blocks_used,
                         "total": g.blocks_total,
                         "util": round(g.utilization(), 3),
                         "batch": len(g.active)}
                        for g in self.gpus
                    ],
                })

            if all(r.is_done for r in self.requests.values()):
                break
            self.now += 1

        m = self._metrics()
        if self.trace_enabled:
            m["trace"] = {
                "steps": self.trace_steps,
                "events": self.events,
                "total_restart_events": self._restart_events_full,
                "blocks_per_gpu": self.cfg.blocks_per_gpu,
                "num_gpus": self.cfg.num_gpus,
                "sample_every": self._sample_every,
            }
        return m

    # ---- metrics --------------------------------------------------------
    def _metrics(self) -> dict:
        all_reqs = list(self.requests.values())
        finished = [r for r in all_reqs if r.is_done]
        n_all = len(all_reqs)
        n = len(finished)
        completion_rate = n / n_all if n_all else 0.0

        if n == 0:
            return {"error": "no requests completed within horizon",
                    "completed": 0, "num_requests": n_all,
                    "completion_rate": 0.0}

        # e2e for finished; unfinished are charged (horizon - arrival) as a
        # penalty so a scheduler cannot look good by finishing only a few.
        e2e_finished = [r.finished_at - r.arrival for r in finished]
        e2e_penalized = list(e2e_finished) + [
            self.now - r.arrival for r in all_reqs if not r.is_done
        ]
        queueing = [
            (r.admitted_at - r.arrival) if r.admitted_at is not None else 0
            for r in finished
        ]
        restarted = [r for r in finished if r.restarts > 0]
        # restart fraction over ALL requests that were ever admitted (honest:
        # not biased by which requests happened to finish)
        ever_admitted = [r for r in all_reqs
                         if r.admitted_at is not None or r.restarts > 0 or r.is_done]
        ever_restarted = [r for r in ever_admitted if r.restarts > 0]
        restart_fraction_admitted = (
            len(ever_restarted) / len(ever_admitted) if ever_admitted else 0.0)

        def pct(vals, p):
            s = sorted(vals)
            if not s:
                return 0.0
            k = min(len(s) - 1, int(p / 100 * len(s)))
            return float(s[k])

        return {
            "completed": n,
            "num_requests": n_all,
            "completion_rate": round(completion_rate, 4),
            "makespan": self.now,
            # headline score: completion-penalized mean e2e (lower is better)
            "mean_e2e": round(sum(e2e_penalized) / n_all, 2),
            "mean_e2e_finished_only": round(sum(e2e_finished) / n, 2),
            "p50_e2e": round(pct(e2e_penalized, 50), 2),
            "p90_e2e": round(pct(e2e_penalized, 90), 2),
            "p99_e2e": round(pct(e2e_penalized, 99), 2),
            "mean_queueing": round(sum(queueing) / n, 2),
            "restart_fraction": round(restart_fraction_admitted, 4),
            "restart_fraction_finished": round(len(restarted) / n, 4),
            "total_restarts": self.total_restarts,
            "wasted_token_steps": int(self.wasted_token_steps),
            "mean_e2e_no_restart": round(
                sum(r.finished_at - r.arrival for r in finished if r.restarts == 0)
                / max(1, len([r for r in finished if r.restarts == 0])), 2),
            "mean_e2e_restarted": round(
                sum(r.finished_at - r.arrival for r in restarted)
                / max(1, len(restarted)), 2) if restarted else 0.0,
        }


def run_scheduler(scheduler: Scheduler, cfg: SimConfig | None = None,
                  params: dict | None = None, seeds: list[int] | None = None) -> dict:
    """Run across multiple seeds and average -- the paper uses 10 seeds and
    reports confidence intervals. We average mean_e2e across seeds for a
    stable score."""
    cfg = cfg or SimConfig()
    seeds = seeds or [0, 1, 2, 3, 4]
    runs = []
    for s in seeds:
        c = SimConfig(**{**cfg.__dict__, "seed": s})
        sim = Simulator(c)
        m = sim.run(scheduler, params)
        runs.append(m)
    # if any run errored, surface it
    errored = [r for r in runs if "error" in r]
    if errored:
        return errored[0]
    keys = ["mean_e2e", "p90_e2e", "p99_e2e", "mean_queueing",
            "restart_fraction", "total_restarts", "completed", "completion_rate"]
    agg = {k: round(sum(r[k] for r in runs) / len(runs), 4) for k in keys}
    agg["per_seed_mean_e2e"] = [r["mean_e2e"] for r in runs]
    agg["num_requests"] = cfg.num_requests
    return agg
