"""
The Researcher + Supervisor loop -- the heart of Mini-Glia, faithful to
Glia (arXiv:2510.27176, §4).

TWO AGENTS, DISTINCT ROLES
--------------------------
Researcher: the only agent with access to the "codebase" (the scheduler
  interface and the telemetry). Each turn it EITHER proposes a hypothesis +
  a new `schedule()` implementation, OR analyses the telemetry from the last
  experiment and reflects. We compile its code in the sandbox, run it in the
  simulator, and hand back real metrics.

Supervisor: has NO code access (exactly as in the paper). It sees only the
  Researcher's stated hypotheses, the experiment telemetry, and the history.
  Its job is to keep the Researcher productive: ask probing questions when
  progress stalls, halt clearly dead directions, and -- the paper's key
  move -- nudge IDEA COMPOSITION ("you saw headroom cut restarts and
  shortest-prefill cut queueing; have you combined them?").

CONTROL FLOW (per iteration)
  1. Researcher proposes {hypothesis, code}.
  2. Sandbox-compile + simulate -> telemetry (or a failure to learn from).
  3. Researcher analyses the telemetry.
  4. Every `supervisor_every` iterations (or on stall), Supervisor intervenes
     with a question/nudge that is injected into the Researcher's context.
  5. Track best-scoring scheduler. Stop on budget or target.

Everything the loop does is emitted through an `emit` callback as a typed
event, so the API layer can stream the reasoning to the browser (SSE) or a
capture script can dump it to JSON for replay.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

from sim.simulator import SimConfig, run_scheduler
from agent.sandbox import make_safe_scheduler, SandboxError
from agent.llm import LLMClient, extract_json, extract_code, sanitize_code


# --------------------------------------------------------------------------
# System prompts. These encode the "general principles of systems research"
# the paper teaches the Researcher via its system prompt.
# --------------------------------------------------------------------------

RESEARCHER_SYSTEM = """You are a Researcher agent that designs BATCH SCHEDULERS for a distributed LLM-inference GPU cluster, working like an expert systems engineer: form a hypothesis, implement it, run an experiment, analyse the telemetry, and refine.

THE SYSTEM YOU ARE OPTIMISING
- Requests arrive over time. Each has a prefill (prompt) length you CAN see, and a decode (output) length you CANNOT see at scheduling time. This is the central difficulty.
- Each GPU has a fixed KV-cache measured in blocks (16 tokens/block). As a request decodes, its KV usage GROWS one token at a time.
- When a GPU runs OUT of free blocks mid-decode, the YOUNGEST request on it is EVICTED and RESTARTED -- all its progress is lost. Restarts waste compute and inflate latency.
- Goal: minimise mean end-to-end latency AND maximise completion rate, primarily by controlling restarts and queueing.

HOW TO READ THE TELEMETRY (think like a systems engineer)
- LOW completion rate and HIGH restart fraction together are the signature of MEMORY PRESSURE: GPUs are being packed so full that decode growth triggers evictions. The first lever to reach for is controlling how full you let a GPU get, NOT how you order requests.
- Only once restarts are under control does the ORDER you admit requests meaningfully affect latency. Reordering requests while memory is still oversubscribed will not fix restarts and can make things worse.
- A change that improves one metric but worsens another (e.g. lower latency but MORE restarts) is a signal to investigate, not to accept blindly. Ask WHY.
- Test ONE idea at a time when you can. If you bundle two changes and the result is mixed, you cannot tell which one helped. Isolate, then combine deliberately.

YOUR DESIGN SURFACE -- you write exactly this function:

    def schedule(pending, gpus, now, params):
        # pending: list of objects with .rid, .prefill_tokens, .waiting_since, .restarts
        # gpus: list of objects with .gpu_id, .blocks_total, .blocks_used, .free_blocks, .num_active, .max_batch
        # now: current integer time-step
        # params: dict of tunable knobs you may read, e.g. params.get("headroom", 0.3)
        # RETURN: dict mapping gpu_id -> list of rids to admit onto that GPU this step
        ...

RULES FOR YOUR CODE
- Pure Python. You may use `sorted`, `min`, `max`, `sum`, comprehensions, lambdas, and `math`.
- NO imports, NO file/network access, NO while-True loops.
- Only admit a request to a GPU that has free blocks and batch room; the simulator enforces this too.
- You do NOT know decode length. You must reason about it indirectly.

OUTPUT FORMAT -- READ CAREFULLY, THIS IS STRICT
Respond with ONE JSON object and NOTHING else. No text before or after it. No markdown fences.
When action is "propose", the "code" field MUST contain the COMPLETE source of a function literally named `schedule` with the signature `def schedule(pending, gpus, now, params):`. Do NOT put explanation or prose inside the code field -- only runnable Python. Do NOT rename the function. Do NOT wrap it in a class.

{
  "action": "propose" | "analyze" | "stop",
  "hypothesis": "<one or two sentences: what you will try and why you expect it to help>",
  "code": "<full python source of schedule(), REQUIRED when action=propose>",
  "params": {"headroom": 0.3},
  "sweep": {"param": "headroom", "min": 0.0, "max": 0.9, "steps": 8},
  "analysis": "<when action=analyze: what the telemetry tells you and what you'll try next>"
}

TUNING A PARAMETER: your first guess for a knob like headroom is often wrong -- too low leaves restarts, too high causes queueing. Instead of guessing repeatedly, include a "sweep" field alongside a "propose" action to test your scheduler across a RANGE of values for one parameter in a single step. You will get back a table of every value and its telemetry, and can then pick the best value and refine or compose from there. Use a sweep as soon as you have a working memory-control scheduler whose parameter you want to tune.

Keep hypotheses concrete and grounded in the mechanism (restarts, memory pressure, queueing, prefill length). Start with the dominant bottleneck the telemetry points to, get it under control, THEN refine and compose additional ideas on top."""


SUPERVISOR_SYSTEM = """You are a Supervisor agent guiding a Researcher who is designing a batch scheduler for an LLM-inference GPU cluster. You do NOT write code and you do NOT see the code -- you see only the Researcher's stated hypotheses and the experiment telemetry (mean latency, completion rate, restart fraction, queueing delay).

Your role, like a senior advisor in a small research team:
- When the Researcher draws a conclusion the telemetry does not support, challenge it specifically. For example, if they treat an idea as helpful when it actually worsened completion or restarts, point that out and ask what the poor completion rate implies about the real bottleneck.
- Guide them to fix the DOMINANT bottleneck first (memory pressure shows up as low completion + high restarts) before optimising secondary things like request ordering.
- Encourage IDEA COMPOSITION once pieces work in isolation: if one idea controlled restarts and a different idea was tried separately, explicitly prompt them to layer the second on top of the first and measure the combination.
- If an experiment failed or was rejected, remind them of the strict output format and ask them to re-propose cleanly.
- Be brief and pointed: one or two sentences. Do NOT write code or name specific parameter values; guide the REASONING and let the Researcher choose the mechanism.

Respond with a single JSON object:
{ "intervention": "<your question or nudge to the Researcher>" }"""


# --------------------------------------------------------------------------
# Events + result containers
# --------------------------------------------------------------------------

@dataclass
class Candidate:
    iteration: int
    hypothesis: str
    code: str
    params: dict
    metrics: dict
    score: float          # lower is better (mean_e2e, with completion penalty)
    valid: bool
    error: str = ""


@dataclass
class LoopResult:
    candidates: list[Candidate] = field(default_factory=list)
    best: Candidate | None = None
    baseline_metrics: dict = field(default_factory=dict)


def _score(metrics: dict) -> float:
    """Lower is better. mean_e2e is already completion-penalised in the sim,
    but we add an explicit penalty for low completion to be safe."""
    if "error" in metrics:
        return float("inf")
    mean_e2e = metrics.get("mean_e2e", float("inf"))
    completion = metrics.get("completion_rate", 0.0)
    # penalise incompleteness: a run that finishes 50% is not half as good
    penalty = (1.0 - completion) * 3000.0
    return mean_e2e + penalty


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------

class GliaLoop:
    def __init__(self, llm: LLMClient, cfg: SimConfig | None = None,
                 budget: int = 8, supervisor_every: int = 2,
                 seeds: list[int] | None = None,
                 emit: Callable[[dict], None] | None = None):
        self.llm = llm
        self.cfg = cfg or SimConfig()
        self.budget = budget
        self.supervisor_every = supervisor_every
        self.seeds = seeds or [0, 1, 2, 3]
        self.emit = emit or (lambda ev: None)
        self.researcher_msgs: list[dict] = []
        self.result = LoopResult()

    # ---- baseline (FCFS) so the agent has a reference point -------------
    def _baseline(self):
        from sim.schedulers import fcfs_greedy
        m = run_scheduler(fcfs_greedy, self.cfg, {}, self.seeds)
        self.result.baseline_metrics = m
        self.emit({"type": "baseline", "metrics": m})
        # seed the researcher's context with the task + baseline telemetry
        self.researcher_msgs.append({
            "role": "user",
            "content": (
                "Here is the baseline scheduler's telemetry (naive FCFS, "
                "packs each GPU greedily by arrival order):\n"
                f"{json.dumps(_telemetry_for_llm(m), indent=2)}\n\n"
                "Propose your first scheduler. Start with a simple, well-motivated "
                "idea and state your hypothesis."
            )
        })

    # ---- agent-requested parameter sweep --------------------------------
    def _run_sweep(self, sched, sweep: dict, base_params: dict, iteration: int):
        """Run `sched` across a range of values for one parameter. Returns a
        table of {value, telemetry} plus the best (value, metrics, score).
        The agent specifies the parameter; we use its range if given, else a
        sensible default grid in [0, 1] for fractional knobs like headroom."""
        param = sweep["param"]
        values = sweep.get("values")
        if not values:
            lo = sweep.get("min", 0.0)
            hi = sweep.get("max", 0.9)
            steps = min(int(sweep.get("steps", 8)), 12)  # cap work
            if steps < 2:
                steps = 8
            values = [round(lo + (hi - lo) * i / (steps - 1), 3) for i in range(steps)]
        # cap total sweep size so a run can't explode cost
        values = values[:12]

        self.emit({"type": "sweep_start", "iteration": iteration,
                   "param": param, "values": values})

        table = []
        best = None  # (value, metrics, score)
        for v in values:
            p = {**base_params, param: v}
            m = run_scheduler(sched, self.cfg, p, self.seeds)
            s = _score(m)
            row = {param: v, **_telemetry_for_llm(m), "score": round(s, 1)}
            table.append(row)
            if best is None or s < best[2]:
                best = (v, m, s)

        self.emit({"type": "sweep_result", "iteration": iteration,
                   "param": param, "table": table,
                   "best_value": best[0], "best_score": round(best[2], 1)})
        return {"table": table, "best": best}

    # ---- one Researcher proposal + evaluation ---------------------------
    def _researcher_turn(self, iteration: int) -> Candidate | None:
        raw = self.llm.complete(RESEARCHER_SYSTEM, self.researcher_msgs)
        self.researcher_msgs.append({"role": "assistant", "content": raw})
        obj = extract_json(raw)
        action = obj.get("action", "propose")

        if action == "stop":
            self.emit({"type": "researcher_stop",
                       "rationale": obj.get("rationale", obj.get("analysis", ""))})
            return None

        hypothesis = obj.get("hypothesis", "").strip()
        code = sanitize_code(obj.get("code", "") or extract_code(raw))
        params = obj.get("params", {}) or {}

        self.emit({"type": "hypothesis", "iteration": iteration,
                   "text": hypothesis, "params": params})

        # compile + evaluate, with up to 2 INVISIBLE retries for formatting slips.
        # A weaker model occasionally emits code the sandbox rejects; rather than
        # burn a numbered iteration on a formatting mistake, we quietly ask it to
        # re-propose and re-sanitise. Only if all retries fail do we surface it.
        sched = None
        last_err = ""
        for attempt in range(3):
            if not code:
                last_err = "no schedule() function found in response"
            else:
                try:
                    sched = make_safe_scheduler(code)
                    break
                except SandboxError as e:
                    last_err = str(e)
            # ask for a clean re-proposal (not counted as an iteration)
            if attempt < 2:
                self.researcher_msgs.append({
                    "role": "user",
                    "content": (
                        f"Your previous code could not be run ({last_err}). "
                        "Re-send ONLY the corrected scheduler. The `code` field "
                        "must contain a single function `def schedule(pending, "
                        "gpus, now, params):` in pure Python -- no imports, no "
                        "json, no markdown, no prose, no class wrapper. Keep the "
                        "same hypothesis."
                    )
                })
                retry_raw = self.llm.complete(RESEARCHER_SYSTEM, self.researcher_msgs)
                self.researcher_msgs.append({"role": "assistant", "content": retry_raw})
                retry_obj = extract_json(retry_raw)
                code = sanitize_code(retry_obj.get("code", "") or extract_code(retry_raw))
                if retry_obj.get("hypothesis"):
                    hypothesis = retry_obj["hypothesis"].strip()

        if sched is None:
            cand = Candidate(iteration, hypothesis, code, params, {}, float("inf"),
                             valid=False, error=last_err)
            self.emit({"type": "experiment_result", "iteration": iteration,
                       "valid": False, "error": last_err})
            self.result.candidates.append(cand)
            return cand

        self.emit({"type": "experiment_start", "iteration": iteration, "code": code})
        metrics = run_scheduler(sched, self.cfg, params, self.seeds)
        score = _score(metrics)
        valid = "error" not in metrics
        cand = Candidate(iteration, hypothesis, code, params, metrics, score, valid,
                         error=metrics.get("error", ""))
        self.result.candidates.append(cand)

        self.emit({"type": "experiment_result", "iteration": iteration,
                   "valid": valid, "metrics": metrics, "score": round(score, 1),
                   "error": metrics.get("error", "")})

        # update best
        if valid and (self.result.best is None or score < self.result.best.score):
            self.result.best = cand
            self.emit({"type": "new_best", "iteration": iteration,
                       "score": round(score, 1),
                       "mean_e2e": metrics.get("mean_e2e"),
                       "completion_rate": metrics.get("completion_rate")})

        # --- optional agent-requested PARAMETER SWEEP ---
        # If the agent asked to sweep a parameter (e.g. "headroom"), we run its
        # SAME scheduler across a range of values and feed all results back, so
        # the agent can pick the best value itself. This mirrors the paper's
        # "rapid search over the (r,m) parameter space" after discovering HRA:
        # the first headroom guess is often wrong, and a sweep lets the agent
        # converge. The agent decides to sweep and interprets the results -- we
        # only execute the runs.
        sweep = obj.get("sweep")
        if valid and isinstance(sweep, dict) and sweep.get("param"):
            swept = self._run_sweep(sched, sweep, params, iteration)
            if swept:
                # surface the best swept config; may become the new best
                best_val, best_metrics, best_score = swept["best"]
                if best_score < cand.score:
                    swept_params = {**params, sweep["param"]: best_val}
                    swept_cand = Candidate(
                        iteration, hypothesis + f" (swept {sweep['param']}={best_val})",
                        code, swept_params, best_metrics, best_score, True)
                    self.result.candidates.append(swept_cand)
                    if self.result.best is None or best_score < self.result.best.score:
                        self.result.best = swept_cand
                        self.emit({"type": "new_best", "iteration": iteration,
                                   "score": round(best_score, 1),
                                   "mean_e2e": best_metrics.get("mean_e2e"),
                                   "completion_rate": best_metrics.get("completion_rate")})
                # feed the whole sweep table back to the agent
                self.researcher_msgs.append({
                    "role": "user",
                    "content": (
                        f"Parameter sweep of '{sweep['param']}' for your scheduler "
                        f"(each row is a value and its telemetry):\n"
                        f"{json.dumps(swept['table'], indent=2)}\n\n"
                        "Analyse the sweep: which value is best and why? Then decide "
                        "whether to refine further or compose another idea on top."
                    )
                })
                return cand

        # feed telemetry back and ask for analysis
        self.researcher_msgs.append({
            "role": "user",
            "content": (
                "Experiment telemetry for your scheduler:\n"
                f"{json.dumps(_telemetry_for_llm(metrics), indent=2)}\n\n"
                "Analyse this: what does it tell you about the bottleneck, and "
                "what will you try next? Respond with action=analyze, then you "
                "may propose again next turn."
            )
        })
        return cand

    # ---- Researcher analysis turn (reflection) --------------------------
    def _analysis_turn(self, iteration: int):
        raw = self.llm.complete(RESEARCHER_SYSTEM, self.researcher_msgs)
        self.researcher_msgs.append({"role": "assistant", "content": raw})
        obj = extract_json(raw)
        analysis = obj.get("analysis", "") or obj.get("hypothesis", "")
        if analysis:
            self.emit({"type": "analysis", "iteration": iteration, "text": analysis})
        # if the researcher actually proposed in the same turn, capture code too
        return obj

    # ---- Supervisor intervention ----------------------------------------
    def _supervisor_turn(self, iteration: int):
        # build a compact view of history for the supervisor (NO code)
        history = []
        for c in self.result.candidates:
            history.append({
                "iteration": c.iteration,
                "hypothesis": c.hypothesis,
                "valid": c.valid,
                "metrics": _telemetry_for_llm(c.metrics) if c.valid else {"error": c.error},
            })
        best = self.result.best
        sup_msg = [{
            "role": "user",
            "content": (
                "Research progress so far (you do not see the code, only "
                "hypotheses and telemetry):\n"
                f"{json.dumps(history, indent=2)}\n\n"
                f"Best score so far: {round(best.score,1) if best else 'none'}.\n"
                "Baseline mean latency: "
                f"{self.result.baseline_metrics.get('mean_e2e')}.\n\n"
                "Intervene: ask one sharp question, point out a contradiction, or "
                "nudge the Researcher to compose ideas that each helped."
            )
        }]
        raw = self.llm.complete(SUPERVISOR_SYSTEM, sup_msg)
        obj = extract_json(raw)
        intervention = obj.get("intervention", "").strip()
        if not intervention:
            return
        self.emit({"type": "supervisor", "iteration": iteration, "text": intervention})
        # inject into the researcher's context as guidance
        self.researcher_msgs.append({
            "role": "user",
            "content": (f"[Supervisor]: {intervention}\n\n"
                        "Take this into account and propose your next scheduler "
                        "(action=propose) with an updated hypothesis.")
        })

    # ---- driver ---------------------------------------------------------
    def run(self) -> LoopResult:
        self.emit({"type": "start", "budget": self.budget,
                   "config": _config_for_llm(self.cfg)})
        self._baseline()

        for it in range(1, self.budget + 1):
            cand = self._researcher_turn(it)
            if cand is None:
                break  # researcher chose to stop

            # reflection turn (analysis of the telemetry)
            if cand.valid:
                self._analysis_turn(it)

            # supervisor intervenes periodically
            if it % self.supervisor_every == 0 and it < self.budget:
                self._supervisor_turn(it)

        best = self.result.best
        baseline_e2e = self.result.baseline_metrics.get("mean_e2e", float("inf"))
        # Honest reporting: a "best" that didn't actually beat the FCFS baseline
        # is NOT a discovery. Flag it so the UI can say "no improvement found"
        # rather than crowning a scheduler that lost to the starting point.
        beats_baseline = bool(best and best.metrics.get("mean_e2e", float("inf")) < baseline_e2e)
        self.emit({"type": "done",
                   "beats_baseline": beats_baseline,
                   "best": {
                       "iteration": best.iteration if best else None,
                       "hypothesis": best.hypothesis if best else None,
                       "code": best.code if best else None,
                       "params": best.params if best else None,
                       "metrics": best.metrics if best else None,
                       "score": round(best.score, 1) if best else None,
                   } if beats_baseline else None,
                   "baseline_metrics": self.result.baseline_metrics})
        return self.result


# --------------------------------------------------------------------------
# Helpers: shape data for the LLM / frontend (hide internal noise)
# --------------------------------------------------------------------------

def _telemetry_for_llm(m: dict) -> dict:
    if "error" in m:
        return {"error": m["error"], "completion_rate": m.get("completion_rate", 0)}
    return {
        "mean_e2e_latency": m.get("mean_e2e"),
        "completion_rate": m.get("completion_rate"),
        "restart_fraction": m.get("restart_fraction"),
        "total_restarts": m.get("total_restarts"),
        "mean_queueing_delay": m.get("mean_queueing"),
        "p99_latency": m.get("p99_e2e"),
    }


def _config_for_llm(c: SimConfig) -> dict:
    return {
        "num_gpus": c.num_gpus,
        "blocks_per_gpu": c.blocks_per_gpu,
        "tokens_per_block": c.tokens_per_block,
        "num_requests": c.num_requests,
        "max_batch": c.max_batch,
    }
