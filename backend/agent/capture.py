"""
Capture a full agent run to a replay JSON.

Runs the GliaLoop against whatever model MODEL points to (Ollama locally, or
mock for a scripted deterministic run), records every streamed event, then
re-runs the WINNING scheduler with tracing on to capture the KV-cache trace
that drives the explainability playback. The result is a single JSON file the
frontend can replay with zero model calls -- this is what the public URL serves
so a founder clicking cold always sees a flawless run.

Usage:
    # deterministic scripted capture (no model needed) -- good for a guaranteed replay
    python -m agent.capture --mock --out runs/replay.json

    # real capture against local Ollama
    MODEL=ollama/qwen2.5-coder:14b python -m agent.capture --out runs/replay.json
"""

from __future__ import annotations

import argparse
import json
import os

from sim.simulator import SimConfig, Simulator, run_scheduler
from agent.sandbox import make_safe_scheduler
from agent.llm import LLMClient
from agent.loop import GliaLoop


# the same scripted arc used in the offline test, for --mock captures
from agent.test_loop_offline import (
    NAIVE, NAIVE_ANALYSIS, HEADROOM, HEADROOM_ANALYSIS,
    SUPERVISOR_1, COMPOSED, COMPOSED_ANALYSIS, FINAL_STOP,
)

MOCK_SCRIPT = [
    NAIVE, NAIVE_ANALYSIS,
    HEADROOM, HEADROOM_ANALYSIS,
    SUPERVISOR_1,
    COMPOSED, COMPOSED_ANALYSIS,
    FINAL_STOP,
]


def capture(out_path: str, use_mock: bool, budget: int, seed: int):
    cfg = SimConfig(seed=seed)

    if use_mock:
        llm = LLMClient(model="mock")
        llm.load_mock_script(MOCK_SCRIPT)
    else:
        llm = LLMClient()  # reads MODEL env var
        print(f"Capturing with model: {llm.model}")

    events: list[dict] = []
    loop = GliaLoop(llm, cfg, budget=budget, supervisor_every=2,
                    seeds=[0, 1, 2, 3], emit=events.append)
    result = loop.run()

    if result.best is None:
        raise SystemExit("capture failed: agent produced no valid scheduler")

    # Rate-limit guard: if the API throttled us during this run, the agent was
    # reasoning on degraded/empty responses -- the result is meaningless. Warn
    # loudly and refuse to save, so a throttled run is never mistaken for a
    # genuine bad-reasoning run (they look identical in the summary otherwise).
    if getattr(llm, "rate_limit_hits", 0) > 0:
        print(f"\n  ⚠️  RATE LIMITED: {llm.rate_limit_hits} API call(s) were "
              "throttled during this run.")
        print("  The agent ran on incomplete responses, so this result is NOT "
              "reliable and will not be saved.")
        print("  Wait ~60 seconds (per-minute limit) and re-run. Check quota at "
              "https://aistudio.google.com/app/apikey")
        raise SystemExit(2)

    # Honest gate: the "best" must actually beat the FCFS baseline. A run where
    # the agent never found an improvement is NOT a usable demo replay -- saving
    # it would present a scheduler that lost to the starting point as a discovery.
    best = result.best
    baseline_e2e = result.baseline_metrics.get("mean_e2e", float("inf"))
    if best.metrics.get("mean_e2e", float("inf")) >= baseline_e2e:
        print(f"\n  NO IMPROVEMENT this run: best mean_e2e "
              f"{best.metrics.get('mean_e2e'):.0f} did NOT beat baseline "
              f"{baseline_e2e:.0f}.")
        print("  This run is not saved as a replay. Re-run to get a run where "
              "the agent finds a real improvement.")
        raise SystemExit(1)

    # re-run the winning scheduler WITH tracing for the explainability playback
    sched = make_safe_scheduler(best.code)
    trace_sim = Simulator(cfg, trace=True)
    trace_metrics = trace_sim.run(sched, best.params)

    # also trace the baseline (FCFS) so the UI can show the before/after contrast
    from sim.schedulers import fcfs_greedy
    base_sim = Simulator(cfg, trace=True)
    base_metrics = base_sim.run(fcfs_greedy, {})

    replay = {
        "meta": {
            "model": "mock" if use_mock else llm.model,
            "budget": budget,
            "seed": seed,
            "config": {
                "num_gpus": cfg.num_gpus,
                "blocks_per_gpu": cfg.blocks_per_gpu,
                "tokens_per_block": cfg.tokens_per_block,
                "num_requests": cfg.num_requests,
            },
        },
        "events": events,
        "best": {
            "iteration": best.iteration,
            "hypothesis": best.hypothesis,
            "code": best.code,
            "params": best.params,
            "metrics": {k: v for k, v in best.metrics.items() if k != "trace"},
        },
        "traces": {
            "composed": _shape_trace(trace_metrics),
            "baseline": _shape_trace(base_metrics),
        },
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # Standards-compliant JSON only. Python's json accepts Infinity/NaN as an
    # extension, but browsers (JSON.parse) and most other parsers reject them.
    # The replay is consumed by a web frontend, so we must (a) replace any
    # non-finite floats with JSON-safe values, then (b) write with
    # allow_nan=False so any that slip through raise here instead of producing
    # a file no browser can load.
    def _finite(obj):
        if isinstance(obj, float):
            if obj == float("inf"):
                return 1e12   # "effectively infinite" but a real JSON number
            if obj == float("-inf"):
                return -1e12
            if obj != obj:    # NaN
                return None
            return obj
        if isinstance(obj, dict):
            return {k: _finite(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_finite(v) for v in obj]
        return obj

    safe_replay = _finite(replay)
    with open(out_path, "w") as f:
        json.dump(safe_replay, f, allow_nan=False)
    size = os.path.getsize(out_path)
    print(f"wrote {out_path} ({size:,} bytes)")
    print(f"  events: {len(events)}")
    print(f"  best: iteration {best.iteration}, mean_e2e {best.metrics['mean_e2e']}")
    print(f"  baseline mean_e2e: {base_metrics['mean_e2e']}")
    print(f"  composed restarts: {trace_metrics['trace']['total_restart_events']:,}")
    print(f"  baseline restarts: {base_metrics['trace']['total_restart_events']:,}")


def _shape_trace(metrics: dict) -> dict:
    """Turn a raw sim trace into a compact, chart-ready structure with per-GPU
    series (for the animated playback) and the causal event log."""
    tr = metrics.get("trace", {})
    steps = tr.get("steps", [])
    num_gpus = tr.get("num_gpus", 4)

    # per-GPU utilization series + restart deltas, aligned by snapshot
    frames = []
    for s in steps:
        frames.append({
            "step": s["step"],
            "util": [g["util"] for g in s["gpus"]],
            "batch": [g["batch"] for g in s["gpus"]],
            "queue": s["queue_depth"],
            "restarts": s.get("restarts_delta", 0),
        })

    return {
        "num_gpus": num_gpus,
        "blocks_per_gpu": tr.get("blocks_per_gpu"),
        "frames": frames,
        "events": tr.get("events", []),
        "total_restarts": tr.get("total_restart_events", 0),
        "mean_e2e": metrics.get("mean_e2e"),
        "completion_rate": metrics.get("completion_rate"),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/replay.json")
    ap.add_argument("--mock", action="store_true",
                    help="use the scripted deterministic arc (no model needed)")
    ap.add_argument("--budget", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    capture(args.out, args.mock, args.budget, args.seed)
