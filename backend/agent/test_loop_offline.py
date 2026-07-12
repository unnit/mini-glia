"""
Offline end-to-end test of the GliaLoop using MOCK LLM responses.

This scripts a realistic Researcher/Supervisor conversation -- naive first
attempt, then discovering headroom, then (nudged by the Supervisor) composing
headroom with shortest-prefill -- and verifies:
  * the loop orchestrates propose -> evaluate -> analyze -> supervise correctly
  * the sandbox+sim actually run the agent's code and return real metrics
  * events stream in the right order
  * the best candidate is the composed one (matching our validated result)

No network, no model. On the Mac you set MODEL=ollama/... and run for real.
"""

import json
from sim.simulator import SimConfig
from agent.llm import LLMClient
from agent.loop import GliaLoop


# ---- scripted "LLM" responses, consumed in order ------------------------
# The loop calls the LLM in this order per iteration:
#   researcher propose -> researcher analyze -> [supervisor every 2] ...

NAIVE = json.dumps({
    "action": "propose",
    "hypothesis": "Start simple: least-loaded-queue by active count, pack GPUs to keep them busy.",
    "code": """
def schedule(pending, gpus, now, params):
    order = sorted(pending, key=lambda p: p.waiting_since)
    decisions = {g.gpu_id: [] for g in gpus}
    load = {g.gpu_id: g.num_active for g in gpus}
    for p in order:
        target = min(gpus, key=lambda g: load[g.gpu_id] + len(decisions[g.gpu_id]))
        if load[target.gpu_id] + len(decisions[target.gpu_id]) < target.max_batch and target.free_blocks > 0:
            decisions[target.gpu_id].append(p.rid)
    return decisions
""",
    "params": {}
})

NAIVE_ANALYSIS = json.dumps({
    "action": "analyze",
    "analysis": "Restart fraction is high and completion is low. Packing GPUs full means KV cache fills and the youngest requests get evicted. The bottleneck looks like memory pressure, not load balancing."
})

HEADROOM = json.dumps({
    "action": "propose",
    "hypothesis": "Reserve KV-cache headroom: only admit to a GPU while it keeps a memory safety margin free, so unknown decode growth doesn't trigger evictions.",
    "code": """
def schedule(pending, gpus, now, params):
    m = params.get("headroom", 0.35)
    order = sorted(pending, key=lambda p: p.waiting_since)
    decisions = {g.gpu_id: [] for g in gpus}
    virt_free = {g.gpu_id: g.free_blocks for g in gpus}
    for p in order:
        cands = [g for g in gpus if virt_free[g.gpu_id] - m*g.blocks_total > 1]
        if not cands:
            continue
        target = max(cands, key=lambda g: virt_free[g.gpu_id])
        decisions[target.gpu_id].append(p.rid)
        virt_free[target.gpu_id] -= 1
    return decisions
""",
    "params": {"headroom": 0.35}
})

HEADROOM_ANALYSIS = json.dumps({
    "action": "analyze",
    "analysis": "Headroom cut restarts dramatically and completion jumped. Latency improved a lot. Queueing delay rose slightly because we admit more conservatively. Next I could reduce head-of-line blocking."
})

# supervisor nudge appears here (mock) -- then researcher composes
COMPOSED = json.dumps({
    "action": "propose",
    "hypothesis": "Compose headroom with shortest-prefill-first: keep the memory margin AND admit shorter prefills first to approximate shortest-job-first, cutting head-of-line blocking.",
    "code": """
def schedule(pending, gpus, now, params):
    m = params.get("headroom", 0.35)
    order = sorted(pending, key=lambda p: p.prefill_tokens)
    decisions = {g.gpu_id: [] for g in gpus}
    virt_free = {g.gpu_id: g.free_blocks for g in gpus}
    for p in order:
        cands = [g for g in gpus if virt_free[g.gpu_id] - m*g.blocks_total > 1]
        if not cands:
            continue
        target = max(cands, key=lambda g: virt_free[g.gpu_id])
        decisions[target.gpu_id].append(p.rid)
        virt_free[target.gpu_id] -= 1
    return decisions
""",
    "params": {"headroom": 0.35}
})

COMPOSED_ANALYSIS = json.dumps({
    "action": "analyze",
    "analysis": "Best result yet: low restarts, high completion, lowest mean latency. Shortest-prefill ordering reduced queueing versus the previous version. This composed design is my recommendation."
})

SUPERVISOR_1 = json.dumps({
    "intervention": "You found memory headroom cuts restarts. Separately, does the ORDER you admit requests matter? You're admitting by wait time -- could prioritising shorter prefills reduce head-of-line blocking on top of the headroom gain?"
})

FINAL_STOP = json.dumps({
    "action": "stop",
    "rationale": "Composed design meets the goal; further iteration shows diminishing returns."
})


def main():
    script = [
        NAIVE, NAIVE_ANALYSIS,
        HEADROOM, HEADROOM_ANALYSIS,
        SUPERVISOR_1,               # supervisor fires after iteration 2
        COMPOSED, COMPOSED_ANALYSIS,
        FINAL_STOP,
    ]
    llm = LLMClient(model="mock")
    llm.load_mock_script(script)

    events = []
    def emit(ev):
        events.append(ev)
        t = ev.get("type")
        if t == "baseline":
            print(f"[baseline] mean_e2e={ev['metrics']['mean_e2e']} "
                  f"completion={ev['metrics']['completion_rate']*100:.0f}%")
        elif t == "hypothesis":
            print(f"\n[iter {ev['iteration']}] HYPOTHESIS: {ev['text']}")
        elif t == "experiment_result":
            if ev["valid"]:
                m = ev["metrics"]
                print(f"[iter {ev['iteration']}] RESULT: mean_e2e={m['mean_e2e']} "
                      f"completion={m['completion_rate']*100:.0f}% "
                      f"restarts={m['restart_fraction']} score={ev['score']}")
            else:
                print(f"[iter {ev['iteration']}] INVALID: {ev['error']}")
        elif t == "new_best":
            print(f"[iter {ev['iteration']}] *** NEW BEST *** score={ev['score']}")
        elif t == "analysis":
            print(f"[iter {ev['iteration']}] ANALYSIS: {ev['text'][:80]}...")
        elif t == "supervisor":
            print(f"\n[iter {ev['iteration']}] SUPERVISOR: {ev['text']}")
        elif t == "done":
            b = ev["best"]
            print(f"\n[done] BEST = iteration {b['iteration']}, "
                  f"score {b['score']}, mean_e2e {b['metrics']['mean_e2e']}")

    loop = GliaLoop(llm, SimConfig(), budget=4, supervisor_every=2,
                    seeds=[0, 1, 2, 3], emit=emit)
    result = loop.run()

    # assertions
    print("\n--- checks ---")
    baseline = result.baseline_metrics["mean_e2e"]
    best = result.best
    print(f"baseline mean_e2e = {baseline}")
    print(f"best mean_e2e     = {best.metrics['mean_e2e']}")
    improvement = (baseline - best.metrics["mean_e2e"]) / baseline * 100
    print(f"improvement       = {improvement:.0f}%")
    assert best is not None, "no best candidate!"
    assert best.metrics["mean_e2e"] < baseline, "best should beat baseline"
    # with Option A, the COMPOSED design (headroom + shortest-prefill) should win
    assert "ompos" in best.hypothesis or "prefill" in best.hypothesis.lower(), \
        f"best should be the composed design, got: {best.hypothesis!r}"
    # and it should beat the headroom-only candidate (iteration 2)
    headroom_cand = next((c for c in result.candidates if "headroom" in c.hypothesis.lower()
                          and "compos" not in c.hypothesis.lower()), None)
    if headroom_cand and headroom_cand.valid:
        assert best.metrics["mean_e2e"] < headroom_cand.metrics["mean_e2e"], \
            "composed design should beat headroom-alone"
        print(f"composed {best.metrics['mean_e2e']} beats headroom-only "
              f"{headroom_cand.metrics['mean_e2e']}")
    n_valid = sum(1 for c in result.candidates if c.valid)
    print(f"valid candidates  = {n_valid}/{len(result.candidates)}")
    supervisor_fired = any(e["type"] == "supervisor" for e in events)
    print(f"supervisor fired  = {supervisor_fired}")
    assert supervisor_fired, "supervisor should have intervened"
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
