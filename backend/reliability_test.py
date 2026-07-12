"""
Run the tuned loop N times and report the spread, so we know whether the
clean-arc result is RELIABLE or a lucky single draw. LLMs are stochastic;
one good run proves little.

Run from backend/ (after copying in the updated llm.py and loop.py):
    PYTHONPATH=. MODEL=ollama/qwen2.5-coder:14b python reliability_test.py
"""
import os, pathlib, statistics
import agent.loop as loop
from sim.simulator import SimConfig
from agent.llm import LLMClient

here = pathlib.Path(__file__).parent
loop.RESEARCHER_SYSTEM = (here / "researcher_prompt.txt").read_text()
loop.SUPERVISOR_SYSTEM = (here / "supervisor_prompt.txt").read_text()

N_RUNS = 3

def run_once(run_idx):
    llm = LLMClient()
    events = {"rejected": 0, "supervisor": 0, "composed_mention": False}
    def emit(ev):
        t = ev.get("type")
        if t == "experiment_result" and not ev.get("valid"):
            events["rejected"] += 1
        if t == "supervisor":
            events["supervisor"] += 1
        if t == "hypothesis":
            txt = ev.get("text", "").lower()
            if "compos" in txt or ("order" in txt and "headroom" in txt) or "on top" in txt:
                events["composed_mention"] = True
    lp = loop.GliaLoop(llm, SimConfig(), budget=6, supervisor_every=2,
                       seeds=[0, 1, 2, 3], emit=emit)
    result = lp.run()
    b = result.best
    base = result.baseline_metrics["mean_e2e"]
    return {
        "best_e2e": b.metrics["mean_e2e"] if b else None,
        "best_completion": b.metrics["completion_rate"] if b else 0,
        "best_iter": b.iteration if b else None,
        "improvement": (base - b.metrics["mean_e2e"]) / base * 100 if b else 0,
        "valid": sum(1 for c in result.candidates if c.valid),
        "total": len(result.candidates),
        "rejected_shown": events["rejected"],
        "composed": events["composed_mention"],
        "clean": bool(b and b.metrics["mean_e2e"] < 1000 and b.metrics["completion_rate"] > 0.80),
    }

def main():
    llm = LLMClient()
    print(f"Reliability test: {N_RUNS} runs with model {llm.model}\n")
    rows = []
    for i in range(N_RUNS):
        print(f"--- run {i+1}/{N_RUNS} ---")
        r = run_once(i)
        rows.append(r)
        print(f"  best mean_e2e={r['best_e2e']:.0f} (iter {r['best_iter']}, +{r['improvement']:.0f}%)  "
              f"completion={r['best_completion']*100:.0f}%  valid={r['valid']}/{r['total']}  "
              f"visible_rejects={r['rejected_shown']}  composed={r['composed']}  clean={'YES' if r['clean'] else 'no'}")

    print("\n=== SUMMARY ACROSS RUNS ===")
    e2es = [r["best_e2e"] for r in rows if r["best_e2e"]]
    print(f"best mean_e2e: min={min(e2es):.0f} max={max(e2es):.0f} "
          f"mean={statistics.mean(e2es):.0f}" +
          (f" stdev={statistics.stdev(e2es):.0f}" if len(e2es) > 1 else ""))
    print(f"clean-arc runs: {sum(1 for r in rows if r['clean'])}/{N_RUNS}")
    print(f"runs mentioning composition: {sum(1 for r in rows if r['composed'])}/{N_RUNS}")
    print(f"total visible rejections across runs: {sum(r['rejected_shown'] for r in rows)}")
    reliable = sum(1 for r in rows if r["clean"]) == N_RUNS
    print(f"\nVERDICT: {'RELIABLE -- clean arc every run' if reliable else 'INCONSISTENT -- needs more work or a stronger model'}")

if __name__ == "__main__":
    main()
