"""
A/B test the tuned prompts against your live Ollama model WITHOUT editing loop.py.

It monkey-patches the two system-prompt strings in agent.loop, then runs a
single live capture so you can see whether the 14B model now finds the clean
arc (naive/ordering -> recognise memory pressure -> headroom -> compose).

Run from the backend/ directory:
    PYTHONPATH=. MODEL=ollama/qwen2.5-coder:14b python test_tuned_prompts.py
"""
import os, json, pathlib
import agent.loop as loop
from sim.simulator import SimConfig
from agent.llm import LLMClient

# load the tuned prompts sitting next to this script
here = pathlib.Path(__file__).parent
loop.RESEARCHER_SYSTEM = (here / "researcher_prompt.txt").read_text()
loop.SUPERVISOR_SYSTEM = (here / "supervisor_prompt.txt").read_text()

def main():
    llm = LLMClient()  # reads MODEL env var
    print(f"Testing tuned prompts with model: {llm.model}\n")

    def emit(ev):
        t = ev.get("type")
        if t == "baseline":
            print(f"[baseline] mean_e2e={ev['metrics']['mean_e2e']:.0f} "
                  f"completion={ev['metrics']['completion_rate']*100:.0f}%")
        elif t == "hypothesis":
            print(f"\n[iter {ev['iteration']}] HYPOTHESIS: {ev['text'][:160]}")
        elif t == "experiment_result":
            if ev["valid"]:
                m = ev["metrics"]
                print(f"  RESULT: mean_e2e={m['mean_e2e']:.0f} "
                      f"done={m['completion_rate']*100:.0f}% "
                      f"restarts={m['restart_fraction']:.2f} score={ev.get('score',0):.0f}")
            else:
                print(f"  REJECTED: {ev.get('error','?')[:120]}")
        elif t == "new_best":
            print(f"  *** NEW BEST *** score={ev['score']:.0f}")
        elif t == "supervisor":
            print(f"  SUPERVISOR: {ev['text'][:160]}")
        elif t == "analysis":
            print(f"  ANALYSIS: {ev['text'][:110]}")
        elif t == "researcher_stop":
            print(f"  STOPPED: {ev.get('rationale','')[:110]}")

    loop_obj = loop.GliaLoop(llm, SimConfig(), budget=6, supervisor_every=2,
                             seeds=[0, 1, 2, 3], emit=emit)
    result = loop_obj.run()

    print("\n--- summary ---")
    if result.best:
        b = result.best
        base = result.baseline_metrics["mean_e2e"]
        imp = (base - b.metrics["mean_e2e"]) / base * 100
        print(f"baseline mean_e2e = {base:.0f}")
        print(f"best     mean_e2e = {b.metrics['mean_e2e']:.0f}  (iteration {b.iteration}, +{imp:.0f}%)")
        print(f"best completion   = {b.metrics['completion_rate']*100:.0f}%")
        print(f"valid candidates  = {sum(1 for c in result.candidates if c.valid)}/{len(result.candidates)}")
        # did it reach the 'clean arc' bar? (roughly: sub-1000 latency, >80% done)
        clean = b.metrics["mean_e2e"] < 1000 and b.metrics["completion_rate"] > 0.80
        print(f"\nReached clean-arc quality: {'YES' if clean else 'NO -- may need another pass'}")
    else:
        print("no valid scheduler found")

if __name__ == "__main__":
    main()
