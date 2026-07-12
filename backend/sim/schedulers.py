"""
Reference schedulers + validation.

These are hand-written to CONFIRM the simulator rewards the insight we want the
agent to discover. If shortest-prefill-first + KV headroom does not beat FCFS
here, the simulator is not faithful to the paper and must be fixed BEFORE any
LLM work. This file is our correctness anchor.
"""

from sim.simulator import SimConfig, run_scheduler, PendingView, GpuView


# --- Baseline 1: FCFS, greedy fill (closest to naive vLLM/Sarathi) ---------
def fcfs_greedy(pending: list[PendingView], gpus: list[GpuView], now: int, params: dict):
    """Admit oldest-waiting requests, packing each GPU as full as it will go.
    No memory headroom -> should trigger restarts under load."""
    order = sorted(pending, key=lambda p: p.waiting_since)
    decisions = {g.gpu_id: [] for g in gpus}
    gi = {g.gpu_id: g for g in gpus}
    # round-robin-ish: send to whichever GPU has the fewest active
    for p in order:
        target = min(gpus, key=lambda g: (len(decisions[g.gpu_id]) + g.num_active))
        g = gi[target.gpu_id]
        if len(decisions[g.gpu_id]) + g.num_active < g.max_batch and g.free_blocks > 0:
            decisions[g.gpu_id].append(p.rid)
    return decisions


# --- Baseline 2: least-loaded queue (LLQ), still no headroom ---------------
def llq(pending: list[PendingView], gpus: list[GpuView], now: int, params: dict):
    order = sorted(pending, key=lambda p: p.waiting_since)
    decisions = {g.gpu_id: [] for g in gpus}
    active_count = {g.gpu_id: g.num_active for g in gpus}
    for p in order:
        target = min(gpus, key=lambda g: active_count[g.gpu_id] + len(decisions[g.gpu_id]))
        if (active_count[target.gpu_id] + len(decisions[target.gpu_id]) < target.max_batch
                and target.free_blocks > 0):
            decisions[target.gpu_id].append(p.rid)
    return decisions


# --- Intermediate step: headroom WITHOUT prefill ordering ------------------
def headroom_only(pending: list[PendingView], gpus: list[GpuView], now: int, params: dict):
    """KV headroom, but admit in arrival order (no shortest-prefill). This is
    the 'first insight' -- it cuts restarts but leaves head-of-line blocking on
    the table, which composing with shortest-prefill later recovers."""
    m = params.get("headroom", 0.35)
    order = sorted(pending, key=lambda p: p.waiting_since)   # arrival order
    decisions = {g.gpu_id: [] for g in gpus}
    virt_free = {g.gpu_id: g.free_blocks for g in gpus}
    for p in order:
        cands = [g for g in gpus if virt_free[g.gpu_id] - m * g.blocks_total > 1]
        if not cands:
            continue
        target = max(cands, key=lambda g: virt_free[g.gpu_id])
        decisions[target.gpu_id].append(p.rid)
        virt_free[target.gpu_id] -= 1
    return decisions


# --- The "discovered" answer: shortest-prefill-first + KV headroom ---------
def headroom_spf(pending: list[PendingView], gpus: list[GpuView], now: int, params: dict):
    """Glia's COMPOSED insight, hand-implemented:
      1. Reserve a KV memory margin `m` (headroom) -> avoid restarts.
      2. Admit SHORTER prefills first (approx SJF) -> cut head-of-line blocking.
    Neither alone is best: shortest-prefill without headroom just crams GPUs
    and restarts more. Together they beat headroom alone -- the composition the
    Supervisor nudges the Researcher toward."""
    m = params.get("headroom", 0.35)
    order = sorted(pending, key=lambda p: p.prefill_tokens)   # shortest first
    decisions = {g.gpu_id: [] for g in gpus}
    virt_free = {g.gpu_id: g.free_blocks for g in gpus}
    for p in order:
        cands = [g for g in gpus if virt_free[g.gpu_id] - m * g.blocks_total > 1]
        if not cands:
            continue
        target = max(cands, key=lambda g: virt_free[g.gpu_id])
        decisions[target.gpu_id].append(p.rid)
        virt_free[target.gpu_id] -= 1
    return decisions


def _fmt(name, m):
    if "error" in m:
        return f"  {name:22s}  ERROR: {m['error']}"
    return (f"  {name:22s}  mean_e2e={m['mean_e2e']:8.1f}  "
            f"p99={m['p99_e2e']:8.1f}  restarts={m['restart_fraction']:.3f}  "
            f"queue={m['mean_queueing']:6.1f}  done={m['completion_rate']*100:5.1f}%")


if __name__ == "__main__":
    cfg = SimConfig()
    seeds = [0, 1, 2, 3, 4]
    print(f"\nWorkload: {cfg.num_requests} reqs, {cfg.num_gpus} GPUs, "
          f"{cfg.blocks_per_gpu} blocks/GPU, {seeds} seeds\n")

    results = {
        "FCFS greedy": run_scheduler(fcfs_greedy, cfg, {}, seeds),
        "LLQ": run_scheduler(llq, cfg, {}, seeds),
        "Headroom only": run_scheduler(headroom_only, cfg, {"headroom": 0.35}, seeds),
        "Headroom+SPF (composed)": run_scheduler(headroom_spf, cfg, {"headroom": 0.35}, seeds),
    }
    for name, m in results.items():
        print(_fmt(name, m))

    # the claims we must verify: (1) headroom beats baseline, (2) composition
    # (headroom + shortest-prefill) beats headroom alone -- the paper's arc.
    base = results["FCFS greedy"]
    ho = results["Headroom only"]
    comp = results["Headroom+SPF (composed)"]
    if all("error" not in r for r in (base, ho, comp)):
        imp_ho = (base["mean_e2e"] - ho["mean_e2e"]) / base["mean_e2e"] * 100
        imp_comp = (base["mean_e2e"] - comp["mean_e2e"]) / base["mean_e2e"] * 100
        comp_vs_ho = (ho["mean_e2e"] - comp["mean_e2e"]) / ho["mean_e2e"] * 100
        print(f"\n  headroom vs baseline:        {imp_ho:+.0f}%")
        print(f"  composed vs baseline:        {imp_comp:+.0f}%")
        print(f"  composed vs headroom-alone:  {comp_vs_ho:+.0f}%")
        ok = imp_ho > 15 and comp_vs_ho > 5
        print(f"  => Sim is {'FAITHFUL: headroom helps AND composition wins' if ok else 'NOT showing the composition effect -- TUNE'}")
        if not ok:
            import sys
            sys.exit(1)  # fail CI: the simulator no longer rewards the discovery arc
