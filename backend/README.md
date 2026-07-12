# Mini-Glia — backend

A miniature reproduction of Glia (arXiv:2510.27176): a Researcher + Supervisor
agent loop that discovers a batch-scheduling policy for an LLM-inference GPU
cluster, grounded in a toy simulator that faithfully reproduces the KV-cache /
restart dynamic from the paper.

This is the backend. It runs the agent loop, evaluates agent-written schedulers
in the simulator, and streams the reasoning + an explainability trace.

## What's here

    backend/
      sim/
        simulator.py     # discrete-event LLM-inference sim (KV cache, restarts,
                         #   head-of-line blocking) + explainability tracing
        schedulers.py    # reference schedulers + validation of the discovery arc
      agent/
        sandbox.py       # safe execution of agent-written scheduler code
        llm.py           # LiteLLM wrapper (Ollama / Gemini / mock) + JSON parsing
        loop.py          # the Researcher + Supervisor loop (the core)
        capture.py       # run once -> replay JSON for the public demo
        test_loop_offline.py  # end-to-end test with a mocked LLM (no network)
      runs/
        replay.json      # a captured deterministic run (regenerate any time)
      requirements.txt

## Setup (macOS, Apple Silicon)

    cd backend
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

Install Ollama and pull a code-capable model (14b is noticeably better at
writing schedulers if you have the RAM; 7b works):

    # from https://ollama.com
    ollama pull qwen2.5-coder:14b
    ollama serve   # if not already running

## Verify everything works (no model needed)

    # 1. sandbox safety checks
    python agent/sandbox.py

    # 2. the discovery arc: FCFS -> headroom -> composed
    python sim/schedulers.py
    #    expect: "FAITHFUL: headroom helps AND composition wins"

    # 3. full agent loop, end-to-end, with a scripted mock LLM
    python agent/test_loop_offline.py
    #    expect: "ALL CHECKS PASSED"

## Run the agent live against Ollama

    MODEL=ollama/qwen2.5-coder:14b python -m agent.capture --out runs/replay_live.json

This runs the real two-agent loop. The model will propose its own schedulers,
so the exact path will differ from the scripted mock — that's expected. If the
model writes code the sandbox rejects, the loop feeds the error back and asks
it to retry. A good run rediscovers headroom and (nudged by the Supervisor)
composes it with prefill ordering.

For a guaranteed-clean deterministic capture (what the public URL should serve):

    python -m agent.capture --mock --out runs/replay.json

## Model configuration

The `MODEL` env var selects the backend via LiteLLM:

    MODEL=ollama/qwen2.5-coder:14b          # local dev
    MODEL=gemini/gemini-flash-lite-latest   # public (set GEMINI_API_KEY)
    MODEL=mock                              # scripted, offline

Copy `.env.example` to `.env` and fill in as needed.

## A note on fidelity

The simulator is a *minimal reproduction* of one dynamic (KV-cache growth →
OOM → youngest-request eviction/restart), plus head-of-line blocking from
compute-heavy prefill. It is tuned so the paper's insight is discoverable and
robust across seeds. It is NOT a physically faithful cluster simulator: no real
GPU compute, no paging/fragmentation, no networking, no tensor parallelism.
Describe it that way — the mechanism is honest, the specific numbers are a toy
model's.
