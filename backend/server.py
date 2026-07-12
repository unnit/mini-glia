"""
Mini-Glia web server.

Serves three things:
  GET  /              -> the dashboard (single-file HTML)
  GET  /api/replay    -> the captured replay.json (so the page auto-loads it)
  GET  /api/run       -> a LIVE agent run, streamed as Server-Sent Events

The replay path is the safe default: it always works, needs no API key, and is
what a first-time visitor sees. The live path runs the real Gemini-backed agent
and streams its reasoning + simulator frames as they happen -- gated behind the
"run it live" button so a cold visit never depends on the model or the network.

Run locally:
    cd backend
    GEMINI_API_KEY=... uvicorn server:app --reload --port 8000
Then open http://localhost:8000
"""
from __future__ import annotations

import asyncio
import json
import os
import queue
import threading
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from sim.simulator import SimConfig, Simulator
from sim.schedulers import fcfs_greedy
from agent.llm import LLMClient
from agent.loop import GliaLoop
from agent.sandbox import make_safe_scheduler

app = FastAPI(title="Mini-Glia")

HERE = Path(__file__).resolve().parent
FRONTEND = HERE.parent / "frontend" / "index.html"
REPLAY = HERE / "runs" / "replay.json"


# ---- shape a live-run trace: reuse capture.py's transform so the server and
# the captured-replay path produce IDENTICAL frame structures (the raw sim
# trace stores per-step snapshots under "steps"; both must convert to "frames").
from agent.capture import _shape_trace  # noqa: E402


def _finite(obj):
    """Same JSON-safety pass capture.py uses: no Infinity/NaN in the stream."""
    if isinstance(obj, float):
        if obj == float("inf"):
            return 1e12
        if obj == float("-inf"):
            return -1e12
        if obj != obj:
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_finite(v) for v in obj]
    return obj


@app.get("/", response_class=HTMLResponse)
def index():
    if FRONTEND.exists():
        return FRONTEND.read_text()
    return HTMLResponse("<h1>frontend/index.html not found</h1>", status_code=404)


@app.get("/api/replay")
def replay():
    if not REPLAY.exists():
        return JSONResponse({"error": "no replay.json captured yet"}, status_code=404)
    # already standards-compliant if produced by the fixed capture.py, but pass
    # through _finite anyway so an older file with Infinity still serves cleanly.
    data = json.loads(REPLAY.read_text())
    return JSONResponse(_finite(data))


@app.get("/api/run")
def run_live():
    """Run the agent live and stream events as SSE. Falls back with a clear
    error event if no API key / model is configured."""
    model = os.environ.get("MODEL", "gemini/gemini-3.1-flash-lite")

    def event_stream():
        q: "queue.Queue[dict | None]" = queue.Queue()

        def emit(ev):
            q.put(_finite(ev))

        def worker():
            try:
                llm = LLMClient(model=model)
                cfg = SimConfig()
                loop = GliaLoop(llm, cfg, budget=6, supervisor_every=2,
                                seeds=[0, 1, 2, 3], emit=emit)
                result = loop.run()

                # after the loop, emit traced baseline + the agent's best
                # ATTEMPT so the UI can animate both panels. Emit the best trace
                # even when it did NOT beat baseline -- otherwise a no-improvement
                # run leaves the panels empty. We separately signal whether it
                # beat baseline so the UI can label it honestly.
                base_sim = Simulator(cfg, trace=True)
                base_m = base_sim.run(fcfs_greedy, {})
                emit({"type": "trace", "which": "baseline", "trace": _shape_trace(base_m)})

                beat = False
                if result.best:
                    beat = result.best.metrics.get("mean_e2e", 1e12) < base_m.get("mean_e2e", 1e12)
                    try:
                        sched = make_safe_scheduler(result.best.code)
                        win_sim = Simulator(cfg, trace=True)
                        win_m = win_sim.run(sched, result.best.params)
                        emit({"type": "trace", "which": "composed", "trace": _shape_trace(win_m)})
                    except Exception as te:
                        emit({"type": "warning", "text": f"could not trace best scheduler: {te}"})

                # explicit end-of-run summary so the UI never shows empty panels
                emit({"type": "done", "beats_baseline": beat,
                      "best": {
                          "iteration": result.best.iteration if result.best else None,
                          "hypothesis": result.best.hypothesis if result.best else None,
                          "code": result.best.code if result.best else None,
                          "params": result.best.params if result.best else {},
                          "metrics": result.best.metrics if result.best else None,
                      } if result.best else None})

                if getattr(llm, "rate_limit_hits", 0) > 0:
                    emit({"type": "warning", "text": "rate limited during run; results may be partial"})
            except Exception as e:  # never hang the stream
                emit({"type": "error", "text": f"{type(e).__name__}: {e}"})
            finally:
                q.put(None)  # sentinel: stream complete

        threading.Thread(target=worker, daemon=True).start()

        while True:
            ev = q.get()
            if ev is None:
                yield "event: end\ndata: {}\n\n"
                break
            yield f"data: {json.dumps(ev)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
