"""
Sandbox for executing agent-written scheduler code.

SECURITY MODEL
--------------
The Researcher agent emits Python source for a `schedule(...)` function. We must
run it, but untrusted LLM-generated code can be malicious or simply broken. We
defend with several layers:

  1. Static rejection of dangerous constructs (import, exec, eval, dunder access,
     open, etc.) before the code is ever compiled.
  2. Execution in a restricted global namespace: no real __builtins__, only a
     small allow-list of safe names the scheduler legitimately needs.
  3. A wall-clock timeout enforced in a worker thread, so an infinite loop in the
     scheduler cannot hang the server forever.
  4. The simulator itself already wraps scheduler calls in try/except, so a
     runtime error becomes a scored failure, not a crash.

This is defence-in-depth for a DEMO, not a claim of bulletproof isolation. For a
hostile multi-tenant setting you would run the code in a separate process with
seccomp / a container / gVisor. State that honestly if asked -- overclaiming
sandbox security to a systems audience is a credibility risk. For this app the
model is trusted-ish (it's your own Gemini/Ollama), the code only ever sees
read-only dataclasses, and the layers below stop the obvious foot-guns.
"""

from __future__ import annotations

import ast
import math
import threading
from typing import Callable


# Names the scheduler code is allowed to reference. Everything else is absent
# from its namespace, so `os`, `open`, `__import__` etc. simply don't exist.
_SAFE_BUILTINS = {
    "len": len, "range": range, "min": min, "max": max, "sum": sum,
    "sorted": sorted, "abs": abs, "round": round, "int": int, "float": float,
    "bool": bool, "list": list, "dict": dict, "set": set, "tuple": tuple,
    "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
    "any": any, "all": all, "reversed": reversed, "print": lambda *a, **k: None,
}

# Substrings that, if present in the source, cause immediate rejection.
# NOTE: `lambda` and `while` are intentionally NOT here. Lambda is needed for
# natural `sorted(key=...)` calls and cannot access anything outside the
# sandbox namespace. While-loops are a hang risk, but the per-call wall-clock
# timeout in make_safe_scheduler() is the real defence against that -- banning
# the keyword would also reject legitimate code and is easily circumvented by
# recursion anyway.
_FORBIDDEN_SUBSTRINGS = [
    "__", "import", "exec", "eval", "compile", "open(", "globals",
    "locals", "getattr", "setattr", "delattr", "vars(", "input(",
    "breakpoint",
]

# AST node types we refuse outright.
_FORBIDDEN_NODES = (
    ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal,
    ast.With, ast.AsyncWith, ast.AsyncFor, ast.AsyncFunctionDef, ast.Await,
)


class SandboxError(Exception):
    pass


def _static_check(src: str) -> None:
    lowered = src
    for bad in _FORBIDDEN_SUBSTRINGS:
        if bad in lowered:
            raise SandboxError(f"forbidden construct in code: {bad!r}")
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        raise SandboxError(f"syntax error: {e}")
    for node in ast.walk(tree):
        if isinstance(node, _FORBIDDEN_NODES):
            raise SandboxError(f"forbidden statement: {type(node).__name__}")
        # block attribute access to dunders (e.g. obj.__class__)
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise SandboxError(f"forbidden attribute access: {node.attr}")


def compile_scheduler(src: str) -> Callable:
    """Validate and compile agent code into a callable `schedule` function.
    Raises SandboxError on any problem."""
    _static_check(src)

    # the only module-level name we inject is `math` (safe, pure) plus builtins
    sandbox_globals: dict = {
        "__builtins__": _SAFE_BUILTINS,
        "math": math,
    }
    try:
        code = compile(src, "<agent_scheduler>", "exec")
        exec(code, sandbox_globals)  # defines schedule() in sandbox_globals
    except Exception as e:
        raise SandboxError(f"failed to load code: {type(e).__name__}: {e}")

    fn = sandbox_globals.get("schedule")
    if fn is None or not callable(fn):
        raise SandboxError("code did not define a callable `schedule(pending, gpus, now, params)`")
    return fn


def make_safe_scheduler(src: str, per_call_timeout: float = 2.0) -> Callable:
    """Return a scheduler wrapper enforcing a per-call wall-clock timeout.

    The simulator calls the scheduler many times; we bound each call. If any
    call exceeds the timeout we raise inside the sim (which scores it as a
    failed candidate) rather than letting a bad loop hang the whole run."""
    fn = compile_scheduler(src)

    def wrapped(pending, gpus, now, params):
        result = {}
        error = {}

        def target():
            try:
                result["value"] = fn(pending, gpus, now, params)
            except Exception as e:  # surface as a scored failure
                error["err"] = f"{type(e).__name__}: {e}"

        t = threading.Thread(target=target, daemon=True)
        t.start()
        t.join(per_call_timeout)
        if t.is_alive():
            raise SandboxError(f"scheduler call exceeded {per_call_timeout}s (likely infinite loop)")
        if "err" in error:
            raise SandboxError(error["err"])
        return result.get("value", {})

    return wrapped


# quick self-test
if __name__ == "__main__":
    # lambda in sort key is now ALLOWED (safe, and needed for natural code)
    good = """
def schedule(pending, gpus, now, params):
    m = params.get("headroom", 0.3)
    order = sorted(pending, key=lambda p: p.prefill_tokens)
    decisions = {g.gpu_id: [] for g in gpus}
    for p in order:
        cands = [g for g in gpus if g.free_blocks - m*g.blocks_total > 1]
        if not cands:
            continue
        target = max(cands, key=lambda g: g.free_blocks)
        decisions[target.gpu_id].append(p.rid)
    return decisions
"""
    fn = make_safe_scheduler(good)
    print("compiled lambda scheduler OK:", callable(fn))

    evil = "def schedule(p,g,n,pp):\n    import os\n    return {}"
    try:
        make_safe_scheduler(evil)
    except SandboxError as e:
        print("correctly rejected import:", e)

    dunder = "def schedule(p,g,n,pp):\n    return {}.__class__"
    try:
        make_safe_scheduler(dunder)
    except SandboxError as e:
        print("correctly rejected dunder:", e)

    # infinite loop -> should be killed by the per-call timeout
    loop = "def schedule(p,g,n,pp):\n    x=0\n    while True:\n        x+=1\n    return {}"
    fn = make_safe_scheduler(loop, per_call_timeout=0.5)
    try:
        fn([], [], 0, {})
    except SandboxError as e:
        print("correctly timed out infinite loop:", e)
