"""
LLM client -- a thin wrapper over LiteLLM so the agent code never hard-codes a
provider. Dev runs against local Ollama; the public demo runs against Gemini
Flash-Lite free tier. Swapping is a matter of the MODEL env var, nothing else.

    MODEL=ollama/qwen2.5-coder:14b        # local dev on the Mac
    MODEL=gemini/gemini-flash-lite-latest  # public URL (free tier)

A MOCK mode (MODEL=mock) lets us unit-test the whole agent loop with no network
and no model -- it replays a scripted sequence of "LLM" responses. That is how
this file was validated in an offline container; on your Mac you'll set MODEL
to a real Ollama tag.
"""

from __future__ import annotations

import json
import os
import time
from typing import Callable

# Quiet LiteLLM's verbose logging. It prints a "Give Feedback / Get Help" block
# on every caught error (e.g. transient rate-limit 429s), which floods the
# terminal even though our own retry logic handles them. We keep our retries
# and just silence the library's chatter.
os.environ.setdefault("LITELLM_LOG", "ERROR")
try:
    import litellm
    litellm.suppress_debug_info = True
    litellm.set_verbose = False
    import logging
    logging.getLogger("LiteLLM").setLevel(logging.ERROR)
except Exception:
    pass  # litellm not installed (offline mock mode) -- fine


class LLMClient:
    # minimum seconds between real API calls, to stay under free-tier
    # per-minute rate limits (e.g. Gemini free tier ~15 RPM -> ~4s spacing).
    # Set MIN_CALL_INTERVAL=0 to disable (e.g. for local Ollama, no limit).
    def __init__(self, model: str | None = None, temperature: float = 0.4):
        self.model = model or os.environ.get("MODEL", "mock")
        self.temperature = temperature
        self._mock_script: list[str] = []
        self._mock_i = 0
        self._last_call_ts = 0.0
        self.rate_limit_hits = 0  # counts rate-limit/quota rejections this session
        # spacing: default 4s for gemini free tier, 0 for ollama/mock
        default_spacing = 4.0 if self.model.startswith("gemini/") else 0.0
        self._min_interval = float(os.environ.get("MIN_CALL_INTERVAL", default_spacing))

    # ---- mock support (offline testing) --------------------------------
    def load_mock_script(self, responses: list[str]):
        """Provide a list of canned assistant responses, consumed in order."""
        self._mock_script = responses
        self._mock_i = 0

    def _mock_complete(self) -> str:
        if self._mock_i < len(self._mock_script):
            r = self._mock_script[self._mock_i]
            self._mock_i += 1
            return r
        # default terminal response
        return json.dumps({"action": "stop", "rationale": "mock exhausted"})

    # ---- real completion ------------------------------------------------
    def complete(self, system: str, messages: list[dict],
                 temperature: float | None = None) -> str:
        """Return the assistant text for a chat completion.

        Robust across providers: some (notably Gemini) can return a null
        content field when a safety filter trips or the model emits only a
        tool call. We coerce that to a clear, parseable message instead of
        letting a None propagate and crash JSON extraction downstream."""
        if self.model == "mock":
            return self._mock_complete()

        # import here so the module loads even if litellm isn't installed
        # (e.g. in the offline container running mock tests)
        from litellm import completion

        # rate-limit spacing: wait if we called too recently
        if self._min_interval > 0:
            elapsed = time.time() - self._last_call_ts
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)

        full_messages = [{"role": "system", "content": system}] + messages

        last_exc = None
        for attempt in range(3):
            self._last_call_ts = time.time()
            try:
                resp = completion(
                    model=self.model,
                    messages=full_messages,
                    temperature=self.temperature if temperature is None else temperature,
                )
            except Exception as e:  # transient network / rate-limit
                last_exc = e
                # detect rate-limit / quota errors specifically so the run can
                # be flagged as unreliable rather than silently degrading.
                err_str = f"{type(e).__name__} {e}".lower()
                is_rate_limit = any(k in err_str for k in
                                    ("ratelimit", "rate limit", "429",
                                     "quota", "resource_exhausted", "resource exhausted"))
                if is_rate_limit:
                    self.rate_limit_hits += 1
                    # longer backoff for rate limits (per-minute windows)
                    if attempt < 2:
                        time.sleep(max(self._min_interval, 15) * (attempt + 1))
                        continue
                    return ('{"action": "analyze", "analysis": "RATE LIMITED: the '
                            'API rejected this call (quota/rate limit). This run is '
                            'unreliable."}')
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
                    continue
                # non-rate-limit failure
                return '{"action": "analyze", "analysis": "LLM call failed: ' \
                       f'{type(e).__name__}. Retrying next turn."' + "}"

            choice = resp["choices"][0]
            content = choice["message"].get("content")
            if content:
                return content

            # content is None/empty -- figure out why and return a usable string
            finish = choice.get("finish_reason", "unknown")
            # a blocked/empty response: tell the model to try a plainer approach
            return ('{"action": "analyze", "analysis": "Previous response was '
                    f'empty (finish_reason={finish}); I will restate my next '
                    'scheduler proposal in plain text without any blocked '
                    'content."}')

        # should not reach here, but never return None
        return '{"action": "stop", "rationale": "LLM unavailable"}'


def extract_json(text: str) -> dict:
    """LLMs wrap JSON in prose or ```json fences. Pull out the first JSON object
    robustly. Returns {} if nothing parseable is found (or input is empty)."""
    if not text:
        return {}
    # strip code fences
    cleaned = text.replace("```json", "```").strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        # take the longest fenced block that looks like JSON
        candidates = [p for p in parts if "{" in p and "}" in p]
        if candidates:
            cleaned = max(candidates, key=len)
    # find the outermost braces
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {}
    blob = cleaned[start:end + 1]
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        # last resort: try to fix trailing commas
        try:
            return json.loads(blob.replace(",}", "}").replace(",]", "]"))
        except json.JSONDecodeError:
            return {}


def extract_code(text: str) -> str:
    """Pull a python code block out of an LLM response. Falls back to the
    largest fenced block, then to the raw text if it defines schedule().
    Returns "" for empty/None input."""
    if not text:
        return ""
    if "```python" in text:
        block = text.split("```python", 1)[1].split("```", 1)[0]
        return block.strip()
    if "```" in text:
        parts = text.split("```")
        code_like = [p for p in parts if "def schedule" in p]
        if code_like:
            return code_like[0].strip()
        # largest block
        blocks = parts[1::2]
        if blocks:
            return max(blocks, key=len).strip()
    if "def schedule" in text:
        return text.strip()
    return ""


def sanitize_code(code: str) -> str:
    """Clean up the `code` field from an LLM's JSON before it hits the sandbox.

    Weaker models make a recurring set of format mistakes even when told not to.
    This repairs the ones we've actually observed, so a good IDEA isn't thrown
    away over a formatting slip:
      * markdown fences embedded inside the JSON string
      * prose or explanation before the actual `def schedule`
      * a stray `import json` / `json.dumps(...)` wrapper (the model sometimes
        re-serialises its own output inside the code field)
      * leading/trailing whitespace and blank lines

    Returns cleaned source, or "" if no `schedule` function can be recovered
    (in which case the loop will ask the model to re-propose)."""
    if not code:
        return ""

    # 1. strip markdown fences if the model wrapped the code again
    if "```" in code:
        # prefer a python-fenced block, else the largest fenced block
        if "```python" in code:
            code = code.split("```python", 1)[1].split("```", 1)[0]
        else:
            parts = code.split("```")
            fenced = parts[1::2]
            if fenced:
                code = max(fenced, key=len)

    # 2. drop everything before the first `def schedule` (removes prose /
    #    stray statements like `import json` that crash compilation)
    idx = code.find("def schedule")
    if idx > 0:
        code = code[idx:]

    # 3. if there is no schedule def at all, we can't use it
    if "def schedule" not in code:
        return ""

    # 4. remove any line that references `json` (the observed NameError) or
    #    is a bare import -- these are never part of a legitimate scheduler
    cleaned_lines = []
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            continue
        if "json." in line or stripped == "json":
            continue
        cleaned_lines.append(line)
    code = "\n".join(cleaned_lines)

    return code.strip()
