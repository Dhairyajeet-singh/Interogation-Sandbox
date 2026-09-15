"""
Deepseek.py   -  shared API client for stages 10 and 11
--------------------------------------------------------
One place that knows how to talk to DeepSeek, so the generator and the
judge cannot drift apart.

The API is OpenAI-compatible, so the openai SDK works against it with a
different base_url. The key comes from the DEEPSEEK_API_KEY environment
variable - never hard-code it, never commit it.

    setx DEEPSEEK_API_KEY sk-...        (Windows, then reopen the shell)
    export DEEPSEEK_API_KEY=sk-...      (Linux / macOS)

WHY THERE IS AN OFFLINE MODE
----------------------------
Everything in stages 10 to 12 has to be testable, reproducible, and
demoable without a network. A dead API or an exhausted quota during a
demo must not take the project down. So every call has a deterministic
offline fallback, and any artefact it produces is stamped
generated_by="offline" so it can never be mistaken for real model output.

NOTES ON deepseek-reasoner
--------------------------
It is the reasoning model, so it is slower and pricier than
deepseek-chat, and it has quirks the client handles:
  - the reply has both reasoning_content and content; only content is the
    answer
  - some deployments reject temperature / top_p / response_format, so the
    client retries without them rather than failing
  - it likes to wrap JSON in markdown fences, which strip_json removes
"""

from __future__ import annotations

import json
import os
import re
import time

DEFAULT_MODEL = "deepseek-reasoner"
BASE_URL = "https://api.deepseek.com"


class DeepseekError(RuntimeError):
    pass


def have_key() -> bool:
    return bool(os.environ.get("DEEPSEEK_API_KEY", "").strip())


def strip_json(text: str) -> str:
    """
    Pull the JSON object out of a reply.

    Models wrap JSON in ```json fences, add a sentence before it, or both.
    Takes the outermost {...} span, which survives all of that.
    """
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise DeepseekError(f"no JSON object in reply: {text[:200]!r}")
    return text[start:end + 1]


class Deepseek:
    """
    Thin wrapper. `offline=True`, or simply no API key, routes every call
    to the caller-supplied fallback instead.
    """

    def __init__(self, model=DEFAULT_MODEL, offline=None, timeout=600,
                 max_retries=3, verbose=True):
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.verbose = verbose
        self.offline = (not have_key()) if offline is None else offline
        self.calls = 0
        self.last_reasoning = None
        self._client = None

        if self.verbose:
            where = "OFFLINE (no DEEPSEEK_API_KEY)" if self.offline else self.model
            print(f"[deepseek] {where}")

    # ------------------------------------------------------------------
    def _sdk(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=os.environ["DEEPSEEK_API_KEY"],
                base_url=BASE_URL,
                timeout=self.timeout,
            )
        return self._client

    def chat(self, system: str, user: str, json_mode=True, temperature=None):
        """
        One completion. Returns the assistant's text.

        Retries on transient failures, and retries once without the
        optional parameters if the model rejects them - deepseek-reasoner
        does not accept all of them on every deployment.
        """
        if self.offline:
            raise DeepseekError("offline: no API call was made")

        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]

        kwargs = {"model": self.model, "messages": messages}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if temperature is not None:
            kwargs["temperature"] = temperature

        last = None
        for attempt in range(self.max_retries):
            try:
                res = self._sdk().chat.completions.create(**kwargs)
                self.calls += 1
                msg = res.choices[0].message
                self.last_reasoning = getattr(msg, "reasoning_content", None)
                return msg.content
            except Exception as exc:
                last = exc
                text = str(exc).lower()
                # the model refused an optional parameter: drop them and retry
                if ("response_format" in text or "temperature" in text
                        or "unsupported" in text or "not support" in text):
                    kwargs.pop("response_format", None)
                    kwargs.pop("temperature", None)
                    if self.verbose:
                        print("[deepseek] retrying without optional params")
                    continue
                if attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    if self.verbose:
                        print(f"[deepseek] {type(exc).__name__}, retry in {wait}s")
                    time.sleep(wait)
                    continue
        raise DeepseekError(f"failed after {self.max_retries} attempts: {last}")

    def chat_json(self, system: str, user: str, fallback=None, **kw):
        """
        chat() that parses the reply as JSON.

        If we are offline, or the call fails, and a fallback callable was
        supplied, use it. Returns (data, source) where source is
        "deepseek-..." or "offline", so callers can stamp their output
        honestly.
        """
        if self.offline:
            if fallback is None:
                raise DeepseekError("offline and no fallback supplied")
            return fallback(), "offline"

        try:
            raw = self.chat(system, user, **kw)
            return json.loads(strip_json(raw)), self.model
        except Exception as exc:
            if fallback is None:
                raise
            if self.verbose:
                print(f"[deepseek] falling back offline: {exc}")
            return fallback(), "offline"