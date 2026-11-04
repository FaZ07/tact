"""
Minimal OpenAI-compatible chat client on urllib -- Groq, a local llama.cpp
server and Ollama all speak this. No SDK, no dependency.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request


def chat(base_url: str, model: str, messages: list[dict], key: str = "",
         max_tokens: int = 160, retries: int = 6) -> str:
    body = json.dumps({"model": model, "messages": messages, "temperature": 0,
                       "max_tokens": max_tokens}).encode()
    headers = {"Content-Type": "application/json", "User-Agent": "kutti/0.1"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    url = base_url.rstrip("/") + "/chat/completions"
    wait = 5.0
    for attempt in range(retries):
        req = urllib.request.Request(url, data=body, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"] or ""
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503) or attempt == retries - 1:
                raise
            after = e.headers.get("retry-after")
            time.sleep(float(after) if after and after.replace(".", "").isdigit()
                       else wait)
            wait *= 2
    raise RuntimeError("unreachable")
