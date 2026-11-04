"""
Keyword router: the floor every model must clear. Its keywords are the
content words of each tool's own name and description -- nothing read off
the templates -- so it is not tuned against the test set.

It only routes; it never fills arguments, so its exact-match score on calls
is zero by construction. What it shows is what keyword routing does on the
near_miss and missing_arg cases: it fires actions.
"""
from __future__ import annotations

import json
import re

from .tools import TOOLS

_STOP = {"a", "an", "the", "for", "of", "on", "to", "by", "its", "in", "and",
         "or", "from", "one", "another", "right", "now", "something", "user",
         "between", "two", "into", "city", "date", "with", "at", "is"}


def _words(s: str) -> list[str]:
    return [w.rstrip("s") for w in re.findall(r"[a-z0-9]+", s.casefold())]


KEYWORDS = {t["name"]: sorted({w for w in _words(t["name"].replace("_", " ") + " "
                                                 + t["description"])
                               if w not in _STOP and len(w) > 2})
            for t in TOOLS}


def route(query: str) -> str | None:
    q = set(_words(query))
    best, best_hits = None, 0
    for t in TOOLS:  # catalog order breaks ties
        hits = sum(1 for k in KEYWORDS[t["name"]] if k in q)
        if hits > best_hits:
            best, best_hits = t["name"], hits
    return best


def predict(rows: list[dict]) -> dict[str, str]:
    out = {}
    for r in rows:
        tool = route(r["query"])
        out[r["id"]] = json.dumps({"tool": tool} if tool is None
                                  else {"tool": tool, "arguments": {}})
    return out
