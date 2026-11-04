"""
The one prompt every system is scored with -- the fine-tuned model, the base
model it started from, and the large API model. Same words, same rules.

The catalog is rendered one line per tool instead of as JSON schema: about a
third of the tokens, which matters for a 0.5B model's context and for the
free-tier token budget of the API baseline.
"""
from __future__ import annotations

import datetime as dt

from .tools import TOOLS

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
            "Saturday", "Sunday"]


def _sig(t: dict) -> str:
    parts = []
    for k, p in t["parameters"].items():
        if "enum" in p:
            kind = "|".join(p["enum"])
        elif p["description"].startswith(("date as", "24-hour")):
            kind = p["description"].split(" as ")[-1]
        else:
            kind = p["type"]
        parts.append(f"{k}: {kind}")
    return f"{t['name']}({', '.join(parts)}) - {t['description']}"


CATALOG = "\n".join(_sig(t) for t in TOOLS)

RULES = """Rules:
1. Reply with exactly one JSON object and nothing else.
2. If one tool fits and every argument is stated, reply
   {"tool": "<name>", "arguments": {<every argument>}}.
3. If no tool fits, or any argument is not stated, reply {"tool": null}.
   Never guess a missing value.
4. Dates are YYYY-MM-DD. "tomorrow" is today plus 1 day, "the day after
   tomorrow" is today plus 2. A weekday name means the next such day after
   today, never today itself. "15 December" means the next 15 December on or
   after today.
5. Times are 24-hour HH:MM.
6. Copy names, places, messages and notes exactly as the user wrote them."""


def system_prompt(today: str) -> str:
    d = dt.date.fromisoformat(today)
    return (f"You are a phone assistant that turns a request into one tool call.\n"
            f"Today is {WEEKDAYS[d.weekday()]}, {today}.\n\n"
            f"Tools:\n{CATALOG}\n\n{RULES}")


def messages(example: dict) -> list[dict]:
    return [{"role": "system", "content": system_prompt(example["today"])},
            {"role": "user", "content": example["query"]}]
