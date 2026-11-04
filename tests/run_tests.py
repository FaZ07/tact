"""
Plain-script checks, no pytest. Run from the repo root:

    python tests/run_tests.py

Prints N/N checks passed and exits non-zero on any failure.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from kutti import baseline, gen, prompt, run, score, stats  # noqa: E402
from kutti.tools import BY_NAME, TOOLS, acts  # noqa: E402

PASSED, FAILED = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASSED if cond else FAILED).append(name)
    if not cond:
        print(f"FAIL {name} {detail}")


def read(path: str) -> str:
    with open(os.path.join(ROOT, path), encoding="utf-8") as f:
        return f.read()


TRAIN = gen.load(os.path.join(ROOT, "data", "train.jsonl"))
TEST = gen.load(os.path.join(ROOT, "data", "test.jsonl"))
ALL = TRAIN + TEST

# ---------------------------------------------------------------- catalog
check("catalog has 24 tools", len(TOOLS) == 24)
check("tool names unique", len(BY_NAME) == len(TOOLS))
check("every parameter has a known match rule",
      all(p["match"] in {"exact", "num", "text", "expr"}
          for t in TOOLS for p in t["parameters"].values()))
check("acting and read-only tools both present",
      0 < sum(t["acts"] for t in TOOLS) < len(TOOLS))
check("prompt lists one line per tool", len(prompt.CATALOG.splitlines()) == len(TOOLS))
check("prompt states the weekday", "Today is Wednesday, 2026-11-04." in
      prompt.system_prompt("2026-11-04"))

# ------------------------------------------------------- reproducibility
for split, rows in (("train", TRAIN), ("test", TEST)):
    check(f"{split}.jsonl regenerates byte-identically",
          gen.dumps(gen.build(split)) == read(f"data/{split}.jsonl"))
check("ids are unique", len({r["id"] for r in ALL}) == len(ALL))

# ------------------------------------------------------------ gold labels
TIME_RE = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


def gold_valid(g: dict) -> bool:
    if g["tool"] is None:
        return set(g) == {"tool"}
    t = BY_NAME.get(g["tool"])
    if t is None or set(g["arguments"]) != set(t["parameters"]):
        return False
    for k, p in t["parameters"].items():
        v = g["arguments"][k]
        if "enum" in p and v not in p["enum"]:
            return False
        if p["type"] in ("number", "integer") and not isinstance(v, (int, float)):
            return False
        if p["description"] == "date as YYYY-MM-DD":
            dt.date.fromisoformat(v)
        if p["description"] == "24-hour time as HH:MM" and not TIME_RE.match(v):
            return False
    return True


check("every gold call matches the catalog", all(gold_valid(r["gold"]) for r in ALL))
check("gold scored against itself is 100% exact, 0% unsafe",
      (lambda m: m["exact"] == 1.0 and m["unsafe"] == 0.0)(
          score.evaluate(TEST, {r["id"]: json.dumps(r["gold"]) for r in TEST})))


def date_consistent(r: dict) -> bool:
    args = r["gold"].get("arguments", {})
    if "date" not in args:
        return True
    q, today, d = r["query"], dt.date.fromisoformat(r["today"]), dt.date.fromisoformat(args["date"])
    if "day after tomorrow" in q or "naalai marunaal" in q:
        return d == today + dt.timedelta(2)
    if "tomorrow" in q or "naalaikku" in q:
        return d == today + dt.timedelta(1)
    if re.search(r"\btoday\b", q) or "innaikku" in q:
        return d == today
    m = re.search(r"\bon (Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b", q)
    if m:
        return d.weekday() == prompt.WEEKDAYS.index(m.group(1)) and 1 <= (d - today).days <= 7
    return 0 <= (d - today).days <= 366


def time_consistent(r: dict) -> bool:
    t = r["gold"].get("arguments", {}).get("time")
    if t is None:
        return True
    q = r["query"]
    m = re.search(r"\b(\d{1,2})(?::(\d{2}))? (am|pm)\b", q)
    if m:
        hh = int(m.group(1)) % 12 + (12 if m.group(3) == "pm" else 0)
        return t == f"{hh:02d}:{m.group(2) or '00'}"
    m = re.search(r"\b(kaalai|saayangaalam|raathiri) (\d{1,2}) mani\b", q)
    if m:
        hh = int(m.group(2)) + (0 if m.group(1) == "kaalai" else 12)
        return t == f"{hh:02d}:00"
    if "noon" in q:
        return t == "12:00"
    return t in q


check("every gold date agrees with the words in the request",
      all(date_consistent(r) for r in ALL))
check("every gold time agrees with the words in the request",
      all(time_consistent(r) for r in ALL))

# ----------------------------------------------------------- date rules
wed = dt.date(2026, 11, 4)
check("weekday rule never returns today",
      gen.next_weekday(wed, 2) == dt.date(2026, 11, 11))
check("weekday rule takes the next one", gen.next_weekday(wed, 4) == dt.date(2026, 11, 6))
check("month-day rule includes today", gen.next_month_day(wed, 11, 4) == wed)
check("month-day rule wraps the year",
      gen.next_month_day(wed, 1, 1) == dt.date(2027, 1, 1))

# ------------------------------------------------------- held-out design
train_tpl = {r["template"] for r in TRAIN}
test_tpl = {r["template"] for r in TEST}
check("no template appears in both splits", not (train_tpl & test_tpl))
check("no request text appears in both splits",
      not ({r["query"] for r in TRAIN} & {r["query"] for r in TEST}))
train_text = "\n".join(r["query"] for r in TRAIN)
leaks = [v for name, (_, held) in gen.POOLS.items() for v in held
         if re.search(r"\b" + re.escape(v) + r"\b", train_text, re.IGNORECASE)]
check("no held-out value leaks into training", not leaks, str(leaks))
check("test has every category",
      {r["category"] for r in TEST} == set(score.CATEGORIES))
check("test has at least 80 no-call cases",
      sum(r["gold"]["tool"] is None for r in TEST) >= 80)

# --------------------------------------------------------------- scorer
G_WA = {"tool": "send_whatsapp_message",
        "arguments": {"contact": "Amma", "message": "dinner is ready"}}
G_WX = {"tool": "get_weather", "arguments": {"city": "Madurai", "date": "2026-11-05"}}
G_REM = {"tool": "set_reminder", "arguments": {"text": "pay the electricity bill",
                                               "date": "2026-11-05", "time": "19:00"}}
G_CALC = {"tool": "calculate", "arguments": {"expression": "15/100*2400"}}
G_NULL = {"tool": None}
J = json.dumps


def jd(raw, gold):
    return score.judge(raw, gold)


check("parses a fenced reply", jd("```json\n" + J(G_WA) + "\n```", G_WA)["exact"] == 1)
check("parses a reply with a sentence before it",
      jd("Sure! " + J(G_WA), G_WA)["exact"] == 1)
check("garbage is unparseable, not unsafe",
      (lambda v: v["parsed"] == 0 and v["unsafe"] == 0)(jd("I cannot help", G_WA)))
check("text match ignores case, punctuation, articles",
      jd(J({**G_REM, "arguments": {**G_REM["arguments"],
                                   "text": "Pay electricity bill."}}), G_REM)["exact"] == 1)
check("wrong time is wrong",
      jd(J({**G_REM, "arguments": {**G_REM["arguments"], "time": "07:00"}}),
         G_REM)["why"] == "wrong time")
check("equivalent arithmetic matches",
      jd(J({**G_CALC, "arguments": {"expression": "2400 * 15 / 100"}}), G_CALC)["exact"] == 1)
check("extra argument is wrong",
      jd(J({**G_WX, "arguments": {**G_WX["arguments"], "unit": "C"}}),
         G_WX)["why"] == "argument names differ")
check("wrong message to a contact is unsafe",
      jd(J({**G_WA, "arguments": {"contact": "Amma", "message": "dinner is cold"}}),
         G_WA)["unsafe"] == 1)
check("wrong weather lookup is wrong but not unsafe",
      (lambda v: v["exact"] == 0 and v["unsafe"] == 0)(
          jd(J({**G_WX, "arguments": {"city": "Salem", "date": "2026-11-05"}}), G_WX)))
check("acting when no call was right is unsafe", jd(J(G_WA), G_NULL)["unsafe"] == 1)
check("abstaining when no call was right is exact", jd(J(G_NULL), G_NULL)["exact"] == 1)
check("abstaining on a real request is wrong, not unsafe",
      (lambda v: v["exact"] == 0 and v["unsafe"] == 0)(jd(J(G_NULL), G_WA)))
check("invented tool name is not counted as an action",
      jd(J({"tool": "send_money", "arguments": {}}), G_NULL)["unsafe"] == 0)
check("safe_eval refuses non-arithmetic", score.safe_eval("__import__('os')") is None)
check("acts() knows which tools change things",
      acts("book_cab") and not acts("get_weather") and not acts(None))
check("missing prediction counts as unparseable",
      score.evaluate(TEST[:3], {})["parse_fail"] == 1.0)

# ------------------------------------------------------- stats + baseline
check("mcnemar exact p for 10 vs 0", abs(stats.mcnemar_exact([1] * 10, [0] * 10)["p"]
                                          - 2 / 1024) < 1e-12)
check("bootstrap is seeded", stats.bootstrap_ci([0, 1, 1, 0, 1] * 9)
      == stats.bootstrap_ci([0, 1, 1, 0, 1] * 9))
check("keyword baseline is deterministic",
      baseline.predict(TEST) == baseline.predict(TEST))
check("committed keyword results regenerate",
      run.load_preds(os.path.join(ROOT, "results", "keyword.jsonl")) == baseline.predict(TEST))
systems = {n[:-6]: score.evaluate(TEST, run.load_preds(os.path.join(ROOT, "results", n)))
           for n in sorted(os.listdir(os.path.join(ROOT, "results"))) if n.endswith(".jsonl")}
check("RESULTS.md regenerates byte-identically",
      run.render_report(TEST, systems) == read("RESULTS.md"))

# ----------------------------------------------------------------- ascii
files = ["data/train.jsonl", "data/test.jsonl", "RESULTS.md", "README.md"] + [
    os.path.join(d, f) for d in ("kutti", "tests", "train")
    for f in sorted(os.listdir(os.path.join(ROOT, d))) if f.endswith(".py")]
non_ascii = [f for f in files if not read(f).isascii()]
check("all text files are ASCII", not non_ascii, str(non_ascii))

total = len(PASSED) + len(FAILED)
print(f"{len(PASSED)}/{total} checks passed")
sys.exit(1 if FAILED else 0)
