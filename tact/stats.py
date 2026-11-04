"""
Stdlib statistics, fully seeded. bootstrap_ci and mcnemar_exact are copied
from GroundTruth's eval/stats.py; cluster_bootstrap_ci is the same idea.

Why clusters: the eight test fills of one template share their phrasing, so
they share their errors. Resampling examples one at a time would treat ~70
distinct phrasings as ~500 independent draws and report an interval far
tighter than the evidence supports. Resampling whole templates does not.
"""
from __future__ import annotations

import math
import random


def bootstrap_ci(indicators: list[int], n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 7) -> tuple[float, float]:
    n = len(indicators)
    if n == 0:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = sorted(sum(indicators[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(n_boot))
    return (means[int((alpha / 2) * n_boot)],
            means[min(int((1 - alpha / 2) * n_boot), n_boot - 1)])


def cluster_bootstrap_ci(indicators: list[int], clusters: list[str],
                         n_boot: int = 2000, alpha: float = 0.05,
                         seed: int = 7) -> tuple[float, float]:
    if not indicators:
        return (0.0, 0.0)
    groups: dict[str, list[int]] = {}
    for value, key in zip(indicators, clusters):
        groups.setdefault(key, []).append(value)
    keys = sorted(groups)
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        drawn: list[int] = []
        for _ in range(len(keys)):
            drawn.extend(groups[keys[rng.randrange(len(keys))]])
        means.append(sum(drawn) / len(drawn))
    means.sort()
    return (means[int((alpha / 2) * n_boot)],
            means[min(int((1 - alpha / 2) * n_boot), n_boot - 1)])


def mcnemar_exact(a_correct: list[int], b_correct: list[int]) -> dict:
    """Exact two-sided McNemar. b = A right & B wrong, c = the reverse."""
    assert len(a_correct) == len(b_correct)
    b = sum(1 for x, y in zip(a_correct, b_correct) if x and not y)
    c = sum(1 for x, y in zip(a_correct, b_correct) if y and not x)
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "p": 1.0}
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return {"b": b, "c": c, "p": min(1.0, 2 * tail)}
