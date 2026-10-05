"""Metrics over a set of run records, and the statistics to compare two sets.

pass^k is the number that matters for an agent people rely on: the chance
that *all* of k independent attempts at a task succeed. A 90% pass@1 agent
whose failures are spread evenly across tasks has a pass^8 near 43%; users
experience pass^k, not pass@1.
"""

from __future__ import annotations

import random
from collections import defaultdict
from math import comb


def by_task(records: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        out[r["task"]].append(r)
    return out


def pass_hat_k(n: int, c: int, k: int) -> float:
    """Unbiased estimate of P(all k of k fresh trials succeed) from c of n."""
    if k > n:
        raise ValueError("k must be <= trials")
    return comb(c, k) / comb(n, k)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimate of P(at least one of k trials succeeds)."""
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def summary(records: list[dict]) -> dict:
    tasks = by_task(records)
    n = min(len(v) for v in tasks.values())
    s = {
        "agent": records[0]["agent"], "tasks": len(tasks), "trials": n, "runs": len(records),
        "pass@1": mean(r["success"] for r in records),
        "state_ok": mean(r["state_ok"] for r in records),
        "answer_ok": mean(r["answer_ok"] for r in records),
        "violation_rate": mean(bool(r["violations"]) for r in records),
        "tasks_with_any_violation": sum(any(r["violations"] for r in v) for v in tasks.values()),
        "tool_calls_per_run": mean(r["tool_calls"] for r in records),
        "tool_error_rate": safe_div(sum(r["tool_errors"] for r in records), sum(r["tool_calls"] for r in records)),
        "unknown_tool_calls": sum(r["unknown_tools"] for r in records),
        "cost_usd": round(sum(r["cost_usd"] for r in records), 4),
        "fallback_runs": sum(bool(r.get("fallback_used")) for r in records),
        "pass^k": {}, "pass@k": {},
    }
    for k in sorted({1, 2, 4, 8, n} & set(range(1, n + 1))):
        s["pass^k"][k] = mean(pass_hat_k(len(v), sum(r["success"] for r in v), k) for v in tasks.values())
        s["pass@k"][k] = mean(pass_at_k(len(v), sum(r["success"] for r in v), k) for v in tasks.values())
    lo, hi = bootstrap_ci(records)
    s["pass@1_ci95"] = (lo, hi)
    return s


def by_category(records: list[dict]) -> dict[str, dict]:
    cats: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        cats[r["category"]].append(r)
    return {c: {"runs": len(v), "pass@1": mean(r["success"] for r in v),
                "violation_rate": mean(bool(r["violations"]) for r in v)} for c, v in sorted(cats.items())}


def judge_agreement(records: list[dict]) -> dict:
    """How a transcript-only grader compares with the state-and-policy truth."""
    tp = sum(r["judge_pass"] and r["success"] for r in records)
    fp = sum(r["judge_pass"] and not r["success"] for r in records)
    fn = sum(not r["judge_pass"] and r["success"] for r in records)
    tn = sum(not r["judge_pass"] and not r["success"] for r in records)
    n = len(records)
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n)
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    fp_viol = sum(r["judge_pass"] and bool(r["violations"]) for r in records)
    return {"agreement": po, "kappa": kappa, "judge_pass_rate": (tp + fp) / n, "true_pass_rate": (tp + fn) / n,
            "false_pass": fp, "false_fail": fn, "false_pass_with_violation": fp_viol, "runs": n}


def bootstrap_ci(records: list[dict], iters: int = 2000, seed: int = 7) -> tuple[float, float]:
    """95% CI for pass@1, resampling *tasks* (trials of one task are not
    independent evidence about the agent)."""
    tasks = list(by_task(records).values())
    rng = random.Random(seed)
    means = []
    for _ in range(iters):
        sample = [tasks[rng.randrange(len(tasks))] for _ in tasks]
        runs = [r["success"] for t in sample for r in t]
        means.append(sum(runs) / len(runs))
    means.sort()
    return means[int(0.025 * iters)], means[int(0.975 * iters) - 1]


def mcnemar(base: list[dict], cand: list[dict]) -> dict:
    """Exact McNemar test on per-task outcomes (majority over trials).

    Paired by task, so a fixed set of easy tasks does not inflate the
    evidence. Returns the discordant counts and a two-sided p-value."""
    b = {t: majority(v) for t, v in by_task(base).items()}
    c = {t: majority(v) for t, v in by_task(cand).items()}
    common = sorted(set(b) & set(c))
    lost = sum(b[t] and not c[t] for t in common)
    gained = sum(c[t] and not b[t] for t in common)
    n = lost + gained
    if n == 0:
        p = 1.0
    else:
        k = min(lost, gained)
        p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"tasks": len(common), "lost": lost, "gained": gained, "p_value": p,
            "lost_tasks": [t for t in common if b[t] and not c[t]]}


def new_violations(base: list[dict], cand: list[dict]) -> list[str]:
    bv = {t for t, v in by_task(base).items() if any(r["violations"] for r in v)}
    return sorted(t for t, v in by_task(cand).items() if any(r["violations"] for r in v) and t not in bv)


def majority(runs: list[dict]) -> bool:
    return sum(r["success"] for r in runs) * 2 > len(runs)


def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def safe_div(a, b) -> float:
    return a / b if b else 0.0
