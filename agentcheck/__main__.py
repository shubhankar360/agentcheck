"""agentcheck — evaluate tool-using agents on end state, policy and answer.

    python -m agentcheck run careful                 # scripted reference policy
    python -m agentcheck run flaky --trials 8
    python -m agentcheck run claude --model claude-opus-5-5 --trials 4   # needs Anthropic credentials
    python -m agentcheck report results/*.jsonl      # markdown report
    python -m agentcheck compare base.jsonl cand.jsonl   # regression gate (exit 1 on regression)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import metrics as M
from .agents import ClaudeAgent, make_scripted
from .runner import load_records, load_tasks, run_suite


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def report(paths: list[Path]) -> str:
    sets = [(p, load_records(p)) for p in paths]
    out = ["# agentcheck results", "",
           "Each run is graded on three things: the **end state** of the world, the **policy** "
           "(zero tolerance), and the **facts in the answer**. Success needs all three.", "",
           "| agent | runs | pass@1 (95% CI) | state ok | answer ok | runs with a policy violation | tool calls / run | tool error rate | cost |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    sums = []
    for p, recs in sets:
        s = M.summary(recs)
        sums.append((s, recs))
        lo, hi = s["pass@1_ci95"]
        out.append(f"| `{s['agent']}` | {s['runs']} | {pct(s['pass@1'])} ({pct(lo)}–{pct(hi)}) | {pct(s['state_ok'])} | "
                   f"{pct(s['answer_ok'])} | {pct(s['violation_rate'])} | {s['tool_calls_per_run']:.1f} | "
                   f"{pct(s['tool_error_rate'])} | ${s['cost_usd']:.2f} |")

    multi = [(s, r) for s, r in sums if s["trials"] > 1]
    if multi:
        out += ["", "## Reliability: pass^k", "",
                "pass^k is the chance that **all** k attempts at a task succeed — what a user who relies on the agent "
                "every day experiences. pass@k (any of k) is shown for contrast; it is what a demo shows.", ""]
        for s, _ in multi:
            ks = sorted(s["pass^k"])
            out.append(f"**`{s['agent']}`** ({s['trials']} trials per task)\n")
            out.append("| k | " + " | ".join(str(k) for k in ks) + " |")
            out.append("| --- | " + " | ".join("---" for _ in ks) + " |")
            out.append("| pass^k | " + " | ".join(pct(s["pass^k"][k]) for k in ks) + " |")
            out.append("| pass@k | " + " | ".join(pct(s["pass@k"][k]) for k in ks) + " |")
            out.append("")

    out += ["## By category", "", "| category | " + " | ".join(f"`{s['agent']}`" for s, _ in sums) + " |",
            "| --- |" + " --- |" * len(sums)]
    cats = sorted({c for _, r in sums for c in M.by_category(r)})
    per = [M.by_category(r) for _, r in sums]
    for c in cats:
        cells = []
        for pc in per:
            v = pc.get(c)
            cells.append("—" if not v else f"{pct(v['pass@1'])}" + (f" · {pct(v['violation_rate'])} viol." if v["violation_rate"] else ""))
        out.append(f"| {c} | " + " | ".join(cells) + " |")

    out += ["", "## Would a transcript-only grader have noticed?", "",
            "`transcript_judge` passes a run when the reply tells the customer they got what they asked for — the "
            "grader you get from \"did the agent resolve the user's request?\" over the conversation alone. "
            "Compared with the state-and-policy ground truth:", "",
            "| agent | judge pass rate | true pass rate | agreement | Cohen's κ | false passes | …of which broke policy | false fails |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    allrecs = []
    for s, r in sums:
        j = M.judge_agreement(r)
        allrecs += r
        out.append(f"| `{s['agent']}` | {pct(j['judge_pass_rate'])} | {pct(j['true_pass_rate'])} | {pct(j['agreement'])} | "
                   f"{j['kappa']:.2f} | {j['false_pass']} | {j['false_pass_with_violation']} | {j['false_fail']} |")
    j = M.judge_agreement(allrecs)
    out.append(f"| **all runs** | {pct(j['judge_pass_rate'])} | {pct(j['true_pass_rate'])} | {pct(j['agreement'])} | "
               f"{j['kappa']:.2f} | {j['false_pass']} | {j['false_pass_with_violation']} | {j['false_fail']} |")
    return "\n".join(out) + "\n"


def compare(base: Path, cand: Path, alpha: float = 0.05) -> int:
    b, c = load_records(base), load_records(cand)
    sb, sc = M.summary(b), M.summary(c)
    mc = M.mcnemar(b, c)
    nv = M.new_violations(b, c)
    print(f"pass@1  {pct(sb['pass@1'])} -> {pct(sc['pass@1'])}   "
          f"tasks lost {mc['lost']}, gained {mc['gained']}, McNemar p = {mc['p_value']:.3f}")
    failed = False
    if nv:
        print(f"FAIL: new policy violations on {', '.join(nv)} (zero tolerance, no significance test)")
        failed = True
    if mc["lost"] > mc["gained"] and mc["p_value"] < alpha:
        print(f"FAIL: significant regression on {', '.join(mc['lost_tasks'])}")
        failed = True
    if not failed:
        msg = "difference is within noise" if mc["lost"] > mc["gained"] else "no regression"
        print(f"ok: {msg}")
    return 1 if failed else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="agentcheck", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("agent", help="careful | hasty | flaky | claude")
    r.add_argument("--trials", type=int, default=1)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--model", default="claude-opus-5-5")
    r.add_argument("--effort", default="medium")
    r.add_argument("--fallback", action="store_true", help="allow server-side refusal fallback (marks affected runs)")
    r.add_argument("--only", nargs="*", help="task ids or categories")
    r.add_argument("--out", type=Path)
    rp = sub.add_parser("report")
    rp.add_argument("paths", nargs="+", type=Path)
    cp = sub.add_parser("compare")
    cp.add_argument("base", type=Path)
    cp.add_argument("cand", type=Path)
    cp.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args(argv)

    if a.cmd == "run":
        tasks = load_tasks()
        if a.only:
            tasks = [t for t in tasks if t["id"] in a.only or t["category"] in a.only]
        agent = (ClaudeAgent(model=a.model, effort=a.effort, fallback=a.fallback) if a.agent == "claude"
                 else make_scripted(a.agent))
        out = a.out or Path("results") / f"{agent.name.replace('@', '-')}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        recs = run_suite(agent, tasks, a.trials, a.seed, out, log=print)
        s = M.summary(recs)
        print(f"\n{s['agent']}: pass@1 {pct(s['pass@1'])}, violations in {pct(s['violation_rate'])} of runs -> {out}")
        return 0
    if a.cmd == "report":
        sys.stdout.write(report(a.paths))
        return 0
    return compare(a.base, a.cand, a.alpha)


if __name__ == "__main__":
    sys.exit(main())
