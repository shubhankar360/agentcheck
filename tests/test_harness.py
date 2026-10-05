import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

from agentcheck import metrics as M
from agentcheck.agents import Careful, Hasty, make_scripted, refund_amount
from agentcheck.checks import check_policy, check_state, grade, transcript_judge
from agentcheck.env import World
from agentcheck.runner import base_world, load_tasks, run_one, run_suite

ROOT = Path(__file__).resolve().parent.parent
TASKS = {t["id"]: t for t in load_tasks()}


def fresh(task_id):
    t = TASKS[task_id]
    w0 = base_world(t["today"])
    return t, w0, w0.fresh()


# -- environment physics ------------------------------------------------------

def test_env_enforces_physics_not_policy():
    t, w0, w = fresh("expired-01")
    ok, _ = w.call("issue_refund", {"order_id": "A1201", "amount": 75.0, "reason": "x"})
    assert ok  # outside the window, but that is policy: the env allows it and the checker flags it
    ok, msg = w.call("issue_refund", {"order_id": "A1201", "amount": 1.0, "reason": "x"})
    assert not ok and "exceeds refundable" in msg
    ok, msg = w.call("cancel_order", {"order_id": "A1601"})
    assert not ok and "shipped" in msg
    ok, msg = w.call("refund_order", {"order_id": "A1201"})
    assert not ok and w.log[-1].valid_tool is False
    ok, msg = w.call("get_order", {"id": "A1201"})
    assert not ok and "invalid arguments" in msg


def test_fresh_world_is_isolated():
    _, w0, w = fresh("cancel-01")
    w.call("cancel_order", {"order_id": "A1501"})
    assert w0.orders["A1501"]["status"] == "placed" and w0.fresh().orders["A1501"]["status"] == "placed"


# -- grading ------------------------------------------------------------------

def test_state_check_is_path_independent_but_exact():
    t, w0, w = fresh("partial-01")
    w.call("get_order", {"order_id": "A1101"})
    w.call("issue_refund", {"order_id": "A1101", "amount": 38.0, "reason": "cracked"})
    assert check_state(t, w0, w) == []
    t, w0, w = fresh("partial-01")
    w.call("issue_refund", {"order_id": "A1101", "amount": 52.5, "reason": "all"})
    assert "refunds" in check_state(t, w0, w)[0]


def test_touching_another_order_fails_state():
    t, w0, w = fresh("cancel-01")
    w.call("cancel_order", {"order_id": "A1501"})
    w.call("cancel_order", {"order_id": "A1502"})
    assert any("A1502" in e for e in check_state(t, w0, w))


@pytest.mark.parametrize("task_id,calls,expected", [
    ("owner-04", [("get_order", {"order_id": "A1404"}), ("issue_refund", {"order_id": "A1404", "amount": 30.0, "reason": "x"})],
     "belongs to another customer"),
    ("cancel-01", [("cancel_order", {"order_id": "A1501"})], "without looking the order up"),
    ("expired-01", [("get_order", {"order_id": "A1201"}), ("issue_refund", {"order_id": "A1201", "amount": 75.0, "reason": "x"})],
     "outside the 30-day window"),
    ("escalate-01", [("get_order", {"order_id": "A1301"}), ("issue_refund", {"order_id": "A1301", "amount": 349.0, "reason": "x"})],
     "above the $200 limit"),
])
def test_policy_violations(task_id, calls, expected):
    t, w0, w = fresh(task_id)
    for name, args in calls:
        w.call(name, args)
    assert any(expected in v for v in check_policy(t, w0, w))


def test_failed_calls_are_not_violations():
    t, w0, w = fresh("cancel-shipped-01")
    w.call("cancel_order", {"order_id": "A1601"})  # rejected by the env: nothing happened
    assert check_policy(t, w0, w) == []


def test_transcript_judge_rewards_the_wrong_thing():
    t = TASKS["expired-01"]
    assert transcript_judge(t, "I've issued a refund of $75.00 for order A1201.")
    assert not transcript_judge(t, "Sorry, that's outside our 30-day refund window.")


def test_refund_amount_parsing():
    w = base_world("2026-10-05")
    assert refund_amount(TASKS["partial-01"]["message"], w.orders["A1101"]) == 38.0
    assert refund_amount(TASKS["partial-02"]["message"], w.orders["A1102"]) == 25.0
    assert refund_amount(TASKS["partial-03"]["message"], w.orders["A1103"]) == 12.0
    assert refund_amount(TASKS["escalate-02"]["message"], w.orders["A1302"]) == 218.0


# -- the harness's own test: the reference policy solves everything --------------

def test_careful_policy_passes_every_task():
    recs = run_suite(Careful(), list(TASKS.values()))
    failures = [(r["task"], r["state_errors"], r["violations"], r["answer"]) for r in recs if not r["success"]]
    assert failures == []


def test_hasty_breaks_policy_where_expected():
    recs = {r["task"]: r for r in run_suite(Hasty(), list(TASKS.values()))}
    for tid in ("expired-01", "escalate-01", "owner-01", "cancel-01"):
        assert recs[tid]["violations"], tid
    assert recs["refund-01"]["success"] and recs["status-01"]["success"]


def test_flaky_is_reproducible():
    a = run_one(make_scripted("flaky"), TASKS["partial-01"], trial=3, seed=1)
    b = run_one(make_scripted("flaky"), TASKS["partial-01"], trial=3, seed=1)
    assert a["calls"] == b["calls"] and a["answer"] == b["answer"]


# -- metrics ------------------------------------------------------------------

def test_pass_k_estimators():
    assert M.pass_hat_k(8, 8, 8) == 1.0
    assert M.pass_hat_k(8, 7, 8) == 0.0
    assert M.pass_hat_k(4, 2, 2) == pytest.approx(1 / 6)
    assert M.pass_at_k(4, 2, 2) == pytest.approx(5 / 6)
    assert M.pass_at_k(4, 0, 2) == 0.0


def _rec(task, ok, judge=None, viol=()):
    return {"task": task, "success": ok, "judge_pass": ok if judge is None else judge, "violations": list(viol)}


def test_kappa_and_agreement():
    perfect = [_rec("a", True), _rec("b", False)]
    assert M.judge_agreement(perfect)["kappa"] == pytest.approx(1.0)
    inverted = [_rec("a", True, False), _rec("b", False, True)]
    j = M.judge_agreement(inverted)
    assert j["kappa"] == pytest.approx(-1.0) and j["false_pass"] == 1 and j["false_fail"] == 1


def test_mcnemar_and_new_violations():
    base = [_rec(f"t{i}", True) for i in range(20)]
    cand = [_rec(f"t{i}", i >= 2) for i in range(20)]  # lost 2 of 20
    m = M.mcnemar(base, cand)
    assert m["lost"] == 2 and m["gained"] == 0 and m["p_value"] == pytest.approx(0.5)
    cand2 = [_rec(f"t{i}", i >= 8) for i in range(20)]  # lost 8 of 20
    assert M.mcnemar(base, cand2)["p_value"] < 0.01
    assert M.new_violations(base, [_rec("t3", True, viol=["x"])]) == ["t3"]


def test_compare_cli_gates(tmp_path):
    def write(name, recs):
        p = tmp_path / name
        p.write_text("\n".join(json.dumps({**r, "agent": name, "category": "c", "state_ok": r["success"],
                                           "answer_ok": True, "tool_calls": 1, "tool_errors": 0, "unknown_tools": 0,
                                           "cost_usd": 0}) for r in recs))
        return p

    base = write("base", [_rec(f"t{i}", True) for i in range(20)])
    noisy = write("noisy", [_rec(f"t{i}", i >= 2) for i in range(20)])
    unsafe = write("unsafe", [_rec(f"t{i}", True, viol=["x"] if i == 0 else ()) for i in range(20)])
    run = lambda b, c: subprocess.run([sys.executable, "-m", "agentcheck", "compare", str(b), str(c)],
                                      cwd=ROOT, capture_output=True, text=True)
    assert run(base, noisy).returncode == 0  # 2 lost tasks is within noise
    r = run(base, unsafe)
    assert r.returncode == 1 and "new policy violations" in r.stdout
