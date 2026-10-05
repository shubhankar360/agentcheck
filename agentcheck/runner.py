from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from .checks import grade, transcript_judge
from .env import World

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def load_tasks(path: Path | None = None) -> list[dict]:
    return json.loads((path or DATA / "tasks.json").read_text())


def base_world(today: str) -> World:
    return World.load(DATA / "world.json", today)


def trial_seed(task_id: str, trial: int, seed: int) -> int:
    # Stable across Python runs (hash() is salted per process).
    return int(hashlib.sha256(f"{seed}:{task_id}:{trial}".encode()).hexdigest()[:12], 16)


def run_one(agent, task: dict, trial: int = 0, seed: int = 0, world: World | None = None) -> dict:
    initial = world or base_world(task["today"])
    w = initial.fresh()
    traj = agent.run(task, w, random.Random(trial_seed(task["id"], trial, seed)))
    g = grade(task, initial, w, traj.answer)
    return {
        "agent": getattr(agent, "name", "agent"), "task": task["id"], "category": task["category"], "trial": trial,
        "success": g.success, "state_ok": g.state_ok, "answer_ok": g.answer_ok, "violations": g.violations,
        "state_errors": g.state_errors, "judge_pass": transcript_judge(task, traj.answer),
        "tool_calls": len(w.log), "tool_errors": sum(1 for c in w.log if not c.ok),
        "unknown_tools": sum(1 for c in w.log if not c.valid_tool),
        "calls": [{"name": c.name, "args": c.args, "ok": c.ok} for c in w.log],
        "answer": traj.answer, "stop_reason": traj.stop_reason, "model": traj.model,
        "input_tokens": traj.input_tokens, "output_tokens": traj.output_tokens, "cost_usd": traj.cost_usd,
        "latency_s": round(traj.latency_s, 4), "fallback_used": traj.fallback_used, "notes": traj.notes,
    }


def run_suite(agent, tasks: list[dict], trials: int = 1, seed: int = 0, out: Path | None = None, log=None) -> list[dict]:
    records = []
    fh = out.open("w") if out else None
    try:
        for t in tasks:
            for k in range(trials):
                r = run_one(agent, t, k, seed)
                records.append(r)
                if fh:
                    fh.write(json.dumps(r) + "\n")
                    fh.flush()
                if log:
                    log(f"{r['task']:<22} trial {k}  {'PASS' if r['success'] else 'fail'}"
                        f"{'  ' + '; '.join(r['violations']) if r['violations'] else ''}")
    finally:
        if fh:
            fh.close()
    return records


def load_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
