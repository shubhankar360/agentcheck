"""Agents under test.

Three scripted policies — deterministic stand-ins that make the harness
testable with no API key and give the metrics something with known
behaviour to measure:

* ``careful`` implements the policy exactly. It must pass every task; if it
  does not, the task or the checker is wrong. That is the harness's own test.
* ``hasty`` does whatever the customer asks, as fast as possible.
* ``flaky`` is ``careful`` with a small chance of a realistic slip at each
  decision point. It is stochastic on purpose, to show what pass^k reveals.

``ClaudeAgent`` runs a real model through the same tools, with the loop
owned here so every step is recorded and capped.
"""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass, field
from datetime import date

from .env import POLICY, TOOLS, World

ORDER_RE = re.compile(r"\b([A-Z]\d{4})\b")
QUOTED = re.compile(r"[\"“]([^\"”]+)[\"”]")


@dataclass
class Trajectory:
    answer: str
    stop_reason: str = "end_turn"
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    model: str = ""
    fallback_used: bool = False
    notes: list[str] = field(default_factory=list)


def parse_intent(msg: str) -> str:
    m = msg.lower()
    if re.search(r"refund|money back", m):
        return "refund"
    if "cancel" in m:
        return "cancel"
    if re.search(r"address|send .* to|deliver .* to", m):
        return "address"
    return "status"


def refund_amount(msg: str, order: dict) -> float:
    """What the customer is asking to have refunded: the whole order unless
    they single out one item after the word "refund"."""
    items = order["items"]
    m = msg.lower()
    if len(items) > 1 and "whole order" not in m:
        tail = m[m.find("refund"):] if "refund" in m else m
        hits = sorted((tail.find(i["name"]), i) for i in items if i["name"] in tail)
        if hits:
            it = hits[0][1]
            qty = 1 if re.search(r"\bone\b|\bthat one\b", m) else it["qty"]
            return round(it["price"] * qty, 2)
    return order["total"]


class Scripted:
    """Shared machinery for the rule-based policies."""

    name = "scripted"

    def __init__(self, slip: float = 0.0):
        self.slip = slip

    def _s(self, rng: random.Random) -> bool:
        return self.slip > 0 and rng.random() < self.slip

    def run(self, task: dict, world: World, rng: random.Random) -> Trajectory:
        t0 = time.perf_counter()
        answer = self.respond(task, world, rng)
        return Trajectory(answer=answer, latency_s=time.perf_counter() - t0, model=self.name)

    def respond(self, task, world, rng) -> str:  # pragma: no cover - overridden
        raise NotImplementedError


class Careful(Scripted):
    name = "careful"

    def respond(self, task, world, rng):
        msg, today = task["message"], date.fromisoformat(task["today"])
        intent = parse_intent(msg)
        m = ORDER_RE.search(msg)
        if not m:
            return "Could you tell me the order number?"
        oid = m.group(1)

        # Slip: act on a cancel/address request before looking the order up.
        if intent in ("cancel", "address") and self._s(rng):
            ok, out = (world.call("cancel_order", {"order_id": oid}) if intent == "cancel"
                       else world.call("update_address", {"order_id": oid, "address": (QUOTED.findall(msg) or [""])[0]}))
            if ok:
                return f"Done — order {oid} has been {'cancelled' if intent == 'cancel' else 'updated with your new address'}."

        if self._s(rng):  # slip: invent a tool name, then recover
            world.call("refund_order" if intent == "refund" else "lookup_order", {"order_id": oid})

        ok, order = world.call("get_order", {"order_id": oid})
        if not ok:
            return f"I couldn't find an order with the number {oid}. Could you double-check it?"
        if order["customer_email"].lower() != task["from"].lower() and not self._s(rng):
            return ("I'm sorry, I can only help the account holder with this order, and your email doesn't match "
                    "the one it was placed with. Please ask them to contact us directly.")

        if intent == "status":
            when = f", on {order['delivered_at']}" if order.get("delivered_at") else ""
            return f"Order {oid} is {order['status']}{when}."

        if intent == "cancel":
            if order["status"] != "placed":
                return f"Order {oid} has already {order['status']}, so it can no longer be cancelled. You can return it once it arrives."
            world.call("cancel_order", {"order_id": oid})
            return f"Order {oid} has been cancelled."

        if intent == "address":
            new = (QUOTED.findall(msg) or [""])[0]
            if order["status"] != "placed":
                return f"Order {oid} has already {order['status']}, so its address can't be changed now."
            world.call("update_address", {"order_id": oid, "address": new})
            return f"I've updated the delivery address for {oid} to {new}."

        # refund
        if order["status"] != "delivered":
            return f"Order {oid} is {order['status']}, so it can't be refunded yet."
        amount = order["total"] if self._s(rng) else refund_amount(msg, order)
        age = (today - date.fromisoformat(order["delivered_at"])).days
        if age > 30 and not self._s(rng):
            return (f"I'm sorry — order {oid} was delivered {age} days ago, which is outside our 30-day refund window, "
                    "so I can't refund it.")
        if amount > 200 and not self._s(rng):
            ok, t = world.call("create_ticket", {"order_id": oid, "summary": f"Refund request ${amount:.2f}"})
            return (f"Refunds over $200 are handled by our refunds team, so I've escalated this as ticket "
                    f"{t['ticket_id']}. They'll be in touch shortly.")
        ok, r = world.call("issue_refund", {"order_id": oid, "amount": amount, "reason": "customer reported a problem"})
        if not ok:
            return f"I wasn't able to process that refund: {r}"
        if self._s(rng):
            return "All sorted — you'll see it soon!"
        return f"I've issued a refund of ${amount:.2f} for order {oid}."


class Hasty(Scripted):
    """Does what is asked. Looks an order up only when it needs a number."""

    name = "hasty"

    def respond(self, task, world, rng):
        msg = task["message"]
        intent = parse_intent(msg)
        m = ORDER_RE.search(msg)
        oid = m.group(1) if m else ""
        if intent == "cancel":
            ok, out = world.call("cancel_order", {"order_id": oid})
            return f"Order {oid} has been cancelled." if ok else f"Sorry, I couldn't cancel it: {out}"
        if intent == "address":
            new = (QUOTED.findall(msg) or [""])[0]
            ok, out = world.call("update_address", {"order_id": oid, "address": new})
            return f"I've updated the address for {oid} to {new}." if ok else f"Sorry, that didn't work: {out}"
        ok, order = world.call("get_order", {"order_id": oid})
        if not ok:
            return f"Sorry, something went wrong: {order}"
        if intent == "status":
            return f"Order {oid} is {order['status']}."
        ok, r = world.call("issue_refund", {"order_id": oid, "amount": order["total"], "reason": "requested"})
        return f"I've issued a refund of ${order['total']:.2f} for order {oid}." if ok else f"Sorry, the refund failed: {r}"


def make_scripted(name: str) -> Scripted:
    if name == "careful":
        return Careful()
    if name == "hasty":
        return Hasty()
    if name == "flaky":
        a = Careful(slip=0.08)
        a.name = "flaky"
        return a
    raise ValueError(f"unknown agent {name}")


# -- a real model ----------------------------------------------------------------

PRICES = {  # USD per million tokens (input, output)
    "claude-opus-5-5": (4.00, 20.00), "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00), "claude-fable-5-1": (10.00, 50.00),
}


class ClaudeAgent:
    """Claude through the Messages API with a hand-owned tool loop.

    The loop is written out rather than delegated to the SDK's tool runner
    because the harness has to see and cap every step: each tool call is
    routed through the instrumented World, and a run that is still calling
    tools after ``max_steps`` is cut off and graded as it stands.

    ``fallback`` is off by default on purpose. Server-side fallback re-runs a
    refused request on another model, which is right for a product and wrong
    for an evaluation: the score would silently mix two models. When it is
    on, runs a fallback answered are marked so they can be excluded.
    """

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "medium", max_steps: int = 12,
                 client=None, fallback: bool = False, strict: bool = False):
        if client is None:
            import anthropic  # optional dependency: only needed for real-model runs

            client = anthropic.Anthropic()
        self.client, self.model, self.effort = client, model, effort
        self.max_steps, self.fallback = max_steps, fallback
        self.tools = [dict(t, strict=True) for t in TOOLS] if strict else TOOLS
        self.name = f"{model}@{effort}"

    def _create(self, **kw):
        if self.fallback:
            return self.client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kw)
        return self.client.messages.create(**kw)

    def run(self, task: dict, world: World, rng: random.Random) -> Trajectory:
        t0 = time.perf_counter()
        system = f"{POLICY}\n\nToday's date is {task['today']}."
        messages: list[dict] = [{"role": "user", "content": f"From: {task['from']}\n\n{task['message']}"}]
        traj = Trajectory(answer="", model=self.model)
        for _ in range(self.max_steps):
            resp = self._create(model=self.model, max_tokens=16000, system=system, tools=self.tools,
                                messages=messages, output_config={"effort": self.effort})
            traj.input_tokens += resp.usage.input_tokens
            traj.output_tokens += resp.usage.output_tokens
            if any(getattr(it, "type", "") == "fallback_message" for it in (getattr(resp.usage, "iterations", None) or [])):
                traj.fallback_used = True
            traj.stop_reason = resp.stop_reason
            if resp.stop_reason == "refusal":
                traj.notes.append("refused")
                break
            messages.append({"role": "assistant", "content": resp.content})
            if resp.stop_reason == "tool_use":
                results = []
                for b in resp.content:
                    if b.type == "tool_use":
                        ok, out = world.call(b.name, dict(b.input or {}))
                        results.append({"type": "tool_result", "tool_use_id": b.id,
                                        "content": json.dumps(out) if ok else str(out), "is_error": not ok})
                messages.append({"role": "user", "content": results})  # all results in one message
                continue
            traj.answer = "\n".join(b.text for b in resp.content if b.type == "text").strip()
            break
        else:
            traj.stop_reason = "max_steps"
            traj.notes.append(f"cut off after {self.max_steps} model calls")
        p_in, p_out = PRICES.get(self.model, (0.0, 0.0))
        traj.cost_usd = round((traj.input_tokens * p_in + traj.output_tokens * p_out) / 1e6, 6)
        traj.latency_s = time.perf_counter() - t0
        return traj
