"""The Claude adapter, driven through the real Anthropic SDK against a local
mock of the Messages API. No key and no network: the test asserts on the
exact requests the SDK sends and on how the loop handles each response."""

import json
import random
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

anthropic = pytest.importorskip("anthropic")

from agentcheck.agents import ClaudeAgent  # noqa: E402
from agentcheck.checks import grade  # noqa: E402
from agentcheck.runner import base_world, load_tasks  # noqa: E402

TASKS = {t["id"]: t for t in load_tasks()}


def msg(content, stop, i=1):
    return {"id": f"msg_{i}", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": content, "stop_reason": stop, "stop_sequence": None,
            "usage": {"input_tokens": 1000, "output_tokens": 100}}


def tool_use(tid, name, args):
    return {"type": "tool_use", "id": tid, "name": name, "input": args}


@pytest.fixture
def mock_api():
    state = {"queue": [], "requests": []}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()}, "body": body})
            resp = state["queue"].pop(0)
            data = json.dumps(resp).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    client = anthropic.Anthropic(api_key="test-key", base_url=f"http://127.0.0.1:{srv.server_port}", max_retries=0)
    yield client, state
    srv.shutdown()


def run(client, task_id, **kw):
    t = TASKS[task_id]
    w0 = base_world(t["today"])
    w = w0.fresh()
    traj = ClaudeAgent(client=client, **kw).run(t, w, random.Random(0))
    return t, w0, w, traj


def test_tool_loop_end_to_end(mock_api):
    client, state = mock_api
    state["queue"] = [
        msg([{"type": "text", "text": "Let me check."}, tool_use("tu_1", "get_order", {"order_id": "A1003"})], "tool_use", 1),
        msg([tool_use("tu_2", "issue_refund", {"order_id": "A1003", "amount": 59.0, "reason": "broken"})], "tool_use", 2),
        msg([{"type": "text", "text": "I've issued a refund of $59.00 for order A1003."}], "end_turn", 3),
    ]
    t, w0, w, traj = run(client, "refund-03")
    assert grade(t, w0, w, traj.answer).success
    assert (traj.input_tokens, traj.output_tokens) == (3000, 300)
    assert traj.cost_usd == pytest.approx((3000 * 4 + 300 * 20) / 1e6)

    first = state["requests"][0]
    assert first["path"] == "/v1/messages" and first["headers"]["x-api-key"] == "test-key"
    body = first["body"]
    assert body["model"] == "claude-opus-5-5" and body["output_config"] == {"effort": "medium"}
    assert "Verify identity" in body["system"] and "2026-10-05" in body["system"]
    assert {tl["name"] for tl in body["tools"]} >= {"get_order", "issue_refund", "create_ticket"}
    assert "tool_choice" not in body  # forced tool use is rejected on current models
    assert body["messages"][0]["content"].startswith("From: aisha.khan@example.com")

    second = state["requests"][1]["body"]["messages"]
    assert second[1]["role"] == "assistant" and second[1]["content"][1]["type"] == "tool_use"
    result = second[2]["content"][0]
    assert result["type"] == "tool_result" and result["tool_use_id"] == "tu_1" and result["is_error"] is False
    assert json.loads(result["content"])["customer_email"] == "aisha.khan@example.com"


def test_parallel_tool_results_go_back_in_one_message(mock_api):
    client, state = mock_api
    state["queue"] = [
        msg([tool_use("a", "get_order", {"order_id": "A1001"}), tool_use("b", "get_order", {"order_id": "A9999"})], "tool_use"),
        msg([{"type": "text", "text": "done"}], "end_turn", 2),
    ]
    run(client, "refund-01")
    results = state["requests"][1]["body"]["messages"][-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["a", "b"]
    assert results[1]["is_error"] is True and "not found" in results[1]["content"]


def test_refusal_and_step_cap(mock_api):
    client, state = mock_api
    state["queue"] = [msg([], "refusal")]
    *_, traj = run(client, "refund-01")
    assert traj.stop_reason == "refusal" and traj.answer == ""
    state["queue"] = [msg([tool_use(f"t{i}", "get_order", {"order_id": "A1001"})], "tool_use", i) for i in range(3)]
    *_, traj = run(client, "refund-01", max_steps=3)
    assert traj.stop_reason == "max_steps" and len(state["requests"]) == 4


def test_fallback_is_opt_in_and_uses_the_beta(mock_api):
    client, state = mock_api
    state["queue"] = [msg([{"type": "text", "text": "ok"}], "end_turn")]
    run(client, "status-01", fallback=True)
    req = state["requests"][-1]
    assert req["body"]["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in req["headers"].get("anthropic-beta", "")
