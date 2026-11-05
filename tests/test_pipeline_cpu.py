"""CPU integration test: real TVCache server + real TVCacheToolAgentLoop
tool path, with the LLM replaced by scripted tool calls.

Needs the tvcache binary: set TVCACHE_BIN (default: ../tvcache-rust/target/release/tvcache).
The server binds a fixed port (7878); the test is skipped if that port is busy.

    TVCACHE_BIN=... pytest -q tests/test_pipeline_cpu.py
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.environ.get("TVCACHE_BIN", os.path.join(HERE, "..", "..", "tvcache-rust", "target", "release", "tvcache"))


def _wait_port(port: int, timeout: float = 10) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.1)
    raise RuntimeError("tvcache server did not start")


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    if not os.path.exists(BIN):
        pytest.skip(f"tvcache binary not found at {BIN}")
    tmp = tmp_path_factory.mktemp("rsched")
    port = 7878
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", port)) == 0:
            pytest.skip("port 7878 busy")
    proc = subprocess.Popen([BIN, "--eviction-policy=none"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _wait_port(port)
    os.environ.update(
        RSCHED_TIMELINE_DIR=str(tmp / "timeline"),
        RSCHED_SQL_DB_ROOT=str(tmp / "db"),
        RSCHED_SANDBOX_ROOT=str(tmp / "sbx"),
        TVCACHE_PORT=str(port),
    )
    subprocess.check_call(
        [sys.executable, "-m", "rsched.data.make_toy_sql", "--db-root", str(tmp / "db"), "--out", str(tmp / "sql")]
    )
    yield SimpleNamespace(tmp=tmp, port=port)
    proc.terminate()
    proc.wait()


def make_loop(workload: str, cache: str = "tvcache"):
    """Construct the real class with veRL's heavy base __init__ stubbed out."""
    from verl.experimental.agent_loop import tool_agent_loop as tal

    from rsched.agent_loop import TVCacheToolAgentLoop

    def fake_base_init(self, *a, **k):
        self.max_parallel_calls = 1
        self.max_tool_response_length = 4096
        self.tool_response_truncate_side = "middle"

    orig = tal.ToolAgentLoop.__init__
    tal.ToolAgentLoop.__init__ = fake_base_init
    try:
        return TVCacheToolAgentLoop(workload=workload, cache=cache)
    finally:
        tal.ToolAgentLoop.__init__ = orig


async def scripted_rollout(workload: str, row: dict, calls: list[dict], final: str, cache: str = "tvcache"):
    """Replace ToolAgentLoop.run with a script: each entry in `calls` is one
    assistant turn with one tool call, then a final assistant message."""
    from verl.experimental.agent_loop import tool_agent_loop as tal

    loop = make_loop(workload, cache)
    loop._script = (calls, final)  # read by the (shared) fake base run below
    if not getattr(tal.ToolAgentLoop.run, "_rsched_fake", False):
        tal.ToolAgentLoop.run = _fake_base_run
    return await loop.run({}, raw_prompt=row["prompt"], extra_info=row["extra_info"], uid="g0", index=0)


async def _fake_base_run(self, sampling_params, priority=0, **kwargs):
    """Stands in for ToolAgentLoop.run: one tool call per assistant turn, then a final message."""
    from verl.experimental.agent_loop.tool_parser import FunctionCall

    calls, final = self._script
    ad = SimpleNamespace(messages=list(kwargs["raw_prompt"]), prompt_ids=[], response_ids=[], tool_calls=[])
    self._agent_data = ad
    for c in calls:
        self._turn += 1
        fc = FunctionCall(name=c["name"], arguments=json.dumps(c["args"]))
        resp, _, _ = await self._call_tool(fc, {}, ad)
        ad.messages.append({"role": "assistant", "content": ""})
        ad.messages.append({"role": "tool", "content": resp.text})
    ad.messages.append({"role": "assistant", "content": final})
    return SimpleNamespace(reward_score=None, messages=ad.messages)


_fake_base_run._rsched_fake = True


def test_sql_and_terminal(env):
    import pandas as pd

    from rsched.analysis.timeline_report import classify, summarize
    from rsched.timeline import load_events

    rows = pd.read_parquet(env.tmp / "sql" / "train.parquet").to_dict("records")
    row = next(r for r in rows if r["extra_info"]["db_id"] == "company")
    gold = row["extra_info"]["gold_sql"]
    explore = {"name": "execute_sql", "args": {"sql": "SELECT name FROM sqlite_master WHERE type='table'"}}
    check = {"name": "execute_sql", "args": {"sql": gold}}

    term_row = {
        "prompt": [{"role": "user", "content": "count errors"}],
        "extra_info": {"tvcache_task_id": "term|count-errors"},
    }
    ls = {"name": "bash", "args": {"command": "ls -R"}}
    good = {"name": "bash", "args": {"command": "grep -c ' ERROR ' logs/app.log > answer.txt"}}
    bad = {"name": "bash", "args": {"command": "echo 3 > answer.txt"}}

    async def main():
        # 4 concurrent siblings per prompt (like GRPO n=4) -> in-flight (timing) misses on the shared first call
        sql = [scripted_rollout("sql", row, [explore, check], f"<solution>{gold}</solution>") for _ in range(4)]
        sql.append(scripted_rollout("sql", row, [explore], "<solution>SELECT 1</solution>"))
        sql.append(scripted_rollout("sql", row, [explore], "no solution"))
        term = [scripted_rollout("terminal", term_row, [ls, good], "done") for _ in range(3)]
        term.append(scripted_rollout("terminal", term_row, [ls, bad], "done"))
        res = await asyncio.gather(*sql, *term)
        # a second wave after the first finished -> warm hits
        res2 = await asyncio.gather(*[scripted_rollout("terminal", term_row, [ls, good], "done") for _ in range(2)])
        return res, res2

    res, res2 = asyncio.run(main())
    rewards = [r.reward_score for r in res]
    assert rewards[:4] == [1.0] * 4, rewards
    assert rewards[4] == 0.0 and rewards[5] == -1.0
    assert rewards[6:9] == [1.0] * 3 and rewards[9] == 0.0
    assert [r.reward_score for r in res2] == [1.0, 1.0]

    ev = load_events(os.environ["RSCHED_TIMELINE_DIR"])
    tools = [e for e in ev if e["ev"] == "tool"]
    assert len(tools) == 4 * 2 + 2 + 4 * 2 + 2 * 2
    rep = {wl: summarize([r for r in classify(tools) if r["workload"] == wl]) for wl in ("sql", "terminal")}
    print(json.dumps(rep, indent=2))
    # second terminal wave must be fully served from cache (tool calls and the test)
    wave2 = sorted(tools, key=lambda e: e["t_start"])[-4:]
    assert all(e["hit"] for e in wave2), wave2
    assert rep["sql"]["classes"].get("hit", 0) + rep["sql"]["classes"].get("timing", 0) >= 4
    assert sum(1 for e in ev if e["ev"] == "rollout_end") == 12


def test_cache_off_baseline(env):
    term_row = {
        "prompt": [{"role": "user", "content": "x"}],
        "extra_info": {"tvcache_task_id": "term|hello-script"},
    }
    calls = [{"name": "bash", "args": {"command": "printf '#!/bin/bash\\necho \"Hello, RL!\"\\n' > hello.sh && chmod +x hello.sh"}}]

    async def main():
        return await asyncio.gather(*[scripted_rollout("terminal", term_row, calls, "ok", cache="off") for _ in range(2)])

    res = asyncio.run(main())
    assert [r.reward_score for r in res] == [1.0, 1.0]
