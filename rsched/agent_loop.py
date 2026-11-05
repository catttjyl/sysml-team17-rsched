"""veRL agent loop with TVCache-backed tool execution and timeline logging.

``TVCacheToolAgentLoop`` subclasses veRL's ``ToolAgentLoop`` (v0.9.1) and
changes three things:

1. Per-rollout lifecycle: ``run()`` opens one ``TvcacheToolExecutor`` (or the
   no-cache ``DirectExecutor``) for the trajectory, keeps the cumulative
   tool-call history, scores the rollout at the end (``reward_score`` on the
   output, so veRL skips its reward manager), and closes the executor.
2. ``_call_tool()``: tool calls for the workload's tools go through
   ``executor.lookup_or_execute(history)`` instead of ``BaseTool.execute``.
3. Timeline events for rollout start/end, each generation turn, and each
   tool call (start, end, hit, cache key) -> ``rsched.timeline``.

veRL instantiates one agent-loop object per trajectory, so instance attributes
are per-rollout state. Register it through
``actor_rollout_ref.rollout.agent.agent_loop_config_path`` (see
configs/agent_loops/*.yaml); the yaml keys below become constructor kwargs.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any, Optional

from omegaconf import DictConfig, OmegaConf

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop
from verl.experimental.agent_loop.tool_parser import FunctionCall
from verl.tools.schemas import ToolResponse

from rsched.timeline import get_timeline
from rsched.workloads import DirectExecutor, cache_path, get_workload, path_key

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _plain(x: Any) -> Any:
    if isinstance(x, DictConfig):
        return OmegaConf.to_container(x, resolve=True)
    return x


class TVCacheToolAgentLoop(ToolAgentLoop):
    def __init__(
        self,
        *args,
        workload: str = "sql",
        workload_config: Optional[dict] = None,
        cache: str = "tvcache",  # "tvcache" | "off"
        tvcache_host: str = "127.0.0.1",
        tvcache_port: int = 7878,
        timeline_dir: Optional[str] = None,
        adaptive_snapshots: bool = True,
        **kwargs,
    ):
        kwargs.pop("name", None)  # registry name from the yaml, not a ctor arg upstream
        super().__init__(*args, **kwargs)
        self.workload = get_workload(workload, **(_plain(workload_config) or {}))
        self.cache_mode = os.environ.get("RSCHED_CACHE", cache)
        self.tvcache_host = os.environ.get("TVCACHE_HOST", tvcache_host)
        self.tvcache_port = int(os.environ.get("TVCACHE_PORT", tvcache_port))
        self.adaptive_snapshots = adaptive_snapshots
        self.timeline = get_timeline(timeline_dir)
        if self.max_parallel_calls != 1:
            # Parallel calls in one turn would make the cumulative history order racy.
            logger.warning("TVCacheToolAgentLoop forces max_parallel_calls=1 (was %s)", self.max_parallel_calls)
            self.max_parallel_calls = 1

        # per-rollout state (one instance per trajectory)
        self._executor = None
        self._history: list = []
        self._agent_data: Optional[AgentData] = None
        self._ctx: dict[str, Any] = {}
        self._turn = 0

    # ------------------------------------------------------------------ lifecycle
    def _make_executor(self, task_id: str):
        if self.cache_mode == "off":
            return DirectExecutor(self.workload.sandbox_cls, task_id)
        from tvclient import TvcacheToolExecutor

        ex = TvcacheToolExecutor(
            self.workload.sandbox_cls,
            self.workload.tool_call_cls,
            task_id,
            host=self.tvcache_host,
            port=self.tvcache_port,
            adaptive=self.adaptive_snapshots,
            fixed_threshold_seconds=0.0 if self.adaptive_snapshots else 1e-9,
        )
        return ex

    async def run(self, sampling_params: dict[str, Any], priority: int = 0, **kwargs) -> AgentLoopOutput:
        task_id = self.workload.task_id(kwargs)
        rollout_id = uuid.uuid4().hex[:12]
        self._ctx = {
            "rollout": rollout_id,
            "task": task_id,
            "uid": str(kwargs.get("uid", "")),  # GRPO group id (same prompt, same step)
            "index": int(kwargs["index"]) if "index" in kwargs else None,
            "workload": self.workload.name,
            "cache": self.cache_mode,
        }
        self._history = []
        self._turn = 0
        t0 = time.time()
        self.timeline.log("rollout_start", priority=int(priority), **self._ctx)

        self._executor = self._make_executor(task_id)
        reward: Optional[float] = None
        n_tools = 0
        try:
            async with self._executor:
                output = await super().run(sampling_params, priority=priority, **kwargs)
                n_tools = len(self._history)
                messages = self._agent_data.messages if self._agent_data is not None else []
                ts = time.time()
                reward = float(
                    await self.workload.score(task_id, self._executor, self._history, messages, kwargs)
                )
                self.timeline.log("score", t_start=ts, t_end=time.time(), reward=reward, **self._ctx)
            output.reward_score = reward
            return output
        finally:
            self.timeline.log(
                "rollout_end",
                t_start=t0,
                reward=reward,
                n_tool_calls=n_tools,
                n_turns=self._turn,
                **self._ctx,
            )
            self._executor = None

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        self._agent_data = agent_data  # keep a handle for scoring after super().run()
        return await super()._handle_pending_state(agent_data, sampling_params)

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        self._turn += 1
        n_ctx = len(agent_data.prompt_ids)
        t0 = time.time()
        self.timeline.log("gen_start", turn=self._turn, ctx_tokens=n_ctx, **self._ctx)
        state = await super()._handle_generating_state(agent_data, sampling_params, ignore_termination)
        self.timeline.log(
            "gen_end",
            turn=self._turn,
            t_start=t0,
            new_tokens=len(agent_data.response_ids),
            n_tool_calls=len(agent_data.tool_calls) if state == AgentState.PROCESSING_TOOLS else 0,
            next_state=state.value,
            **self._ctx,
        )
        return state

    # ------------------------------------------------------------------ tools
    def _truncate(self, text: str) -> str:
        n = self.max_tool_response_length
        if not text or len(text) <= n:
            return text
        if self.tool_response_truncate_side == "left":
            return "(truncated)..." + text[-n:]
        if self.tool_response_truncate_side == "right":
            return text[:n] + "...(truncated)"
        half = n // 2
        return text[:half] + "...(truncated)..." + text[-half:]

    async def _call_tool(
        self, tool_call: FunctionCall, tools_kwargs: dict[str, Any], agent_data: AgentData
    ) -> tuple[ToolResponse, float, dict]:
        name = tool_call.name
        if name not in self.workload.tool_names or self._executor is None:
            return await super()._call_tool(tool_call, tools_kwargs, agent_data)

        try:
            args = json.loads(tool_call.arguments)
            if not isinstance(args, dict):
                raise TypeError("arguments must be a JSON object")
        except (json.JSONDecodeError, TypeError) as e:
            self.timeline.log("tool_invalid", turn=self._turn, tool=name, error=str(e), **self._ctx)
            return ToolResponse(text=f"Invalid JSON in arguments for '{name}': {e}"), 0.0, {}

        call = self.workload.make_tool_call(name, args)
        self._history.append(call)
        ex = self._executor
        key = path_key(self._ctx["task"], self._history)
        depth = len(cache_path(self._history))
        n_exec0 = ex.total_executions
        sb0 = getattr(ex, "rollout_sandbox", None)
        t0 = time.time()
        self.timeline.log("tool_issue", turn=self._turn, seq=len(self._history), key=key, **self._ctx)
        error = None
        try:
            text = await ex.lookup_or_execute(self._history)
        except Exception as e:  # sandbox raised: state unknown, drop the call from history
            self._history.pop()
            error = f"{type(e).__name__}: {e}"
            text = f"Error executing tool '{name}': {e}"
        t1 = time.time()
        executed = ex.total_executions - n_exec0
        hit = error is None and executed == 0
        sb1 = getattr(ex, "rollout_sandbox", None)
        if hit:
            source = "hit"
        elif error is not None:
            source = "error"
        elif self.cache_mode == "off":
            source = "direct"
        elif sb1 is sb0 and sb0 is not None:
            source = "live"  # continued in this rollout's own sandbox
        elif executed < depth:
            source = "fork"  # resumed from a cached snapshot at depth - executed
        else:
            source = "fresh"  # replayed from the root
        self.timeline.log(
            "tool",
            turn=self._turn,
            seq=len(self._history),
            tool=name,
            key=key,
            depth=depth,
            mutates=call.will_mutate_state(),
            t_start=t0,
            t_end=t1,
            hit=hit,
            executed=executed,
            source=source,
            uncacheable=text.startswith("Failed to execute") if text else False,
            error=error,
            out_chars=len(text or ""),
            **self._ctx,
        )
        return ToolResponse(text=self._truncate(text)), 0.0, {}
