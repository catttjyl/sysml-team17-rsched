"""Workload interface + the no-cache executor used as a baseline.

A workload bundles everything task-specific that the agent loop needs:
  * which tool names it handles and how to turn (name, args) into a tvclient
    ``ToolCall`` (the unit of the TVCache key),
  * the ``Sandbox`` class tool calls execute in,
  * how to derive the TVCache task id from a dataset row,
  * how to score a finished rollout.

Nothing here imports veRL, so workloads can be tested without a GPU stack.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Type

from tvclient import Sandbox, TestToolCall, ToolCall


def cache_path(history: List[ToolCall]) -> List[str]:
    """The exact key tvclient's executor looks up: stateful prefix + current call."""
    prefix, current = history[:-1], history[-1]
    return [c.serialize() for c in prefix if c.will_mutate_state()] + [current.serialize()]


def path_key(task_id: str, history: List[ToolCall]) -> str:
    """Short stable hash of (task, cache path) for logs and offline analysis."""
    blob = json.dumps([task_id, cache_path(history)])
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


class Workload(ABC):
    name: str = "base"
    tool_names: tuple[str, ...] = ()
    sandbox_cls: Type[Sandbox]
    tool_call_cls: Type[ToolCall]

    def __init__(self, **config: Any):
        self.config = config

    @abstractmethod
    def task_id(self, row_kwargs: dict) -> str:
        """TVCache task id (one Tool Call Graph per id) for a dataset row."""

    @abstractmethod
    def make_tool_call(self, tool_name: str, args: dict) -> ToolCall:
        ...

    @abstractmethod
    async def score(
        self, task_id: str, executor: Any, history: List[ToolCall], messages: list, row_kwargs: dict
    ) -> float:
        """Reward for a finished rollout. ``executor`` is still open."""


class DirectExecutor:
    """Same interface as ``tvclient.TvcacheToolExecutor`` but never touches the
    cache: one private sandbox per rollout, every call executed. Used for the
    cache-off baseline (we still log cache keys, so the potential hit rate can
    be computed offline)."""

    def __init__(self, sandbox_cls: Type[Sandbox], task_id: str):
        self.sandbox_cls = sandbox_cls
        self.task_name = task_id
        self.sandbox: Optional[Sandbox] = None
        self.total_calls = 0
        self.total_executions = 0

    async def __aenter__(self) -> "DirectExecutor":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def _ensure(self) -> Sandbox:
        if self.sandbox is None:
            self.sandbox = self.sandbox_cls(task_name=self.task_name)
            await self.sandbox.start()
        return self.sandbox

    async def lookup_or_execute(self, history: List[ToolCall]) -> str:
        self.total_calls += 1
        self.total_executions += 1
        sb = await self._ensure()
        try:
            return await sb.execute(history[-1])
        except Exception:
            await self.close()
            raise

    async def test(self, history: List[ToolCall]) -> str:
        result = await self.lookup_or_execute(history + [TestToolCall("")])
        await self.close()
        return result

    async def close(self) -> None:
        if self.sandbox is not None:
            sb, self.sandbox = self.sandbox, None
            try:
                await sb.stop()
            except Exception:
                pass
