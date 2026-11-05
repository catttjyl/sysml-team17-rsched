"""Text-to-SQL workload (SkyRL-SQL-653 style).

The agent calls ``execute_sql(sql)`` against a SQLite database opened
read-only, so every call is *stateless* (``will_mutate_state() == False``):
the cache key is just the current query, scoped to the task. The sandbox is a
handle to the .sqlite file; "forking" it is free.

TVCache task id format: ``sql|<source>|<db_id>|<row index>``. The source
(synsql / spider / bird / toy) picks the sub-directory under the DB root, using
the same layout as SkyRL's OmniSQL download.
"""

from __future__ import annotations

import asyncio
import os
import re
import sqlite3
import time
from typing import Any, List, Optional

from tvclient import Sandbox, TestToolCall, ToolCall

from rsched.workloads.base import Workload

# Same layout SkyRL's SQLEnv uses for the OmniSQL data.zip.
SOURCE_SUBDIRS = {
    "synsql": "SynSQL-2.5M/databases",
    "spider": "spider/database",
    "bird": "bird/train/train_databases",
    "toy": "toy",
}

_DB_ROOT: Optional[str] = os.environ.get("RSCHED_SQL_DB_ROOT")
QUERY_TIMEOUT_S = float(os.environ.get("RSCHED_SQL_TIMEOUT", "5"))
MAX_ROWS = 50


def set_db_root(path: str) -> None:
    global _DB_ROOT
    _DB_ROOT = path


def parse_task_id(task_id: str) -> tuple[str, str]:
    kind, source, db_id, _ = task_id.split("|", 3)
    assert kind == "sql", task_id
    return source, db_id


def db_file_for(source: str, db_id: str) -> str:
    if _DB_ROOT is None:
        raise RuntimeError("SQL DB root not set (RSCHED_SQL_DB_ROOT or workload db_root)")
    sub = SOURCE_SUBDIRS.get(source, source)
    return os.path.join(_DB_ROOT, sub, db_id, f"{db_id}.sqlite")


def normalize_sql(sql: str) -> str:
    """Whitespace/semicolon normalization only -- semantics preserved."""
    sql = re.sub(r"\s+", " ", sql.strip())
    return sql.rstrip("; ")


def run_query(db_file: str, sql: str, timeout_s: float = QUERY_TIMEOUT_S):
    """Returns (columns, rows) or raises sqlite3.Error / TimeoutError."""
    conn = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True, check_same_thread=False)
    deadline = time.monotonic() + timeout_s
    timed_out = {"v": False}

    def _progress() -> int:
        if time.monotonic() > deadline:
            timed_out["v"] = True
            return 1  # abort the statement
        return 0

    conn.set_progress_handler(_progress, 10_000)
    try:
        cur = conn.execute(sql)
        rows = cur.fetchall()
        cols = [d[0] for d in (cur.description or [])]
        return cols, rows
    except sqlite3.OperationalError:
        if timed_out["v"]:
            raise TimeoutError
        raise
    finally:
        conn.close()


def format_result(cols: list, rows: list) -> str:
    n = len(rows)
    shown = rows[:MAX_ROWS]
    lines = ["\t".join(map(str, cols))] if cols else []
    lines += ["\t".join(map(str, r)) for r in shown]
    out = "\n".join(lines) if lines else "(empty result)"
    if n > MAX_ROWS:
        out += f"\n... ({n} rows total, showing first {MAX_ROWS})"
    return out


class SQLToolCall(ToolCall):
    def __init__(self, sql: str):
        self.sql = normalize_sql(sql)

    def to_dict(self) -> dict:
        return {"tool": "execute_sql", "sql": self.sql}

    def will_mutate_state(self) -> bool:
        return False  # DB is opened read-only


class SQLSandbox(Sandbox):
    """env_id is just the db file path; no process, nothing to snapshot."""

    async def start(self) -> None:
        if self.env_id is None:
            self.env_id = db_file_for(*parse_task_id(self.task_name))
        if not os.path.exists(self.env_id):
            raise FileNotFoundError(self.env_id)

    async def fork(self) -> "SQLSandbox":
        return SQLSandbox(task_name=self.task_name, env_id=self.env_id)

    async def execute(self, tool_call: ToolCall) -> str:
        if isinstance(tool_call, TestToolCall):
            return "no-op"  # SQL reward is computed outside the cache (see score())
        assert isinstance(tool_call, SQLToolCall)
        try:
            cols, rows = await asyncio.to_thread(run_query, self.env_id, tool_call.sql)
        except TimeoutError:
            # Prefix tells tvclient not to cache it (timeouts are load-dependent).
            return f"Failed to execute: query timed out after {QUERY_TIMEOUT_S}s"
        except sqlite3.Error as e:
            return f"Error executing SQL: {e}"  # deterministic -> cacheable
        return format_result(cols, rows)

    async def stop(self) -> None:
        pass


SOLUTION_RE = re.compile(r"<solution>(.*?)</solution>", re.S)


def extract_solution(messages: list) -> Optional[str]:
    text = "".join(
        m.get("content") or "" for m in messages if m.get("role") == "assistant" and isinstance(m.get("content"), str)
    )
    found = SOLUTION_RE.findall(text)
    if not found:
        return None
    sol = found[-1].strip()
    sol = re.sub(r"^```(?:sql)?\s*|\s*```$", "", sol).strip()
    return sol or None


def score_sql(db_file: str, pred_sql: Optional[str], gold_sql: str) -> float:
    """SkyRL semantics: -1 bad format, 0 wrong, 1 same result set (order-insensitive)."""
    if pred_sql is None:
        return -1.0
    try:
        _, pred = run_query(db_file, pred_sql, timeout_s=30)
        _, gold = run_query(db_file, gold_sql, timeout_s=30)
    except Exception:
        return 0.0
    return 1.0 if frozenset(pred) == frozenset(gold) else 0.0


class SQLWorkload(Workload):
    name = "sql"
    tool_names = ("execute_sql",)
    sandbox_cls = SQLSandbox
    tool_call_cls = SQLToolCall

    def __init__(self, db_root: Optional[str] = None, **config: Any):
        super().__init__(**config)
        if db_root:
            set_db_root(db_root)

    def task_id(self, row_kwargs: dict) -> str:
        info = row_kwargs.get("extra_info") or {}
        return info["tvcache_task_id"]

    def make_tool_call(self, tool_name: str, args: dict) -> ToolCall:
        return SQLToolCall(str(args.get("sql", "")))

    async def score(self, task_id: str, executor: Any, history: List[ToolCall], messages: list, row_kwargs: dict = None) -> float:
        info = (row_kwargs or {}).get("extra_info") or {}
        gold = info.get("gold_sql")
        if gold is None:
            return 0.0
        db_file = db_file_for(*parse_task_id(task_id))
        return await asyncio.to_thread(score_sql, db_file, extract_solution(messages), gold)
