"""Per-process JSONL event log for rollout / tool-call timelines.

Every agent-loop worker process (a Ray actor) appends to its own file
``<dir>/<host>-<pid>.jsonl`` so no cross-process locking is needed. Events use
wall-clock ``time.time()`` so files from different workers on the same node
can be merged and sorted; ``mono`` (perf_counter) is included for precise
intra-process durations.

Event types (field ``ev``):
  rollout_start / rollout_end   one per rollout (trajectory)
  gen_start / gen_end           one pair per LLM generation turn
  tool                          one per tool call (has t_start, t_end, hit, key)

Directory: ``RSCHED_TIMELINE_DIR`` env var, else the value passed to
``get_timeline()``, else ``./timelines``.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
from typing import Any, Optional

_LOCK = threading.Lock()
_INSTANCE: Optional["Timeline"] = None


class Timeline:
    def __init__(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        self.path = os.path.join(directory, f"{socket.gethostname()}-{os.getpid()}.jsonl")
        # line-buffered so a killed job still leaves usable traces
        self._fh = open(self.path, "a", buffering=1)
        self._lock = threading.Lock()

    def log(self, ev: str, **fields: Any) -> None:
        rec = {"ev": ev, "t": time.time(), "mono": time.perf_counter(), "pid": os.getpid()}
        rec.update(fields)
        line = json.dumps(rec, default=str)
        with self._lock:
            self._fh.write(line + "\n")

    def close(self) -> None:
        with self._lock:
            self._fh.close()


def get_timeline(directory: Optional[str] = None) -> Timeline:
    """Process-wide singleton."""
    global _INSTANCE
    with _LOCK:
        if _INSTANCE is None:
            d = os.environ.get("RSCHED_TIMELINE_DIR") or directory or "./timelines"
            _INSTANCE = Timeline(d)
        return _INSTANCE


def load_events(directory: str) -> list[dict]:
    """Read and time-sort all events under ``directory``."""
    events = []
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".jsonl"):
            continue
        with open(os.path.join(directory, name)) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass  # truncated last line of a killed job
    events.sort(key=lambda e: e["t"])
    return events
