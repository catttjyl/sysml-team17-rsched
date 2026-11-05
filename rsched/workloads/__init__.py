from rsched.workloads.base import DirectExecutor, Workload, cache_path, path_key

_REGISTRY = {
    "sql": "rsched.workloads.sql:SQLWorkload",
    "terminal": "rsched.workloads.terminal:TerminalWorkload",
}
_CACHE: dict = {}


def get_workload(name: str, **config) -> Workload:
    """Process-wide singleton per (name, config)."""
    key = (name, tuple(sorted(config.items())))
    if key not in _CACHE:
        import importlib

        mod, cls = _REGISTRY[name].split(":")
        _CACHE[key] = getattr(importlib.import_module(mod), cls)(**config)
    return _CACHE[key]


__all__ = ["DirectExecutor", "Workload", "cache_path", "path_key", "get_workload"]
