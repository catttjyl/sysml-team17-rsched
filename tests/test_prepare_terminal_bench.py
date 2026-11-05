"""Converter test on small fixture tasks. Needs no GPU, tvclient or containers."""

from __future__ import annotations

import importlib.util
import json
import os
import sys

import pandas as pd

from rsched.data import prepare_terminal_bench as ptb

HERE = os.path.dirname(os.path.abspath(__file__))
CLIENT = "  client:\n    build:\n      context: {context}\n      dockerfile: Dockerfile\n    image: x\n"


def _task(root, name, difficulty="easy", context=".", extra="", dockerfile=True):
    d = root / name
    (d / context).mkdir(parents=True, exist_ok=True)
    (d / "task.yaml").write_text(f"instruction: |-\n  Do {name}\ndifficulty: {difficulty}\n")
    (d / "docker-compose.yaml").write_text("services:\n" + CLIENT.format(context=context) + extra)
    if dockerfile:
        (d / context / "Dockerfile").write_text("FROM scratch\n")


def _load_terminal_tasks():
    # Load the module directly, since importing rsched.workloads requires tvclient.
    path = os.path.join(HERE, "..", "rsched", "workloads", "terminal_tasks.py")
    spec = importlib.util.spec_from_file_location("_terminal_tasks", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_converts_and_reports_tasks(tmp_path, monkeypatch, capsys):
    tasks = tmp_path / "original-tasks"
    _task(tasks, "a-root")
    _task(tasks, "b-client-dir", context="client")
    _task(tasks, "c-settings", extra="    tmpfs:\n      - /root\n")
    _task(tasks, "d-val")
    _task(tasks, "e-other-group", difficulty="medium")
    _task(tasks, "f-multi", extra="  db:\n    image: y\n")
    _task(tasks, "g-no-dockerfile", dockerfile=False)
    (tasks / "not-a-task").mkdir()
    out = tmp_path / "out"

    argv = ["prog", "--tasks-dir", str(tasks), "--difficulty", "easy", "--out", str(out), "--val-tasks", "1"]
    monkeypatch.setattr(sys, "argv", argv)
    ptb.main()

    train = pd.read_parquet(out / "train.parquet")
    val = pd.read_parquet(out / "validation.parquet")
    assert [r["task"] for r in train["extra_info"]] == ["a-root", "b-client-dir", "c-settings"]
    assert [r["task"] for r in val["extra_info"]] == ["d-val"]
    row = train.iloc[0]
    assert row["agent_name"] == "tvcache_terminal_agent"
    assert row["extra_info"]["tvcache_task_id"] == "term|a-root"
    assert row["extra_info"]["difficulty"] == "easy"
    assert row["prompt"][1]["content"] == "Do a-root"

    printed = capsys.readouterr().out
    assert "easy: 6 tasks, kept 4 (3 train, 1 validation)" in printed
    assert "skipped 1 (multi_service): f-multi" in printed
    assert "skipped 1 (no_dockerfile): g-no-dockerfile" in printed
    assert "c-settings: tmpfs" in printed

    # tasks.json must load through the same path the workers use.
    monkeypatch.setenv("RSCHED_TERMINAL_TASKS", str(out / "tasks.json"))
    loaded = _load_terminal_tasks().load_tasks()
    task = loaded["b-client-dir"]
    assert task.image == "tb1/b-client-dir:d28711d"
    assert task.build_context == "client"
    assert task.task_dir == str(tasks / "b-client-dir")
    assert not loaded["count-errors"].image  # built-in tasks stay folder-based
    assert len(json.load(open(out / "tasks.json"))) == 4
