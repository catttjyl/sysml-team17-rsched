"""Small Terminal-Bench-style task set for bring-up.

Each task = instruction + initial files + a bash test (exit 0 = pass) run in
the final sandbox state. These exist so the pipeline runs end-to-end on
Perlmutter without Docker; real Terminal-Bench tasks (which ship Docker
images) need a podman-hpc sandbox backend -- see README "Next steps".

Tasks can also be loaded from a JSON file (``RSCHED_TERMINAL_TASKS``) with the
same fields, which is how we will plug in converted Terminal-Bench tasks.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field


@dataclass
class TerminalTask:
    task_id: str
    instruction: str
    files: dict[str, str] = field(default_factory=dict)
    setup: str = ""  # bash run once in a fresh sandbox after files are written
    test: str = "true"  # bash, exit 0 == pass
    executable: tuple[str, ...] = ()
    # Container tasks (Terminal-Bench 1), from prepare_terminal_bench. Empty for the built-in tasks.
    image: str = ""
    task_dir: str = ""
    build_context: str = "."
    dockerfile: str = "Dockerfile"
    difficulty: str = ""


_LOG = "\n".join(
    f"2026-11-0{1 + i % 3} 12:{i:02d}:00 {lvl} worker-{i % 4} {msg}"
    for i, (lvl, msg) in enumerate(
        [
            ("INFO", "started"),
            ("ERROR", "disk full"),
            ("INFO", "retry"),
            ("WARN", "slow request"),
            ("ERROR", "timeout"),
            ("INFO", "ok"),
            ("ERROR", "disk full"),
            ("INFO", "done"),
            ("ERROR", "connection reset"),
            ("WARN", "slow request"),
        ]
    )
)

BUILTIN_TASKS: list[TerminalTask] = [
    TerminalTask(
        "count-errors",
        "The file logs/app.log contains application logs. Count the lines whose level is ERROR "
        "and write just that number to answer.txt.",
        files={"logs/app.log": _LOG + "\n"},
        test='[ "$(tr -d "[:space:]" < answer.txt)" = "4" ]',
    ),
    TerminalTask(
        "top-error",
        "Look at logs/app.log. Find the most frequent ERROR message (the text after the worker name) "
        "and write it to top_error.txt.",
        files={"logs/app.log": _LOG + "\n"},
        test='grep -qx "disk full" top_error.txt',
    ),
    TerminalTask(
        "fix-python-bug",
        "The script calc.py should print the average of the numbers in data.txt, but it is buggy. "
        "Fix calc.py so that `python3 calc.py` prints 5.5.",
        files={
            "calc.py": "nums = [int(x) for x in open('data.txt').read().split()]\nprint(sum(nums) // len(nums))\n",
            "data.txt": " ".join(str(i) for i in range(1, 11)) + "\n",
        },
        test='[ "$(python3 calc.py)" = "5.5" ]',
    ),
    TerminalTask(
        "csv-column-sum",
        "sales.csv has a header row. Compute the total of the 'amount' column and write it "
        "(as an integer) to total.txt.",
        files={"sales.csv": "id,region,amount\n1,east,100\n2,west,250\n3,east,75\n4,north,575\n"},
        test='[ "$(tr -d "[:space:]" < total.txt)" = "1000" ]',
    ),
    TerminalTask(
        "rename-ext",
        "In the directory docs/, rename every file ending in .txt to end in .md instead. Do not touch other files.",
        files={"docs/a.txt": "a\n", "docs/b.txt": "b\n", "docs/keep.rst": "k\n", "docs/sub/c.txt": "c\n"},
        test="[ -f docs/a.md ] && [ -f docs/b.md ] && [ -f docs/sub/c.md ] && [ -f docs/keep.rst ] "
        "&& [ -z \"$(find docs -name '*.txt')\" ]",
    ),
    TerminalTask(
        "largest-file",
        "Find the largest file (by bytes) anywhere under data/ and write its path relative to the "
        "current directory (e.g. data/x/y.bin) to largest.txt.",
        setup="mkdir -p data/a data/b/c && head -c 1200 /dev/zero > data/a/one.bin && "
        "head -c 5000 /dev/zero > data/b/c/big.bin && head -c 300 /dev/zero > data/b/two.bin",
        test='[ "$(tr -d "[:space:]" < largest.txt | sed "s#^\\./##")" = "data/b/c/big.bin" ]',
    ),
    TerminalTask(
        "hello-script",
        "Create an executable shell script named hello.sh that prints exactly 'Hello, RL!' when run as ./hello.sh.",
        test='[ -x hello.sh ] && [ "$(./hello.sh)" = "Hello, RL!" ]',
    ),
    TerminalTask(
        "json-edit",
        "config.json contains a JSON object. Set the field 'replicas' to 3 and add a field "
        "'env' with value 'prod'. Keep all other fields unchanged and the file valid JSON.",
        files={"config.json": json.dumps({"name": "svc", "replicas": 1, "port": 8080}, indent=2) + "\n"},
        test="python3 -c \"import json;c=json.load(open('config.json'));"
        "assert c=={'name':'svc','replicas':3,'port':8080,'env':'prod'}\"",
    ),
    TerminalTask(
        "unique-words",
        "words.txt contains one word per line, with duplicates. Write the distinct words, sorted "
        "alphabetically, one per line, to unique.txt.",
        files={"words.txt": "pear\napple\nfig\napple\npear\nbanana\nfig\n"},
        test='[ "$(cat unique.txt)" = "$(printf "apple\\nbanana\\nfig\\npear")" ]',
    ),
    TerminalTask(
        "archive-src",
        "Create a gzip-compressed tarball named src.tar.gz containing the src/ directory "
        "(paths inside the archive should start with src/).",
        files={"src/main.py": "print('hi')\n", "src/util/helpers.py": "X = 1\n"},
        test="tar -tzf src.tar.gz | grep -q '^src/main.py$' && tar -tzf src.tar.gz | grep -q '^src/util/helpers.py$'",
    ),
    TerminalTask(
        "make-test-pass",
        "Run the tests with `python3 -m unittest -q` and fix the bug in mathlib.py so the tests pass. "
        "Do not modify test_mathlib.py.",
        files={
            "mathlib.py": "def is_prime(n):\n    if n < 2:\n        return False\n"
            "    for d in range(2, n):\n        if n % d == 0:\n            return True\n    return False\n",
            "test_mathlib.py": "import unittest\nfrom mathlib import is_prime\n\n"
            "class T(unittest.TestCase):\n    def test(self):\n"
            "        self.assertEqual([n for n in range(20) if is_prime(n)], [2,3,5,7,11,13,17,19])\n",
        },
        test="grep -q '\\[2,3,5,7,11,13,17,19\\]' test_mathlib.py && python3 -m unittest -q 2>/dev/null",
    ),
    TerminalTask(
        "env-file",
        "Read settings.env (KEY=VALUE lines) and write the value of DB_HOST followed by a colon and "
        "the value of DB_PORT (e.g. host:1234) to dsn.txt.",
        files={"settings.env": "APP=demo\nDB_HOST=db.internal\n# comment\nDB_PORT=5433\n"},
        test='[ "$(tr -d "[:space:]" < dsn.txt)" = "db.internal:5433" ]',
    ),
]


def load_tasks() -> dict[str, TerminalTask]:
    tasks = {t.task_id: t for t in BUILTIN_TASKS}
    path = os.environ.get("RSCHED_TERMINAL_TASKS")
    if path:
        with open(path) as fh:
            for d in json.load(fh):
                d["executable"] = tuple(d.get("executable", ()))
                t = TerminalTask(**d)
                tasks[t.task_id] = t
    return tasks
