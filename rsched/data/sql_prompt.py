"""Prompt + row builder shared by the SQL data scripts.

SkyRL-SQL uses ``<sql>..</sql>`` tags parsed by its own env. In veRL we use
native function calling (hermes format) with an ``execute_sql`` tool, and keep
SkyRL's ``<solution>..</solution>`` convention for the final answer so the
reward is the same.
"""

from __future__ import annotations

import re

SYSTEM = "You are a data science expert who answers questions by writing SQLite queries."

USER_TEMPLATE = """Below you are given a database schema and a natural language question. Write a SQLite query that answers the question.

Database Engine:
SQLite

Database Schema:
{schema}

External Knowledge:
{knowledge}

Question:
{question}

Instructions:
- You may call the `execute_sql` tool to explore the data or check a draft query; you have at most {max_turns} turns.
- Output only the information the question asks for: no missing and no extra columns.
- When you are done, give the final query inside <solution>...</solution> (exactly one solution block, no tool call in that turn)."""


def build_messages(schema: str, question: str, knowledge: str = "", max_turns: int = 6) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(
                schema=schema.strip(), question=question.strip(), knowledge=(knowledge or "").strip(), max_turns=max_turns
            ),
        },
    ]


def build_row(
    *, source: str, db_id: str, idx: int, schema: str, question: str, gold_sql: str, knowledge: str = "", split: str,
    max_turns: int = 6, agent_name: str = "tvcache_sql_agent",
) -> dict:
    return {
        "data_source": f"rsched_sql_{source}",
        "agent_name": agent_name,
        "prompt": build_messages(schema, question, knowledge, max_turns),
        "ability": "text2sql",
        "reward_model": {"style": "rule", "ground_truth": gold_sql},
        "extra_info": {
            "index": idx,
            "split": split,
            "db_id": db_id,
            "source": source,
            "gold_sql": gold_sql,
            "tvcache_task_id": f"sql|{source}|{db_id}|{split}-{idx}",
        },
    }


_SECTION = r"{name}:\s*\n(.*?)(?=\n\s*(?:This schema describes|External Knowledge:|Question:|Instructions:|Format:)|\Z)"


def parse_skyrl_prompt(text: str) -> tuple[str, str, str]:
    """Pull (schema, knowledge, question) out of a SkyRL-SQL user prompt."""

    def grab(name: str) -> str:
        m = re.search(_SECTION.format(name=re.escape(name)), text, re.S)
        return m.group(1).strip() if m else ""

    schema, knowledge, question = grab("Database Schema"), grab("External Knowledge"), grab("Question")
    if not schema or not question:
        raise ValueError("could not parse SkyRL prompt")
    return schema, knowledge, question
