"""Trace-driven simulator of the rollout loop: replay recorded timelines under
alternative schedules and predict hit rate, tool time and step makespan."""

from rsched.sim.engine import Policy, SimResult, simulate
from rsched.sim.metrics import summarize
from rsched.sim.trace import Run, load_run

__all__ = ["Policy", "Run", "SimResult", "load_run", "simulate", "summarize"]
