"""Agents with no model behind them, for testing the harness and the graders before any paid run.

- OracleAgent replays the task's hand-written correct trajectory: the state grader must pass it
  on every task, or the task or the grader is broken.
- NullAgent does nothing and says so: it must fail every task that requires a change and pass
  only the ones where doing nothing is correct.
- LiarAgent does nothing and claims it did what was asked: the state grader must fail it exactly
  where the null agent fails, whatever its narrative says. A grader that reads the transcript
  is what it is built to fool.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from agent_eval.agent import Trajectory
from agent_eval.env import OrderEnv
from agent_eval.tasks import Task


@dataclass
class OracleAgent:
    name: str = "oracle"

    def run(self, task: Task, env: OrderEnv) -> Trajectory:
        trajectory = Trajectory(served_model=None)
        for message in task.messages:
            trajectory.transcript.append({"role": "user", "text": message})
        for name, arguments in task.oracle:
            result, is_error = env.call(name, arguments)
            trajectory.tool_calls.append({"name": name, "input": arguments, "result": result, "is_error": is_error})
            trajectory.transcript.append({"role": "tool", "text": f"{name}({json.dumps(arguments)}) -> {result}"})
        trajectory.transcript.append({"role": "assistant", "text": "Done."})
        return trajectory


@dataclass
class NullAgent:
    name: str = "null"

    def run(self, task: Task, env: OrderEnv) -> Trajectory:
        trajectory = Trajectory(served_model=None)
        trajectory.transcript.append({"role": "user", "text": task.messages[0]})
        trajectory.transcript.append({"role": "assistant", "text": "I'm sorry, I can't help with that."})
        return trajectory


CLAIMS = {
    "act": "All done: I've taken care of that for you, and you'll see it reflected on your account shortly.",
    "decline": "I've passed this to a colleague who will follow up with you.",
    "clarify": "All done: I've taken care of that for you.",
}


@dataclass
class LiarAgent:
    name: str = "liar"

    def run(self, task: Task, env: OrderEnv) -> Trajectory:
        trajectory = Trajectory(served_model=None)
        trajectory.transcript.append({"role": "user", "text": task.messages[0]})
        trajectory.transcript.append({"role": "assistant", "text": CLAIMS[task.category]})
        return trajectory
