"""Agents under test, and the trajectory they leave behind.

ClaudeAgent is the real one: a manual tool loop over the Messages API, so every call, tool result,
token count and stop reason lands in the trajectory. The scripted agents need no model; they exist
to prove the harness and the graders before a paid run (see scripted.py).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

import anthropic

from agent_eval.env import SYSTEM_PROMPT, TOOLS, OrderEnv
from agent_eval.tasks import Task

# $ per million tokens: input, output. Cache writes cost 1.25x input and cache reads 0.1x input.
# Haiku 5.5's rates are for prompts up to 100K tokens, which every request here stays under.
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


def cost_usd(model: str, usage: dict) -> float:
    price_in, price_out = PRICES[model]
    return (usage["input_tokens"] * price_in
            + usage["cache_creation_input_tokens"] * price_in * 1.25
            + usage["cache_read_input_tokens"] * price_in * 0.1
            + usage["output_tokens"] * price_out) / 1_000_000


def usage_dict(usage: anthropic.types.Usage) -> dict:
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_creation_input_tokens": usage.cache_creation_input_tokens or 0,
        "cache_read_input_tokens": usage.cache_read_input_tokens or 0,
    }


class ServedModelMismatch(Exception):
    """The response came from a different model than the one requested: the attempt measures nothing."""


def is_requested_model(requested: str, served: str) -> bool:
    """The model asked for, or the dated snapshot an alias resolves to (claude-haiku-4-5 -> claude-haiku-4-5-20251001)."""
    return served == requested or re.fullmatch(re.escape(requested) + r"-\d{8}", served) is not None


@dataclass
class Trajectory:
    status: str = "completed"  # completed | truncated | refusal | step_limit | stopped:<reason>
    transcript: list[dict] = field(default_factory=list)  # {"role": user|assistant|tool, "text": ...}
    tool_calls: list[dict] = field(default_factory=list)  # {"name", "input", "result", "is_error"}
    calls: list[dict] = field(default_factory=list)  # per model call: usage, cost, latency, stop_reason
    served_model: str | None = None

    @property
    def cost_usd(self) -> float:
        return sum(call["cost_usd"] for call in self.calls)


@dataclass
class ClaudeAgent:
    client: anthropic.Anthropic
    model: str = "claude-haiku-5-5"
    effort: str | None = "medium"  # None: send no effort (Claude Haiku 4.5 rejects the parameter)
    max_tokens: int = 8000
    max_model_calls: int = 20

    @property
    def name(self) -> str:
        return f"{self.model}@{self.effort}" if self.effort else self.model

    def run(self, task: Task, env: OrderEnv) -> Trajectory:
        trajectory = Trajectory()
        replies = list(task.messages[1:])
        messages: list[dict] = [{"role": "user", "content": task.messages[0]}]
        trajectory.transcript.append({"role": "user", "text": task.messages[0]})

        for _ in range(self.max_model_calls):
            started = time.monotonic()
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
                tools=TOOLS,
                messages=messages,
                **({"output_config": {"effort": self.effort}} if self.effort else {}),
            )
            latency = time.monotonic() - started
            if not is_requested_model(self.model, response.model):
                raise ServedModelMismatch(f"asked for {self.model}, served by {response.model}")
            trajectory.served_model = response.model
            usage = usage_dict(response.usage)
            trajectory.calls.append({"stop_reason": response.stop_reason, "usage": usage,
                                     "cost_usd": cost_usd(self.model, usage), "latency_s": round(latency, 3),
                                     "request_id": response._request_id})

            # The whole content goes back, thinking blocks included: the model expects its own turn verbatim.
            messages.append({"role": "assistant", "content": response.content})
            text = "".join(block.text for block in response.content if block.type == "text").strip()
            if text:
                trajectory.transcript.append({"role": "assistant", "text": text})

            if response.stop_reason == "tool_use":
                results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    result, is_error = env.call(block.name, dict(block.input))
                    trajectory.tool_calls.append({"name": block.name, "input": dict(block.input),
                                                  "result": result, "is_error": is_error})
                    trajectory.transcript.append({"role": "tool",
                                                  "text": f"{block.name}({json.dumps(block.input)}) -> {result}"})
                    results.append({"type": "tool_result", "tool_use_id": block.id,
                                    "content": result, "is_error": is_error})
                messages.append({"role": "user", "content": results})
                continue

            if response.stop_reason == "end_turn":
                if replies:  # the scripted customer answers whatever the agent said or asked
                    reply = replies.pop(0)
                    messages.append({"role": "user", "content": reply})
                    trajectory.transcript.append({"role": "user", "text": reply})
                    continue
                return trajectory

            trajectory.status = {"max_tokens": "truncated", "refusal": "refusal"}.get(
                response.stop_reason, f"stopped:{response.stop_reason}")
            return trajectory

        trajectory.status = "step_limit"
        return trajectory
