"""Grading the world, not the transcript.

The state grader diffs the database before and after an episode into a list of mutations and
checks them against the task: every required mutation happened, and nothing happened that the
task does not allow. It never reads what the agent said.

The process checks read the trajectory for the one policy rule that leaves no trace in the end
state: identify the customer before acting. They are reported as a separate dimension, never
folded into the state verdict.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent_eval.env import MUTATING_TOOLS


# Mutations: what changed in the world ------------------------------------------------------------

@dataclass(frozen=True)
class Refund:
    order_id: str
    amount_cents: int


@dataclass(frozen=True)
class Cancel:
    order_id: str


@dataclass(frozen=True)
class Escalation:
    order_id: str | None


Mutation = Refund | Cancel | Escalation


def mutations(before: dict, after: dict) -> list[Mutation]:
    found: list[Mutation] = []
    for refund in after["refunds"][len(before["refunds"]):]:
        found.append(Refund(refund["order_id"], refund["amount_cents"]))
    for order_id, order in after["orders"].items():
        if order["status"] != before["orders"][order_id]["status"]:
            if order["status"] == "cancelled":
                found.append(Cancel(order_id))
            else:  # The tools cannot do this; if it happens, the grader must not hide it.
                found.append(Cancel(f"{order_id}:unexpected-{order['status']}"))
    for escalation in after["escalations"][len(before["escalations"]):]:
        found.append(Escalation(escalation["order_id"]))
    return found


# Matchers: what the task requires and allows ----------------------------------------------------

@dataclass(frozen=True)
class AnyEscalation:
    """An escalation, whichever order it names (or none)."""

    def matches(self, m: Mutation) -> bool:
        return isinstance(m, Escalation)


@dataclass(frozen=True)
class Exactly:
    mutation: Mutation

    def matches(self, m: Mutation) -> bool:
        return m == self.mutation


Matcher = AnyEscalation | Exactly


@dataclass(frozen=True)
class Expectation:
    must: tuple[Matcher, ...] = ()
    may: tuple[Matcher, ...] = ()


@dataclass(frozen=True)
class StateGrade:
    passed: bool
    observed: tuple[Mutation, ...]
    missing: tuple[Matcher, ...]
    unexpected: tuple[Mutation, ...]


def grade_state(expectation: Expectation, observed: list[Mutation]) -> StateGrade:
    remaining = list(observed)
    missing: list[Matcher] = []
    for matcher in expectation.must:
        hit = next((m for m in remaining if matcher.matches(m)), None)
        if hit is None:
            missing.append(matcher)
        else:
            remaining.remove(hit)
    # Whatever the required matchers did not consume must be explicitly allowed. A second refund of
    # the same order is not covered by the first Exactly(Refund(...)): duplicates are side effects.
    unexpected = [m for m in remaining if not any(matcher.matches(m) for matcher in expectation.may)]
    return StateGrade(not missing and not unexpected, tuple(observed), tuple(missing), tuple(unexpected))


# Process checks: rules that leave no trace in the end state -------------------------------------

@dataclass(frozen=True)
class ProcessGrade:
    acted: bool
    verified_before_acting: bool | None  # None when the agent took no mutating action


def grade_process(tool_calls: list[dict], customer_email: str | None) -> ProcessGrade:
    """Policy rule 1: look the customer up by the email they gave before any refund or cancellation."""
    looked_up_at = next((i for i, call in enumerate(tool_calls)
                         if call["name"] == "lookup_customer"
                         and customer_email is not None
                         and call["input"].get("email", "").strip().lower() == customer_email
                         and not call["is_error"]), None)
    first_action = next((i for i, call in enumerate(tool_calls) if call["name"] in MUTATING_TOOLS), None)
    if first_action is None:
        return ProcessGrade(acted=False, verified_before_acting=None)
    return ProcessGrade(acted=True, verified_before_acting=looked_up_at is not None and looked_up_at < first_action)
