"""Runs agents over the task catalogue and writes one row per (task, repetition).

    python -m agent_eval.run --agent oracle                      # free: proves the harness
    python -m agent_eval.run --agent claude --reps 5 --judge both --budget-usd 3

Rows that produced a scorable episode go to results.jsonl. Attempts that never did (an API error
after the SDK's retries, the wrong model answering) go to errors.jsonl: plumbing is not scored as
a model failure. Every trajectory is saved, so any surprising score can be read, not guessed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import anthropic

from agent_eval.agent import ClaudeAgent, ServedModelMismatch
from agent_eval.env import OrderEnv
from agent_eval.grade import grade_process, grade_state, mutations
from agent_eval.judge import Judge
from agent_eval.scripted import LiarAgent, NullAgent, OracleAgent
from agent_eval.tasks import TASKS, TASKS_BY_ID


def jsonable(value):
    if hasattr(value, "__dataclass_fields__"):
        return {"type": type(value).__name__, **{k: jsonable(v) for k, v in vars(value).items()}}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["claude", "oracle", "null", "liar"], required=True)
    parser.add_argument("--model", default="claude-haiku-5-5")
    parser.add_argument("--effort", default="medium", help="low|medium|high|xhigh|max, or none to omit it")
    parser.add_argument("--tasks", default="all", help="comma-separated task ids, or all")
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--judge", choices=["none", "narrative", "full", "both"], default="none")
    parser.add_argument("--budget-usd", type=float, default=1.0, help="hard stop for model spend, agent and judge together")
    parser.add_argument("--out", default="runs")
    args = parser.parse_args(argv)

    tasks = TASKS if args.tasks == "all" else tuple(TASKS_BY_ID[t] for t in args.tasks.split(","))
    needs_api = args.agent == "claude" or args.judge != "none"
    client = anthropic.Anthropic() if needs_api else None
    agent = {"claude": lambda: ClaudeAgent(client, args.model, None if args.effort == "none" else args.effort), "oracle": OracleAgent,
             "null": NullAgent, "liar": LiarAgent}[args.agent]()
    judge = Judge(client) if args.judge != "none" else None
    views = {"none": [], "narrative": ["narrative"], "full": ["full"], "both": ["narrative", "full"]}[args.judge]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out) / f"{stamp}-{agent.name.replace('@', '-')}"
    (out / "trajectories").mkdir(parents=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))

    spent = 0.0
    stopped_for_budget = False
    stopped_for_auth = False
    consecutive_auth_errors = 0
    with (out / "results.jsonl").open("w") as results, (out / "errors.jsonl").open("w") as errors:
        # Repetitions are the outer loop, so a budget stop leaves tasks at most one repetition apart; the report
        # then computes pass^k over the smallest count and flags the run as uneven.
        for rep in range(1, args.reps + 1):
            for task in tasks:
                if spent >= args.budget_usd:
                    stopped_for_budget = True
                    break
                env = OrderEnv(task.seed)
                before = env.snapshot()
                started = time.monotonic()
                try:
                    trajectory = agent.run(task, env)
                except (anthropic.APIError, ServedModelMismatch) as e:
                    errors.write(json.dumps({"task": task.id, "rep": rep, "class": type(e).__name__, "error": str(e)}) + "\n")
                    # A key that stopped working fails every remaining attempt; stop instead of logging them all.
                    consecutive_auth_errors = consecutive_auth_errors + 1 if isinstance(e, anthropic.AuthenticationError) else 0
                    if consecutive_auth_errors >= 3:
                        stopped_for_auth = True
                        break
                    continue
                consecutive_auth_errors = 0
                observed = mutations(before, env.snapshot())
                state = grade_state(task.expect, observed)
                process = grade_process(trajectory.tool_calls, task.customer_email)
                spent += trajectory.cost_usd

                verdicts = {}
                for view in views:
                    try:
                        verdicts[view] = judge.grade(trajectory.transcript, view)
                        spent += verdicts[view]["cost_usd"]
                    except (anthropic.APIError, ServedModelMismatch, json.JSONDecodeError) as e:
                        verdicts[view] = {"view": view, "verdict": None, "reason": f"{type(e).__name__}: {e}", "cost_usd": 0.0}

                row = {
                    "task": task.id, "category": task.category, "tier": task.tier, "agent": agent.name, "rep": rep,
                    "status": trajectory.status, "served_model": trajectory.served_model,
                    "state_pass": state.passed, "observed": jsonable(state.observed),
                    "missing": jsonable(state.missing), "unexpected": jsonable(state.unexpected),
                    "acted": process.acted, "verified_before_acting": process.verified_before_acting,
                    "tool_calls": len(trajectory.tool_calls), "model_calls": len(trajectory.calls),
                    "agent_cost_usd": round(trajectory.cost_usd, 6),
                    "usage": {k: sum(c["usage"][k] for c in trajectory.calls) for k in
                              ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")},
                    "agent_latency_s": round(sum(c["latency_s"] for c in trajectory.calls), 3),
                    "wall_s": round(time.monotonic() - started, 3),  # the episode plus judging it
                    "judge": {view: {k: v[k] for k in ("verdict", "reason", "cost_usd")} for view, v in verdicts.items()},
                }
                results.write(json.dumps(row) + "\n")
                results.flush()
                (out / "trajectories" / f"{task.id}-rep{rep}.json").write_text(json.dumps(
                    {"row": row, "transcript": trajectory.transcript, "tool_calls": trajectory.tool_calls,
                     "calls": trajectory.calls, "judge": verdicts}, indent=2))
                print(f"rep {rep} {task.id:<22} state={'PASS' if state.passed else 'fail':<4} "
                      f"status={trajectory.status:<10} spent=${spent:.4f}", flush=True)
            if stopped_for_budget or stopped_for_auth:
                break

    summary = {"out": str(out), "spent_usd": round(spent, 4), "stopped_for_budget": stopped_for_budget,
               "stopped_for_auth": stopped_for_auth}
    (out / "run.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
