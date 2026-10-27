"""Re-grades saved episodes with a judge variant, without running the agent again.

    python -m agent_eval.regrade runs/<run-dir> --view full --prompt strict --label full-strict-repeat
    python -m agent_eval.regrade runs/<run-dir> --view narrative --prompt lenient --label narrative-lenient

Trajectories are data: a new grader, or the same grader a second time, costs only the judge calls.
Writes runs/<run-dir>/regrade-<label>.jsonl and prints agreement with the state grader and with the
verdict the same view gave during the original run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anthropic

from agent_eval.agent import ServedModelMismatch
from agent_eval.judge import Judge


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--view", choices=["narrative", "full"], required=True)
    parser.add_argument("--prompt", choices=["strict", "lenient"], default="strict")
    parser.add_argument("--label", required=True)
    parser.add_argument("--budget-usd", type=float, default=0.5)
    args = parser.parse_args(argv)

    run = Path(args.run)
    judge = Judge(anthropic.Anthropic(), prompt=args.prompt)
    out = run / f"regrade-{args.label}.jsonl"
    spent, rows = 0.0, []
    with out.open("w") as sink:
        for path in sorted((run / "trajectories").glob("*.json")):
            if spent >= args.budget_usd:
                print(f"budget reached at ${spent:.4f}")
                break
            saved = json.loads(path.read_text())
            try:
                verdict = judge.grade(saved["transcript"], args.view)
            except (anthropic.APIError, ServedModelMismatch, json.JSONDecodeError) as e:
                verdict = {"verdict": None, "reason": f"{type(e).__name__}: {e}", "cost_usd": 0.0}
            spent += verdict["cost_usd"]
            original = saved.get("judge", {}).get(args.view, {}).get("verdict")
            row = {"task": saved["row"]["task"], "rep": saved["row"]["rep"], "state_pass": saved["row"]["state_pass"],
                   "view": args.view, "prompt": args.prompt, "verdict": verdict["verdict"], "reason": verdict["reason"],
                   "original_verdict": original, "cost_usd": verdict["cost_usd"]}
            rows.append(row)
            sink.write(json.dumps(row) + "\n")

    graded = [r for r in rows if r["verdict"] is not None]
    agree_state = sum((r["verdict"] == "pass") == r["state_pass"] for r in graded)
    false_pass = sum(r["verdict"] == "pass" and not r["state_pass"] for r in graded)
    false_fail = sum(r["verdict"] == "fail" and r["state_pass"] for r in graded)
    compared = [r for r in graded if r["original_verdict"] is not None]
    same = sum(r["verdict"] == r["original_verdict"] for r in compared)
    summary = {"run": str(run), "label": args.label, "view": args.view, "prompt": args.prompt,
               "episodes": len(rows), "graded": len(graded), "agrees_with_state": agree_state,
               "passed_state_failures": false_pass, "failed_state_passes": false_fail,
               "same_as_original_verdict": same, "compared_with_original": len(compared), "spent_usd": round(spent, 4)}
    (run / f"regrade-{args.label}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
