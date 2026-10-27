"""Summarises a run directory. Every number is recomputed from the per-episode rows.

    python -m agent_eval.report runs/<run-dir>
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from agent_eval.agent import PRICES, cost_usd


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


FAILURE_SHAPES = ("harmful action", "escalated instead of acting", "stopped to ask", "did not act", "declined without handoff")


def failure_shape(row: dict, final_text: str) -> str:
    """Sorts a failed episode by what it did to the world. Order matters: the most severe shape wins."""
    unexpected = {m["type"] for m in row["unexpected"]}
    missing_action = any(m["type"] == "Exactly" for m in row["missing"])
    if unexpected & {"Refund", "Cancel"}:
        return "harmful action"
    if missing_action and "Escalation" in unexpected:
        return "escalated instead of acting"
    if missing_action:
        return "stopped to ask" if final_text.rstrip().endswith("?") else "did not act"
    return "declined without handoff"


def summarise(rows: list[dict]) -> dict:
    scored = [r for r in rows if r["status"] == "completed"]
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in scored:
        by_task[r["task"]].append(r)
    reps = {len(v) for v in by_task.values()}
    k = min(reps) if reps else 0

    passes = sum(r["state_pass"] for r in scored)
    lo, hi = wilson(passes, len(scored))
    complete = {t: v for t, v in by_task.items() if len(v) >= k}
    pass_hat_k = sum(all(r["state_pass"] for r in v[:k]) for v in complete.values())
    pass_at_k = sum(any(r["state_pass"] for r in v[:k]) for v in complete.values())

    categories = {}
    for category in sorted({r["category"] for r in scored}):
        rs = [r for r in scored if r["category"] == category]
        ok = sum(r["state_pass"] for r in rs)
        categories[category] = {"passed": ok, "episodes": len(rs), "ci95": wilson(ok, len(rs))}

    tiers = {}
    for tier in sorted({r.get("tier", "base") for r in scored}):
        rs = [r for r in scored if r.get("tier", "base") == tier]
        ok = sum(r["state_pass"] for r in rs)
        tasks_in_tier = {t: v for t, v in by_task.items() if v[0].get("tier", "base") == tier and len(v) >= k}
        tiers[tier] = {"passed": ok, "episodes": len(rs), "ci95": wilson(ok, len(rs)),
                       "pass_hat_k": sum(all(r["state_pass"] for r in v[:k]) for v in tasks_in_tier.values()),
                       "tasks": len(tasks_in_tier)}

    shapes = defaultdict(int)
    for r in scored:
        if not r["state_pass"]:
            shapes[failure_shape(r, r.get("final_text", ""))] += 1
    accepting_questions = passes + shapes["stopped to ask"]

    acted = [r for r in scored if r["acted"]]
    judges = {}
    for view in sorted({v for r in scored for v in r["judge"]}):
        cells = defaultdict(int)
        for r in scored:
            verdict = r["judge"].get(view, {}).get("verdict")
            if verdict is None:
                cells["no verdict"] += 1
                continue
            cells[f"state {'pass' if r['state_pass'] else 'fail'} / judge {verdict}"] += 1
        graded = sum(v for key, v in cells.items() if key != "no verdict")
        agree = cells["state pass / judge pass"] + cells["state fail / judge fail"]
        judges[view] = {"cells": dict(cells), "agreement": agree / graded if graded else None,
                        "judge_passes_state_failures": cells["state fail / judge pass"],
                        "judge_fails_state_passes": cells["state pass / judge fail"],
                        "cost_usd": round(sum(r["judge"].get(view, {}).get("cost_usd", 0.0) for r in scored), 4)}

    agent_cost = sum(r["agent_cost_usd"] for r in rows)
    # The same tokens priced as if nothing had been read from or written to the cache: separates what the
    # model costs from what caching saved, which differs by model (Haiku 4.5 cannot cache a prefix this short).
    uncached = 0.0
    for r in rows:
        model = r["agent"].split("@")[0]
        if model in PRICES:
            u = r["usage"]
            uncached += cost_usd(model, {"input_tokens": u["input_tokens"] + u["cache_read_input_tokens"] + u["cache_creation_input_tokens"],
                                         "output_tokens": u["output_tokens"], "cache_creation_input_tokens": 0,
                                         "cache_read_input_tokens": 0})
    return {
        "agent": sorted({r["agent"] for r in rows}),
        "episodes": len(rows),
        "status": dict(sorted({s: sum(r["status"] == s for r in rows) for s in {r["status"] for r in rows}}.items())),
        "tasks": len(by_task), "reps_per_task": k,
        "pass_at_1": {"passed": passes, "episodes": len(scored), "rate": passes / len(scored) if scored else None,
                      "ci95": (lo, hi)},
        "pass_hat_k": {"k": k, "tasks_all_passed": pass_hat_k, "tasks": len(complete)},
        "pass_at_k": {"k": k, "tasks_any_passed": pass_at_k, "tasks": len(complete)},
        "per_task": {t: {"passed": sum(r["state_pass"] for r in v), "episodes": len(v)} for t, v in sorted(by_task.items())},
        "uneven_reps": len(reps) > 1,
        "categories": categories,
        "failure_shapes": {shape: shapes[shape] for shape in FAILURE_SHAPES if shapes[shape]},
        "pass_at_1_if_asking_counts": {"passed": accepting_questions, "episodes": len(scored)},
        "tiers": tiers,
        "process": {"acted": len(acted), "verified_before_acting": sum(bool(r["verified_before_acting"]) for r in acted)},
        "judges": judges,
        "cost": {"agent_usd": round(agent_cost, 4),
                 "agent_usd_per_successful_episode": round(agent_cost / passes, 5) if passes else None,
                 "agent_usd_if_uncached": round(uncached, 4),
                 "agent_usd_per_successful_episode_if_uncached": round(uncached / passes, 5) if passes else None,
                 "judge_usd": round(sum(j["cost_usd"] for j in judges.values()), 4)},
        "tokens": {k_: sum(r["usage"][k_] for r in rows) for k_ in rows[0]["usage"]} if rows else {},
        "cache_read_share": (lambda t: t["cache_read_input_tokens"] / max(1, t["input_tokens"] + t["cache_read_input_tokens"]
                                                                      + t["cache_creation_input_tokens"]))(
            {k_: sum(r["usage"][k_] for r in rows) for k_ in rows[0]["usage"]}) if rows else None,
        "median_agent_latency_s": statistics.median(r["agent_latency_s"] for r in rows) if rows else None,
    }


def main(argv: list[str]) -> int:
    run = Path(argv[1])
    rows = [json.loads(line) for line in (run / "results.jsonl").read_text().splitlines() if line]
    for row in rows:  # the final reply lives in the trajectory, not in the row
        trajectory = run / "trajectories" / f"{row['task']}-rep{row['rep']}.json"
        if trajectory.exists():
            said = [line["text"] for line in json.loads(trajectory.read_text())["transcript"] if line["role"] == "assistant"]
            row["final_text"] = said[-1] if said else ""
    errors = [json.loads(line) for line in (run / "errors.jsonl").read_text().splitlines() if line]
    summary = summarise(rows)
    summary["errors"] = len(errors)
    (run / "summary.json").write_text(json.dumps(summary, indent=2))

    s = summary
    k = s["reps_per_task"]
    p1 = s["pass_at_1"]
    print(f"agent {', '.join(s['agent'])}: {s['episodes']} episodes, {s['tasks']} tasks x {k} reps, "
          f"{s['errors']} infrastructure errors, status {s['status']}")
    if p1["rate"] is not None:
        print(f"pass@1 {p1['passed']}/{p1['episodes']} = {pct(p1['rate'])} (95% CI {pct(p1['ci95'][0])}-{pct(p1['ci95'][1])})")
        print(f"pass^{k} {s['pass_hat_k']['tasks_all_passed']}/{s['pass_hat_k']['tasks']} tasks   "
              f"pass@{k} {s['pass_at_k']['tasks_any_passed']}/{s['pass_at_k']['tasks']} tasks")
    for name, c in s["categories"].items():
        print(f"  {name:<8} {c['passed']}/{c['episodes']} (95% CI {pct(c['ci95'][0])}-{pct(c['ci95'][1])})")
    for name, t in s["tiers"].items():
        print(f"  tier {name:<5} {t['passed']}/{t['episodes']} (95% CI {pct(t['ci95'][0])}-{pct(t['ci95'][1])}), pass^{k} {t['pass_hat_k']}/{t['tasks']} tasks")
    if s["uneven_reps"]:
        print(f"WARNING: tasks have different numbers of scored reps; pass^k and pass@k use the first {k} of each")
    print("per task:", " ".join(f"{t}={n['passed']}/{n['episodes']}" for t, n in s["per_task"].items()))
    if s["failure_shapes"]:
        print("failures by shape:", ", ".join(f"{k_} {v}" for k_, v in s["failure_shapes"].items()))
        q = s["pass_at_1_if_asking_counts"]
        print(f"pass@1 if stopping to ask for confirmation counted as correct: {q['passed']}/{q['episodes']} = {pct(q['passed'] / q['episodes'])}")
    print(f"process: verified the customer before acting in {s['process']['verified_before_acting']}/{s['process']['acted']} episodes that acted")
    for view, j in s["judges"].items():
        print(f"judge[{view}]: agreement {pct(j['agreement']) if j['agreement'] is not None else 'n/a'}; "
              f"passed {j['judge_passes_state_failures']} that the state failed, failed {j['judge_fails_state_passes']} that the state passed; {j['cells']}")
    print(f"cost: agent ${s['cost']['agent_usd']} (${s['cost']['agent_usd_per_successful_episode']} per successful episode), "
          f"judge ${s['cost']['judge_usd']}; tokens {s['tokens']}; cache-read share {pct(s['cache_read_share'] or 0)}; "
          f"median agent latency {s['median_agent_latency_s']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
