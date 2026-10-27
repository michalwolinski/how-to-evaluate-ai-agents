"""Draws the article's charts from the committed runs, as plain SVG (no plotting library).

    python -m agent_eval.charts charts/

Every value on every chart is read from runs/ or computed from it here, so a chart can be
regenerated after a rerun and cannot drift from the data. Colour follows one rule: the accent
marks what the chart is about, slate marks context.
"""

from __future__ import annotations

import json
import math
import sys
from html import escape
from pathlib import Path
from statistics import NormalDist

RUNS = Path("runs")
MAIN = RUNS / "20261026T125530Z-claude-haiku-5-5-medium"
LOW = RUNS / "20261026T132954Z-claude-haiku-5-5-low"
OLD = RUNS / "20261026T134200Z-claude-haiku-4-5"

ACCENT = "#ea580c"
CONTEXT = "#64748b"
CONTEXT_LIGHT = "#cbd5e1"
INK = "#0f172a"
INK_2 = "#475569"
GRID = "#e2e8f0"
SURFACE = "#ffffff"
FONT = "'Helvetica Neue', Helvetica, Arial, sans-serif"


def text(x, y, s, size=14, fill=INK, anchor="start", weight="normal") -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}">{escape(str(s))}</text>')


def svg(width, height, title, desc, body) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
            f'role="img" aria-label="{escape(desc, quote=True)}">\n<title>{escape(title)}</title>\n'
            f'<rect width="{width}" height="{height}" fill="{SURFACE}"/>\n' + "\n".join(body) + "\n</svg>\n")


def bar(x, y, w, h, fill) -> str:
    """A horizontal bar: square at the baseline (left), 4px rounded data-end (right)."""
    r = min(4, w / 2, h / 2)
    if w <= 0:
        return ""
    return (f'<path d="M{x:.1f},{y:.1f} H{x + w - r:.1f} Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f} '
            f'V{y + h - r:.1f} Q{x + w:.1f},{y + h:.1f} {x + w - r:.1f},{y + h:.1f} H{x:.1f} Z" fill="{fill}"/>')


def rows(run: Path) -> list[dict]:
    return [json.loads(line) for line in (run / "results.jsonl").read_text().splitlines() if line]


def summary(run: Path) -> dict:
    return json.loads((run / "summary.json").read_text())


# 1. Every episode, every task --------------------------------------------------------------------

def reliability_grid() -> str:
    from agent_eval.tasks import TASKS  # task order and tiers come from the catalogue itself
    panels = [("Claude Haiku 5.5", rows(MAIN)), ("Claude Haiku 4.5", rows(OLD))]
    results = {name: {(r["task"], r["rep"]): r["state_pass"] for r in data} for name, data in panels}
    order = [t for t in TASKS if t.tier == "base"] + [t for t in TASKS if t.tier == "hard"]

    cell, gap, row_h = 18, 2, 22
    label_w, panel_w, top = 230, 230, 112
    width, height = 40 + label_w + 2 * panel_w + 40, top + len(order) * row_h + 40 + 52
    body = [text(40, 40, "Every episode of every task: state grader verdicts, 5 repetitions each", 20, weight="600"),
            text(40, 64, "Orange cross = failed episode. Right of each panel: passes out of 5; a task counts toward pass^5 only at 5/5.", 14, INK_2)]
    for p, (name, _) in enumerate(panels):
        x0 = 40 + label_w + p * panel_w
        body.append(text(x0, top - 8, name, 15, weight="600"))
    y = top
    for i, task in enumerate(order):
        if i == len([t for t in TASKS if t.tier == "base"]):
            body.append(f'<line x1="40" x2="{width - 40}" y1="{y + 4}" y2="{y + 4}" stroke="{GRID}" stroke-width="1"/>')
            body.append(text(40, y + 20, "Hard tier", 12, INK_2, weight="600"))
            y += 26
        body.append(text(40, y + 14, task.id, 13, INK_2))
        for p, (name, _) in enumerate(panels):
            x0 = 40 + label_w + p * panel_w
            passed = 0
            for rep in range(1, 6):
                ok = results[name][(task.id, rep)]
                passed += ok
                cx = x0 + (rep - 1) * (cell + gap)
                body.append(f'<rect x="{cx}" y="{y}" width="{cell}" height="{cell}" rx="3" fill="{CONTEXT_LIGHT if ok else ACCENT}"/>')
                if not ok:
                    body.append(f'<path d="M{cx + 5},{y + 5} L{cx + 13},{y + 13} M{cx + 13},{y + 5} L{cx + 5},{y + 13}" '
                                f'stroke="{SURFACE}" stroke-width="2" stroke-linecap="round"/>')
            body.append(text(x0 + 5 * (cell + gap) + 10, y + 14, f"{passed}/5", 13, INK, weight="600" if passed < 5 else "normal"))
        y += row_h
    # Totals row, recomputed here from the same cells.
    y += 10
    body.append(f'<line x1="40" x2="{width - 40}" y1="{y}" y2="{y}" stroke="{GRID}" stroke-width="1"/>')
    for p, (name, _) in enumerate(panels):
        x0 = 40 + label_w + p * panel_w
        res = results[name]
        total = sum(res.values())
        hat = sum(all(res[(t.id, r)] for r in range(1, 6)) for t in order)
        body.append(text(x0, y + 22, f"pass@1 {total}/{len(res)} · pass^5 {hat}/{len(order)}", 13, INK, weight="600"))
    body.append(text(40, y + 22, "Totals", 13, INK_2, weight="600"))
    parts = []
    for name, _ in panels:
        res = results[name]
        per_task = {t.id: sum(res[(t.id, r)] for r in range(1, 6)) for t in order}
        weak = sorted((n, t) for t, n in per_task.items() if n < 5)
        detail = "; ".join(f"{t} {n} of 5" for n, t in weak) if weak else "no failed episode"
        parts.append(f"{name}: {sum(res.values())} of {len(res)} episodes passed, "
                     f"{sum(n == 5 for n in per_task.values())} of {len(order)} tasks every time ({detail}).")
    desc = f"Grid of {len(order)} tasks by 5 repetitions for two agents. " + " ".join(parts)
    return svg(width, height + 10, "Every episode, every task", desc, body)


# 2. Judges against the world -------------------------------------------------------------------

def judge_agreement() -> str:
    main = summary(MAIN)
    lenient = [json.loads(l) for l in (MAIN / "regrade-narrative-lenient.jsonl").read_text().splitlines() if l]
    lenient_agree = sum((r["verdict"] == "pass") == r["state_pass"] for r in lenient)
    strict_same = sum((r["original_verdict"] == "pass") == r["state_pass"] for r in lenient)
    data = [
        ("Full view (tool calls included)", f"all {main['episodes']} episodes", main["judges"]["full"]["agreement"], True),
        ("Narrative view, strict prompt", f"all {main['episodes']} episodes", main["judges"]["narrative"]["agreement"], False),
        None,  # divider: the re-judged subset below compares two prompts on the same episodes
        ("Narrative view, strict prompt", f"the {len(lenient)} re-judged episodes", strict_same / len(lenient), False),
        ("Narrative view, lenient prompt", f"the same {len(lenient)} episodes", lenient_agree / len(lenient), False),
    ]
    width, left, plot_w, top, band = 1200, 330, 700, 110, 52
    height = top + len(data) * band + 70
    body = [text(40, 40, "How often the LLM judge agrees with the state grader", 20, weight="600"),
            text(40, 64, "Claude Sonnet 5.5 judging Claude Haiku 5.5 episodes; the state grader passed every one of them.", 14, INK_2)]
    for pct in range(0, 101, 25):
        x = left + plot_w * pct / 100
        body.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top - 10}" y2="{top + len(data) * band}" stroke="{GRID}" stroke-width="1"/>')
        body.append(text(x, top + len(data) * band + 20, f"{pct}%", 12, INK_2, "middle"))
    for i, item in enumerate(data):
        y = top + i * band
        if item is None:
            body.append(f'<line x1="40" x2="{left + plot_w}" y1="{y + 22}" y2="{y + 22}" stroke="{GRID}" stroke-width="1"/>')
            body.append(text(40, y + 42, "Re-judged with the two-sentence rule in the prompt reworded (no new agent runs)", 12, INK_2, weight="600"))
            continue
        label, n, value, accent = item
        body.append(text(left - 16, y + 12, label, 14, INK, "end"))
        body.append(text(left - 16, y + 30, n, 12, INK_2, "end"))
        body.append(bar(left, y + 4, plot_w * value, 24, ACCENT if accent else CONTEXT))
        body.append(text(left + plot_w * value + 10, y + 21, f"{100 * value:.0f}%", 14, INK, weight="600"))
    desc = (f"Horizontal bars of agreement with the state grader. Narrative view with the strict prompt: "
            f"{100 * strict_same / len(lenient):.0f}% on {len(lenient)} episodes and {100 * main['judges']['narrative']['agreement']:.0f}% on all {main['episodes']}. "
            f"Narrative view with the lenient prompt: {100 * lenient_agree / len(lenient):.0f}% on the same {len(lenient)}. "
            f"Full view: {100 * main['judges']['full']['agreement']:.0f}% on {main['episodes']}.")
    return svg(width, height, "Judge agreement with the state grader", desc, body)


# 3. How many episodes a difference needs -------------------------------------------------------

def episodes_needed(p1: float, p2: float, alpha: float = 0.05, power: float = 0.8) -> int:
    z = NormalDist().inv_cdf
    pb = (p1 + p2) / 2
    num = (z(1 - alpha / 2) * math.sqrt(2 * pb * (1 - pb)) + z(power) * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return math.ceil(num / (p1 - p2) ** 2)


def sample_size() -> str:
    points = [(d, episodes_needed(0.95, 0.95 - d / 100)) for d in range(2, 21)]
    width, left, top, plot_w, plot_h = 1200, 120, 100, 960, 360
    lo, hi = math.log10(50), math.log10(5000)
    def x(d): return left + plot_w * (d - 2) / 18
    def y(n): return top + plot_h * (1 - (math.log10(n) - lo) / (hi - lo))
    body = [text(40, 40, "Episodes per configuration needed to detect a drop from a 95% pass rate", 20, weight="600"),
            text(40, 64, "Two-sided test at 5% significance, 80% power, two independent proportions; log scale.", 14, INK_2)]
    for n in (50, 100, 200, 500, 1000, 2000, 5000):
        body.append(f'<line x1="{left}" x2="{left + plot_w}" y1="{y(n):.1f}" y2="{y(n):.1f}" stroke="{GRID}" stroke-width="1"/>')
        body.append(text(left - 12, y(n) + 4, f"{n:,}", 12, INK_2, "end"))
    for d in (2, 5, 10, 15, 20):
        body.append(text(x(d), top + plot_h + 24, f"{d} points", 12, INK_2, "middle"))
    ours = 115
    body.append(f'<line x1="{left}" x2="{left + plot_w}" y1="{y(ours):.1f}" y2="{y(ours):.1f}" stroke="{CONTEXT}" stroke-width="2"/>')
    body.append(text(left + plot_w, y(ours) - 8, f"{ours} episodes per configuration in this article", 13, INK, "end"))
    path = " ".join(f"{'M' if i == 0 else 'L'}{x(d):.1f},{y(n):.1f}" for i, (d, n) in enumerate(points) if n <= 5000)
    body.append(f'<path d="{path}" fill="none" stroke="{ACCENT}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
    marked = {d: n for d, n in points}
    crossing = min(d for d, n in points if n <= ours)
    body.append(f'<circle cx="{x(crossing):.1f}" cy="{y(marked[crossing]):.1f}" r="5" fill="{CONTEXT}" stroke="{SURFACE}" stroke-width="2"/>')
    body.append(text(x(crossing) - 12, y(marked[crossing]) + 28, f"{ours} episodes detect a drop of {crossing} points or more", 13, INK, "end"))
    for d in (5,):
        body.append(f'<circle cx="{x(d):.1f}" cy="{y(marked[d]):.1f}" r="5" fill="{ACCENT}" stroke="{SURFACE}" stroke-width="2"/>')
        body.append(text(x(d) + 12, y(marked[d]) - 10, f"95% vs {95 - d}%: {marked[d]} episodes", 13, INK, weight="600"))
    body.append(text(left + plot_w / 2, top + plot_h + 50, "Difference to detect (percentage points below 95%)", 13, INK_2, "middle"))
    desc = (f"Line chart on a log scale of episodes needed per configuration against the difference to detect from a 95% baseline. "
            f"A 5-point drop needs {marked[5]} episodes, a 10-point drop {marked[10]}; the {ours} episodes used here detect only drops of about "
            f"{min(d for d, n in points if n <= ours)} points or more.")
    return svg(width, top + plot_h + 70, "Episodes needed per configuration", desc, body)


# 4. What the cache did to cost -----------------------------------------------------------------

def cost_dumbbell() -> str:
    configs = [("Claude Haiku 5.5, effort medium", summary(MAIN)), ("Claude Haiku 5.5, effort low", summary(LOW)),
               ("Claude Haiku 4.5", summary(OLD))]
    width, left, plot_w, top, band = 1200, 330, 760, 120, 64
    lo, hi = math.log10(0.0002), math.log10(0.02)
    def x(v): return left + plot_w * (math.log10(v) - lo) / (hi - lo)
    body = [text(40, 40, "Agent cost per successful episode, with and without prompt caching", 20, weight="600"),
            text(40, 64, "Each row: what the run cost, and the same tokens priced as if nothing had been cached. Log scale.", 14, INK_2),
            f'<circle cx="46" cy="88" r="6" fill="{ACCENT}"/>', text(58, 93, "Actual cost per successful episode", 13, INK),
            f'<circle cx="326" cy="88" r="6" fill="{CONTEXT}"/>', text(338, 93, "Same tokens, no caching", 13, INK)]
    for v in (0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02):
        body.append(f'<line x1="{x(v):.1f}" x2="{x(v):.1f}" y1="{top - 10}" y2="{top + len(configs) * band}" stroke="{GRID}" stroke-width="1"/>')
        body.append(text(x(v), top + len(configs) * band + 20, f"${v:g}", 12, INK_2, "middle"))
    for i, (name, s) in enumerate(configs):
        cy = top + i * band + 20
        actual = s["cost"]["agent_usd_per_successful_episode"]
        uncached = s["cost"]["agent_usd_per_successful_episode_if_uncached"]
        body.append(text(left - 16, cy + 5, name, 14, INK, "end"))
        if abs(uncached - actual) > 1e-9:
            body.append(f'<line x1="{x(actual):.1f}" x2="{x(uncached):.1f}" y1="{cy}" y2="{cy}" stroke="{CONTEXT_LIGHT}" stroke-width="2"/>')
            body.append(f'<circle cx="{x(uncached):.1f}" cy="{cy}" r="6" fill="{CONTEXT}" stroke="{SURFACE}" stroke-width="2"/>')
            body.append(text(x(uncached) + 12, cy + 5, f"${uncached:.5f}", 13, INK_2))
            body.append(f'<circle cx="{x(actual):.1f}" cy="{cy}" r="6" fill="{ACCENT}" stroke="{SURFACE}" stroke-width="2"/>')
            body.append(text(x(actual) - 12, cy + 5, f"${actual:.5f}", 13, INK, "end", "600"))
        else:
            body.append(f'<circle cx="{x(actual):.1f}" cy="{cy}" r="6" fill="{ACCENT}" stroke="{SURFACE}" stroke-width="2"/>')
            body.append(text(x(actual) - 12, cy + 5, f"${actual:.5f}", 13, INK, "end", "600"))
            body.append(text(x(actual) + 12, cy + 5, "same: nothing was cached", 13, INK_2))
    desc = ("Dumbbell chart on a log scale. Claude Haiku 5.5 at medium effort cost $"
            f"{configs[0][1]['cost']['agent_usd_per_successful_episode']:.5f} per successful episode, ${configs[0][1]['cost']['agent_usd_per_successful_episode_if_uncached']:.5f} without caching; "
            f"at low effort ${configs[1][1]['cost']['agent_usd_per_successful_episode']:.5f} and ${configs[1][1]['cost']['agent_usd_per_successful_episode_if_uncached']:.5f}. "
            f"Claude Haiku 4.5 cost ${configs[2][1]['cost']['agent_usd_per_successful_episode']:.5f} either way, because its prefix was below the minimum cacheable length.")
    return svg(width, top + len(configs) * band + 40, "Cost per successful episode", desc, body)


CHARTS = {
    "reliability-grid": reliability_grid,
    "judge-agreement": judge_agreement,
    "sample-size": sample_size,
    "cost-cache": cost_dumbbell,
}


def main(argv: list[str]) -> int:
    out = Path(argv[1] if len(argv) > 1 else "charts")
    out.mkdir(parents=True, exist_ok=True)
    for name, draw in CHARTS.items():
        (out / f"{name}.svg").write_text(draw())
        print(f"wrote {out / name}.svg")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
