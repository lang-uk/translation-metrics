"""TrueSkill ranking from pairwise judgments.

Computes TrueSkill ratings from three sources:
  1. Human evaluation (arena_task.__all__.incomplete.jsonl)
  2. LLM judge — translation quality (results/llm_judge/)
  3. LLM judge — literary quality (results/llm_judge_literary/)

Usage:
    python src/trueskill_rank.py           # compute & print all
    python src/trueskill_rank.py --plot    # also save plots

Requirements:
    pip install trueskill matplotlib
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import trueskill

ROOT = Path(__file__).resolve().parent.parent

HUMAN_EVAL_PATH = ROOT / "human_eval" / "arena_task.__all__.incomplete_2.jsonl"
SYSTEM_KEY_PATH = ROOT / "human_eval" / "system_key.json"
LLM_JUDGE_PATH = ROOT / "results" / "llm_judge" / "raw_judgments.jsonl"
LLM_LITERARY_PATH = ROOT / "results" / "llm_judge_literary" / "raw_judgments.jsonl"
RESULTS_DIR = ROOT / "results" / "trueskill"
PLOTS_DIR = ROOT / "plots" / "trueskill"

AI_SYSTEMS = {"gpt_5_2", "lapa_translations_combine", "deepl"}

ANON_TO_REAL = {}
REAL_TO_ANON = {}


def load_system_key():
    global ANON_TO_REAL, REAL_TO_ANON
    with open(SYSTEM_KEY_PATH) as f:
        REAL_TO_ANON = json.load(f)
    ANON_TO_REAL = {v: k for k, v in REAL_TO_ANON.items()}


def short_name(s: str) -> str:
    return (
        s.replace("UK_", "")
        .replace("_Kolhosp_tvaryn", "")
        .replace("_Khutir_tvaryn", "")
        .replace("_Skotoferma", "")
        .replace("_Ferma_rai_dlia_tvaryn", "")
        .replace("_Skotokhutir", "")
        .replace("_translations_combine", "")
    )


def is_ai(s: str) -> bool:
    return s in AI_SYSTEMS


# ── Load pairwise matches ────────────────────────────────────────────

def load_human_matches() -> list[tuple[str, str, str]]:
    """Return list of (winner, loser, 'tie') from human eval."""
    matches = []
    with open(HUMAN_EVAL_PATH) as f:
        for line in f:
            entries = json.loads(line.strip())
            for e in entries:
                ans = e.get("answer", {})
                verdict = ans.get("judgment", "")
                if verdict in ("bad_data", "error", ""):
                    continue
                left_anon = ans["left_system"]
                right_anon = ans["right_system"]
                left = ANON_TO_REAL.get(left_anon, left_anon)
                right = ANON_TO_REAL.get(right_anon, right_anon)

                if verdict == "system1":
                    matches.append((left, right, "win"))
                elif verdict == "system2":
                    matches.append((right, left, "win"))
                elif verdict == "tie":
                    matches.append((left, right, "tie"))
    return matches


def load_llm_matches(path: Path) -> list[tuple[str, str, str]]:
    """Return list of (winner, loser, 'tie') from LLM judge JSONL."""
    matches = []
    with open(path) as f:
        for line in f:
            j = json.loads(line.strip())
            verdict = j.get("verdict", "")
            if verdict == "error":
                continue
            left = j["real_left"]
            right = j["real_right"]

            if verdict == "system1":
                matches.append((left, right, "win"))
            elif verdict == "system2":
                matches.append((right, left, "win"))
            elif verdict == "tie":
                matches.append((left, right, "tie"))
    return matches


# ── TrueSkill computation ────────────────────────────────────────────

def compute_trueskill(matches: list[tuple[str, str, str]]) -> dict:
    env = trueskill.TrueSkill(draw_probability=0.15)
    ratings = defaultdict(lambda: env.create_rating())

    for a, b, outcome in matches:
        ra, rb = ratings[a], ratings[b]
        if outcome == "win":
            ra_new, rb_new = env.rate_1vs1(ra, rb)
        elif outcome == "tie":
            ra_new, rb_new = env.rate_1vs1(ra, rb, drawn=True)
        else:
            continue
        ratings[a] = ra_new
        ratings[b] = rb_new

    result = {}
    for sys, r in ratings.items():
        result[sys] = {
            "mu": r.mu,
            "sigma": r.sigma,
            "conservative": r.mu - 2 * r.sigma,
        }
    return dict(sorted(result.items(), key=lambda x: -x[1]["conservative"]))


def compute_win_rates(matches: list[tuple[str, str, str]]) -> dict:
    wins = defaultdict(int)
    losses = defaultdict(int)
    ties = defaultdict(int)

    for a, b, outcome in matches:
        if outcome == "win":
            wins[a] += 1
            losses[b] += 1
        elif outcome == "tie":
            ties[a] += 1
            ties[b] += 1

    result = {}
    all_sys = set(wins) | set(losses) | set(ties)
    for sys in all_sys:
        total = wins[sys] + losses[sys] + ties[sys]
        result[sys] = {
            "wins": wins[sys],
            "losses": losses[sys],
            "ties": ties[sys],
            "total": total,
            "win_rate": wins[sys] / total if total else 0,
        }
    return result


# ── Display ──────────────────────────────────────────────────────────

def print_rankings(name: str, ts: dict, wr: dict, n_matches: int):
    print(f"\n{'=' * 72}")
    print(f"  {name}  ({n_matches} matches)")
    print(f"{'=' * 72}")
    print(f"  {'#':>2}  {'System':<35} {'TrueSkill':>10} {'μ':>7} {'σ':>6} {'Win%':>6}  {'W/L/T':>10}")
    print(f"  {'-' * 70}")

    for i, (sys, ts_data) in enumerate(ts.items(), 1):
        name_str = short_name(sys)
        tag = " *" if is_ai(sys) else ""
        con = ts_data["conservative"]
        mu = ts_data["mu"]
        sigma = ts_data["sigma"]
        w = wr.get(sys, {})
        win_rate = w.get("win_rate", 0) * 100
        wlt = f"{w.get('wins', 0)}/{w.get('losses', 0)}/{w.get('ties', 0)}"
        print(f"  {i:>2}. {name_str:<35} {con:>9.2f}  {mu:>6.2f} {sigma:>5.2f} {win_rate:>5.1f}%  {wlt:>10}{tag}")


# ── Plot ─────────────────────────────────────────────────────────────

def plot_comparison(all_results: dict, show=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    C_HUMAN = "#4878CF"
    C_AI = "#E24A33"

    sources = list(all_results.keys())
    n_sources = len(sources)

    fig, axes = plt.subplots(1, n_sources, figsize=(7 * n_sources, 6), sharey=False)
    if n_sources == 1:
        axes = [axes]

    for ax, source_name in zip(axes, sources):
        ts = all_results[source_name]["trueskill"]
        systems = list(ts.keys())
        cons = [ts[s]["conservative"] for s in systems]
        mus = [ts[s]["mu"] for s in systems]
        sigmas = [ts[s]["sigma"] for s in systems]
        labels = [short_name(s).replace("_", " ") for s in systems]
        colors = [C_AI if is_ai(s) else C_HUMAN for s in systems]

        y = range(len(systems))
        ax.barh(y, cons, color=colors, edgecolor="white", linewidth=0.5, alpha=0.7)
        ax.errorbar(mus, y, xerr=[[2 * s for s in sigmas], [2 * s for s in sigmas]],
                     fmt="o", color="black", markersize=4, capsize=3, linewidth=1)

        ax.set_yticks(list(y))
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("TrueSkill (μ − 2σ)")
        ax.set_title(source_name)

        for i, (c, m) in enumerate(zip(cons, mus)):
            ax.text(c - 0.3, i, f"{c:.1f}", va="center", ha="right", fontsize=8)

    from matplotlib.patches import Patch
    axes[-1].legend(
        handles=[Patch(color=C_HUMAN, label="Human"), Patch(color=C_AI, label="AI")],
        loc="lower right",
    )

    fig.suptitle("TrueSkill Rankings: Human vs LLM Judges", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "trueskill_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved → {PLOTS_DIR / 'trueskill_comparison.png'}")

    if show:
        matplotlib.use("TkAgg")
        plt.show()


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TrueSkill ranking from pairwise judgments")
    parser.add_argument("--plot", action="store_true", help="Generate comparison plot")
    parser.add_argument("--show", action="store_true", help="Show plot interactively")
    args = parser.parse_args()

    load_system_key()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}

    sources = []
    if HUMAN_EVAL_PATH.exists():
        sources.append(("Human Eval", load_human_matches()))
    if LLM_JUDGE_PATH.exists():
        sources.append(("LLM Judge (translation)", load_llm_matches(LLM_JUDGE_PATH)))
    if LLM_LITERARY_PATH.exists():
        sources.append(("LLM Judge (literary)", load_llm_matches(LLM_LITERARY_PATH)))

    for name, matches in sources:
        ts = compute_trueskill(matches)
        wr = compute_win_rates(matches)
        print_rankings(name, ts, wr, len(matches))

        all_results[name] = {
            "n_matches": len(matches),
            "trueskill": ts,
            "win_rates": {s: d for s, d in wr.items()},
        }

    output_file = RESULTS_DIR / "trueskill_rankings.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved → {output_file}")

    if args.plot or args.show:
        plot_comparison(all_results, show=args.show)


if __name__ == "__main__":
    main()
