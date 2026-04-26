"""Visualise stylometry results.

Usage:
    python -m src.stylometry.plot             # save PNGs to plots/stylometry/
    python -m src.stylometry.plot --show      # also open interactively
"""

import argparse
import json
from pathlib import Path

from . import _common as common

ROOT = common.ROOT
RESULTS_DIR = ROOT / "results" / "stylometry"
PLOTS_DIR = ROOT / "plots" / "stylometry"

C_HUMAN = "#4878CF"
C_AI = "#E24A33"
C_GREY = "#8B8B8B"


def _is_ai(name: str) -> bool:
    low = name.lower()
    return any(k in low for k in ("gpt", "lapa", "deepl"))


def _display(name: str) -> str:
    return common.short_name(name).replace("_", " ")


def _load(filename: str) -> dict:
    with open(RESULTS_DIR / filename) as f:
        return json.load(f)


def _bar_colors(names: list[str]) -> list[str]:
    return [C_AI if _is_ai(n) else C_HUMAN for n in names]


def _legend(ax, plt):
    from matplotlib.patches import Patch
    ax.legend(
        handles=[Patch(color=C_HUMAN, label="Human"),
                 Patch(color=C_AI, label="AI")],
        loc="lower right",
    )


# ── Individual panels ────────────────────────────────────────────────

def _plot_discourse_particles(plt):
    data = _load("discourse_particles.json")
    systems = [s for s in common.ALL_SYSTEMS if s in data]
    systems.sort(key=lambda s: -data[s]["per_1k"])
    vals = [data[s]["per_1k"] for s in systems]
    labels = [_display(s) for s in systems]
    colors = _bar_colors(systems)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.barh(range(len(systems)), vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(systems)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Discourse particles per 1,000 words")
    ax.set_title("Discourse Particle Density")
    for bar, v in zip(bars, vals):
        ax.text(v + 0.1, bar.get_y() + bar.get_height() / 2, f"{v:.1f}",
                va="center", fontsize=9)
    _legend(ax, plt)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "discourse_particles.png", dpi=150)
    plt.close(fig)
    print("  discourse_particles.png")


def _plot_cosine_delta(plt):
    data = _load("cosine_delta.json")
    pw = data.get("_pairwise", {})

    systems = [s for s in common.ALL_SYSTEMS if s in data and s != "_pairwise"]
    systems.sort(key=lambda s: -data[s]["mean_distance"])
    vals = [data[s]["mean_distance"] for s in systems]
    labels = [_display(s) for s in systems]
    colors = _bar_colors(systems)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), gridspec_kw={"width_ratios": [1, 1.3]})

    ax = axes[0]
    bars = ax.barh(range(len(systems)), vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(systems)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Mean pairwise Cosine Delta")
    ax.set_title("Cosine Delta (stylistic distance)")
    for bar, v in zip(bars, vals):
        ax.text(v + 0.003, bar.get_y() + bar.get_height() / 2, f"{v:.3f}",
                va="center", fontsize=8)
    _legend(ax, plt)

    if pw:
        all_short = [common.short_name(s) for s in common.ALL_SYSTEMS]
        n = len(all_short)
        matrix = [[0.0] * n for _ in range(n)]
        for key, val in pw.items():
            a_str, b_str = key.split("|")
            a_sn, b_sn = common.short_name(a_str), common.short_name(b_str)
            if a_sn in all_short and b_sn in all_short:
                i, j = all_short.index(a_sn), all_short.index(b_sn)
                matrix[i][j] = val
                matrix[j][i] = val

        import numpy as np
        mat = np.array(matrix)
        dlabels = [_display(s) for s in common.ALL_SYSTEMS]

        ax2 = axes[1]
        triu_vals = mat[np.triu_indices(n, k=1)]
        vmin, vmax = float(np.min(triu_vals)), float(np.max(triu_vals))
        im = ax2.imshow(mat, cmap="RdYlGn_r", vmin=vmin - 0.02, vmax=vmax + 0.02,
                        aspect="equal")
        ax2.set_xticks(range(n))
        ax2.set_yticks(range(n))
        ax2.set_xticklabels(dlabels, rotation=45, ha="right", fontsize=7.5)
        ax2.set_yticklabels(dlabels, fontsize=7.5)
        for i in range(n):
            for j in range(n):
                if i != j:
                    ax2.text(j, i, f"{mat[i][j]:.2f}", ha="center", va="center",
                             fontsize=6, color="black")
        h_count = len(common.HUMAN_SYSTEMS)
        ax2.axhline(h_count - 0.5, color="black", linewidth=1.5, linestyle="--")
        ax2.axvline(h_count - 0.5, color="black", linewidth=1.5, linestyle="--")
        fig.colorbar(im, ax=ax2, shrink=0.8, label="Cosine Delta (distance)")
        ax2.set_title("Pairwise Cosine Delta")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "cosine_delta.png", dpi=150)
    plt.close(fig)
    print("  cosine_delta.png")


def _plot_chrf_bleu(plt):
    data = _load("chrf.json")
    systems = [s for s in common.ALL_SYSTEMS if s in data]
    systems.sort(key=lambda s: -data[s]["mean_pairwise_chrf"])

    labels = [_display(s) for s in systems]
    chrf_vals = [data[s]["mean_pairwise_chrf"] for s in systems]
    bleu_vals = [data[s]["mean_pairwise_bleu"] for s in systems]

    import numpy as np
    x = np.arange(len(systems))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    bars1 = ax.bar(x - width / 2, chrf_vals, width, label="chrF",
                   color=[C_AI if _is_ai(s) else C_HUMAN for s in systems],
                   edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, bleu_vals, width, label="BLEU",
                   color=[C_AI if _is_ai(s) else C_HUMAN for s in systems],
                   edgecolor="white", linewidth=0.5, alpha=0.55)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Mean Pairwise chrF & BLEU (higher = more similar to others)")
    ax.legend(["chrF", "BLEU"], loc="upper right")

    for bar, v in zip(bars1, chrf_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.3, f"{v:.1f}",
                ha="center", va="bottom", fontsize=7)
    for bar, v in zip(bars2, bleu_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.3, f"{v:.1f}",
                ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "chrf_bleu.png", dpi=150)
    plt.close(fig)
    print("  chrf_bleu.png")


def _plot_lexical_diversity(plt):
    data = _load("mtld.json")
    systems = [s for s in common.ALL_SYSTEMS if s in data]
    systems.sort(key=lambda s: -data[s]["mtld"])

    labels = [_display(s) for s in systems]
    colors = _bar_colors(systems)

    metrics = {
        "MTLD": ("mtld", "Higher = more diverse vocabulary"),
        "Hapax ratio": ("hapax_ratio", "Higher = more unique words"),
        "Top-100 concentration": ("top100_concentration", "Lower = less repetitive"),
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (title, (key, xlabel)) in zip(axes, metrics.items()):
        vals = [data[s][key] for s in systems]
        bars = ax.barh(range(len(systems)), vals, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(len(systems)))
        ax.set_yticklabels(labels if ax == axes[0] else [_display(s) for s in systems], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        fmt = ".1f" if key == "mtld" else ".3f" if key == "hapax_ratio" else ".1%"
        for bar, v in zip(bars, vals):
            txt = f"{v:{fmt}}" if "%" not in fmt else f"{v:.1%}"
            ax.text(v + (max(vals) - min(vals)) * 0.01, bar.get_y() + bar.get_height() / 2,
                    txt, va="center", fontsize=8)
    _legend(axes[0], plt)

    fig.suptitle("Lexical Diversity & Vocabulary Profile", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "lexical_diversity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  lexical_diversity.png")


def _plot_diminutives(plt):
    data = _load("diminutives.json")
    systems = [s for s in common.ALL_SYSTEMS if s in data]
    systems.sort(key=lambda s: -data[s]["per_1k"])
    vals = [data[s]["per_1k"] for s in systems]
    labels = [_display(s) for s in systems]
    colors = _bar_colors(systems)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.barh(range(len(systems)), vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(systems)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Diminutives per 1,000 words")
    ax.set_title("Diminutive Morphology Density")
    for bar, v in zip(bars, vals):
        ax.text(v + 0.02, bar.get_y() + bar.get_height() / 2, f"{v:.2f}",
                va="center", fontsize=9)
    _legend(ax, plt)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "diminutives.png", dpi=150)
    plt.close(fig)
    print("  diminutives.png")


def _plot_word_ratio(plt):
    data = _load("word_ratio.json")
    systems = [s for s in common.ALL_SYSTEMS if s in data]
    systems.sort(key=lambda s: data[s]["std"])
    stds = [data[s]["std"] for s in systems]
    labels = [_display(s) for s in systems]
    colors = _bar_colors(systems)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.barh(range(len(systems)), stds, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(systems)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Std of segment word-count ratio (lower = more uniform)")
    ax.set_title("Word Ratio Variability")
    for bar, v in zip(bars, stds):
        ax.text(v + 0.002, bar.get_y() + bar.get_height() / 2, f"{v:.3f}",
                va="center", fontsize=9)
    _legend(ax, plt)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "word_ratio.png", dpi=150)
    plt.close(fig)
    print("  word_ratio.png")


def _plot_convergence_summary(plt):
    """Single grouped-bar chart summarising AI-AI vs Human-Human across key metrics."""
    import numpy as np

    data_cd = _load("cosine_delta.json")
    data_chrf = _load("chrf.json")
    data_mtld = _load("mtld.json")
    data_dp = _load("discourse_particles.json")
    data_dim = _load("diminutives.json")

    def _group_stats(data, key, systems):
        return [data[s][key] for s in systems if s in data]

    def _mean(xs):
        return sum(xs) / len(xs) if xs else 0

    h_sys = [s for s in common.HUMAN_SYSTEMS if s != common.DYBKO_KEY]
    a_sys = common.AI_SYSTEMS

    metrics_data = [
        ("chrF\n(similarity)", "mean_pairwise_chrf", data_chrf, False),
        ("MTLD\n(lex diversity)", "mtld", data_mtld, False),
        ("Discourse\nparticles/1k", "per_1k", data_dp, False),
        ("Diminutives\n/1k", "per_1k", data_dim, False),
    ]

    names = [m[0] for m in metrics_data]
    human_vals = []
    ai_vals = []
    for label, key, d, invert in metrics_data:
        h = _mean(_group_stats(d, key, h_sys))
        a = _mean(_group_stats(d, key, a_sys))
        human_vals.append(h)
        ai_vals.append(a)

    x = np.arange(len(names))
    width = 0.32

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_h = ax.bar(x - width / 2, human_vals, width, label="Human", color=C_HUMAN,
                    edgecolor="white", linewidth=0.5)
    bars_a = ax.bar(x + width / 2, ai_vals, width, label="AI", color=C_AI,
                    edgecolor="white", linewidth=0.5)

    for bar, v in zip(bars_h, human_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(human_vals) * 0.01,
                f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    for bar, v in zip(bars_a, ai_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(human_vals) * 0.01,
                f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Mean value")
    ax.set_title("Human vs AI: Stylometric Profile (excl. Dybko)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "convergence_summary.png", dpi=150)
    plt.close(fig)
    print("  convergence_summary.png")


# ── Main ─────────────────────────────────────────────────────────────

def plot_all(show=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating stylometry plots...")
    _plot_discourse_particles(plt)
    _plot_cosine_delta(plt)
    _plot_chrf_bleu(plt)
    _plot_lexical_diversity(plt)
    _plot_diminutives(plt)
    _plot_word_ratio(plt)
    _plot_convergence_summary(plt)
    print(f"\nAll plots saved → {PLOTS_DIR}/")

    if show:
        matplotlib.use("TkAgg")
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot stylometry results")
    parser.add_argument("--show", action="store_true", help="Open plots interactively")
    args = parser.parse_args()
    plot_all(show=args.show)


if __name__ == "__main__":
    main()
