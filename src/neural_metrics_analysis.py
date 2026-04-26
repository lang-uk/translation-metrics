"""Analysis and visualization of all neural MT metrics.

Covers:
  - Reference-free: COMETKiwi-22, COMETKiwi-XL, XCOMET-XXL, MetricX-24 QE
  - Round-robin: COMET-22, XCOMET, MetricX-24

Usage:
    python src/neural_metrics_analysis.py              # print analysis
    python src/neural_metrics_analysis.py --plot       # also save plots
    python src/neural_metrics_analysis.py --plot --show
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMET_DIR = ROOT / "results" / "comet"
METRICX_DIR = ROOT / "results" / "metricx"
PLOTS_DIR = ROOT / "plots" / "neural_metrics"

HUMAN_SYSTEMS = [
    "UK_1947_Ivan_Cherniatynskyi_Kolhosp_tvaryn",
    "UK_1984_Iryna_Dybko_Khutir_tvaryn",
    "UK_1991_Oleksii_Drozdovskyi_Skotoferma",
    "UK_1991_Yurii_Shevchuk_Ferma_rai_dlia_tvaryn",
    "UK_1992_Natalia_Okolitenko_Skotokhutir",
    "UK_2020_Bohdana_Nosenok_Kolhosp_tvaryn",
    "UK_2021_Viacheslav_Stelmakh_Kolhosp_tvaryn",
]
AI_SYSTEMS = ["gpt_5_2", "lapa_translations_combine", "deepl"]
ALL_SYSTEMS = HUMAN_SYSTEMS + AI_SYSTEMS
DYBKO_KEY = "UK_1984_Iryna_Dybko_Khutir_tvaryn"

C_HUMAN = "#4878CF"
C_AI = "#E24A33"
C_GREY = "#8B8B8B"


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


def display_name(s: str) -> str:
    return short_name(s).replace("_", " ")


def is_ai(s: str) -> bool:
    return s in AI_SYSTEMS


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0


def _std(xs):
    if len(xs) < 2:
        return 0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


# ── Data loading ─────────────────────────────────────────────────────

REF_FREE_METRICS = {
    "COMETKiwi-22": ("comet", "reference_free_cometkiwi-22.json", False),
    "COMETKiwi-XL": ("comet", "reference_free_cometkiwi-xl.json", False),
    "XCOMET-XXL": ("comet", "reference_free_xcomet.json", False),
    "MetricX-24 QE": ("metricx", "metricx24_qe.json", True),
}

ROUND_ROBIN_METRICS = {
    "COMET-22": ("comet", "round_robin_comet-22.json", False),
    "XCOMET": ("comet", "round_robin_xcomet.json", False),
    "MetricX-24": ("metricx", "metricx24_round_robin.json", True),
}


def _load(subdir: str, filename: str) -> dict:
    base = COMET_DIR if subdir == "comet" else METRICX_DIR
    path = base / filename
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


# ── Reference-free analysis ──────────────────────────────────────────

def analyze_ref_free():
    print("\n" + "=" * 80)
    print("  REFERENCE-FREE METRICS (per-system scores)")
    print("=" * 80)

    results = {}

    for metric_name, (subdir, filename, inverted) in REF_FREE_METRICS.items():
        data = _load(subdir, filename)
        if not data:
            print(f"\n  {metric_name}: no data found")
            continue

        arrow = "↓ lower=better" if inverted else "↑ higher=better"
        sorted_sys = sorted(
            [s for s in ALL_SYSTEMS if s in data],
            key=lambda s: data[s]["mean"],
            reverse=not inverted,
        )

        print(f"\n  {metric_name} ({arrow}):")
        print(f"  {'#':>3}  {'System':<35} {'Mean':>8} {'Std':>8}")
        print(f"  {'-' * 60}")

        for i, s in enumerate(sorted_sys, 1):
            tag = " *" if is_ai(s) else ""
            print(f"  {i:>3}. {display_name(s):<35} {data[s]['mean']:>8.4f} {data[s]['std']:>8.4f}{tag}")

        h_scores = [data[s]["mean"] for s in HUMAN_SYSTEMS if s in data and s != DYBKO_KEY]
        a_scores = [data[s]["mean"] for s in AI_SYSTEMS if s in data]

        print(f"\n  Human (excl Dybko): {_mean(h_scores):.4f} ± {_std(h_scores):.4f}")
        print(f"  AI:                 {_mean(a_scores):.4f} ± {_std(a_scores):.4f}")

        results[metric_name] = {
            "inverted": inverted,
            "systems": {s: data[s]["mean"] for s in sorted_sys},
            "human_mean": _mean(h_scores),
            "ai_mean": _mean(a_scores),
        }

    return results


# ── Round-robin analysis ─────────────────────────────────────────────

def analyze_round_robin():
    print("\n" + "=" * 80)
    print("  ROUND-ROBIN METRICS (mean score when using each other system as reference)")
    print("=" * 80)

    results = {}

    for metric_name, (subdir, filename, inverted) in ROUND_ROBIN_METRICS.items():
        data = _load(subdir, filename)
        if not data:
            print(f"\n  {metric_name}: no data found")
            continue

        arrow = "↓ lower=better" if inverted else "↑ higher=better"

        sys_means = {}
        for s in ALL_SYSTEMS:
            scores = []
            for ref in ALL_SYSTEMS:
                if s == ref:
                    continue
                pair_key = f"{s}_vs_{ref}"
                if pair_key in data:
                    scores.append(data[pair_key]["mean"])
            if scores:
                sys_means[s] = _mean(scores)

        sorted_sys = sorted(
            sys_means.keys(),
            key=lambda s: sys_means[s],
            reverse=not inverted,
        )

        print(f"\n  {metric_name} round-robin ({arrow}):")
        print(f"  {'#':>3}  {'System':<35} {'Mean RR':>10}")
        print(f"  {'-' * 55}")

        for i, s in enumerate(sorted_sys, 1):
            tag = " *" if is_ai(s) else ""
            print(f"  {i:>3}. {display_name(s):<35} {sys_means[s]:>10.4f}{tag}")

        h_scores = [sys_means[s] for s in HUMAN_SYSTEMS if s in sys_means and s != DYBKO_KEY]
        a_scores = [sys_means[s] for s in AI_SYSTEMS if s in sys_means]

        print(f"\n  Human (excl Dybko): {_mean(h_scores):.4f} ± {_std(h_scores):.4f}")
        print(f"  AI:                 {_mean(a_scores):.4f} ± {_std(a_scores):.4f}")

        # Convergence: AI-AI vs H-H pairwise scores
        ai_ai, h_h, h_ai = [], [], []
        for hyp in ALL_SYSTEMS:
            for ref in ALL_SYSTEMS:
                if hyp == ref:
                    continue
                pair_key = f"{hyp}_vs_{ref}"
                if pair_key not in data:
                    continue
                val = data[pair_key]["mean"]
                if is_ai(hyp) and is_ai(ref):
                    ai_ai.append(val)
                elif not is_ai(hyp) and not is_ai(ref):
                    h_h.append(val)
                else:
                    h_ai.append(val)

        if ai_ai and h_h:
            print(f"\n  Convergence (pairwise scores as reference):")
            print(f"    AI-AI:    {_mean(ai_ai):.4f}  (n={len(ai_ai)})")
            print(f"    H-H:     {_mean(h_h):.4f}  (n={len(h_h)})")
            print(f"    Cross:   {_mean(h_ai):.4f}  (n={len(h_ai)})")

        results[metric_name] = {
            "inverted": inverted,
            "systems": {s: sys_means[s] for s in sorted_sys},
            "human_mean": _mean(h_scores),
            "ai_mean": _mean(a_scores),
            "convergence": {
                "ai_ai": _mean(ai_ai) if ai_ai else None,
                "h_h": _mean(h_h) if h_h else None,
                "cross": _mean(h_ai) if h_ai else None,
            },
        }

    return results


# ── Visualization ────────────────────────────────────────────────────

def plot_all(ref_free_results, round_robin_results, show=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    _plot_ref_free_comparison(plt, Patch, ref_free_results)
    _plot_round_robin_comparison(plt, Patch, round_robin_results)
    _plot_convergence(plt, Patch, round_robin_results)
    _plot_round_robin_heatmaps(plt)

    print(f"\nAll plots saved → {PLOTS_DIR}/")
    if show:
        plt.show()


def _plot_ref_free_comparison(plt, Patch, results):
    """Side-by-side bar charts of all ref-free metrics."""
    metrics = [m for m in REF_FREE_METRICS if m in results]
    n = len(metrics)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, metric_name in zip(axes, metrics):
        r = results[metric_name]
        inv = r["inverted"]
        systems = list(r["systems"].keys())
        vals = [r["systems"][s] for s in systems]
        labels = [display_name(s) for s in systems]
        colors = [C_AI if is_ai(s) else C_HUMAN for s in systems]

        ax.barh(range(len(systems)), vals, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(len(systems)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        arrow = "← better" if inv else "better →"
        ax.set_xlabel(f"Mean score ({arrow})")
        ax.set_title(metric_name, fontsize=11)

        for i, v in enumerate(vals):
            fmt = f"{v:.3f}" if abs(v) < 1 else f"{v:.2f}"
            offset = -0.01 * max(vals) if not inv else 0.01 * max(vals)
            ax.text(v + offset, i, fmt, va="center", fontsize=7,
                    ha="right" if not inv else "left")

    axes[0].legend(
        handles=[Patch(color=C_HUMAN, label="Human"), Patch(color=C_AI, label="AI")],
        loc="lower right" if not results[metrics[0]]["inverted"] else "lower left",
    )

    fig.suptitle("Reference-Free Neural Metrics", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "ref_free_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ref_free_comparison.png")


def _plot_round_robin_comparison(plt, Patch, results):
    """Side-by-side bar charts of round-robin metrics."""
    metrics = [m for m in ROUND_ROBIN_METRICS if m in results]
    n = len(metrics)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, metric_name in zip(axes, metrics):
        r = results[metric_name]
        inv = r["inverted"]
        systems = list(r["systems"].keys())
        vals = [r["systems"][s] for s in systems]
        labels = [display_name(s) for s in systems]
        colors = [C_AI if is_ai(s) else C_HUMAN for s in systems]

        ax.barh(range(len(systems)), vals, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(len(systems)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        arrow = "← better" if inv else "better →"
        ax.set_xlabel(f"Mean round-robin score ({arrow})")
        ax.set_title(metric_name, fontsize=11)

        for i, v in enumerate(vals):
            fmt = f"{v:.3f}" if abs(v) < 1 else f"{v:.2f}"
            offset = -0.005 * max(abs(x) for x in vals) if not inv else 0.005 * max(abs(x) for x in vals)
            ax.text(v + offset, i, fmt, va="center", fontsize=7,
                    ha="right" if not inv else "left")

    axes[0].legend(
        handles=[Patch(color=C_HUMAN, label="Human"), Patch(color=C_AI, label="AI")],
        loc="lower right" if not results[metrics[0]]["inverted"] else "lower left",
    )

    fig.suptitle("Round-Robin Reference-Based Neural Metrics", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "round_robin_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  round_robin_comparison.png")


def _plot_convergence(plt, Patch, results):
    """Grouped bars: AI-AI vs H-H vs Cross for each round-robin metric."""
    import numpy as np

    metrics = [m for m in results if results[m].get("convergence", {}).get("ai_ai") is not None]
    if not metrics:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(metrics))
    width = 0.25

    ai_vals = [results[m]["convergence"]["ai_ai"] for m in metrics]
    hh_vals = [results[m]["convergence"]["h_h"] for m in metrics]
    cross_vals = [results[m]["convergence"]["cross"] for m in metrics]

    bars1 = ax.bar(x - width, ai_vals, width, label="AI–AI", color=C_AI, edgecolor="white")
    bars2 = ax.bar(x, hh_vals, width, label="Human–Human", color=C_HUMAN, edgecolor="white")
    bars3 = ax.bar(x + width, cross_vals, width, label="Cross", color=C_GREY, edgecolor="white")

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            v = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.002,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    labels = []
    for m in metrics:
        inv = results[m]["inverted"]
        labels.append(f"{m}\n({'lower=better' if inv else 'higher=better'})")
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Mean pairwise score")
    ax.set_title("Round-Robin Convergence: AI-AI vs Human-Human")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "convergence_round_robin.png", dpi=150)
    plt.close(fig)
    print("  convergence_round_robin.png")


def _plot_round_robin_heatmaps(plt):
    """10x10 pairwise heatmap for each round-robin metric."""
    import numpy as np

    for metric_name, (subdir, filename, inverted) in ROUND_ROBIN_METRICS.items():
        data = _load(subdir, filename)
        if not data:
            continue

        n = len(ALL_SYSTEMS)
        matrix = np.full((n, n), np.nan)

        for i, hyp in enumerate(ALL_SYSTEMS):
            for j, ref in enumerate(ALL_SYSTEMS):
                if i == j:
                    continue
                pair_key = f"{hyp}_vs_{ref}"
                if pair_key in data:
                    matrix[i][j] = data[pair_key]["mean"]

        # Symmetrize: average (A_vs_B, B_vs_A) for display
        sym = np.full((n, n), np.nan)
        for i in range(n):
            for j in range(i + 1, n):
                vals = []
                if not np.isnan(matrix[i][j]):
                    vals.append(matrix[i][j])
                if not np.isnan(matrix[j][i]):
                    vals.append(matrix[j][i])
                if vals:
                    avg = sum(vals) / len(vals)
                    sym[i][j] = avg
                    sym[j][i] = avg

        labels = [display_name(s) for s in ALL_SYSTEMS]
        triu_vals = sym[np.triu_indices(n, k=1)]
        triu_vals = triu_vals[~np.isnan(triu_vals)]
        if len(triu_vals) == 0:
            continue

        vmin, vmax = float(np.nanmin(triu_vals)), float(np.nanmax(triu_vals))

        cmap = "RdYlGn" if not inverted else "RdYlGn_r"

        fig, ax = plt.subplots(figsize=(9, 8))
        masked = np.ma.masked_where(np.isnan(sym) | np.eye(n, dtype=bool), sym)
        im = ax.imshow(masked, cmap=cmap, vmin=vmin - 0.005, vmax=vmax + 0.005,
                        aspect="equal")

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)

        for i in range(n):
            for j in range(n):
                if i != j and not np.isnan(sym[i][j]):
                    fmt = f"{sym[i][j]:.3f}" if abs(sym[i][j]) < 1 else f"{sym[i][j]:.2f}"
                    ax.text(j, i, fmt, ha="center", va="center", fontsize=6, color="black")

        h_count = len(HUMAN_SYSTEMS)
        ax.axhline(h_count - 0.5, color="black", linewidth=1.5, linestyle="--")
        ax.axvline(h_count - 0.5, color="black", linewidth=1.5, linestyle="--")

        arrow = "lower=better" if inverted else "higher=better"
        fig.colorbar(im, ax=ax, shrink=0.8, label=f"Mean score ({arrow})")
        ax.set_title(f"{metric_name}: Pairwise Round-Robin Scores")
        fig.tight_layout()

        safe_name = metric_name.lower().replace("-", "_").replace(" ", "_")
        fig.savefig(PLOTS_DIR / f"heatmap_{safe_name}.png", dpi=150)
        plt.close(fig)
        print(f"  heatmap_{safe_name}.png")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze and plot neural MT metrics")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    args = parser.parse_args()

    ref_free = analyze_ref_free()
    round_robin = analyze_round_robin()

    if args.plot or args.show:
        plot_all(ref_free, round_robin, show=args.show)


if __name__ == "__main__":
    main()
