"""LaBSE full-corpus cross-lingual semantic similarity.

Computes cosine similarity between sentence-transformers/LaBSE embeddings
for all valid segments across all 10 translation systems:
  - Similarity to English source (semantic fidelity)
  - Similarity to best human (Stelmakh 2021)
  - Pairwise similarity + cluster averages (AI-AI, H-H, H-AI)

Idempotent: skips if result file exists (unless --force).

Usage:
    python src/labse_evaluate.py                  # compute (GPU)
    python src/labse_evaluate.py --force           # recompute
    python src/labse_evaluate.py --plot            # visualize from saved results
    python src/labse_evaluate.py --plot --show     # open plots interactively

Requirements:
    Compute: pip install sentence-transformers numpy torch
    Plot:    pip install matplotlib numpy
"""

import argparse
import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TRANSLATIONS_PATH = ROOT / "results" / "unite-models" / "parsed_translations.json"
RESULTS_DIR = ROOT / "results"
RESULT_FILE = RESULTS_DIR / "semantic_similarity_results_labse_full.json"
PLOTS_DIR = ROOT / "plots" / "labse"

EN_KEY = "EN_1945_George_Orwell_Animal_Farm"
BEST_HUMAN_KEY = "UK_2021_Viacheslav_Stelmakh_Kolhosp_tvaryn"

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

LABSE_MODEL = "sentence-transformers/LaBSE"

AI_SHORT = {
    "gpt_5_2", "lapa", "deepl",
}

DISPLAY_NAMES = {
    "1947_Ivan_Cherniatynskyi": "Cherniatynskyi '47",
    "1984_Iryna_Dybko": "Dybko '84",
    "1991_Oleksii_Drozdovskyi": "Drozdovskyi '91",
    "1991_Yurii_Shevchuk": "Shevchuk '91",
    "1992_Natalia_Okolitenko": "Okolitenko '92",
    "2020_Bohdana_Nosenok": "Nosenok '20",
    "2021_Viacheslav_Stelmakh": "Stelmakh '21",
    "gpt_5_2": "GPT-5.2",
    "lapa": "LaPa",
    "deepl": "DeepL",
}


def short_name(s):
    return (
        s.replace("UK_", "")
        .replace("_Kolhosp_tvaryn", "")
        .replace("_Khutir_tvaryn", "")
        .replace("_Skotoferma", "")
        .replace("_Ferma_rai_dlia_tvaryn", "")
        .replace("_Skotokhutir", "")
        .replace("_translations_combine", "")
    )


def display_name(short):
    return DISPLAY_NAMES.get(short, short)


def is_ai(short):
    return any(short.startswith(a) or short == a for a in AI_SHORT)


def cos_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ── Plotting ────────────────────────────────────────────────────────

def plot_all(show=False):
    """Generate all plots from saved results JSON."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    if not RESULT_FILE.exists():
        print(f"No results found at {RESULT_FILE}. Run without --plot first.")
        return

    with open(RESULT_FILE) as f:
        data = json.load(f)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    C_HUMAN = "#4878CF"
    C_AI = "#E24A33"

    _plot_source_similarity(plt, data, C_HUMAN, C_AI)
    _plot_heatmap(plt, mcolors, data)
    _plot_cluster_bars(plt, data, C_HUMAN, C_AI)

    if show:
        plt.show()
    print(f"Plots saved → {PLOTS_DIR}/")


def _plot_source_similarity(plt, data, c_human, c_ai):
    """Bar chart: mean cosine similarity to English source per system."""
    src = data["similarity_to_source"]
    names = list(src.keys())
    means = [src[n]["mean"] for n in names]
    stds = [src[n]["std"] for n in names]
    colors = [c_ai if is_ai(n) else c_human for n in names]
    labels = [display_name(n) for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(range(len(names)), means, xerr=stds, color=colors,
                   edgecolor="white", linewidth=0.5, capsize=3)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Mean cosine similarity to English source")
    ax.set_title("LaBSE: Semantic Fidelity to Source")

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c_human, label="Human"),
                       Patch(color=c_ai, label="AI")],
              loc="lower right")

    ax.set_xlim(left=min(means) - 0.02)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "source_similarity.png", dpi=150)
    plt.close(fig)
    print("  source_similarity.png")


def _plot_heatmap(plt, mcolors, data):
    """10x10 pairwise similarity heatmap."""
    pw = data["pairwise_similarity"]

    all_short = [short_name(s) for s in ALL_SYSTEMS]
    n = len(all_short)
    matrix = np.ones((n, n))

    for key, val in pw.items():
        a_str, b_str = key.split(" <-> ")
        if a_str in all_short and b_str in all_short:
            i, j = all_short.index(a_str), all_short.index(b_str)
            matrix[i][j] = val
            matrix[j][i] = val

    labels = [display_name(s) for s in all_short]

    fig, ax = plt.subplots(figsize=(9, 8))
    vmin = np.min(matrix[np.triu_indices(n, k=1)])
    vmax = np.max(matrix[np.triu_indices(n, k=1)])
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=vmin - 0.005, vmax=vmax + 0.005,
                   aspect="equal")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    for i in range(n):
        for j in range(n):
            if i != j:
                ax.text(j, i, f"{matrix[i][j]:.3f}", ha="center", va="center",
                        fontsize=6.5, color="black")

    h_count = len([s for s in all_short if not is_ai(s)])
    ax.axhline(h_count - 0.5, color="black", linewidth=1.5, linestyle="--")
    ax.axvline(h_count - 0.5, color="black", linewidth=1.5, linestyle="--")

    fig.colorbar(im, ax=ax, shrink=0.8, label="Mean cosine similarity")
    ax.set_title("LaBSE: Pairwise Similarity Between Translation Systems")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "pairwise_heatmap.png", dpi=150)
    plt.close(fig)
    print("  pairwise_heatmap.png")


def _plot_cluster_bars(plt, data, c_human, c_ai):
    """Grouped bar chart: AI-AI vs Human-Human vs Cross cluster averages."""
    cl = data["cluster_averages"]
    groups = []
    vals = []
    colors = []

    if cl.get("ai_ai") is not None:
        groups.append("AI–AI")
        vals.append(cl["ai_ai"])
        colors.append(c_ai)
    groups.append("Human–Human")
    vals.append(cl["human_human"])
    colors.append(c_human)
    groups.append("Human–AI")
    vals.append(cl["human_ai"])
    colors.append("#8B8B8B")

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(groups, vals, color=colors, edgecolor="white", width=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.001,
                f"{v:.4f}", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Mean pairwise cosine similarity")
    ax.set_title("LaBSE: Cluster Averages")
    ymin = min(vals) - 0.01
    ax.set_ylim(bottom=ymin)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "cluster_averages.png", dpi=150)
    plt.close(fig)
    print("  cluster_averages.png")


# ── Compute ─────────────────────────────────────────────────────────

def compute(force=False):
    """Run LaBSE embedding + cosine similarity computation."""
    if RESULT_FILE.exists() and not force:
        print(f"{RESULT_FILE.name} already exists. Use --force to recompute.")
        return

    from sentence_transformers import SentenceTransformer

    with open(TRANSLATIONS_PATH) as f:
        trans = json.load(f)["translations"]
    en = trans[EN_KEY]

    valid = []
    for i, seg in enumerate(en):
        txt = seg.strip()
        if not txt or txt == "---" or len(txt.split()) < 3:
            continue
        if all(trans[s][i].strip() and trans[s][i].strip() != "---" for s in ALL_SYSTEMS):
            valid.append(i)

    print(f"Valid segments: {len(valid)}")
    print(f"Loading {LABSE_MODEL}...")
    model = SentenceTransformer(LABSE_MODEL)

    all_texts = []
    for i in valid:
        all_texts.append(en[i])
        for sys_key in ALL_SYSTEMS:
            all_texts.append(trans[sys_key][i])

    print(f"Embedding {len(all_texts)} texts...")
    t0 = time.time()
    embeddings = model.encode(all_texts, batch_size=128, show_progress_bar=True)
    print(f"Done in {time.time() - t0:.0f}s")

    stride = 1 + len(ALL_SYSTEMS)
    ai_short = {short_name(k) for k in AI_SYSTEMS}

    sim_to_source = {s: [] for s in ALL_SYSTEMS}
    sim_to_best = {s: [] for s in ALL_SYSTEMS if s != BEST_HUMAN_KEY}
    pairwise_all = {(a, b): [] for a, b in combinations(ALL_SYSTEMS, 2)}

    for seg_pos, seg_idx in enumerate(valid):
        base = seg_pos * stride
        en_emb = embeddings[base]
        sys_embs = {
            sys_key: embeddings[base + 1 + j]
            for j, sys_key in enumerate(ALL_SYSTEMS)
        }

        for sys_key in ALL_SYSTEMS:
            sim_to_source[sys_key].append(cos_sim(en_emb, sys_embs[sys_key]))

        best_emb = sys_embs[BEST_HUMAN_KEY]
        for sys_key in ALL_SYSTEMS:
            if sys_key != BEST_HUMAN_KEY:
                sim_to_best[sys_key].append(cos_sim(best_emb, sys_embs[sys_key]))

        for a, b in combinations(ALL_SYSTEMS, 2):
            pairwise_all[(a, b)].append(cos_sim(sys_embs[a], sys_embs[b]))

    source_stats = {}
    for sys_key in sorted(ALL_SYSTEMS, key=lambda s: -np.mean(sim_to_source[s])):
        sc = sim_to_source[sys_key]
        source_stats[short_name(sys_key)] = {
            "mean": float(np.mean(sc)),
            "median": float(np.median(sc)),
            "std": float(np.std(sc)),
        }

    best_stats = {}
    for sys_key in sorted(sim_to_best, key=lambda s: -np.mean(sim_to_best[s])):
        sc = sim_to_best[sys_key]
        best_stats[short_name(sys_key)] = {
            "mean": float(np.mean(sc)),
            "median": float(np.median(sc)),
            "std": float(np.std(sc)),
        }

    pw_summary = {}
    human_human, human_ai, ai_ai = [], [], []
    for (a, b), sims in pairwise_all.items():
        key = f"{short_name(a)} <-> {short_name(b)}"
        mean_sim = float(np.mean(sims))
        pw_summary[key] = mean_sim
        a_ai = short_name(a) in ai_short
        b_ai = short_name(b) in ai_short
        if a_ai and b_ai:
            ai_ai.append(mean_sim)
        elif a_ai or b_ai:
            human_ai.append(mean_sim)
        else:
            human_human.append(mean_sim)

    print("\n--- Similarity to English source (top 5) ---")
    for i, (name, stats) in enumerate(
        sorted(source_stats.items(), key=lambda x: -x[1]["mean"])
    ):
        if i >= 5:
            break
        print(f"  {name:40} {stats['mean']:.4f}")

    print("\n--- Pairwise cluster averages ---")
    if ai_ai:
        print(f"  AI-AI:       {np.mean(ai_ai):.4f}  (n={len(ai_ai)})")
    print(f"  Human-AI:    {np.mean(human_ai):.4f}  (n={len(human_ai)})")
    print(f"  Human-Human: {np.mean(human_human):.4f}  (n={len(human_human)})")

    output = {
        "model": LABSE_MODEL,
        "n_segments": len(valid),
        "segment_indices": valid,
        "similarity_to_source": source_stats,
        "similarity_to_best_human": best_stats,
        "pairwise_similarity": pw_summary,
        "cluster_averages": {
            "ai_ai": float(np.mean(ai_ai)) if ai_ai else None,
            "human_ai": float(np.mean(human_ai)),
            "human_human": float(np.mean(human_human)),
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULT_FILE, "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nSaved → {RESULT_FILE}")


# ── CLI ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LaBSE semantic similarity: compute and visualize",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python src/labse_evaluate.py                  # compute (needs GPU)
  python src/labse_evaluate.py --force           # recompute from scratch
  python src/labse_evaluate.py --plot            # save plots to plots/labse/
  python src/labse_evaluate.py --plot --show     # also open interactively
""",
    )
    parser.add_argument("--force", action="store_true", help="Recompute from scratch")
    parser.add_argument("--plot", action="store_true", help="Generate plots from saved results")
    parser.add_argument("--show", action="store_true", help="Open plots interactively (with --plot)")
    args = parser.parse_args()

    if args.plot:
        plot_all(show=args.show)
    else:
        compute(force=args.force)


if __name__ == "__main__":
    main()
