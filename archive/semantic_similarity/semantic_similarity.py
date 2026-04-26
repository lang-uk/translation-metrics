"""Semantic similarity analysis using cross-lingual sentence embeddings.

Uses LaBSE (Language-agnostic BERT Sentence Embeddings) to compute cosine
similarity between translations and the English source, between translations
and the best human translation, and pairwise among all translations — for
the 100 segments where automatic quality metrics disagree most.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer

TRANSLATIONS_PATH = (
    Path(__file__).parent.parent
    / "results"
    / "unite-models"
    / "parsed_translations.json"
)
XCOMET_SCORES_PATH = (
    Path(__file__).parent.parent
    / "results"
    / "comet"
    / "reference_free_xcomet.json"
)
OUTPUT_PATH = (
    Path(__file__).parent.parent
    / "results"
    / "semantic_similarity_results.json"
)
PLOTS_DIR = Path(__file__).parent.parent / "plots"

EN_KEY = "EN_1945_George_Orwell_Animal_Farm"
BEST_HUMAN_KEY = "UK_2021_Viacheslav_Stelmakh_Kolhosp_tvaryn"

ALL_SYSTEMS = [
    "UK_1947_Ivan_Cherniatynskyi_Kolhosp_tvaryn",
    "UK_1984_Iryna_Dybko_Khutir_tvaryn",
    "UK_1991_Oleksii_Drozdovskyi_Skotoferma",
    "UK_1991_Yurii_Shevchuk_Ferma_rai_dlia_tvaryn",
    "UK_1992_Natalia_Okolitenko_Skotokhutir",
    "UK_2020_Bohdana_Nosenok_Kolhosp_tvaryn",
    "UK_2021_Viacheslav_Stelmakh_Kolhosp_tvaryn",
    "lapa_translations_combine",
    "gpt_5_2",
    "deepl",
]

LLM_KEYS = {"lapa_translations_combine", "gpt_5_2", "deepl"}

TOP_N = 100
EMBEDDING_MODEL = "sentence-transformers/LaBSE"


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def find_most_disagreeable_segments(xcomet_path, translations, n=100):
    with open(xcomet_path) as f:
        xcomet = json.load(f)

    systems = [s for s in xcomet if "scores" in xcomet[s]]
    n_segments = len(xcomet[systems[0]]["scores"])
    en = translations[EN_KEY]

    segment_variance = []
    for i in range(n_segments):
        scores = [
            xcomet[s]["scores"][i]
            for s in systems
            if i < len(xcomet[s]["scores"])
        ]
        if len(scores) < 2:
            continue
        en_text = en[i].strip() if i < len(en) else ""
        if en_text and en_text != "---" and len(en_text) > 10:
            segment_variance.append((i, np.var(scores)))

    segment_variance.sort(key=lambda x: -x[1])
    return [idx for idx, _ in segment_variance[:n]]


def get_embeddings(model, texts):
    sanitized = [t if t.strip() else "[empty]" for t in texts]
    return model.encode(sanitized, show_progress_bar=True, batch_size=64)


def short_name(system):
    return (
        system.replace("UK_", "")
        .replace("_Kolhosp_tvaryn", "")
        .replace("_Khutir_tvaryn", "")
        .replace("_Skotoferma", "")
        .replace("_Ferma_rai_dlia_tvaryn", "")
        .replace("_Skotokhutir", "")
    )


def main():
    with open(TRANSLATIONS_PATH) as f:
        data = json.load(f)
    translations = data["translations"]

    print("Finding 100 most metric-disagreeable segments...")
    segment_indices = find_most_disagreeable_segments(
        XCOMET_SCORES_PATH, translations, TOP_N
    )
    print(f"Selected {len(segment_indices)} segments\n")

    print(f"Loading LaBSE model ({EMBEDDING_MODEL})...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("Model loaded.\n")

    all_texts = []
    for seg_idx in segment_indices:
        all_texts.append(translations[EN_KEY][seg_idx])
        for system in ALL_SYSTEMS:
            all_texts.append(translations[system][seg_idx])

    all_texts = [t if t.strip() else "[empty]" for t in all_texts]
    total = len(all_texts)
    print(
        f"Embedding {total} texts "
        f"({len(segment_indices)} segments x 9 systems)..."
    )

    all_embeddings = get_embeddings(model, all_texts)
    print(f"Got {len(all_embeddings)} embeddings\n")

    # Organize into per-segment dicts
    texts_per_seg = 1 + len(ALL_SYSTEMS)
    emb_by_seg = {}
    for i, seg_idx in enumerate(segment_indices):
        base = i * texts_per_seg
        emb_by_seg[seg_idx] = {"EN": np.array(all_embeddings[base])}
        for j, system in enumerate(ALL_SYSTEMS):
            emb_by_seg[seg_idx][system] = np.array(
                all_embeddings[base + 1 + j]
            )

    # =================================================================
    # Analysis 1: Similarity to English source
    # =================================================================
    print("=" * 80)
    print("ANALYSIS 1: Cosine similarity to ENGLISH SOURCE")
    print("  (Higher = more semantically faithful to the original)")
    print("=" * 80)

    sim_to_source = {sys: [] for sys in ALL_SYSTEMS}
    for seg_idx in segment_indices:
        en_emb = emb_by_seg[seg_idx]["EN"]
        for system in ALL_SYSTEMS:
            sim = cosine_similarity(
                en_emb, emb_by_seg[seg_idx][system]
            )
            sim_to_source[system].append(sim)

    source_stats = {}
    for system in sorted(
        ALL_SYSTEMS, key=lambda s: -np.mean(sim_to_source[s])
    ):
        scores = sim_to_source[system]
        stats = {
            "mean": float(np.mean(scores)),
            "median": float(np.median(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
        }
        source_stats[system] = stats
        name = short_name(system)
        print(
            f"  {name:45} mean={stats['mean']:.4f}"
            f"  std={stats['std']:.4f}"
            f"  median={stats['median']:.4f}"
        )

    # =================================================================
    # Analysis 2: Similarity to best human (Stelmakh 2021)
    # =================================================================
    print()
    print("=" * 80)
    print("ANALYSIS 2: Cosine similarity to STELMAKH 2021")
    print("  (Higher = more similar to the metric-favorite human)")
    print("=" * 80)

    sim_to_best = {
        sys: [] for sys in ALL_SYSTEMS if sys != BEST_HUMAN_KEY
    }
    for seg_idx in segment_indices:
        best_emb = emb_by_seg[seg_idx][BEST_HUMAN_KEY]
        for system in ALL_SYSTEMS:
            if system == BEST_HUMAN_KEY:
                continue
            sim = cosine_similarity(
                best_emb, emb_by_seg[seg_idx][system]
            )
            sim_to_best[system].append(sim)

    best_stats = {}
    for system in sorted(
        sim_to_best.keys(), key=lambda s: -np.mean(sim_to_best[s])
    ):
        scores = sim_to_best[system]
        stats = {
            "mean": float(np.mean(scores)),
            "median": float(np.median(scores)),
            "std": float(np.std(scores)),
        }
        best_stats[system] = stats
        name = short_name(system)
        print(
            f"  {name:45} mean={stats['mean']:.4f}"
            f"  std={stats['std']:.4f}"
            f"  median={stats['median']:.4f}"
        )

    # =================================================================
    # Analysis 3: Pairwise similarity
    # =================================================================
    print()
    print("=" * 80)
    print("ANALYSIS 3: Pairwise similarity (mean across 100 segments)")
    print("=" * 80)

    pairwise = {}
    for i, sys_a in enumerate(ALL_SYSTEMS):
        for sys_b in ALL_SYSTEMS[i + 1 :]:
            sims = []
            for seg_idx in segment_indices:
                sim = cosine_similarity(
                    emb_by_seg[seg_idx][sys_a],
                    emb_by_seg[seg_idx][sys_b],
                )
                sims.append(sim)
            key = (
                f"{short_name(sys_a)} <-> {short_name(sys_b)}"
            )
            pairwise[key] = float(np.mean(sims))

    human_human_sims = []
    human_lapa_sims = []
    llm_short = {short_name(k) for k in LLM_KEYS}
    for pair, sim in sorted(pairwise.items(), key=lambda x: -x[1]):
        is_lapa = any(ln in pair for ln in llm_short)
        marker = " [HUMAN-LLM]" if is_lapa else ""
        if is_lapa:
            human_lapa_sims.append(sim)
        else:
            human_human_sims.append(sim)
        print(f"  {pair:70} {sim:.4f}{marker}")

    print()
    print("-" * 80)
    hh = np.mean(human_human_sims)
    hl = np.mean(human_lapa_sims)
    print(
        f"  Human-Human avg pairwise similarity:"
        f" {hh:.4f}  (n={len(human_human_sims)} pairs)"
    )
    print(
        f"  Human-LLM   avg pairwise similarity:"
        f" {hl:.4f}  (n={len(human_lapa_sims)} pairs)"
    )
    print(f"  Difference (H-H minus H-LLM):        {hh - hl:+.4f}")

    # =================================================================
    # Summary
    # =================================================================
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\nSimilarity to English source (ranking):")
    for rank, (sys, stats) in enumerate(
        sorted(source_stats.items(), key=lambda x: -x[1]["mean"]), 1
    ):
        name = short_name(sys)
        marker = " <-- LLM" if "lapa" in sys else ""
        print(f"  {rank}. {name:45} {stats['mean']:.4f}{marker}")

    print("\nSimilarity to Stelmakh 2021 (ranking):")
    for rank, (sys, stats) in enumerate(
        sorted(best_stats.items(), key=lambda x: -x[1]["mean"]), 1
    ):
        name = short_name(sys)
        marker = " <-- LLM" if "lapa" in sys else ""
        print(f"  {rank}. {name:45} {stats['mean']:.4f}{marker}")

    # =================================================================
    # Plots
    # =================================================================
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    system_order = sorted(
        source_stats.keys(), key=lambda s: -source_stats[s]["mean"]
    )
    names = [short_name(s) for s in system_order]
    colors = [
        "#e74c3c" if "lapa" in s else "#3498db" for s in system_order
    ]

    # Plot 1: Similarity to English source (bar chart)
    fig, ax = plt.subplots(figsize=(12, 6))
    means = [source_stats[s]["mean"] for s in system_order]
    stds = [source_stats[s]["std"] for s in system_order]
    bars = ax.bar(names, means, yerr=stds, color=colors, capsize=4)
    ax.set_title(
        "Semantic Similarity to English Source\n"
        "(LaBSE cross-lingual embeddings, 100 most disagreeable segments)",
        fontweight="bold",
    )
    ax.set_ylabel("Cosine Similarity")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, fc="#3498db", label="Human"),
            plt.Rectangle((0, 0), 1, 1, fc="#e74c3c", label="LLM (lapa)"),
        ],
        loc="lower right",
    )
    plt.tight_layout()
    path1 = PLOTS_DIR / "semantic_similarity_to_source.png"
    plt.savefig(path1, dpi=300, bbox_inches="tight")
    print(f"\nSaved plot to {path1}")

    # Plot 2: Similarity to Stelmakh 2021 (bar chart)
    best_order = sorted(
        best_stats.keys(), key=lambda s: -best_stats[s]["mean"]
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    names2 = [short_name(s) for s in best_order]
    means2 = [best_stats[s]["mean"] for s in best_order]
    stds2 = [best_stats[s]["std"] for s in best_order]
    colors2 = [
        "#e74c3c" if "lapa" in s else "#3498db" for s in best_order
    ]
    ax.bar(names2, means2, yerr=stds2, color=colors2, capsize=4)
    ax.set_title(
        "Semantic Similarity to Stelmakh 2021 (best-ranked human)\n"
        "(LaBSE cross-lingual embeddings, 100 most disagreeable segments)",
        fontweight="bold",
    )
    ax.set_ylabel("Cosine Similarity")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, fc="#3498db", label="Human"),
            plt.Rectangle((0, 0), 1, 1, fc="#e74c3c", label="LLM (lapa)"),
        ],
        loc="lower right",
    )
    plt.tight_layout()
    path2 = PLOTS_DIR / "semantic_similarity_to_stelmakh.png"
    plt.savefig(path2, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {path2}")

    # Plot 3: Grouped bar — source sim vs Stelmakh sim side-by-side
    all_sys_ordered = sorted(
        ALL_SYSTEMS, key=lambda s: -source_stats[s]["mean"]
    )
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(all_sys_ordered))
    width = 0.35
    src_means = [source_stats[s]["mean"] for s in all_sys_ordered]
    best_means = [
        best_stats[s]["mean"] if s in best_stats else np.nan
        for s in all_sys_ordered
    ]
    bar1 = ax.bar(
        x - width / 2, src_means, width,
        label="vs English source", color="#2ecc71",
    )
    bar2 = ax.bar(
        x + width / 2, best_means, width,
        label="vs Stelmakh 2021", color="#9b59b6",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [short_name(s) for s in all_sys_ordered], rotation=45, ha="right"
    )
    for i, s in enumerate(all_sys_ordered):
        if "lapa" in s:
            ax.get_xticklabels()[i].set_color("#e74c3c")
            ax.get_xticklabels()[i].set_fontweight("bold")
    ax.set_title(
        "Semantic Similarity: Source Fidelity vs Human Consensus\n"
        "(100 most disagreeable segments)",
        fontweight="bold",
    )
    ax.set_ylabel("Cosine Similarity")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path3 = PLOTS_DIR / "semantic_similarity_grouped.png"
    plt.savefig(path3, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {path3}")

    # Save full results
    results = {
        "model": EMBEDDING_MODEL,
        "n_segments": len(segment_indices),
        "segment_indices": segment_indices,
        "similarity_to_source": {
            short_name(k): v for k, v in source_stats.items()
        },
        "similarity_to_best_human": {
            short_name(k): v for k, v in best_stats.items()
        },
        "pairwise_similarity": pairwise,
        "human_human_avg": float(hh),
        "human_lapa_avg": float(hl),
        "per_segment_source_similarity": {
            short_name(sys): sim_to_source[sys] for sys in ALL_SYSTEMS
        },
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
