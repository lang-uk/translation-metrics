"""Semantic similarity analysis using OpenAI text embeddings.

Uses text-embedding-3-small (general-purpose, NOT trained on parallel corpora)
to provide an independent comparison against the LaBSE results.
Includes analysis vs English source, vs best human (Stelmakh 2021),
vs worst-performing human (Dybko 1984), and pairwise among all translations.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from openai import OpenAI

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
    / "semantic_similarity_openai_results.json"
)
PLOTS_DIR = Path(__file__).parent.parent / "plots"

EN_KEY = "EN_1945_George_Orwell_Animal_Farm"
BEST_HUMAN_KEY = "UK_2021_Viacheslav_Stelmakh_Kolhosp_tvaryn"
WORST_HUMAN_KEY = "UK_1984_Iryna_Dybko_Khutir_tvaryn"

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
EMBEDDING_MODEL = "text-embedding-3-small"


def cosine_similarity(a, b):
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


def get_embeddings(client, texts):
    sanitized = [t if t.strip() else "[empty]" for t in texts]
    response = client.embeddings.create(input=sanitized, model=EMBEDDING_MODEL)
    return [item.embedding for item in response.data]


def short_name(system):
    return (
        system.replace("UK_", "")
        .replace("_Kolhosp_tvaryn", "")
        .replace("_Khutir_tvaryn", "")
        .replace("_Skotoferma", "")
        .replace("_Ferma_rai_dlia_tvaryn", "")
        .replace("_Skotokhutir", "")
    )


def compute_ref_similarity(emb_by_seg, segment_indices, ref_key, label):
    """Compute similarity of all systems to a reference system."""
    others = [s for s in ALL_SYSTEMS if s != ref_key]
    sim_to_ref = {sys: [] for sys in others}
    for seg_idx in segment_indices:
        ref_emb = emb_by_seg[seg_idx][ref_key]
        for system in others:
            sim = cosine_similarity(ref_emb, emb_by_seg[seg_idx][system])
            sim_to_ref[system].append(sim)

    stats = {}
    print(f"\n{'=' * 80}")
    print(f"Cosine similarity to {label}")
    print("=" * 80)
    for system in sorted(others, key=lambda s: -np.mean(sim_to_ref[s])):
        scores = sim_to_ref[system]
        s = {
            "mean": float(np.mean(scores)),
            "median": float(np.median(scores)),
            "std": float(np.std(scores)),
        }
        stats[system] = s
        name = short_name(system)
        print(
            f"  {name:45} mean={s['mean']:.4f}"
            f"  std={s['std']:.4f}"
            f"  median={s['median']:.4f}"
        )
    return sim_to_ref, stats


def plot_ref_bar(stats, ref_label, filename, model_label):
    order = sorted(stats.keys(), key=lambda s: -stats[s]["mean"])
    names = [short_name(s) for s in order]
    means = [stats[s]["mean"] for s in order]
    stds = [stats[s]["std"] for s in order]
    colors = ["#e74c3c" if "lapa" in s else "#3498db" for s in order]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(names, means, yerr=stds, color=colors, capsize=4)
    ax.set_title(
        f"Semantic Similarity to {ref_label}\n"
        f"({model_label}, 100 most disagreeable segments)",
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
    path = PLOTS_DIR / filename
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {path}")
    return path


def main():
    with open(TRANSLATIONS_PATH) as f:
        data = json.load(f)
    translations = data["translations"]

    print("Finding 100 most metric-disagreeable segments...")
    segment_indices = find_most_disagreeable_segments(
        XCOMET_SCORES_PATH, translations, TOP_N
    )
    print(f"Selected {len(segment_indices)} segments\n")

    client = OpenAI()

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

    all_embeddings = []
    batch_size = 200
    for i in range(0, total, batch_size):
        batch = all_texts[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size
        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} texts)...")
        embeddings = get_embeddings(client, batch)
        all_embeddings.extend(embeddings)

    print(f"Got {len(all_embeddings)} embeddings\n")

    texts_per_seg = 1 + len(ALL_SYSTEMS)
    emb_by_seg = {}
    for i, seg_idx in enumerate(segment_indices):
        base = i * texts_per_seg
        emb_by_seg[seg_idx] = {"EN": np.array(all_embeddings[base])}
        for j, system in enumerate(ALL_SYSTEMS):
            emb_by_seg[seg_idx][system] = np.array(
                all_embeddings[base + 1 + j]
            )

    # === Analysis 1: Similarity to English source ===
    sim_to_source = {sys: [] for sys in ALL_SYSTEMS}
    print("=" * 80)
    print("ANALYSIS 1: Cosine similarity to ENGLISH SOURCE")
    print("  (Higher = more semantically faithful to the original)")
    print("=" * 80)
    for seg_idx in segment_indices:
        en_emb = emb_by_seg[seg_idx]["EN"]
        for system in ALL_SYSTEMS:
            sim = cosine_similarity(en_emb, emb_by_seg[seg_idx][system])
            sim_to_source[system].append(sim)

    source_stats = {}
    for system in sorted(ALL_SYSTEMS, key=lambda s: -np.mean(sim_to_source[s])):
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

    # === Analysis 2: Similarity to best human (Stelmakh 2021) ===
    _, best_stats = compute_ref_similarity(
        emb_by_seg, segment_indices,
        BEST_HUMAN_KEY, "STELMAKH 2021 (best-ranked human)",
    )

    # === Analysis 3: Similarity to worst human (Dybko 1984) ===
    _, worst_stats = compute_ref_similarity(
        emb_by_seg, segment_indices,
        WORST_HUMAN_KEY, "DYBKO 1984 (worst-ranked human / cultural adapter)",
    )

    # === Analysis 4: Pairwise similarity ===
    print(f"\n{'=' * 80}")
    print("ANALYSIS 4: Pairwise similarity (mean across 100 segments)")
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
            key = f"{short_name(sys_a)} <-> {short_name(sys_b)}"
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

    # === Summary ===
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print("=" * 80)

    for label, st in [
        ("English source", source_stats),
        ("Stelmakh 2021 (best)", best_stats),
        ("Dybko 1984 (worst)", worst_stats),
    ]:
        print(f"\nSimilarity to {label} (ranking):")
        for rank, (sys, s) in enumerate(
            sorted(st.items(), key=lambda x: -x[1]["mean"]), 1
        ):
            name = short_name(sys)
            marker = " <-- LLM" if "lapa" in sys else ""
            print(f"  {rank}. {name:45} {s['mean']:.4f}{marker}")

    # === Plots ===
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    model_label = "OpenAI text-embedding-3-small"

    plot_ref_bar(
        source_stats, "English Source",
        "openai_similarity_to_source.png", model_label,
    )
    plot_ref_bar(
        best_stats, "Stelmakh 2021 (best-ranked human)",
        "openai_similarity_to_stelmakh.png", model_label,
    )
    plot_ref_bar(
        worst_stats, "Dybko 1984 (worst-ranked / cultural adapter)",
        "openai_similarity_to_dybko.png", model_label,
    )

    # Plot 4: Grouped bar — source vs best vs worst side-by-side
    all_sys = sorted(ALL_SYSTEMS, key=lambda s: -source_stats[s]["mean"])
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(all_sys))
    w = 0.25
    ax.bar(
        x - w, [source_stats[s]["mean"] for s in all_sys], w,
        label="vs English source", color="#2ecc71",
    )
    ax.bar(
        x, [best_stats[s]["mean"] if s in best_stats else np.nan for s in all_sys], w,
        label="vs Stelmakh 2021 (best)", color="#9b59b6",
    )
    ax.bar(
        x + w, [worst_stats[s]["mean"] if s in worst_stats else np.nan for s in all_sys], w,
        label="vs Dybko 1984 (worst)", color="#e67e22",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [short_name(s) for s in all_sys], rotation=45, ha="right"
    )
    for i, s in enumerate(all_sys):
        if "lapa" in s:
            ax.get_xticklabels()[i].set_color("#e74c3c")
            ax.get_xticklabels()[i].set_fontweight("bold")
    ax.set_title(
        "Semantic Similarity: Source vs Best Human vs Worst Human\n"
        f"({model_label}, 100 most disagreeable segments)",
        fontweight="bold",
    )
    ax.set_ylabel("Cosine Similarity")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    p4 = PLOTS_DIR / "openai_similarity_grouped.png"
    plt.savefig(p4, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {p4}")

    # Save results
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
        "similarity_to_worst_human": {
            short_name(k): v for k, v in worst_stats.items()
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
