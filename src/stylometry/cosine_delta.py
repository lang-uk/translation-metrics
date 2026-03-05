"""Cosine Delta stylometric distance (closed-class word features).

Methodology (adapted from Evert et al. 2017):
  1. Tokenize each translation, POS-tag with pymorphy3 (uk).
  2. Keep only closed-class POS: CONJ, PREP, PRCL, NPRO.
     (pymorphy3's NPRO includes pronominal adverbs — this broadens
     the feature set beyond strict function words, but the effect is
     consistent across systems and does not bias the comparison.)
  3. Build per-system lemma frequency vectors.
  4. Filter: keep features with total count >= 10 across all systems
     (removes noise from hapax-level features; signal is robust to
     this threshold — see audit).
  5. Z-score normalise across the corpus, then cosine distance.

AI-AI mean distance (~0.48) is roughly half the H-H mean distance
(~1.07), the clearest evidence that machine translations converge
on a shared stylistic profile.
"""

from collections import Counter
from math import sqrt

import pymorphy3

from ._common import (
    ALL_SYSTEMS, AI_SYSTEMS, HUMAN_SYSTEMS, RANKED_SYSTEMS, DYBKO_KEY,
    build_full_text, load_translations, save_results, short_name, uk_words,
)

FUNCTION_POS = {"CONJ", "PREP", "PRCL", "NPRO"}
MIN_TOTAL_FREQ = 10

RESULT_FILE = "cosine_delta.json"


def _cosine_distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sqrt(sum(x * x for x in a))
    nb = sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 1.0
    return 1.0 - dot / (na * nb)


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs):
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def _extract_lemma_freqs(morph, words_per_system):
    lemma_counts = {}
    global_counts: Counter = Counter()
    for sys, words in words_per_system.items():
        counter: Counter = Counter()
        for w in words:
            parsed = morph.parse(w)[0]
            if parsed.tag.POS in FUNCTION_POS:
                counter[parsed.normal_form] += 1
        lemma_counts[sys] = counter
        global_counts.update(counter)

    vocab = sorted(w for w, c in global_counts.items() if c >= MIN_TOTAL_FREQ)
    return lemma_counts, vocab


def compute(translations: dict, en_segments: list[str]) -> dict:
    print("  Initializing pymorphy3 (uk)...")
    morph = pymorphy3.MorphAnalyzer(lang="uk")

    words_per_system = {}
    for sys_key in ALL_SYSTEMS:
        full_text = build_full_text(translations[sys_key], en_segments)
        words_per_system[sys_key] = uk_words(full_text)

    print("  Extracting & POS-tagging...")
    lemma_counts, vocab = _extract_lemma_freqs(morph, words_per_system)
    systems = list(ALL_SYSTEMS)
    N = len(systems)
    D = len(vocab)
    print(f"  {D} features after filtering (min total freq {MIN_TOTAL_FREQ})")

    freq_matrix = [[0.0] * D for _ in range(N)]
    for i, sys in enumerate(systems):
        total = sum(lemma_counts[sys].values())
        if total == 0:
            continue
        for j, w in enumerate(vocab):
            freq_matrix[i][j] = lemma_counts[sys].get(w, 0) / total

    means = [sum(freq_matrix[i][j] for i in range(N)) / N for j in range(D)]
    stds_pop = [
        sqrt(sum((freq_matrix[i][j] - means[j]) ** 2 for i in range(N)) / N)
        for j in range(D)
    ]
    stds_pop = [s if s > 0 else 1.0 for s in stds_pop]
    z_matrix = [
        [(freq_matrix[i][j] - means[j]) / stds_pop[j] for j in range(D)]
        for i in range(N)
    ]

    pairwise = {}
    for i in range(N):
        for j in range(i + 1, N):
            d = _cosine_distance(z_matrix[i], z_matrix[j])
            pairwise[f"{systems[i]}|{systems[j]}"] = d

    per_system = {}
    for i, sys_a in enumerate(systems):
        dists = []
        for j, sys_b in enumerate(systems):
            if i == j:
                continue
            key = (
                f"{sys_a}|{sys_b}" if f"{sys_a}|{sys_b}" in pairwise
                else f"{sys_b}|{sys_a}"
            )
            dists.append(pairwise[key])
        per_system[sys_a] = {"mean_distance": _mean(dists)}

    human_no_dybko = [s for s in HUMAN_SYSTEMS if s != DYBKO_KEY]
    ai_pairs = [
        v for k, v in pairwise.items()
        if k.split("|")[0] in AI_SYSTEMS and k.split("|")[1] in AI_SYSTEMS
    ]
    hh_pairs = [
        v for k, v in pairwise.items()
        if k.split("|")[0] in human_no_dybko and k.split("|")[1] in human_no_dybko
    ]

    ai_z = {systems.index(s) for s in AI_SYSTEMS}
    h_z = {systems.index(s) for s in human_no_dybko}
    top_features = []
    for j, feat in enumerate(vocab):
        ai_mean_z = _mean([z_matrix[i][j] for i in ai_z])
        h_mean_z = _mean([z_matrix[i][j] for i in h_z])
        top_features.append({
            "feature": feat, "ai_z": round(ai_mean_z, 3),
            "human_z": round(h_mean_z, 3),
            "diff": round(abs(ai_mean_z - h_mean_z), 3),
        })
    top_features.sort(key=lambda x: -x["diff"])

    results = {
        **per_system,
        "_pairwise": pairwise,
        "_n_features": D,
        "_group_stats": {
            "ai_ai_mean": _mean(ai_pairs), "ai_ai_std": _std(ai_pairs),
            "hh_mean": _mean(hh_pairs), "hh_std": _std(hh_pairs),
        },
        "_top_discriminating_features": top_features[:20],
    }
    return results


def print_summary(results: dict):
    nf = results["_n_features"]
    print(f"\nCosine Delta — mean pairwise distance ({nf} closed-class features)")
    print("  Higher = more stylistically isolated:")
    ranked = sorted(RANKED_SYSTEMS, key=lambda s: results[s]["mean_distance"], reverse=True)
    for i, sys in enumerate(ranked, 1):
        d = results[sys]["mean_distance"]
        tag = " <-- AI" if sys in AI_SYSTEMS else ""
        print(f"  {i:2}. {short_name(sys):35} {d:.4f}{tag}")

    gs = results["_group_stats"]
    print(f"\n  AI-AI: {gs['ai_ai_mean']:.4f} ± {gs['ai_ai_std']:.4f}")
    print(f"  H-H:   {gs['hh_mean']:.4f} ± {gs['hh_std']:.4f}")
    print(f"  Ratio:  {gs['hh_mean']/gs['ai_ai_mean']:.2f}x")

    print("\n  Top 10 discriminating features (|AI_z - H_z|):")
    for f in results["_top_discriminating_features"][:10]:
        direction = "AI+" if f["ai_z"] > f["human_z"] else "H+"
        print(f"    {f['feature']:20} diff={f['diff']:.3f}  ({direction})")


if __name__ == "__main__":
    data = load_translations()
    trans = data["translations"]
    from ._common import EN_KEY
    results = compute(trans, trans[EN_KEY])
    print_summary(results)
    save_results(results, RESULT_FILE)
