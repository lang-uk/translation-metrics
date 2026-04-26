"""Pairwise surface overlap: chrF + BLEU.

chrF (character n-gram F-score) and BLEU (word n-gram precision) are
independent surface similarity measures at different granularities.
Both are computed pairwise between all Ukrainian translations.

AI translations cluster at high mutual overlap on both metrics,
while human translations spread across a wide range -- direct
evidence of surface-form convergence.
"""

import sacrebleu

from ._common import (
    ALL_SYSTEMS, AI_SYSTEMS, HUMAN_SYSTEMS, RANKED_SYSTEMS, DYBKO_KEY,
    is_valid_segment, load_translations, save_results, short_name,
)

RESULT_FILE = "chrf.json"


def _valid_pairs(sys_a_segs, sys_b_segs, en_segments):
    a_valid, b_valid = [], []
    for i in range(min(len(sys_a_segs), len(sys_b_segs))):
        if i < len(en_segments) and is_valid_segment(en_segments[i]):
            a_valid.append(sys_a_segs[i].strip())
            b_valid.append(sys_b_segs[i].strip())
    return a_valid, b_valid


def _corpus_chrf(sys_segs, ref_segs, en_segs):
    hyp, ref = _valid_pairs(sys_segs, ref_segs, en_segs)
    return float(sacrebleu.corpus_chrf(hyp, [ref]).score)


def _corpus_bleu(sys_segs, ref_segs, en_segs):
    hyp, ref = _valid_pairs(sys_segs, ref_segs, en_segs)
    return float(sacrebleu.corpus_bleu(hyp, [ref]).score)


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs):
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def compute(translations: dict, en_segments: list[str]) -> dict:
    pairwise_chrf = {}
    pairwise_bleu = {}

    for i, sys_a in enumerate(ALL_SYSTEMS):
        for j in range(i + 1, len(ALL_SYSTEMS)):
            sys_b = ALL_SYSTEMS[j]
            key = f"{sys_a}|{sys_b}"
            pairwise_chrf[key] = _corpus_chrf(
                translations[sys_a], translations[sys_b], en_segments,
            )
            pairwise_bleu[key] = _corpus_bleu(
                translations[sys_a], translations[sys_b], en_segments,
            )

    per_system = {}
    for sys_key in ALL_SYSTEMS:
        chrf_scores, bleu_scores = [], []
        for key, score in pairwise_chrf.items():
            a, b = key.split("|")
            if a == sys_key or b == sys_key:
                chrf_scores.append(score)
        for key, score in pairwise_bleu.items():
            a, b = key.split("|")
            if a == sys_key or b == sys_key:
                bleu_scores.append(score)
        per_system[sys_key] = {
            "mean_pairwise_chrf": _mean(chrf_scores),
            "mean_pairwise_bleu": _mean(bleu_scores),
        }

    human_no_dybko = [s for s in HUMAN_SYSTEMS if s != DYBKO_KEY]

    def _group_pairwise(pairs_dict, group):
        vals = []
        for key, score in pairs_dict.items():
            a, b = key.split("|")
            if a in group and b in group:
                vals.append(score)
        return vals

    ai_chrf = _group_pairwise(pairwise_chrf, AI_SYSTEMS)
    h_chrf = _group_pairwise(pairwise_chrf, human_no_dybko)
    ai_bleu = _group_pairwise(pairwise_bleu, AI_SYSTEMS)
    h_bleu = _group_pairwise(pairwise_bleu, human_no_dybko)

    results = {
        **per_system,
        "_pairwise_chrf": pairwise_chrf,
        "_pairwise_bleu": pairwise_bleu,
        "_group_stats": {
            "chrf_ai_ai_mean": _mean(ai_chrf), "chrf_ai_ai_std": _std(ai_chrf),
            "chrf_hh_mean": _mean(h_chrf), "chrf_hh_std": _std(h_chrf),
            "bleu_ai_ai_mean": _mean(ai_bleu), "bleu_ai_ai_std": _std(ai_bleu),
            "bleu_hh_mean": _mean(h_bleu), "bleu_hh_std": _std(h_bleu),
        },
    }
    return results


def print_summary(results: dict):
    print("\nMean pairwise similarity per system (higher = more like others):")
    print(f"  {'#':>2}  {'System':35} {'chrF':>7} {'BLEU':>7}")
    print("  " + "-" * 55)
    ranked = sorted(
        RANKED_SYSTEMS,
        key=lambda s: results[s]["mean_pairwise_chrf"], reverse=True,
    )
    for i, sys in enumerate(ranked, 1):
        r = results[sys]
        tag = " <--AI" if sys in AI_SYSTEMS else ""
        print(
            f"  {i:2}. {short_name(sys):35}"
            f" {r['mean_pairwise_chrf']:6.2f}"
            f"  {r['mean_pairwise_bleu']:6.2f}{tag}"
        )

    gs = results["_group_stats"]
    print(f"\n  Intra-group convergence (excl. Dybko):")
    print(
        f"    chrF   AI-AI: {gs['chrf_ai_ai_mean']:.2f} ± {gs['chrf_ai_ai_std']:.2f}"
        f"    H-H: {gs['chrf_hh_mean']:.2f} ± {gs['chrf_hh_std']:.2f}"
    )
    print(
        f"    BLEU   AI-AI: {gs['bleu_ai_ai_mean']:.2f} ± {gs['bleu_ai_ai_std']:.2f}"
        f"    H-H: {gs['bleu_hh_mean']:.2f} ± {gs['bleu_hh_std']:.2f}"
    )


if __name__ == "__main__":
    data = load_translations()
    trans = data["translations"]
    from ._common import EN_KEY
    results = compute(trans, trans[EN_KEY])
    print_summary(results)
    save_results(results, RESULT_FILE)
