"""Segment-level word ratio (translation / source word count).

Computes per-segment word count ratio against the English source.
The std of this ratio signals mechanical consistency: LaPa (0.127) and
DeepL (0.132) produce segments of very predictable length relative to
the source, while human translators freely expand or compress (std 0.16-0.34).
"""

from ._common import (
    ALL_SYSTEMS, AI_SYSTEMS, HUMAN_SYSTEMS, RANKED_SYSTEMS, DYBKO_KEY,
    is_valid_segment, load_translations, save_results, short_name, uk_words,
)

RESULT_FILE = "word_ratio.json"


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs):
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def _median(xs):
    s = sorted(xs)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def compute(translations: dict, en_segments: list[str]) -> dict:
    results = {}
    for sys_key in ALL_SYSTEMS:
        ratios = []
        for en, tr in zip(en_segments, translations[sys_key]):
            if not is_valid_segment(en) or not is_valid_segment(tr):
                continue
            en_words = uk_words(en)
            tr_words = uk_words(tr)
            if len(en_words) > 0:
                ratios.append(len(tr_words) / len(en_words))

        results[sys_key] = {
            "mean": _mean(ratios),
            "std": _std(ratios),
            "median": _median(ratios),
            "n_segments": len(ratios),
        }

    human_no_dybko = [s for s in HUMAN_SYSTEMS if s != DYBKO_KEY]
    ai_vals = [results[s]["std"] for s in AI_SYSTEMS]
    h_vals = [results[s]["std"] for s in human_no_dybko]
    results["_group_stats"] = {
        "ai_mean_std": _mean(ai_vals),
        "human_mean_std": _mean(h_vals),
    }
    return results


def print_summary(results: dict):
    print("\nWord ratio std (lower = more mechanically uniform):")
    ranked = sorted(RANKED_SYSTEMS, key=lambda s: results[s]["std"])
    for i, sys in enumerate(ranked, 1):
        r = results[sys]
        tag = " <-- AI" if sys in AI_SYSTEMS else ""
        print(f"  {i:2}. {short_name(sys):35} std={r['std']:.4f}  mean={r['mean']:.3f}{tag}")

    gs = results["_group_stats"]
    print(f"\n  Human mean std (excl Dybko): {gs['human_mean_std']:.4f}")
    print(f"  AI mean std:                 {gs['ai_mean_std']:.4f}")


if __name__ == "__main__":
    data = load_translations()
    trans = data["translations"]
    from ._common import EN_KEY
    results = compute(trans, trans[EN_KEY])
    print_summary(results)
    save_results(results, RESULT_FILE)
