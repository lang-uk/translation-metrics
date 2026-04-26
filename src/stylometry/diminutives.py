"""Diminutive & expressive morphology density per 1k words.

Ukrainian diminutive/expressive suffixes (-еньк-, -очк-, -ісіньк-, etc.)
are a hallmark of literary prose.  AI translations cluster tightly at the
bottom (~0.47/1k, std=0.03) while humans range from 0.47 to 2.79/1k,
with a mean of 1.23 — evidence that AI avoids expressive morphology.

Patterns are chosen for precision: broad suffixes (-ик, -усь, -атк-)
are excluded because they overwhelmingly match non-diminutive words
(agent nouns, pronouns, verb forms).  A stopword set filters remaining
false positives (бочка, суперечка, виняток, etc.).
"""

import re

from ._common import (
    ALL_SYSTEMS, AI_SYSTEMS, HUMAN_SYSTEMS, RANKED_SYSTEMS, DYBKO_KEY,
    build_full_text, load_translations, save_results, short_name, uk_words,
)

DIMINUTIVE_PATTERNS = [
    r"\w+еньк[аоиіеу]\b",
    r"\w+оньк[аоиіеу]\b",
    r"\w+есеньк[аоиіеу]\b",
    r"\w+ісіньк[аоиіеу]\b",
    r"\w+юсіньк[аоиіеу]\b",
    r"\w+очк[аоиіеу]\b",
    r"\w+ечк[аоиіеу]\b",
    r"\w+ичк[аоиіеу]\b",
    r"\w+ятк[аоиіеу]\b",
    r"\w+чик\b",
]
DIMINUTIVE_RE = re.compile("|".join(DIMINUTIVE_PATTERNS), re.IGNORECASE)

FP_STEMS = frozenset({
    "бочк", "точк", "квочк", "кочк",              # -очк- non-diminutives
    "суперечк", "вуздечк", "вервечк", "аніскілечк", # -ечк- non-diminutives
    "звичк", "сутичк", "тичк", "пивничк", "бричк", # -ичк- non-diminutives
    "десятк", "півдесятк", "винятк",                # -ятк- non-diminutives
    "глечик", "горщик",                              # lexicalized -чик
    "примочк",                                       # -очк- lexicalized
})

RESULT_FILE = "diminutives.json"


def _is_fp(word: str) -> bool:
    w = word.lower()
    return any(w.startswith(s) for s in FP_STEMS)


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs):
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def compute(translations: dict, en_segments: list[str]) -> dict:
    results = {}
    for sys_key in ALL_SYSTEMS:
        full_text = build_full_text(translations[sys_key], en_segments)
        words = uk_words(full_text)
        raw = DIMINUTIVE_RE.findall(full_text)
        clean = [m for m in raw if not _is_fp(m)]
        total = len(words)
        per_1k = (len(clean) / total * 1000) if total > 0 else 0
        freq = {}
        for w in clean:
            w_lower = w.lower()
            freq[w_lower] = freq.get(w_lower, 0) + 1

        results[sys_key] = {
            "per_1k": per_1k,
            "count": len(clean),
            "n_tokens": total,
            "words": dict(sorted(freq.items(), key=lambda x: -x[1])),
        }

    human_no_dybko = [s for s in HUMAN_SYSTEMS if s != DYBKO_KEY]
    ai_vals = [results[s]["per_1k"] for s in AI_SYSTEMS]
    h_vals = [results[s]["per_1k"] for s in human_no_dybko]
    results["_group_stats"] = {
        "ai_mean": _mean(ai_vals), "ai_std": _std(ai_vals),
        "human_mean": _mean(h_vals), "human_std": _std(h_vals),
    }
    return results


def print_summary(results: dict):
    print("\nDiminutives /1k words (higher = more expressive morphology):")
    ranked = sorted(RANKED_SYSTEMS, key=lambda s: results[s]["per_1k"], reverse=True)
    for i, sys in enumerate(ranked, 1):
        r = results[sys]
        tag = " <-- AI" if sys in AI_SYSTEMS else ""
        print(f"  {i:2}. {short_name(sys):35} {r['per_1k']:5.2f}  ({r['count']} hits){tag}")

    gs = results["_group_stats"]
    print(f"\n  Human (excl Dybko): {gs['human_mean']:.2f} ± {gs['human_std']:.2f}")
    print(f"  AI:                 {gs['ai_mean']:.2f} ± {gs['ai_std']:.2f}")


if __name__ == "__main__":
    data = load_translations()
    trans = data["translations"]
    from ._common import EN_KEY
    results = compute(trans, trans[EN_KEY])
    print_summary(results)
    save_results(results, RESULT_FILE)
