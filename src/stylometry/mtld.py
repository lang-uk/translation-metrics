"""Lexical diversity & vocabulary profile.

Combines MTLD (length-independent diversity) with frequency-spectrum
statistics that show AI relies on a narrower, more common vocabulary:

  - MTLD / MATTR: lexical diversity  (AI lower than most humans)
  - Hapax ratio:  unique-word density (AI lower → fewer rare words)
  - Top-100 concentration: what share of tokens fall into the 100 most
    frequent word types  (AI higher → heavier reuse of common words)

Group-level convergence: AI systems cluster tightly on all measures
while human translators span a wide range.
"""

import re
from collections import Counter

from lexicalrichness import LexicalRichness

from ._common import (
    ALL_SYSTEMS, AI_SYSTEMS, HUMAN_SYSTEMS, RANKED_SYSTEMS, DYBKO_KEY,
    build_full_text, load_translations, save_results, short_name,
)

RESULT_FILE = "mtld.json"


def _word_tokens(full_text: str) -> list[str]:
    """Lowercased word tokens via simple regex (no external tokenizer needed)."""
    return re.findall(r"[а-яіїєґa-z'ʼ]+", full_text.lower())


def _frequency_stats(tokens: list[str]) -> dict:
    freq = Counter(tokens)
    n_tokens = len(tokens)
    n_types = len(freq)
    hapax = sum(1 for c in freq.values() if c == 1)
    top100_tokens = sum(c for _, c in freq.most_common(100))
    return {
        "hapax_count": hapax,
        "hapax_ratio": hapax / n_tokens,
        "hapax_type_pct": hapax / n_types,
        "top100_concentration": top100_tokens / n_tokens,
    }


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
        lr = LexicalRichness(full_text)
        tokens = _word_tokens(full_text)
        fstats = _frequency_stats(tokens)

        results[sys_key] = {
            "mtld": float(lr.mtld(threshold=0.72)),
            "mattr": float(lr.mattr(window_size=100)),
            "hdd": float(lr.hdd(draws=42)),
            "ttr": float(lr.ttr),
            "n_words": lr.words,
            "n_types": lr.terms,
            **fstats,
        }

    human_no_dybko = [s for s in HUMAN_SYSTEMS if s != DYBKO_KEY]
    for metric in ("mtld", "mattr", "hapax_ratio", "top100_concentration"):
        ai_vals = [results[s][metric] for s in AI_SYSTEMS]
        h_vals = [results[s][metric] for s in human_no_dybko]
        results[f"_group_{metric}"] = {
            "ai_mean": _mean(ai_vals),
            "ai_std": _std(ai_vals),
            "human_mean": _mean(h_vals),
            "human_std": _std(h_vals),
        }

    return results


def print_summary(results: dict):
    print("\nLexical diversity & vocabulary profile:")
    header = f"  {'#':>2}  {'System':35} {'MTLD':>6} {'MATTR':>7} {'Hapax':>7} {'Top100%':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    ranked = sorted(RANKED_SYSTEMS, key=lambda s: results[s]["mtld"], reverse=True)
    for i, sys in enumerate(ranked, 1):
        r = results[sys]
        tag = " <--AI" if sys in AI_SYSTEMS else ""
        print(
            f"  {i:2}. {short_name(sys):35} {r['mtld']:6.1f}"
            f"  {r['mattr']:.4f}"
            f"  {r['hapax_ratio']:.4f}"
            f"  {r['top100_concentration']:7.1%}{tag}"
        )

    print("\n  Group convergence (excl. Dybko):")
    for metric, label in [
        ("mtld", "MTLD"), ("mattr", "MATTR"),
        ("hapax_ratio", "Hapax/N"), ("top100_concentration", "Top100%"),
    ]:
        gs = results[f"_group_{metric}"]
        print(
            f"    {label:12}  AI: {gs['ai_mean']:.4f} ± {gs['ai_std']:.4f}"
            f"    Human: {gs['human_mean']:.4f} ± {gs['human_std']:.4f}"
        )


if __name__ == "__main__":
    data = load_translations()
    trans = data["translations"]
    from ._common import EN_KEY
    results = compute(trans, trans[EN_KEY])
    print_summary(results)
    save_results(results, RESULT_FILE)
