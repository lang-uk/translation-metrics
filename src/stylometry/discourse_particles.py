"""Discourse particle density per 1k words.

Ukrainian discourse particles (ж, таки, ось, аж, ну, хіба, ніби, etc.)
are pragmatic markers that signal emphasis, surprise, hedging, or speaker
attitude.  Human literary translators use them at roughly 2x the rate of
AI systems (mean 8.4 vs 4.1 per 1k), making this one of the clearest
human-vs-AI separators.

Excluded from the set: "та" (conjunction "and"), "бо" (conjunction
"because"), "тож"/"адже" (conjunctions) — these inflate counts without
reflecting genuine pragmatic richness.
"""

from collections import Counter

from ._common import (
    ALL_SYSTEMS, AI_SYSTEMS, HUMAN_SYSTEMS, RANKED_SYSTEMS, DYBKO_KEY,
    build_full_text, load_translations, save_results, short_name, uk_words,
)

DISCOURSE_PARTICLES = {
    "ж", "же",                                            # emphasis
    "таки",                                                # emphasis ("все-таки")
    "ось", "он", "от",                                     # deictic
    "аж",                                                  # intensifier
    "ну",                                                  # filler / initiator
    "мов", "наче", "немов", "немовби", "неначе", "мовби",  # comparative hedging
    "ніби",                                                # hedging ("as if")
    "ото", "отож", "авжеж",                                # confirmative
    "еге", "угу",                                          # backchannels
    "хіба", "невже",                                       # rhetorical / surprise
    "хай", "нехай",                                        # optative
    "геть",                                                # emphatic intensifier
}

RESULT_FILE = "discourse_particles.json"


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
        lower_words = [w.lower() for w in words]
        total = len(words)

        counts = Counter(w for w in lower_words if w in DISCOURSE_PARTICLES)
        total_particles = sum(counts.values())
        per_1k = (total_particles / total * 1000) if total > 0 else 0

        results[sys_key] = {
            "per_1k": per_1k,
            "count": total_particles,
            "n_tokens": total,
            "words": dict(counts.most_common()),
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
    print("\nDiscourse particles /1k words (higher = more pragmatic richness):")
    ranked = sorted(RANKED_SYSTEMS, key=lambda s: results[s]["per_1k"], reverse=True)
    for i, sys in enumerate(ranked, 1):
        r = results[sys]
        tag = " <-- AI" if sys in AI_SYSTEMS else ""
        top3 = ", ".join(f"{w}={c}" for w, c in list(r["words"].items())[:3])
        print(f"  {i:2}. {short_name(sys):35} {r['per_1k']:6.2f}  ({top3}){tag}")

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
