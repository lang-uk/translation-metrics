"""CLI entry point: python -m src.stylometry"""

import argparse
import time

from . import _common as common
from . import discourse_particles, cosine_delta, chrf, mtld, word_ratio, diminutives

METRICS = {
    "discourse_particles": discourse_particles,
    "cosine_delta": cosine_delta,
    "chrf": chrf,
    "mtld": mtld,
    "word_ratio": word_ratio,
    "diminutives": diminutives,
}


def main():
    parser = argparse.ArgumentParser(description="Stylometric analysis of translations")
    parser.add_argument("--all", action="store_true", help="Run all metrics")
    parser.add_argument(
        "--metric", action="append", choices=list(METRICS.keys()),
        help="Run specific metric(s); can be repeated",
    )
    args = parser.parse_args()

    if not args.all and not args.metric:
        parser.print_help()
        return

    to_run = list(METRICS.keys()) if args.all else args.metric

    print("Loading translations...")
    data = common.load_translations()
    trans = data["translations"]
    en_segments = trans[common.EN_KEY]

    t_total = time.time()

    for name in to_run:
        mod = METRICS[name]
        print(f"\n{'=' * 60}")
        print(f"  {name}")
        print(f"{'=' * 60}")
        t0 = time.time()
        results = mod.compute(trans, en_segments)
        mod.print_summary(results)
        common.save_results(results, mod.RESULT_FILE)
        print(f"  ({time.time() - t0:.1f}s)")

    print(f"\nTotal: {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
