"""Neural MT metric evaluation: COMET and XCOMET.

Computes reference-free scores (COMETKiwi-22, COMETKiwi-XL, XCOMET-XXL)
and round-robin reference-based scores (COMET-22, XCOMET) for all 10
translation systems.

Idempotent: skips systems/pairs already present in result files, so safe to
re-run after interruption or when adding new systems.

Usage:
    python src/comet_evaluate.py --all                  # run everything
    python src/comet_evaluate.py --task ref_free         # all ref-free metrics
    python src/comet_evaluate.py --task round_robin      # all round-robin pairs
    python src/comet_evaluate.py --task ref_free --force  # recompute from scratch

Requirements (GPU recommended):
    pip install unbabel-comet numpy torch
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TRANSLATIONS_PATH = ROOT / "results" / "unite-models" / "parsed_translations.json"
COMET_DIR = ROOT / "results" / "comet"
RESULTS_DIR = ROOT / "results"

EN_KEY = "EN_1945_George_Orwell_Animal_Farm"
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

REF_FREE_METRICS = {
    "cometkiwi-22": ("Unbabel/wmt22-cometkiwi-da", "reference_free_cometkiwi-22.json"),
    "cometkiwi-xl": ("Unbabel/wmt23-cometkiwi-da-xl", "reference_free_cometkiwi-xl.json"),
    "xcomet": ("Unbabel/XCOMET-XXL", "reference_free_xcomet.json"),
}

ROUND_ROBIN_METRICS = {
    "comet-22": ("Unbabel/wmt22-comet-da", "round_robin_comet-22.json"),
    "xcomet": ("Unbabel/XCOMET-XXL", "round_robin_xcomet.json"),
}


def load_translations():
    with open(TRANSLATIONS_PATH) as f:
        return json.load(f)


def score_stats(scores):
    return {
        "scores": [float(x) for x in scores],
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "median": float(np.median(scores)),
        "num_segments": len(scores),
    }


def load_comet_model(model_path):
    from comet import download_model, load_from_checkpoint

    t0 = time.time()
    print(f"  Loading {model_path}...")
    ckpt = download_model(model_path)
    model = load_from_checkpoint(ckpt)
    print(f"  Loaded in {time.time() - t0:.0f}s")
    return model


def predict(model, data, batch_size=8, use_gpu=True):
    gpus = 1 if use_gpu else 0
    t0 = time.time()
    output = model.predict(data, batch_size=batch_size, gpus=gpus)
    print(f"  Predicted {len(data)} segments in {time.time() - t0:.0f}s")
    return list(output.scores)


# ─────────────────────────────────────────────────────────────────────
# Reference-free evaluation (COMETKiwi-22, COMETKiwi-XL, XCOMET)
# ─────────────────────────────────────────────────────────────────────
def task_ref_free(force=False):
    """Score all 10 systems with each reference-free metric."""
    print("\n" + "=" * 70)
    print("TASK: Reference-free evaluation — all systems, all metrics")
    print("=" * 70)

    COMET_DIR.mkdir(parents=True, exist_ok=True)
    trans = load_translations()["translations"]
    en = trans[EN_KEY]

    for metric_name, (model_path, filename) in REF_FREE_METRICS.items():
        result_file = COMET_DIR / filename
        results = {} if force else (
            json.load(open(result_file)) if result_file.exists() else {}
        )

        missing = [s for s in ALL_SYSTEMS if s not in results]
        if not missing:
            print(f"\n  {metric_name}: all 10 systems present. Skipping.")
            continue

        print(f"\n  --- {metric_name} ({len(missing)} systems to score) ---")
        model = load_comet_model(model_path)

        for sys_key in missing:
            print(f"  Scoring {sys_key}...")
            data = [{"src": s, "mt": h} for s, h in zip(en, trans[sys_key])]
            scores = predict(model, data)
            results[sys_key] = score_stats(scores)
            print(f"    mean={results[sys_key]['mean']:.4f}")

            with open(result_file, "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        del model
        print(f"  Saved → {result_file}")


# ─────────────────────────────────────────────────────────────────────
# Round-robin reference-based (COMET-22, XCOMET)
# ─────────────────────────────────────────────────────────────────────
def task_round_robin(force=False):
    """Compute all N*(N-1) round-robin pairs for each ref-based metric."""
    print("\n" + "=" * 70)
    print("TASK: Round-robin evaluation — all pairs, all metrics")
    print("=" * 70)

    COMET_DIR.mkdir(parents=True, exist_ok=True)
    trans = load_translations()["translations"]
    en = trans[EN_KEY]

    all_pairs = []
    for hyp_key in ALL_SYSTEMS:
        for ref_key in ALL_SYSTEMS:
            if hyp_key != ref_key:
                all_pairs.append((hyp_key, ref_key, f"{hyp_key}_vs_{ref_key}"))

    print(f"  Total pairs for 10 systems: {len(all_pairs)}")

    for metric_name, (model_path, filename) in ROUND_ROBIN_METRICS.items():
        result_file = COMET_DIR / filename
        results = {} if force else (
            json.load(open(result_file)) if result_file.exists() else {}
        )

        needed = [(h, r, pk) for h, r, pk in all_pairs if pk not in results]
        if not needed:
            print(f"\n  {metric_name}: all {len(all_pairs)} pairs present. Skipping.")
            continue

        print(f"\n  --- {metric_name} ({len(needed)}/{len(all_pairs)} pairs to compute) ---")
        model = load_comet_model(model_path)

        for i, (hyp_key, ref_key, pair_key) in enumerate(needed):
            print(f"  [{i+1}/{len(needed)}] {pair_key}")
            data = [
                {"src": s, "ref": r, "mt": h}
                for s, r, h in zip(en, trans[ref_key], trans[hyp_key])
            ]
            scores = predict(model, data)
            results[pair_key] = score_stats(scores)
            print(f"    mean={results[pair_key]['mean']:.4f}")

            if (i + 1) % 5 == 0 or i == len(needed) - 1:
                with open(result_file, "w") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

        del model
        print(f"  Saved → {result_file}")


# ─────────────────────────────────────────────────────────────────────
TASKS = {
    "ref_free": task_ref_free,
    "round_robin": task_round_robin,
}


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate translations with COMET / XCOMET neural MT metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Tasks:
  ref_free      Reference-free scoring (COMETKiwi-22, COMETKiwi-XL, XCOMET-XXL)
                for all 10 systems.
  round_robin   Round-robin reference-based scoring (COMET-22, XCOMET) for all
                10×9 = 90 directed pairs.

All tasks are idempotent — existing results are preserved unless --force is set.
""",
    )
    parser.add_argument("--task", choices=list(TASKS.keys()), help="Run a single task")
    parser.add_argument("--all", action="store_true", help="Run all tasks")
    parser.add_argument("--force", action="store_true", help="Recompute from scratch (ignore existing results)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    args = parser.parse_args()

    if not args.task and not args.all:
        parser.print_help()
        return

    t_total = time.time()

    if args.all:
        for func in TASKS.values():
            func(force=args.force)
    else:
        TASKS[args.task](force=args.force)

    print(f"\n{'=' * 70}")
    print(f"Total time: {time.time() - t_total:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
