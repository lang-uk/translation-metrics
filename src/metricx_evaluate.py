"""MetricX translation evaluation: QE (reference-free) and round-robin.

Computes MetricX-24 hybrid scores for all 10 translation systems in two modes:
  - QE (reference-free): source + candidate
  - Round-robin (reference-based): source + candidate + reference, all N*(N-1) pairs

MetricX scores are 0–25 where LOWER is better (opposite of COMET).

Idempotent: skips systems/pairs already present in result files.

Usage:
    python src/metricx_evaluate.py --all                # run everything
    python src/metricx_evaluate.py --task qe             # QE for all systems
    python src/metricx_evaluate.py --task round_robin    # round-robin pairs
    python src/metricx_evaluate.py --task qe --force     # recompute from scratch

Requirements (GPU strongly recommended):
    pip install torch transformers numpy
"""

import argparse
import json
import sys
import time
from pathlib import Path

import importlib.util

import numpy as np
import torch
import transformers

ROOT = Path(__file__).resolve().parent.parent

_spec = importlib.util.spec_from_file_location(
    "_metricx_models", ROOT / "lib" / "metricx_local" / "models.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
MT5ForRegression = _mod.MT5ForRegression

TRANSLATIONS_PATH = ROOT / "results" / "unite-models" / "parsed_translations.json"
RESULTS_DIR = ROOT / "results" / "metricx"

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

MODEL_PATH = "google/metricx-24-hybrid-xl-v2p6"
TOKENIZER_NAME = "google/mt5-xl"
MAX_INPUT_LENGTH = 512 #original 1536


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


def load_metricx_model():
    """Load MetricX-24 model and tokenizer."""
    t0 = time.time()
    print(f"  Loading tokenizer {TOKENIZER_NAME}...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    print(f"  Loading model {MODEL_PATH}...")
    model = MT5ForRegression.from_pretrained(MODEL_PATH, torch_dtype="auto")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"  Loaded on {device} in {time.time() - t0:.0f}s")
    return model, tokenizer, device


def predict_metricx(model, tokenizer, device, inputs, batch_size=4):
    """Run MetricX prediction on a list of input strings.

    Returns list of scores (0–25, lower is better).
    """
    all_scores = []
    t0 = time.time()

    for start in range(0, len(inputs), batch_size):
        batch_texts = inputs[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        # MetricX removes the trailing EOS token
        input_ids = encoded["input_ids"][:, :-1].to(device)
        attention_mask = encoded["attention_mask"][:, :-1].to(device)

        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            scores = output.predictions.cpu().numpy().tolist()
            all_scores.extend(scores)

    elapsed = time.time() - t0
    print(f"  Predicted {len(inputs)} segments in {elapsed:.0f}s")
    return all_scores


def format_qe_input(source, hypothesis):
    return f"source: {source} candidate: {hypothesis}"


def format_ref_input(source, hypothesis, reference):
    return f"source: {source} candidate: {hypothesis} reference: {reference}"


# ─────────────────────────────────────────────────────────────────────
# QE (reference-free) for all 10 systems
# ─────────────────────────────────────────────────────────────────────
def task_qe(force=False, batch_size=4):
    """Score all 10 systems with MetricX-24 QE (reference-free)."""
    print("\n" + "=" * 70)
    print("TASK: MetricX-24 QE (reference-free) — all systems")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_file = RESULTS_DIR / "metricx24_qe.json"
    results = {} if force else (
        json.load(open(result_file)) if result_file.exists() else {}
    )

    trans = load_translations()["translations"]
    en = trans[EN_KEY]

    missing = [s for s in ALL_SYSTEMS if s not in results]
    if not missing:
        print("  All 10 systems present. Skipping.")
        return

    print(f"  Systems to score: {len(missing)}")
    model, tokenizer, device = load_metricx_model()

    for sys_key in missing:
        print(f"\n  Scoring {sys_key}...")
        inputs = [
            format_qe_input(en[i], trans[sys_key][i])
            for i in range(len(en))
        ]
        scores = predict_metricx(model, tokenizer, device, inputs, batch_size)
        results[sys_key] = score_stats(scores)
        print(f"    mean={results[sys_key]['mean']:.4f} (lower=better)")

        with open(result_file, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n  Saved → {result_file}")


# ─────────────────────────────────────────────────────────────────────
# Round-robin reference-based for all pairs
# ─────────────────────────────────────────────────────────────────────
def task_round_robin(force=False, batch_size=4):
    """Compute all N*(N-1) round-robin MetricX-24 ref-based pairs."""
    print("\n" + "=" * 70)
    print("TASK: MetricX-24 round-robin (reference-based) — all pairs")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_file = RESULTS_DIR / "metricx24_round_robin.json"
    results = {} if force else (
        json.load(open(result_file)) if result_file.exists() else {}
    )

    trans = load_translations()["translations"]
    en = trans[EN_KEY]

    all_pairs = []
    for hyp_key in ALL_SYSTEMS:
        for ref_key in ALL_SYSTEMS:
            if hyp_key != ref_key:
                pair_key = f"{hyp_key}_vs_{ref_key}"
                if pair_key not in results:
                    all_pairs.append((hyp_key, ref_key, pair_key))

    total = len(ALL_SYSTEMS) * (len(ALL_SYSTEMS) - 1)
    if not all_pairs:
        print(f"  All {total} pairs present. Skipping.")
        return

    print(f"  Pairs to compute: {len(all_pairs)}/{total}")
    model, tokenizer, device = load_metricx_model()

    for i, (hyp_key, ref_key, pair_key) in enumerate(all_pairs):
        print(f"\n  [{i+1}/{len(all_pairs)}] {pair_key}")
        inputs = [
            format_ref_input(en[j], trans[hyp_key][j], trans[ref_key][j])
            for j in range(len(en))
        ]
        scores = predict_metricx(model, tokenizer, device, inputs, batch_size)
        results[pair_key] = score_stats(scores)
        print(f"    mean={results[pair_key]['mean']:.4f}")

        if (i + 1) % 5 == 0 or i == len(all_pairs) - 1:
            with open(result_file, "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n  Saved → {result_file}")


# ─────────────────────────────────────────────────────────────────────
TASKS = {
    "qe": task_qe,
    "round_robin": task_round_robin,
}


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate translations with MetricX-24 (QE and round-robin)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Tasks:
  qe            MetricX-24 QE (reference-free) for all 10 systems.
                Scores are 0–25 where LOWER is better.
  round_robin   MetricX-24 reference-based round-robin for all 10×9 = 90
                directed pairs.

All tasks are idempotent — existing results are preserved unless --force is set.
""",
    )
    parser.add_argument("--task", choices=list(TASKS.keys()), help="Run a single task")
    parser.add_argument("--all", action="store_true", help="Run all tasks")
    parser.add_argument("--force", action="store_true", help="Recompute from scratch")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (default: 1, increase if GPU has >16GB)") #original 4
    args = parser.parse_args()

    if not args.task and not args.all:
        parser.print_help()
        return

    t_total = time.time()

    if args.all:
        for func in TASKS.values():
            func(force=args.force, batch_size=args.batch_size)
    else:
        TASKS[args.task](force=args.force, batch_size=args.batch_size)

    print(f"\n{'=' * 70}")
    print(f"Total time: {time.time() - t_total:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
