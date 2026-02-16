"""Translation evaluation using various metrics."""

import json
import os
from pathlib import Path
from token import COMMENT
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config import (
    DEFAULT_CONFIG,
    DUAL_MODE_METRICS,
    REFERENCE_BASED_METRICS,
    REFERENCE_FREE_METRICS,
)


class COMETMetricLoader:
    """Loads and manages different translation evaluation metrics."""

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize metric loader.

        Args:
            cache_dir: Directory to cache models
        """
        self.cache_dir = cache_dir or DEFAULT_CONFIG["cache_dir"]
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        self.loaded_metrics = {}

    def load_metric(
        self, metric_name: str, metric_path: str, mode: str = "reference-free"
    ):
        """
        Load a specific metric.

        Args:
            metric_name: Short name for the metric
            metric_path: HuggingFace model path
            mode: 'reference-free' or 'reference-based'
        """
        if metric_name in self.loaded_metrics:
            print(f"Metric {metric_name} already loaded.")
            return self.loaded_metrics[metric_name]

        print(f"Loading {metric_name} ({metric_path})...")

        try:
            # COMET/COMETKIWI metrics
            if "comet" in metric_path.lower():
                from comet import download_model, load_from_checkpoint

                model_path = download_model(
                    metric_path, saving_directory=self.cache_dir
                )
                model = load_from_checkpoint(model_path)
                self.loaded_metrics[metric_name] = {
                    "model": model,
                    "type": "comet",
                    "mode": mode,
                }
                print(f"✓ Loaded {metric_name}")
                return self.loaded_metrics[metric_name]

            # XCOMET
            elif "xcomet" in metric_path.lower():
                from comet import download_model, load_from_checkpoint

                model_path = download_model(
                    metric_path, saving_directory=self.cache_dir
                )
                model = load_from_checkpoint(model_path)
                self.loaded_metrics[metric_name] = {
                    "model": model,
                    "type": "xcomet",
                    "mode": mode,
                }
                print(f"✓ Loaded {metric_name}")
                return self.loaded_metrics[metric_name]

            else:
                print(f"⚠ Unknown metric type for {metric_name}. Skipping.")
                return None

        except Exception as e:
            print(f"✗ Error loading {metric_name}: {e}")
            return None

    def unload_metric(self, metric_name: str):
        """Unload a metric to free memory."""
        if metric_name in self.loaded_metrics:
            del self.loaded_metrics[metric_name]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"Unloaded {metric_name}")


class TranslationEvaluator:
    """Evaluates translations using various metrics."""

    def __init__(self, cache_dir: Optional[str] = None, device: str = "cuda"):
        """
        Initialize the evaluator.

        Args:
            cache_dir: Directory to cache models
            device: Device to run models on ('cuda' or 'cpu')
        """
        self.cache_dir = cache_dir or DEFAULT_CONFIG["cache_dir"]
        self.device = device if torch.cuda.is_available() else "cpu"
        self.metric_loader = COMETMetricLoader(cache_dir=self.cache_dir)

        if self.device == "cpu":
            print("⚠ CUDA not available, using CPU. Evaluation will be slower.")

    def evaluate_reference_free(
        self,
        sources: List[str],
        hypotheses: List[str],
        metric_name: str,
        metric_path: str,
        batch_size: int = 8,
    ) -> Dict[str, float]:
        """
        Evaluate using a reference-free metric.

        Args:
            sources: List of source sentences
            hypotheses: List of hypothesis translations
            metric_name: Short name for the metric
            metric_path: HuggingFace model path
            batch_size: Batch size for evaluation

        Returns:
            Dictionary with scores and statistics
        """
        # Load metric
        metric = self.metric_loader.load_metric(
            metric_name, metric_path, mode="reference-free"
        )
        if metric is None:
            return {"error": f"Could not load metric {metric_name}"}

        # # Filter out empty segments
        # valid_pairs = [(src, hyp) for src, hyp in zip(sources, hypotheses)
        #                if src.strip() and hyp.strip() and hyp.strip() != '---']

        valid_pairs = list(zip(sources, hypotheses))

        if not valid_pairs:
            return {"error": "No valid source-hypothesis pairs"}

        sources_valid, hypotheses_valid = zip(*valid_pairs)

        try:
            if metric["type"] in ["comet", "xcomet"]:
                # COMET-style evaluation
                data = [
                    {"src": src, "mt": hyp}
                    for src, hyp in zip(sources_valid, hypotheses_valid)
                ]

                model_output = metric["model"].predict(
                    data, batch_size=batch_size, gpus=1 if self.device == "cuda" else 0
                )

                scores = model_output.scores

            else:
                return {"error": f"Unsupported metric type: {metric['type']}"}

            return {
                "scores": scores,
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
                "median": float(np.median(scores)),
                "num_segments": len(scores),
            }

        except Exception as e:
            return {"error": f"Error during evaluation: {str(e)}"}

    def evaluate_reference_based(
        self,
        sources: List[str],
        references: List[str],
        hypotheses: List[str],
        metric_name: str,
        metric_path: str,
        batch_size: int = 8,
    ) -> Dict[str, float]:
        """
        Evaluate using a reference-based metric.

        Args:
            sources: List of source sentences
            references: List of reference translations
            hypotheses: List of hypothesis translations
            metric_name: Short name for the metric
            metric_path: HuggingFace model path
            batch_size: Batch size for evaluation

        Returns:
            Dictionary with scores and statistics
        """
        # Load metric
        metric = self.metric_loader.load_metric(
            metric_name, metric_path, mode="reference-based"
        )
        if metric is None:
            return {"error": f"Could not load metric {metric_name}"}

        # # Filter out empty segments
        # valid_triples = [
        #     (src, ref, hyp)
        #     for src, ref, hyp in zip(sources, references, hypotheses)
        #     if src.strip() and ref.strip() and hyp.strip()
        #     and ref.strip() != '---' and hyp.strip() != '---'
        # ]

        valid_triples = list(zip(sources, references, hypotheses))

        if not valid_triples:
            return {"error": "No valid source-reference-hypothesis triples"}

        sources_valid, references_valid, hypotheses_valid = zip(*valid_triples)

        try:
            if metric["type"] in ["comet", "xcomet"]:
                # COMET-style evaluation
                data = [
                    {"src": src, "ref": ref, "mt": hyp}
                    for src, ref, hyp in zip(
                        sources_valid, references_valid, hypotheses_valid
                    )
                ]

                model_output = metric["model"].predict(
                    data, batch_size=batch_size, gpus=1 if self.device == "cuda" else 0
                )

                scores = model_output.scores

            else:
                return {"error": f"Unsupported metric type: {metric['type']}"}

            return {
                "scores": scores,
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
                "median": float(np.median(scores)),
                "num_segments": len(scores),
            }

        except Exception as e:
            return {"error": f"Error during evaluation: {str(e)}"}

    def evaluate_round_robin(
        self,
        sources: List[str],
        translations: Dict[str, List[str]],
        metric_name: str,
        metric_path: str,
        batch_size: int = 8,
    ) -> Dict[str, Dict]:
        """
        Evaluate translations in round-robin fashion (each as reference).

        Args:
            sources: List of source sentences
            translations: Dictionary of {lang_code: [translations]}
            metric_name: Short name for the metric
            metric_path: HuggingFace model path
            batch_size: Batch size for evaluation

        Returns:
            Dictionary with results for each reference-hypothesis pair
        """
        results = {}
        translation_ids = list(translations.keys())

        for ref_id in translation_ids:
            ref_translations = translations[ref_id]

            for hyp_id in translation_ids:
                if ref_id == hyp_id:
                    continue

                hyp_translations = translations[hyp_id]

                pair_key = f"{ref_id}_vs_{hyp_id}"
                print(
                    f"\nEvaluating: {pair_key} (reference: {ref_id}, hypothesis: {hyp_id})"
                )

                result = self.evaluate_reference_based(
                    sources=sources,
                    references=ref_translations,
                    hypotheses=hyp_translations,
                    metric_name=metric_name,
                    metric_path=metric_path,
                    batch_size=batch_size,
                )

                results[pair_key] = result

        return results

    def cleanup(self):
        """Clean up loaded models and free memory."""
        for metric_name in list(self.metric_loader.loaded_metrics.keys()):
            self.metric_loader.unload_metric(metric_name)


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Translation evaluation module for TMX data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This module is primarily intended to be imported and used programmatically.

For command-line evaluation, use main.py instead:
  python main.py <tmx_file> --mode reference-free --metrics cometkiwi-xl

For examples, see:
  python examples.py

Available metrics:
  Reference-free: cometkiwi-xxl, cometkiwi-xl, metricx-qe, cometkiwi-22, xcomet, metricx-hybrid
  Reference-based: metricx-23, comet-22, xcomet, metricx-hybrid
        """,
    )

    parser.add_argument(
        "--list-metrics", action="store_true", help="List all available metrics"
    )

    parser.add_argument(
        "--check-cuda", action="store_true", help="Check CUDA availability"
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        default=".cache",
        help="Model cache directory (default: .cache)",
    )

    args = parser.parse_args()

    if args.list_metrics:
        print("\n" + "=" * 80)
        print("AVAILABLE METRICS")
        print("=" * 80)

        print("\nReference-Free Metrics (source + hypothesis):")
        for name, path in REFERENCE_FREE_METRICS.items():
            print(f"  - {name:20} {path}")

        print("\nReference-Based Metrics (source + reference + hypothesis):")
        for name, path in REFERENCE_BASED_METRICS.items():
            print(f"  - {name:20} {path}")

        print("\nDual-Mode Metrics (can be used in both scenarios):")
        for name, path in DUAL_MODE_METRICS.items():
            print(f"  - {name:20} {path}")

        print("\n" + "=" * 80)
        sys.exit(0)

    if args.check_cuda:
        print("\n" + "=" * 80)
        print("CUDA AVAILABILITY CHECK")
        print("=" * 80)

        try:
            print(f"\nPyTorch version: {torch.__version__}")
            cuda_available = torch.cuda.is_available()
            print(f"CUDA available: {cuda_available}")

            if cuda_available:
                print(f"CUDA version: {torch.version.cuda}")
                print(f"Number of GPUs: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                    print(
                        f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB"
                    )
            else:
                print("\n⚠ CUDA not available - will use CPU (slower)")
                print("  To enable GPU acceleration, install PyTorch with CUDA:")
                print("  https://pytorch.org/get-started/locally/")
        except Exception as e:
            print(f"\n✗ Error checking CUDA: {e}")

        print("\n" + "=" * 80)
        sys.exit(0)

    # Default message if no arguments provided
    print("\nTranslation Evaluator Module")
    print(
        "\nThis module is primarily intended to be imported and used programmatically."
    )
    print("\nFor command-line evaluation, use:")
    print("  python main.py <tmx_file> --mode reference-free --metrics cometkiwi-xl")
    print("\nFor usage examples, see:")
    print("  python examples.py")
    print("\nFor more options, run:")
    print("  python evaluator.py --help")
    print("  python evaluator.py --list-metrics")
    print("  python evaluator.py --check-cuda")
