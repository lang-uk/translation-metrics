"""Main script for running TMX translation evaluation."""

import argparse
import json
import sys
from pathlib import Path

from config import (
    DEFAULT_CONFIG,
    DUAL_MODE_METRICS,
    REFERENCE_BASED_METRICS,
    REFERENCE_FREE_METRICS,
)
from custom_translations_loader import CustomTranslationsLoader
from evaluator import TranslationEvaluator
from tmx_parser import TMXParser
from visualizer import ResultsVisualizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate translations from TMX files using various metrics"
    )

    parser.add_argument("tmx_file", type=str, help="Path to TMX file")

    parser.add_argument(
        "--mode",
        type=str,
        choices=["reference-free", "reference-based", "round-robin", "all"],
        default="all",
        help="Evaluation mode",
    )

    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        help="Specific metrics to use (default: all available for the mode)",
    )

    parser.add_argument(
        "--output-dir", type=str, default="results", help="Output directory for results"
    )

    parser.add_argument(
        "--cache-dir", type=str, default=".cache", help="Cache directory for models"
    )

    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size for evaluation"
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device to run models on",
    )

    parser.add_argument(
        "--visualize", action="store_true", help="Generate visualizations"
    )

    parser.add_argument(
        "--no-save", action="store_true", help="Do not save results to disk"
    )

    parser.add_argument(
        "--custom-translations",
        type=str,
        nargs="+",
        help="Custom translation file(s) to evaluate (JSON, JSONL, CSV, Parquet, TXT)",
    )

    parser.add_argument(
        "--custom-translations-dir",
        type=str,
        help="Directory containing custom translation files",
    )

    parser.add_argument(
        "--source-field",
        type=str,
        default="source",
        help="Field name for source text in custom translations (default: source)",
    )

    parser.add_argument(
        "--translation-field",
        type=str,
        default="translation",
        help="Field name for translations in custom translations (default: translation)",
    )

    parser.add_argument(
        "--newlines-strategy",
        type=str,
        choices=["combine", "separate"],
        default="combine",
        help="How to handle newlines in TMX entries: 'combine' (replace with spaces) or 'separate' (split into multiple segments)",
    )

    return parser.parse_args()


def run_reference_free_evaluation(
    parser: TMXParser,
    evaluator: TranslationEvaluator,
    metrics: dict,
    batch_size: int,
    output_dir: Path,
):
    """Run reference-free evaluation."""
    print("\n" + "=" * 80)
    print("REFERENCE-FREE EVALUATION")
    print("=" * 80)

    source_lang = parser.get_source_language()
    target_langs = parser.get_target_languages()
    sources = parser.get_source_segments()

    # Get all languages/systems to evaluate (including custom)
    all_langs = parser.get_all_languages()
    systems_to_evaluate = [lang for lang in all_langs if lang != source_lang]

    # Identify custom systems for clearer output
    stats = parser.get_statistics()
    custom_systems = stats.get("custom_systems", [])

    all_results = {}

    for metric_name, metric_path in metrics.items():
        print(f"\n{'=' * 80}")
        print(f"Evaluating with {metric_name}")
        print(f"{'=' * 80}")

        metric_results = {}

        for system in systems_to_evaluate:
            is_custom = system in custom_systems
            label = f"{system} [CUSTOM]" if is_custom else system
            print(f"\n→ Evaluating {label}...")
            hypotheses = parser.get_target_segments(system)

            results = evaluator.evaluate_reference_free(
                sources=sources,
                hypotheses=hypotheses,
                metric_name=metric_name,
                metric_path=metric_path,
                batch_size=batch_size,
            )

            metric_results[system] = results

            if "error" not in results:
                print(f"  ✓ Mean score: {results['mean']:.4f}")
            else:
                print(f"  ✗ {results['error']}")

        all_results[metric_name] = metric_results

        # Save intermediate results
        output_file = output_dir / f"reference_free_{metric_name}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(metric_results, f, ensure_ascii=False, indent=2)
        print(f"\n→ Results saved to {output_file}")

    return all_results


def run_round_robin_evaluation(
    parser: TMXParser,
    evaluator: TranslationEvaluator,
    metrics: dict,
    batch_size: int,
    output_dir: Path,
):
    """Run round-robin (each translation as reference) evaluation."""
    print("\n" + "=" * 80)
    print("ROUND-ROBIN EVALUATION")
    print("=" * 80)

    sources = parser.get_source_segments()
    source_lang = parser.get_source_language()

    # Get all languages/systems to evaluate (including custom)
    all_langs = parser.get_all_languages()
    systems_to_evaluate = [lang for lang in all_langs if lang != source_lang]

    # Prepare translations dictionary
    translations = {}
    for system in systems_to_evaluate:
        translations[system] = parser.get_target_segments(system)

    print(f"\n→ Evaluating {len(systems_to_evaluate)} systems in round-robin fashion")
    print(f"  Systems: {', '.join(systems_to_evaluate)}")

    all_results = {}

    for metric_name, metric_path in metrics.items():
        print(f"\n{'=' * 80}")
        print(f"Evaluating with {metric_name}")
        print(f"{'=' * 80}")

        results = evaluator.evaluate_round_robin(
            sources=sources,
            translations=translations,
            metric_name=metric_name,
            metric_path=metric_path,
            batch_size=batch_size,
        )

        all_results[metric_name] = results

        # Save intermediate results
        output_file = output_dir / f"round_robin_{metric_name}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n→ Results saved to {output_file}")

    return all_results


def main():
    """Main execution function."""
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("TMX TRANSLATION EVALUATION")
    print("=" * 80)

    # Parse TMX file
    print(
        f"\n→ Parsing TMX file: {args.tmx_file} (newlines strategy: {args.newlines_strategy})"
    )
    parser = TMXParser(args.tmx_file, args.newlines_strategy)
    parser.parse()

    # Load custom translations if provided
    if args.custom_translations or args.custom_translations_dir:
        print(f"\n→ Loading custom translations...")
        custom_loader = CustomTranslationsLoader()

        try:
            # Load from directory
            if args.custom_translations_dir:
                system_names = custom_loader.load_from_directory(
                    args.custom_translations_dir,
                    source_field=args.source_field,
                    translation_field=args.translation_field,
                )
                print(f"  ✓ Loaded {len(system_names)} systems from directory")

            # Load from individual files
            if args.custom_translations:
                for filepath in args.custom_translations:
                    system_name = custom_loader.load_from_file(
                        filepath,
                        source_field=args.source_field,
                        translation_field=args.translation_field,
                    )
                    print(f"  ✓ Loaded system: {system_name}")

            # Add custom translations to parser
            custom_translations = custom_loader.get_all_translations()
            parser.add_multiple_custom_translations(custom_translations)

            print(
                f"\n✓ Successfully integrated {len(custom_translations)} custom translation system(s)"
            )

        except Exception as e:
            print(f"\n✗ Error loading custom translations: {e}")
            print("Continuing with TMX translations only...")

    parser.print_statistics()

    # Export parsed data
    if not args.no_save:
        parser.export_to_json(output_dir / "parsed_translations.json")
        parser.export_to_csv(output_dir / "parsed_translations.csv")

    # Initialize evaluator
    print(f"\n→ Initializing evaluator (device: {args.device})...")
    evaluator = TranslationEvaluator(cache_dir=args.cache_dir, device=args.device)

    # Determine which metrics to use
    if args.mode in ["reference-free", "all"]:
        if args.metrics:
            ref_free_metrics = {
                k: v for k, v in REFERENCE_FREE_METRICS.items() if k in args.metrics
            }
        else:
            ref_free_metrics = REFERENCE_FREE_METRICS

        if ref_free_metrics:
            ref_free_results = run_reference_free_evaluation(
                parser=parser,
                evaluator=evaluator,
                metrics=ref_free_metrics,
                batch_size=args.batch_size,
                output_dir=output_dir,
            )

    if args.mode in ["reference-based", "round-robin", "all"]:
        if args.metrics:
            ref_based_metrics = {
                k: v for k, v in REFERENCE_BASED_METRICS.items() if k in args.metrics
            }
        else:
            ref_based_metrics = REFERENCE_BASED_METRICS

        if ref_based_metrics and args.mode in ["round-robin", "all"]:
            round_robin_results = run_round_robin_evaluation(
                parser=parser,
                evaluator=evaluator,
                metrics=ref_based_metrics,
                batch_size=args.batch_size,
                output_dir=output_dir,
            )

    # Cleanup
    print("\n→ Cleaning up...")
    evaluator.cleanup()

    # Generate visualizations
    if args.visualize:
        print("\n→ Generating visualizations...")
        visualizer = ResultsVisualizer(results_dir=str(output_dir))

        # Load and visualize results
        for results_file in output_dir.glob("*.json"):
            if results_file.name.startswith(
                "reference_free_"
            ) or results_file.name.startswith("round_robin_"):
                try:
                    results = visualizer.load_results(results_file)
                    metric_name = results_file.stem

                    # Generate plots
                    visualizer.plot_score_distributions(
                        results=results,
                        metric_name=metric_name,
                        save_path=visualizer.figures_dir
                        / f"{metric_name}_distributions.png",
                    )

                    if "round_robin" in results_file.name:
                        visualizer.plot_comparison_matrix(
                            results=results,
                            metric_name=metric_name,
                            save_path=visualizer.figures_dir
                            / f"{metric_name}_matrix.png",
                        )

                    # Generate summary report
                    visualizer.generate_summary_report(
                        results=results,
                        output_path=output_dir / f"{metric_name}_summary.txt",
                    )

                except Exception as e:
                    print(f"  ⚠ Error visualizing {results_file.name}: {e}")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")

    if args.visualize:
        print(f"Figures saved to: {output_dir / 'figures'}")


if __name__ == "__main__":
    main()
