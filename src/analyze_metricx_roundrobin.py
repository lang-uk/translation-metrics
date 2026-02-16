"""Analyze MetricX round-robin evaluation results.

This script reads MetricX prediction scores from round-robin evaluation
(where each system is evaluated against all others as references) and
generates comparison matrices and statistics.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def load_roundrobin_results(
    results_dir: str, version: int = 24
) -> Dict[str, Dict[str, List[float]]]:
    """
    Load round-robin prediction results.

    Args:
        results_dir: Directory with prediction results
        version: MetricX version (23 or 24)

    Returns:
        Nested dictionary: {hypothesis_system: {reference_system: [scores]}}
    """
    results_dir = Path(results_dir)
    pattern = f"*_vs_*_metricx{version}_ref_scores.jsonl"

    results = {}

    for file in results_dir.glob(pattern):
        # Parse filename: hypothesis_vs_reference_metricx24_ref_scores.jsonl
        parts = file.stem.replace(f"_metricx{version}_ref_scores", "").split("_vs_")
        if len(parts) != 2:
            print(f"⚠ Skipping file with unexpected format: {file.name}")
            continue

        hyp_system, ref_system = parts[0], parts[1]

        # Read scores
        scores = []
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                if "prediction" in entry:
                    scores.append(entry["prediction"])

        if not scores:
            print(f"⚠ No scores found in {file.name}")
            continue

        # Store in nested dictionary
        if hyp_system not in results:
            results[hyp_system] = {}
        results[hyp_system][ref_system] = scores

        print(f"✓ Loaded {len(scores)} scores: {hyp_system} vs {ref_system}")

    return results


def create_comparison_matrix(
    results: Dict[str, Dict[str, List[float]]],
) -> Tuple[List[str], np.ndarray]:
    """
    Create a comparison matrix from round-robin results.

    Args:
        results: Nested dictionary of results

    Returns:
        Tuple of (system_names, matrix) where matrix[i, j] is the mean score
        when system i is evaluated with system j as reference
    """
    # Get all unique systems
    systems = sorted(
        set(results.keys())
        | set(ref for hyp_results in results.values() for ref in hyp_results.keys())
    )

    n = len(systems)
    matrix = np.full((n, n), np.nan)

    for i, hyp_sys in enumerate(systems):
        for j, ref_sys in enumerate(systems):
            if hyp_sys == ref_sys:
                # Diagonal - no self-comparison
                continue

            if hyp_sys in results and ref_sys in results[hyp_sys]:
                scores = results[hyp_sys][ref_sys]
                matrix[i, j] = np.mean(scores)

    return systems, matrix


def print_comparison_matrix(systems: List[str], matrix: np.ndarray):
    """Print the comparison matrix in a readable format."""
    n = len(systems)

    # Shorten system names for display
    short_names = []
    for sys in systems:
        # Try to extract meaningful parts
        parts = sys.split("_")
        if len(parts) >= 3:
            # e.g., UK_2020_Name -> 2020_Name
            short_names.append("_".join(parts[1:3]))
        else:
            short_names.append(sys[:15])

    # Print header
    print("\nComparison Matrix (Hypothesis vs Reference)")
    print("=" * 80)
    print(f"{'Hypothesis':<20} | ", end="")
    for name in short_names:
        print(f"{name:>12} ", end="")
    print()
    print("-" * 80)

    # Print rows
    for i, hyp_name in enumerate(short_names):
        print(f"{hyp_name:<20} | ", end="")
        for j in range(n):
            if i == j:
                print(f"{'---':>12} ", end="")
            elif np.isnan(matrix[i, j]):
                print(f"{'N/A':>12} ", end="")
            else:
                print(f"{matrix[i, j]:>12.4f} ", end="")
        print()

    print("=" * 80)
    print("\nNote: Lower scores are better (error score)")
    print("      Rows = Hypothesis, Columns = Reference")


def calculate_statistics(results: Dict[str, Dict[str, List[float]]]) -> Dict:
    """Calculate statistics for each system."""
    stats = {}

    for hyp_sys, ref_results in results.items():
        all_scores = []
        for scores in ref_results.values():
            all_scores.extend(scores)

        if all_scores:
            stats[hyp_sys] = {
                "mean": np.mean(all_scores),
                "std": np.std(all_scores),
                "min": np.min(all_scores),
                "max": np.max(all_scores),
                "median": np.median(all_scores),
                "num_comparisons": len(ref_results),
                "num_scores": len(all_scores),
            }

    return stats


def print_statistics(stats: Dict):
    """Print statistics for each system."""
    print("\nSystem Statistics (Across All References)")
    print("=" * 80)
    print(
        f"{'System':<30} {'Mean':>10} {'Std':>10} {'Median':>10} {'Min':>10} {'Max':>10}"
    )
    print("-" * 80)

    # Sort by mean score (lower is better)
    sorted_systems = sorted(stats.items(), key=lambda x: x[1]["mean"])

    for system, stat in sorted_systems:
        # Shorten name
        short_name = system[:28] if len(system) <= 30 else system[:27] + "..."
        print(
            f"{short_name:<30} {stat['mean']:>10.4f} {stat['std']:>10.4f} "
            f"{stat['median']:>10.4f} {stat['min']:>10.4f} {stat['max']:>10.4f}"
        )

    print("=" * 80)
    print(f"\nBest system (lowest mean score): {sorted_systems[0][0]}")
    print(f"Worst system (highest mean score): {sorted_systems[-1][0]}")


def save_results(systems: List[str], matrix: np.ndarray, stats: Dict, output_file: str):
    """Save results to JSON file."""
    output_data = {
        "systems": systems,
        "comparison_matrix": matrix.tolist(),
        "statistics": stats,
    }

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Results saved to {output_path}")


def export_to_csv(systems: List[str], matrix: np.ndarray, output_file: str):
    """Export comparison matrix to CSV."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        # Header
        f.write("Hypothesis," + ",".join(systems) + "\n")

        # Rows
        for i, hyp_sys in enumerate(systems):
            row = [hyp_sys]
            for j in range(len(systems)):
                if i == j:
                    row.append("")
                elif np.isnan(matrix[i, j]):
                    row.append("N/A")
                else:
                    row.append(f"{matrix[i, j]:.4f}")
            f.write(",".join(row) + "\n")

    print(f"✓ Matrix exported to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze MetricX round-robin evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze round-robin results
  python analyze_metricx_roundrobin.py --input-dir metricx_results

  # Save results to JSON and CSV
  python analyze_metricx_roundrobin.py --input-dir metricx_results \\
      --output results/roundrobin_analysis.json \\
      --export-csv results/roundrobin_matrix.csv

  # Analyze MetricX-23 results
  python analyze_metricx_roundrobin.py --input-dir metricx_results --version 23
        """,
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default="metricx_results",
        help="Directory with MetricX prediction results (default: metricx_results)",
    )

    parser.add_argument(
        "--version",
        type=int,
        choices=[23, 24],
        default=24,
        help="MetricX version (default: 24)",
    )

    parser.add_argument("--output", type=str, help="Output JSON file for results")

    parser.add_argument(
        "--export-csv", type=str, help="Export comparison matrix to CSV"
    )

    parser.add_argument(
        "--no-print", action="store_true", help="Do not print results to console"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("MetricX Round-Robin Analysis")
    print("=" * 80)
    print(f"\nInput directory: {args.input_dir}")
    print(f"MetricX version: {args.version}")

    # Load results
    print(f"\n→ Loading round-robin results...")
    results = load_roundrobin_results(args.input_dir, args.version)

    if not results:
        print("\n✗ No round-robin results found!")
        print(
            f"  Expected files matching: *_vs_*_metricx{args.version}_ref_scores.jsonl"
        )
        return

    print(f"\n✓ Loaded results for {len(results)} hypothesis systems")

    # Create comparison matrix
    print(f"\n→ Creating comparison matrix...")
    systems, matrix = create_comparison_matrix(results)
    print(f"✓ Matrix size: {len(systems)} × {len(systems)}")

    # Calculate statistics
    print(f"\n→ Calculating statistics...")
    stats = calculate_statistics(results)

    # Print results
    if not args.no_print:
        print_comparison_matrix(systems, matrix)
        print_statistics(stats)

    # Save results
    if args.output:
        save_results(systems, matrix, stats, args.output)

    if args.export_csv:
        export_to_csv(systems, matrix, args.export_csv)

    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
