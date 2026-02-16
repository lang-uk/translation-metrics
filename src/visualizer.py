"""Visualization and analysis of translation evaluation results."""

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

ORIGINAL_TEXT_PATH = "17099743/Animal_farm_combine.txt"


class ResultsVisualizer:
    """Visualizes translation evaluation results."""

    def __init__(self, results_dir: str = "results"):
        """
        Initialize visualizer.

        Args:
            results_dir: Directory containing results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir = self.results_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def load_results(self, results_file: str, sectioned: bool = False) -> Dict:
        """
        Load results from JSON file.

        Args:
            results_file: Path to results JSON file

        Returns:
            Dictionary with results
        """
        with open(results_file, "r", encoding="utf-8") as f:
            results = json.load(f)
            if sectioned:
                original_text = self.load_original_text(ORIGINAL_TEXT_PATH)
                return self.parse_results_with_indices(results, original_text)
            return results

    def load_original_text(self, text_path: str = ORIGINAL_TEXT_PATH) -> List[str]:
        """
        Load original text from txt file.

        Args:
            text_path: Path to original text txt file

        Returns:
            List of original text lines
        """
        with open(text_path, "r", encoding="utf-8") as f:
            return f.readlines()

    def plot_score_distributions(
        self,
        results: Dict,
        metric_name: str,
        save_path: Optional[str] = None,
        sectioned: bool = False,
    ):
        """
        Plot score distributions for different translation systems.

        Args:
            results: Dictionary with evaluation results
            metric_name: Name of the metric
            save_path: Path to save figure
            sectioned: Whether to plot sectioned results
        """
        # Extract data
        system_names = []
        system_scores = []

        for system, data in results.items():
            if "error" not in data and "scores" in data:
                system_names.append(system)
                system_scores.append(data["scores"])

        print(len(system_names))
        print(len(system_scores))
        print(len(system_scores[0]))

        if not system_names:
            print("No valid data to plot")
            return

        import os

        # Determine output base name (if provided)
        base_path = None
        if save_path:
            base_path, _ = os.path.splitext(save_path)
        # 1. Box Plot
        fig_box, ax_box = plt.subplots(figsize=(10, 6))
        ax_box.boxplot(system_scores, labels=system_names)
        ax_box.set_title(f"Box Plot of Scores - {metric_name}", fontweight="bold")
        ax_box.set_ylabel("Score")
        ax_box.tick_params(axis="x", rotation=45)
        ax_box.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            box_path = (
                f"{base_path}_boxplot_{'cleaned' if sectioned else 'original'}.png"
            )
            plt.savefig(box_path, dpi=300, bbox_inches="tight")
            print(f"Saved figure to {box_path}")

        VIOLIN_PLOT_ORDER = [
            "UK_1947_Ivan_Cherniatynskyi_Kolhosp_tvaryn",
            "UK_1984_Iryna_Dybko_Khutir_tvaryn",
            "UK_1991_Oleksii_Drozdovskyi_Skotoferma",
            "UK_1991_Yurii_Shevchuk_Ferma_rai_dlia_tvaryn",
            "UK_1992_Natalia_Okolitenko_Skotokhutir",
            "UK_2020_Bohdana_Nosenok_Kolhosp_tvaryn",
            "UK_2021_Viacheslav_Stelmakh_Kolhosp_tvaryn",
            "lapa_translations_combine",
        ]

        # 2. Violin Plot
        fig_violin, ax_violin = plt.subplots(figsize=(10, 6))
        positions = range(1, len(VIOLIN_PLOT_ORDER) + 1)
        ax_violin.violinplot(
            [system_scores[system_names.index(system)] for system in VIOLIN_PLOT_ORDER],
            positions=positions,
            showmeans=True,
        )
        ax_violin.set_xticks(positions)
        ax_violin.set_xticklabels(VIOLIN_PLOT_ORDER, rotation=45, ha="right")
        ax_violin.set_title(f"Violin Plot of Scores - {metric_name}", fontweight="bold")
        ax_violin.set_ylabel("Score")
        ax_violin.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            violin_path = (
                f"{base_path}_violin_{'cleaned' if sectioned else 'original'}.png"
            )
            plt.savefig(violin_path, dpi=300, bbox_inches="tight")
            print(f"Saved figure to {violin_path}")

        # 3. Histogram
        fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
        for scores, name in zip(system_scores, system_names):
            ax_hist.hist(scores, alpha=0.5, label=name, bins=20)
        ax_hist.set_title(f"Score Histograms - {metric_name}", fontweight="bold")
        ax_hist.set_xlabel("Score")
        ax_hist.set_ylabel("Frequency")
        ax_hist.legend()
        ax_hist.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            hist_path = f"{base_path}_hist_{'cleaned' if sectioned else 'original'}.png"
            plt.savefig(hist_path, dpi=300, bbox_inches="tight")
            print(f"Saved figure to {hist_path}")

    def plot_comparison_matrix(
        self, results: Dict, metric_name: str, save_path: Optional[str] = None
    ):
        """
        Plot comparison matrix (heatmap) for round-robin evaluation.

        Args:
            results: Dictionary with round-robin results
            metric_name: Name of the metric
            save_path: Path to save figure
        """
        # Extract system names
        systems = set()
        for key in results.keys():
            if "_vs_" in key:
                ref, hyp = key.split("_vs_")
                systems.add(ref)
                systems.add(hyp)

        systems = sorted(list(systems))

        # Create matrix
        matrix = np.zeros((len(systems), len(systems)))

        for i, ref_sys in enumerate(systems):
            for j, hyp_sys in enumerate(systems):
                if i == j:
                    matrix[i, j] = np.nan
                else:
                    key = f"{ref_sys}_vs_{hyp_sys}"
                    if key in results and "error" not in results[key]:
                        matrix[i, j] = results[key]["mean"]

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(12, 10))

        # Display 1's on the diagonal for self-comparison
        # matrix_with_diag = matrix.copy()
        # np.fill_diagonal(matrix_with_diag, 1.0)
        matrix_with_diag = matrix

        # No mask, show diagonal values
        sns.heatmap(
            matrix_with_diag,
            annot=True,
            fmt=".4f",
            # cmap="RdYlGn",
            xticklabels=systems,
            yticklabels=systems,
            cbar_kws={"label": "Score"},
            ax=ax,
            vmin=np.nanmin(matrix_with_diag),
            vmax=np.nanmax(matrix_with_diag),
        )

        ax.set_title(
            f"Round-Robin Comparison Matrix - {metric_name}\n",
            # f"(Reference = Y-axis, Hypothesis = X-axis)",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        ax.set_xlabel(
            "Hypothesis (System being evaluated)", fontsize=12, fontweight="bold"
        )
        ax.set_ylabel("Reference (Ground truth)", fontsize=12, fontweight="bold")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved figure to {save_path}")

    def plot_system_rankings(
        self, results_dict: Dict[str, Dict], save_path: Optional[str] = None
    ):
        """
        Plot system rankings across multiple metrics.

        Args:
            results_dict: Dictionary of {metric_name: {system: results}}
            save_path: Path to save figure
        """
        # Prepare data
        metrics = []
        systems = set()

        for metric_name, results in results_dict.items():
            for system in results.keys():
                if "error" not in results[system]:
                    systems.add(system)
            metrics.append(metric_name)

        systems = sorted(list(systems))

        # Create DataFrame
        data = []
        for metric_name, results in results_dict.items():
            for system in systems:
                if system in results and "error" not in results[system]:
                    data.append(
                        {
                            "Metric": metric_name,
                            "System": system,
                            "Score": results[system]["mean"],
                        }
                    )

        df = pd.DataFrame(data)

        if df.empty:
            print("No valid data to plot")
            return

        # Plot
        fig, ax = plt.subplots(figsize=(14, 8))

        # Pivot for grouped bar chart
        pivot_df = df.pivot(index="System", columns="Metric", values="Score")
        pivot_df.plot(kind="bar", ax=ax, width=0.8)

        ax.set_title(
            "System Scores Across Different Metrics",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        ax.set_xlabel("Translation System", fontsize=12, fontweight="bold")
        ax.set_ylabel("Score", fontsize=12, fontweight="bold")
        ax.legend(title="Metric", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3, axis="y")

        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved figure to {save_path}")

    def generate_summary_report(
        self, results: Dict, output_path: str, sectioned: bool = False
    ):
        """
        Generate a text summary report.

        Args:
            results: Dictionary with evaluation results
            output_path: Path to save report
            sectioned: Whether to generate summary report for sectioned results
        """
        output_path = str(output_path).replace(
            "summary.txt", f"summary_{'cleaned' if sectioned else 'original'}.txt"
        )
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("TRANSLATION EVALUATION SUMMARY REPORT\n")
            f.write("=" * 80 + "\n\n")

            for system, data in results.items():
                f.write(f"\n{'-' * 80}\n")
                f.write(f"System: {system}\n")
                f.write(f"{'-' * 80}\n")

                if "error" in data:
                    f.write(f"ERROR: {data['error']}\n")
                else:
                    f.write(f"Number of segments: {data['num_segments']}\n")
                    f.write(f"Mean score:         {data['mean']:.6f}\n")
                    f.write(f"Std deviation:      {data['std']:.6f}\n")
                    f.write(f"Median score:       {data['median']:.6f}\n")
                    f.write(f"Min score:          {data['min']:.6f}\n")
                    f.write(f"Max score:          {data['max']:.6f}\n")

            f.write(f"\n{'=' * 80}\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")

        print(f"Summary report saved to {output_path}")

    def create_detailed_analysis(
        self,
        sources: List[str],
        translations: Dict[str, List[str]],
        results: Dict[str, Dict],
        output_path: str,
        top_n: int = 10,
        bottom_n: int = 10,
    ):
        """
        Create detailed analysis with best and worst examples.

        Args:
            sources: List of source sentences
            translations: Dictionary of translations
            results: Dictionary with evaluation results
            output_path: Path to save analysis
            top_n: Number of top examples to show
            bottom_n: Number of bottom examples to show
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("=" * 100 + "\n")
            f.write("DETAILED TRANSLATION ANALYSIS\n")
            f.write("=" * 100 + "\n\n")

            for system, data in results.items():
                if "error" in data or "scores" not in data:
                    continue

                f.write(f"\n{'=' * 100}\n")
                f.write(f"System: {system}\n")
                f.write(f"{'=' * 100}\n\n")

                scores = np.array(data["scores"])
                indices = np.argsort(scores)

                # Best examples
                f.write(f"\nTOP {top_n} BEST TRANSLATIONS:\n")
                f.write("-" * 100 + "\n")

                for rank, idx in enumerate(indices[-top_n:][::-1], 1):
                    if idx < len(sources):
                        f.write(f"\nRank {rank} - Score: {scores[idx]:.4f}\n")
                        f.write(f"Source: {sources[idx]}\n")

                        # Extract system name from results key
                        if "_vs_" in system:
                            _, hyp_sys = system.split("_vs_")
                            if hyp_sys in translations:
                                f.write(f"Translation: {translations[hyp_sys][idx]}\n")
                        elif system in translations:
                            f.write(f"Translation: {translations[system][idx]}\n")

                        f.write("-" * 100 + "\n")

                # Worst examples
                f.write(f"\n\nBOTTOM {bottom_n} WORST TRANSLATIONS:\n")
                f.write("-" * 100 + "\n")

                for rank, idx in enumerate(indices[:bottom_n], 1):
                    if idx < len(sources):
                        f.write(f"\nRank {rank} - Score: {scores[idx]:.4f}\n")
                        f.write(f"Source: {sources[idx]}\n")

                        # Extract system name from results key
                        if "_vs_" in system:
                            _, hyp_sys = system.split("_vs_")
                            if hyp_sys in translations:
                                f.write(f"Translation: {translations[hyp_sys][idx]}\n")
                        elif system in translations:
                            f.write(f"Translation: {translations[system][idx]}\n")

                        f.write("-" * 100 + "\n")

        print(f"Detailed analysis saved to {output_path}")

    def parse_results_with_indices(
        self, results: Dict, original_text: List[str]
    ) -> Dict:
        """
        Parse results with indices - filter out scores corresponding to invalid lines.
        """
        invalid_indices = {
            i
            for i, line in enumerate(original_text)
            if line.strip() == "---" or not line.strip()
        }

        print(f"Filtering out {len(invalid_indices)} invalid indices")

        results_with_filtered_indices = {}

        for doc_id, doc_data in results.items():
            filtered_doc = {}

            # Only filter the scores list, keep other computed statistics as-is for now
            if "scores" in doc_data and isinstance(doc_data["scores"], list):
                original_scores = doc_data["scores"]
                filtered_scores = [
                    score
                    for i, score in enumerate(original_scores)
                    if i not in invalid_indices
                ]

                # Recalculate statistics based on filtered scores
                if filtered_scores:
                    filtered_doc["scores"] = filtered_scores
                    filtered_doc["num_segments"] = len(filtered_scores)
                    filtered_doc["mean"] = sum(filtered_scores) / len(filtered_scores)
                    filtered_doc["median"] = sorted(filtered_scores)[
                        len(filtered_scores) // 2
                    ]

                    # Calculate standard deviation
                    mean_val = filtered_doc["mean"]
                    variance = sum((x - mean_val) ** 2 for x in filtered_scores) / len(
                        filtered_scores
                    )
                    filtered_doc["std"] = variance**0.5

                    filtered_doc["min"] = min(filtered_scores)
                    filtered_doc["max"] = max(filtered_scores)
                else:
                    filtered_doc["error"] = "No valid scores after filtering"
            else:
                # Keep non-scores data as-is
                filtered_doc = doc_data.copy()

            results_with_filtered_indices[doc_id] = filtered_doc

        return results_with_filtered_indices


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Visualize translation evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize results from a JSON file
  python visualizer.py results/reference_free_cometkiwi-xl.json --plot distributions
  
  # Generate all visualizations
  python visualizer.py results/round_robin_comet-22.json --plot all
  
  # Generate summary report only
  python visualizer.py results/reference_free_cometkiwi-xl.json --report-only
  
  # Specify output directory
  python visualizer.py results/reference_free_cometkiwi-xl.json --output-dir my_viz
        """,
    )

    parser.add_argument("results_file", nargs="?", help="Path to results JSON file")

    parser.add_argument(
        "--plot",
        choices=["distributions", "matrix", "rankings", "all"],
        default="all",
        help="Type of plot to generate (default: all)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for visualizations (default: results)",
    )

    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Generate text report only, skip visualizations",
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display plots (only save to files)",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List available result files in results directory",
    )

    parser.add_argument(
        "--original-text",
        type=str,
        default="17099743/Animal_farm_combine.txt",
        help="Path to original text txt file",
    )

    parser.add_argument(
        "--sectioned",
        action="store_true",
        help="Parse results with sectioned indices",
    )

    args = parser.parse_args()

    # List available files if requested
    if args.list:
        results_dir = Path("results")
        if results_dir.exists():
            json_files = list(results_dir.glob("*.json"))
            if json_files:
                print("\nAvailable result files:")
                for f in json_files:
                    print(f"  - {f}")
                print("\nUsage: python visualizer.py <file> [options]")
            else:
                print("\n⚠ No JSON result files found in results/")
        else:
            print("\n⚠ Results directory does not exist")
        sys.exit(0)

    # Check if results file is provided
    if not args.results_file:
        print("Error: results_file is required")
        print("\nUsage: python visualizer.py <results_file> [options]")
        print("   or: python visualizer.py --list (to see available files)")
        parser.print_help()
        sys.exit(1)

    # Check if file exists
    results_path = Path(args.results_file)
    if not results_path.exists():
        print(f"✗ Error: File not found: {args.results_file}")
        sys.exit(1)

    # Initialize visualizer
    print(f"\n→ Loading results from {args.results_file}...")
    visualizer = ResultsVisualizer(results_dir=args.output_dir)

    try:
        results = visualizer.load_results(args.results_file, sectioned=args.sectioned)
        metric_name = results_path.stem

        print(f"✓ Loaded results for {metric_name}")
        print(f"  Found {len(results)} system(s)")

        # Generate summary report
        report_path = Path(args.output_dir) / f"{metric_name}_summary.txt"
        print(f"\n→ Generating summary report...")
        visualizer.generate_summary_report(
            results=results, output_path=report_path, sectioned=args.sectioned
        )
        print(f"✓ Summary saved to {report_path}")

        # Generate visualizations unless report-only
        if not args.report_only:
            figures_dir = visualizer.figures_dir

            # Check if this is round-robin results
            is_round_robin = "_vs_" in list(results.keys())[0] if results else False

            if args.plot in ["distributions", "all"]:
                print(f"\n→ Generating distribution plots...")
                plot_path = figures_dir / f"{metric_name}_distributions.png"
                visualizer.plot_score_distributions(
                    results=results,
                    metric_name=metric_name,
                    save_path=plot_path,
                    sectioned=args.sectioned,
                )

            if args.plot in ["matrix", "all"] and is_round_robin:
                print(f"\n→ Generating comparison matrix...")
                plot_path = figures_dir / f"{metric_name}_matrix.png"
                visualizer.plot_comparison_matrix(
                    results=results, metric_name=metric_name, save_path=plot_path
                )
        print("\n✓ Done!")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
