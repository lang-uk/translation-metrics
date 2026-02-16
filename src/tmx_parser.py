"""TMX file parser for extracting translation units."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


class TMXParser:
    """Parser for TMX (Translation Memory eXchange) files."""

    def __init__(self, tmx_file_path: str, newlines_strategy: str = "combine"):
        """
        Initialize the TMX parser.

        Args:
            tmx_file_path: Path to the TMX file
            newlines_strategy: How to handle newlines - "combine" (replace with spaces) or "separate" (split on newlines)
        """
        self.tmx_file_path = Path(tmx_file_path)
        self.newlines_strategy = newlines_strategy
        self.tree = None
        self.root = None
        self.translations = None

    def parse(self) -> Dict[str, List[str]]:
        """
        Parse the TMX file and extract all translations.

        Returns:
            Dictionary with language codes as keys and lists of segments as values
        """
        self.tree = ET.parse(self.tmx_file_path)
        self.root = self.tree.getroot()

        # Extract all translation units
        translation_units = []

        for tu in self.root.findall(".//tu"):
            unit = {}
            for tuv in tu.findall("tuv"):
                lang = tuv.get("{http://www.w3.org/XML/1998/namespace}lang")
                seg = tuv.find("seg")
                if seg is not None and seg.text:
                    unit[lang] = seg.text.strip()

            if unit:  # Only add if there's at least one translation
                translation_units.append(unit)

        # Organize by language
        all_langs = set()
        for unit in translation_units:
            all_langs.update(unit.keys())

        self.translations = {lang: [] for lang in all_langs}

        if self.newlines_strategy == "separate":
            # Handle separate strategy: split multiline entries
            self._parse_with_separate_strategy(translation_units, all_langs)
        else:
            # Handle combine strategy: replace newlines with spaces
            self._parse_with_combine_strategy(translation_units, all_langs)

        return self.translations

    def _parse_with_combine_strategy(self, translation_units, all_langs):
        """Parse with combine strategy: replace newlines with spaces."""
        for unit in translation_units:
            for lang in all_langs:
                # Use empty string or "---" placeholder for missing translations
                text = unit.get(lang, "")
                # Replace newlines with spaces
                text = text.replace("\n", "").replace("\r", "")
                self.translations[lang].append(text)

    def _parse_with_separate_strategy(self, translation_units, all_langs):
        """Parse with separate strategy: split multiline entries into multiple segments."""
        # First pass: collect all split segments for each language
        split_translations = {lang: [] for lang in all_langs}

        for unit in translation_units:
            # Get all texts for this unit, split by newlines
            unit_splits = {}
            max_splits = 1

            for lang in all_langs:
                text = unit.get(lang, "")
                if text:
                    # Split on newlines and filter empty lines
                    lines = [line.strip() for line in text.split("\n") if line.strip()]
                    if not lines:  # If all lines were empty, keep original
                        lines = [text.strip()]
                else:
                    lines = [""]

                unit_splits[lang] = lines
                max_splits = max(max_splits, len(lines))

            # Ensure all languages have the same number of segments by padding with empty strings
            for lang in all_langs:
                lines = unit_splits[lang]
                if len(lines) < max_splits:
                    # Pad with empty strings
                    lines.extend([""] * (max_splits - len(lines)))
                split_translations[lang].extend(lines)

        # Update self.translations with the split results
        self.translations = split_translations

    def get_source_language(self) -> str:
        """
        Identify the source language (assumes it starts with 'EN').

        Returns:
            Source language code
        """
        if self.translations is None:
            self.parse()

        for lang in self.translations.keys():
            if lang.startswith("EN"):
                return lang

        raise ValueError("No source language (starting with 'EN') found in TMX file")

    def get_target_languages(self) -> List[str]:
        """
        Get all target language codes (assumes they start with 'UK' for Ukrainian).

        Returns:
            List of target language codes
        """
        if self.translations is None:
            self.parse()

        target_langs = [
            lang for lang in self.translations.keys() if lang.startswith("UK")
        ]
        return sorted(target_langs)

    def export_to_dataframe(self) -> pd.DataFrame:
        """
        Export translations to a pandas DataFrame.

        Returns:
            DataFrame with all translations
        """
        if self.translations is None:
            self.parse()

        df = pd.DataFrame(self.translations)
        df.index.name = "segment_id"
        return df

    def get_source_segments(self) -> List[str]:
        """
        Get all source language segments.

        Returns:
            List of source segments
        """
        source_lang = self.get_source_language()
        return self.translations[source_lang]

    def get_target_segments(self, lang_code: str) -> List[str]:
        """
        Get all segments for a specific target language.

        Args:
            lang_code: Target language code

        Returns:
            List of target segments
        """
        if lang_code not in self.translations:
            raise ValueError(f"Language code '{lang_code}' not found in translations")
        return self.translations[lang_code]

    def filter_valid_segments(self) -> Dict[str, List[Tuple[int, str]]]:
        """
        Filter out segments that are empty or contain only "---".

        Returns:
            Dictionary with language codes and lists of (index, segment) tuples
        """
        source_lang = self.get_source_language()
        target_langs = self.get_target_languages()

        filtered = {lang: [] for lang in [source_lang] + target_langs}

        for idx, source_seg in enumerate(self.translations[source_lang]):
            # Check if source is valid
            if not source_seg or source_seg.strip() in ["", "---"]:
                continue

            # Check if at least one target is valid
            has_valid_target = False
            for target_lang in target_langs:
                target_seg = self.translations[target_lang][idx]
                if target_seg and target_seg.strip() not in ["", "---"]:
                    has_valid_target = True
                    break

            if has_valid_target:
                filtered[source_lang].append((idx, source_seg))
                for target_lang in target_langs:
                    target_seg = self.translations[target_lang][idx]
                    filtered[target_lang].append((idx, target_seg))

        return filtered

    def export_to_json(self, output_path: str):
        """
        Export translations to a JSON file.

        Args:
            output_path: Path to output JSON file
        """
        if self.translations is None:
            self.parse()

        output_data = {
            "source_language": self.get_source_language(),
            "target_languages": self.get_target_languages(),
            "num_segments": len(next(iter(self.translations.values()))),
            "translations": self.translations,
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"Exported translations to {output_path}")

    def export_to_csv(self, output_path: str):
        """
        Export translations to a CSV file.

        Args:
            output_path: Path to output CSV file
        """
        df = self.export_to_dataframe()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path, index=True, encoding="utf-8")
        print(f"Exported translations to {output_path}")

    def add_custom_translations(
        self, system_name: str, translations: List[str], validate_length: bool = True
    ):
        """
        Add custom translations as a new "language/system".

        Args:
            system_name: Name for the custom translation system
            translations: List of translations
            validate_length: Whether to validate translation count matches existing
        """
        if self.translations is None:
            self.parse()

        # Get expected length from source
        source_lang = self.get_source_language()
        expected_length = len(self.translations[source_lang])

        if validate_length and len(translations) != expected_length:
            raise ValueError(
                f"Translation count mismatch for '{system_name}': "
                f"expected {expected_length}, got {len(translations)}"
            )

        # Check for name conflicts
        if system_name in self.translations:
            print(
                f"⚠ Warning: System name '{system_name}' already exists. "
                f"Renaming to '{system_name}_custom'"
            )
            system_name = f"{system_name}_custom"

        self.translations[system_name] = translations
        print(
            f"✓ Added custom translations: {system_name} ({len(translations)} segments)"
        )

    def add_multiple_custom_translations(
        self, custom_translations: Dict[str, List[str]], validate_length: bool = True
    ):
        """
        Add multiple custom translation systems.

        Args:
            custom_translations: Dictionary of {system_name: [translations]}
            validate_length: Whether to validate translation counts
        """
        for system_name, translations in custom_translations.items():
            self.add_custom_translations(system_name, translations, validate_length)

    def get_all_languages(self) -> List[str]:
        """
        Get all language/system codes (including custom translations).

        Returns:
            List of all language/system codes
        """
        if self.translations is None:
            self.parse()
        return list(self.translations.keys())

    def get_statistics(self) -> Dict:
        """
        Get statistics about the TMX file.

        Returns:
            Dictionary with statistics
        """
        if self.translations is None:
            self.parse()

        source_lang = self.get_source_language()
        target_langs = self.get_target_languages()

        # Get all languages including custom ones
        all_langs = self.get_all_languages()
        custom_langs = [
            lang for lang in all_langs if lang not in [source_lang] + target_langs
        ]

        stats = {
            "source_language": source_lang,
            "target_languages": target_langs,
            "custom_systems": custom_langs,
            "num_segments": len(self.translations[source_lang]),
            "languages_coverage": {},
        }

        # Calculate coverage for each language
        for lang in all_langs:
            segments = self.translations[lang]
            non_empty = sum(
                1 for seg in segments if seg and seg.strip() not in ["", "---"]
            )
            coverage = (non_empty / len(segments)) * 100 if segments else 0
            stats["languages_coverage"][lang] = {
                "total_segments": len(segments),
                "non_empty_segments": non_empty,
                "coverage_percentage": round(coverage, 2),
            }

        return stats

    def print_statistics(self):
        """Print statistics about the TMX file."""
        stats = self.get_statistics()

        print(f"\n{'=' * 60}")
        print(f"TMX File Statistics: {self.tmx_file_path.name}")
        print(f"{'=' * 60}")
        print(f"\nSource Language: {stats['source_language']}")
        print(f"Target Languages: {', '.join(stats['target_languages'])}")
        if stats["custom_systems"]:
            print(f"Custom Systems: {', '.join(stats['custom_systems'])}")
        print(f"Total Segments: {stats['num_segments']}")
        print(f"\nLanguage Coverage:")
        print(f"{'-' * 60}")

        # Print in order: source, targets, customs
        for lang in (
            [stats["source_language"]]
            + stats["target_languages"]
            + stats["custom_systems"]
        ):
            if lang in stats["languages_coverage"]:
                coverage = stats["languages_coverage"][lang]
                marker = "[CUSTOM]" if lang in stats["custom_systems"] else ""
                print(
                    f"{lang:40} {marker:10} {coverage['non_empty_segments']:5}/{coverage['total_segments']:5} "
                    f"({coverage['coverage_percentage']:6.2f}%)"
                )

        print(f"{'=' * 60}\n")


if __name__ == "__main__":
    # Example usage with argparse
    import argparse

    parser_cli = argparse.ArgumentParser(
        description="Parse TMX files and extract translation data"
    )

    parser_cli.add_argument(
        "tmx_file",
        nargs="?",
        default="17099743/Animal-Farm_preface.tmx",
        help="Path to TMX file (default: 17099743/Animal-Farm_preface.tmx)",
    )

    parser_cli.add_argument(
        "--output-json",
        type=str,
        default="results/translations.json",
        help="Output path for JSON export (default: results/translations.json)",
    )

    parser_cli.add_argument(
        "--output-csv",
        type=str,
        default="results/translations.csv",
        help="Output path for CSV export (default: results/translations.csv)",
    )

    parser_cli.add_argument(
        "--no-export", action="store_true", help="Skip exporting to files"
    )

    parser_cli.add_argument(
        "--stats-only",
        action="store_true",
        help="Only print statistics without exporting",
    )

    args = parser_cli.parse_args()

    parser = TMXParser(args.tmx_file)
    parser.parse()
    parser.print_statistics()

    # Export to different formats unless disabled
    if not args.no_export and not args.stats_only:
        parser.export_to_json(args.output_json)
        parser.export_to_csv(args.output_csv)
