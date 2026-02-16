"""Load custom translations from various file formats for evaluation."""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd


class CustomTranslationsLoader:
    """Load and manage custom translations from various file formats."""

    def __init__(self):
        """Initialize the custom translations loader."""
        self.custom_systems = {}  # {system_name: [translations]}
        self.source_segments = None

    def load_from_json(
        self,
        filepath: Union[str, Path],
        system_name: Optional[str] = None,
        source_field: str = "source",
        translation_field: str = "translation",
    ) -> str:
        """
        Load translations from a JSON file.

        Args:
            filepath: Path to JSON file
            system_name: Name for this translation system (default: filename)
            source_field: Field name for source text (default: "source")
            translation_field: Field name for translation (default: "translation")

        Returns:
            System name used
        """
        filepath = Path(filepath)
        if system_name is None:
            system_name = filepath.stem

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        sources = []
        translations = []

        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    # Try different field name variations
                    src = (
                        item.get(source_field)
                        or item.get("source")
                        or item.get("text")
                        or item.get("english")
                        or item.get("src")
                        or item.get("en")
                    )
                    trans = (
                        item.get(translation_field)
                        or item.get("translation")
                        or item.get("target")
                        or item.get("ukrainian")
                        or item.get("mt")
                        or item.get("uk")
                    )

                    if src is not None and trans is not None:
                        sources.append(str(src))
                        translations.append(str(trans))
        elif isinstance(data, dict):
            # Check if it's a dict with keys as sources
            if "translations" in data:
                # Format: {"translations": [{source: ..., translation: ...}]}
                return self.load_from_json(
                    json.dumps(data["translations"]),
                    system_name,
                    source_field,
                    translation_field,
                )
            else:
                # Format: {source: translation}
                for src, trans in data.items():
                    sources.append(str(src))
                    translations.append(str(trans))

        if not translations:
            raise ValueError(f"No translations found in {filepath}")

        # Store sources if this is the first system
        if self.source_segments is None:
            self.source_segments = sources
        elif len(sources) != len(self.source_segments):
            raise ValueError(
                f"Translation count mismatch for {system_name}: "
                f"expected {len(self.source_segments)}, got {len(sources)}"
            )

        self.custom_systems[system_name] = translations
        print(
            f"✓ Loaded {len(translations)} translations from {filepath} as '{system_name}'"
        )

        return system_name

    def load_from_jsonl(
        self,
        filepath: Union[str, Path],
        system_name: Optional[str] = None,
        source_field: str = "source",
        translation_field: str = "translation",
    ) -> str:
        """
        Load translations from a JSONL file.

        Args:
            filepath: Path to JSONL file
            system_name: Name for this translation system (default: filename)
            source_field: Field name for source text
            translation_field: Field name for translation

        Returns:
            System name used
        """
        filepath = Path(filepath)
        if system_name is None:
            system_name = filepath.stem

        sources = []
        translations = []

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    # Try different field name variations
                    src = (
                        item.get(source_field)
                        or item.get("source")
                        or item.get("text")
                        or item.get("english")
                        or item.get("src")
                        or item.get("en")
                    )
                    trans = (
                        item.get(translation_field)
                        or item.get("translation")
                        or item.get("target")
                        or item.get("ukrainian")
                        or item.get("mt")
                        or item.get("uk")
                    )

                    if src is not None and trans is not None:
                        sources.append(str(src))
                        translations.append(str(trans))

        if not translations:
            raise ValueError(f"No translations found in {filepath}")

        # Store sources if this is the first system
        if self.source_segments is None:
            self.source_segments = sources
        elif len(sources) != len(self.source_segments):
            raise ValueError(
                f"Translation count mismatch for {system_name}: "
                f"expected {len(self.source_segments)}, got {len(sources)}"
            )

        self.custom_systems[system_name] = translations
        print(
            f"✓ Loaded {len(translations)} translations from {filepath} as '{system_name}'"
        )

        return system_name

    def load_from_csv(
        self,
        filepath: Union[str, Path],
        system_name: Optional[str] = None,
        source_field: str = "source",
        translation_field: str = "translation",
    ) -> str:
        """
        Load translations from a CSV file.

        Args:
            filepath: Path to CSV file
            system_name: Name for this translation system (default: filename)
            source_field: Field name for source text
            translation_field: Field name for translation

        Returns:
            System name used
        """
        filepath = Path(filepath)
        if system_name is None:
            system_name = filepath.stem

        df = pd.read_csv(filepath)

        # Try to find source and translation columns
        src_col = None
        trans_col = None

        for col in df.columns:
            col_lower = col.lower()
            if col == source_field or col_lower in [
                "source",
                "text",
                "english",
                "src",
                "en",
            ]:
                src_col = col
            if col == translation_field or col_lower in [
                "translation",
                "target",
                "ukrainian",
                "mt",
                "uk",
            ]:
                trans_col = col

        if src_col is None or trans_col is None:
            raise ValueError(
                f"Could not find source ('{source_field}') or translation ('{translation_field}') "
                f"columns in {filepath}. Available columns: {list(df.columns)}"
            )

        sources = df[src_col].astype(str).tolist()
        translations = df[trans_col].astype(str).tolist()

        if not translations:
            raise ValueError(f"No translations found in {filepath}")

        # Store sources if this is the first system
        if self.source_segments is None:
            self.source_segments = sources
        elif len(sources) != len(self.source_segments):
            raise ValueError(
                f"Translation count mismatch for {system_name}: "
                f"expected {len(self.source_segments)}, got {len(sources)}"
            )

        self.custom_systems[system_name] = translations
        print(
            f"✓ Loaded {len(translations)} translations from {filepath} as '{system_name}'"
        )

        return system_name

    def load_from_parquet(
        self,
        filepath: Union[str, Path],
        system_name: Optional[str] = None,
        source_field: str = "source",
        translation_field: str = "translation",
    ) -> str:
        """
        Load translations from a Parquet file.

        Args:
            filepath: Path to Parquet file
            system_name: Name for this translation system (default: filename)
            source_field: Field name for source text
            translation_field: Field name for translation

        Returns:
            System name used
        """
        filepath = Path(filepath)
        if system_name is None:
            system_name = filepath.stem

        df = pd.read_parquet(filepath)

        # Try to find source and translation columns
        src_col = None
        trans_col = None

        for col in df.columns:
            col_lower = col.lower()
            if col == source_field or col_lower in [
                "source",
                "text",
                "english",
                "src",
                "en",
            ]:
                src_col = col
            if col == translation_field or col_lower in [
                "translation",
                "target",
                "ukrainian",
                "mt",
                "uk",
            ]:
                trans_col = col

        if src_col is None or trans_col is None:
            raise ValueError(
                f"Could not find source ('{source_field}') or translation ('{translation_field}') "
                f"columns in {filepath}. Available columns: {list(df.columns)}"
            )

        sources = df[src_col].astype(str).tolist()
        translations = df[trans_col].astype(str).tolist()

        if not translations:
            raise ValueError(f"No translations found in {filepath}")

        # Store sources if this is the first system
        if self.source_segments is None:
            self.source_segments = sources
        elif len(sources) != len(self.source_segments):
            raise ValueError(
                f"Translation count mismatch for {system_name}: "
                f"expected {len(self.source_segments)}, got {len(sources)}"
            )

        self.custom_systems[system_name] = translations
        print(
            f"✓ Loaded {len(translations)} translations from {filepath} as '{system_name}'"
        )

        return system_name

    def load_from_txt(
        self,
        filepath: Union[str, Path],
        system_name: Optional[str] = None,
        source_file: Optional[Union[str, Path]] = None,
    ) -> str:
        """
        Load translations from a TXT file (one translation per line).

        Args:
            filepath: Path to TXT file with translations
            system_name: Name for this translation system (default: filename)
            source_file: Optional separate file with source sentences

        Returns:
            System name used
        """
        filepath = Path(filepath)
        if system_name is None:
            system_name = filepath.stem

        with open(filepath, "r", encoding="utf-8") as f:
            translations = [line.strip() for line in f if line.strip()]

        if not translations:
            raise ValueError(f"No translations found in {filepath}")

        # Load sources if provided
        if source_file:
            source_file = Path(source_file)
            with open(source_file, "r", encoding="utf-8") as f:
                sources = [line.strip() for line in f if line.strip()]

            if len(sources) != len(translations):
                raise ValueError(
                    f"Source and translation count mismatch: "
                    f"{len(sources)} sources vs {len(translations)} translations"
                )

            if self.source_segments is None:
                self.source_segments = sources

        # Check length if sources already exist
        if self.source_segments and len(translations) != len(self.source_segments):
            raise ValueError(
                f"Translation count mismatch for {system_name}: "
                f"expected {len(self.source_segments)}, got {len(translations)}"
            )

        self.custom_systems[system_name] = translations
        print(
            f"✓ Loaded {len(translations)} translations from {filepath} as '{system_name}'"
        )

        return system_name

    def load_from_file(
        self,
        filepath: Union[str, Path],
        system_name: Optional[str] = None,
        source_field: str = "source",
        translation_field: str = "translation",
        source_file: Optional[Union[str, Path]] = None,
    ) -> str:
        """
        Auto-detect format and load translations from file.

        Args:
            filepath: Path to file
            system_name: Name for this translation system
            source_field: Field name for source text
            translation_field: Field name for translation
            source_file: Optional separate source file (for TXT format)

        Returns:
            System name used
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        suffix = filepath.suffix.lower()

        if suffix == ".json":
            return self.load_from_json(
                filepath, system_name, source_field, translation_field
            )
        elif suffix == ".jsonl":
            return self.load_from_jsonl(
                filepath, system_name, source_field, translation_field
            )
        elif suffix == ".csv":
            return self.load_from_csv(
                filepath, system_name, source_field, translation_field
            )
        elif suffix == ".parquet":
            return self.load_from_parquet(
                filepath, system_name, source_field, translation_field
            )
        elif suffix == ".txt":
            return self.load_from_txt(filepath, system_name, source_file)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def load_from_directory(
        self,
        directory: Union[str, Path],
        source_field: str = "source",
        translation_field: str = "translation",
        source_file: Optional[Union[str, Path]] = None,
    ) -> List[str]:
        """
        Load all translation files from a directory.

        Args:
            directory: Path to directory
            source_field: Field name for source text
            translation_field: Field name for translation
            source_file: Optional separate source file (for TXT format)

        Returns:
            List of system names loaded
        """
        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        if not directory.is_dir():
            raise ValueError(f"Not a directory: {directory}")

        system_names = []
        supported_extensions = [".json", ".jsonl", ".csv", ".parquet", ".txt"]

        for filepath in sorted(directory.iterdir()):
            if filepath.suffix.lower() in supported_extensions:
                try:
                    system_name = self.load_from_file(
                        filepath,
                        source_field=source_field,
                        translation_field=translation_field,
                        source_file=source_file,
                    )
                    system_names.append(system_name)
                except Exception as e:
                    print(f"⚠ Warning: Failed to load {filepath}: {e}")

        if not system_names:
            print(f"⚠ Warning: No translation files found in {directory}")

        return system_names

    def get_system_names(self) -> List[str]:
        """Get list of loaded system names."""
        return list(self.custom_systems.keys())

    def get_translations(self, system_name: str) -> List[str]:
        """Get translations for a specific system."""
        if system_name not in self.custom_systems:
            raise ValueError(f"System '{system_name}' not found")
        return self.custom_systems[system_name]

    def get_source_segments(self) -> Optional[List[str]]:
        """Get source segments if available."""
        return self.source_segments

    def get_all_translations(self) -> Dict[str, List[str]]:
        """Get all translations as a dictionary."""
        return self.custom_systems.copy()

    def merge_with_tmx(
        self, tmx_translations: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        """
        Merge custom translations with TMX translations.

        Args:
            tmx_translations: Dictionary from TMX parser

        Returns:
            Merged dictionary with all translations
        """
        merged = tmx_translations.copy()

        for system_name, translations in self.custom_systems.items():
            if system_name in merged:
                print(
                    f"⚠ Warning: System name '{system_name}' conflicts with TMX language code. "
                    f"Renaming to '{system_name}_custom'"
                )
                system_name = f"{system_name}_custom"

            merged[system_name] = translations

        return merged


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Load custom translations for evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load from single file
  python custom_translations_loader.py --file model_translations.json
  
  # Load from directory
  python custom_translations_loader.py --directory ./custom_translations
  
  # Load with custom field names
  python custom_translations_loader.py --file data.csv --source-field "english" --translation-field "ukrainian"
  
  # Load TXT with separate source file
  python custom_translations_loader.py --file translations.txt --source-file sources.txt

Supported formats: JSON, JSONL, CSV, Parquet, TXT
        """,
    )

    parser.add_argument("--file", type=str, help="Path to translation file")

    parser.add_argument(
        "--directory", type=str, help="Path to directory with translation files"
    )

    parser.add_argument(
        "--source-field",
        type=str,
        default="source",
        help="Field name for source text (default: source)",
    )

    parser.add_argument(
        "--translation-field",
        type=str,
        default="translation",
        help="Field name for translation (default: translation)",
    )

    parser.add_argument(
        "--source-file",
        type=str,
        help="Separate file with source sentences (for TXT format)",
    )

    parser.add_argument(
        "--system-name",
        type=str,
        help="Name for the translation system (default: filename)",
    )

    args = parser.parse_args()

    if not args.file and not args.directory:
        parser.print_help()
        print("\n✗ Error: Either --file or --directory must be specified")
        exit(1)

    loader = CustomTranslationsLoader()

    try:
        if args.file:
            system_name = loader.load_from_file(
                args.file,
                system_name=args.system_name,
                source_field=args.source_field,
                translation_field=args.translation_field,
                source_file=args.source_file,
            )
            print(f"\n✓ Successfully loaded system: {system_name}")

        elif args.directory:
            system_names = loader.load_from_directory(
                args.directory,
                source_field=args.source_field,
                translation_field=args.translation_field,
                source_file=args.source_file,
            )
            print(f"\n✓ Successfully loaded {len(system_names)} systems:")
            for name in system_names:
                print(f"  - {name}")

        print(
            f"\nTotal translations per system: {len(next(iter(loader.custom_systems.values())))}"
        )
        print(f"Systems loaded: {', '.join(loader.get_system_names())}")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
