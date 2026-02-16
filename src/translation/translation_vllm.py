#!/usr/bin/env python3
"""
translate_dataset.py

Translate one or more columns of English text in a CSV file or JSON file with conversations
to a target language using any text2text-capable Hugging Face model, and save the results 
to a new file in the results/ folder.

For CSV files:
    Translates specified columns to target language.

For JSON files:
    Reads conversations and translates all 'value' keys within them.

Example:
    python translate_dataset.py \
      --model_name google/gemma-3-12b-it \
      --input_file data/my_english_sentences.csv \
      --text_columns sentence title \
      --target_language Ukrainian \
      --batch_size 16 \
      --max_new_tokens 60

    python translate_dataset.py \
      --model_name google/gemma-3-12b-it \
      --input_file data/conversations.json \
      --target_language Ukrainian \
      --batch_size 16 \
      --max_new_tokens 60
"""

import argparse
import json
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams


class ModelManager:
    """Manages the LLM model initialization and inference."""

    def __init__(
        self,
        model_name: str,
        num_gpus: int = 2,
        max_model_len: int = 8192,
        max_num_seqs: int = 4,
    ):
        self.model_name = model_name
        self.num_gpus = num_gpus
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self.llm = None

    def initialize(self):
        """Initialize the LLM model."""
        if self.llm is None:
            self.llm = LLM(
                model=self.model_name,
                tensor_parallel_size=self.num_gpus,
                max_model_len=self.max_model_len,
                max_num_seqs=self.max_num_seqs,
            )

    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ):
        """Generate translations for a batch of prompts."""
        if self.llm is None:
            self.initialize()

        sampling_params = SamplingParams(
            max_tokens=max_new_tokens, temperature=temperature, top_p=top_p
        )
        return self.llm.generate(prompts, sampling_params)


class TranslationPromptBuilder:
    """Builds translation prompts for different scenarios."""

    @staticmethod
    def build_strict_translation_prompt(
        target_language: str, include_code_instruction: bool = True
    ) -> str:
        """Build a strict translation prompt."""
        instruction = f"Translate the following English text to {target_language}."

        if include_code_instruction:
            instruction += " If you encounter code blocks, do not translate them, just return the text as is."

        instruction += " Output only the translated text without any additional words or formatting, start with the translated text:\n"
        return instruction


class DataProcessor(ABC):
    """Abstract base class for data processors."""

    def __init__(self, input_file: str, output_folder: str):
        self.input_file = input_file
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)

    @abstractmethod
    def load_data(self) -> Any:
        """Load data from the input file."""
        pass

    @abstractmethod
    def extract_texts_to_translate(self) -> List[Dict[str, Any]]:
        """Extract texts that need to be translated."""
        pass

    @abstractmethod
    def save_results(self, translated_data: Any, output_path: str) -> None:
        """Save the translated results."""
        pass

    def generate_output_path(
        self, model_name: str, target_language: str, **kwargs
    ) -> str:
        """Generate output file path."""
        base = os.path.splitext(os.path.basename(self.input_file))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model = model_name.replace("/", "_")
        safe_lang = target_language.replace(" ", "_")

        # Add additional suffixes from kwargs
        suffixes = []
        for key, value in kwargs.items():
            if value:
                if isinstance(value, list):
                    suffixes.append("_".join(value))
                else:
                    suffixes.append(str(value))

        filename_parts = [base] + suffixes + [safe_model, safe_lang, timestamp]
        filename = "_".join(filename_parts)

        return os.path.join(
            self.output_folder, f"{filename}.{self.get_file_extension()}"
        )

    @abstractmethod
    def get_file_extension(self) -> str:
        """Get the file extension for the output file."""
        pass


class JSONDataProcessor(DataProcessor):
    """Processes JSON files with conversations."""

    def load_data(self) -> Dict[str, Any]:
        """Load JSON data from file."""
        with open(self.input_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def extract_texts_to_translate(self) -> List[Dict[str, Any]]:
        """Extract all entries containing 'value' keys from conversations."""
        data = self.load_data()
        entries_to_translate = []

        def find_entries_with_values(obj, path=""):
            if isinstance(obj, dict):
                if "value" in obj and isinstance(obj["value"], str):
                    entries_to_translate.append(
                        {
                            "path": path,
                            "entry_data": obj,
                            "value_paths": [f"{path}.value"],
                        }
                    )
                else:
                    for key, value in obj.items():
                        current_path = f"{path}.{key}" if path else key
                        find_entries_with_values(value, current_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    current_path = f"{path}[{i}]"
                    find_entries_with_values(item, current_path)

        if "conversations" in data:
            find_entries_with_values(data["conversations"], "conversations")
        else:
            find_entries_with_values(data)

        return entries_to_translate

    def save_results(self, translated_data: Any, output_path: str) -> None:
        """Save translated JSON data."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(translated_data, f, ensure_ascii=False, indent=2)

    def get_file_extension(self) -> str:
        return "json"


class CSVDataProcessor(DataProcessor):
    """Processes CSV files."""

    def __init__(self, input_file: str, output_folder: str, text_columns: List[str]):
        super().__init__(input_file, output_folder)
        self.text_columns = text_columns

    def load_data(self) -> pd.DataFrame:
        """Load CSV data from file."""
        df = pd.read_csv(self.input_file)
        missing = [c for c in self.text_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Columns not found in {self.input_file}: {missing}")
        return df

    def extract_texts_to_translate(self) -> List[Dict[str, Any]]:
        """Extract texts from specified columns."""
        df = self.load_data()
        entries = []

        for idx, row in df.iterrows():
            for col in self.text_columns:
                entries.append(
                    {
                        "row_index": idx,
                        "column": col,
                        "text": str(row[col]),
                        "entry_data": {
                            "row": idx,
                            "column": col,
                            "text": str(row[col]),
                        },
                    }
                )

        return entries

    def save_results(self, translated_data: pd.DataFrame, output_path: str) -> None:
        """Save translated CSV data."""
        translated_data.to_csv(output_path, index=False)

    def get_file_extension(self) -> str:
        return "csv"


class Translator:
    """Main translation orchestrator."""

    def __init__(
        self, model_manager: ModelManager, prompt_builder: TranslationPromptBuilder
    ):
        self.model_manager = model_manager
        self.prompt_builder = prompt_builder

    def translate_batch(
        self, texts: List[str], target_language: str, max_new_tokens: int
    ) -> List[str]:
        """Translate a batch of texts."""
        instruction = self.prompt_builder.build_strict_translation_prompt(
            target_language
        )
        prompts = [f'{instruction}"{t}"' for t in texts]

        outputs = self.model_manager.generate(prompts, max_new_tokens)

        translations = []
        for i, (out, prompt) in enumerate(zip(outputs, prompts)):
            try:
                generated = out.outputs[0].text.strip()
                if generated.startswith(prompt):
                    cleaned = generated[len(prompt) :].strip()
                else:
                    cleaned = generated
                translations.append(cleaned)
            except Exception as e:
                print(f"❌ Error processing output for item {i}: {e}")
                translations.append(f"[TRANSLATION_ERROR: {str(e)}]")

        return translations


class JSONTranslator(Translator):
    """Specialized translator for JSON files."""

    def translate_dataset(
        self,
        data_processor: JSONDataProcessor,
        target_language: str,
        batch_size: int,
        max_new_tokens: int,
        save_interval: Optional[int] = None,
        save_only_translated: bool = True,
    ) -> str:
        """Translate JSON dataset."""
        # Extract entries to translate
        entries_to_translate = data_processor.extract_texts_to_translate()

        if not entries_to_translate:
            print("⚠️  No entries with 'value' keys found in the JSON structure")
            return None

        print(f"📝 Found {len(entries_to_translate)} entries with values to translate")

        # Generate output path
        output_path = data_processor.generate_output_path(
            self.model_manager.model_name, target_language
        )

        # Set save interval
        total_entries = len(entries_to_translate)
        if save_interval is None:
            save_interval = max(1, min(100, total_entries // 10))

        print(f"💾 Will save progress every {save_interval} entries")
        print(
            f"📁 Output format: {'Only translated entries' if save_only_translated else 'Full structure with translations'}"
        )

        # Process entries in batches
        translated_entries_map = {}

        for start_idx in tqdm(
            range(0, total_entries, batch_size), desc="Translating entries"
        ):
            end_idx = min(start_idx + batch_size, total_entries)
            batch_entries = entries_to_translate[start_idx:end_idx]

            # Extract texts and translate
            texts_to_translate = []
            entry_mapping = []

            for entry in batch_entries:
                entry_data = entry["entry_data"]
                values_in_entry = []
                for key, value in entry_data.items():
                    if key == "value" and isinstance(value, str):
                        values_in_entry.append(value)

                if values_in_entry:
                    texts_to_translate.extend(values_in_entry)
                    entry_mapping.append(
                        {"entry": entry, "value_count": len(values_in_entry)}
                    )

            if not texts_to_translate:
                continue

            # Translate batch
            translations = self.translate_batch(
                texts_to_translate, target_language, max_new_tokens
            )

            # Process results
            output_idx = 0
            for entry_info in entry_mapping:
                entry = entry_info["entry"]
                value_count = entry_info["value_count"]

                translated_entry = json.loads(json.dumps(entry["entry_data"]))

                for key, value in translated_entry.items():
                    if key == "value" and isinstance(value, str):
                        if output_idx < len(translations):
                            translated_entry[key] = translations[output_idx]
                            output_idx += 1
                        else:
                            print(
                                f"❌ Not enough translations for entry {entry['path']}"
                            )
                            break

                translated_entries_map[entry["path"]] = translated_entry

            # Save incrementally
            if (
                start_idx + batch_size
            ) % save_interval == 0 or start_idx + batch_size >= total_entries:
                print(
                    f"💾 Saving progress... ({len(translated_entries_map)} entries completed)"
                )
                if save_only_translated:
                    self._save_translations_only(
                        data_processor, translated_entries_map, output_path
                    )
                else:
                    self._save_full_structure(
                        data_processor, translated_entries_map, output_path
                    )

        print(f"✅ All translations completed and saved to {output_path}")
        return output_path

    def _save_translations_only(
        self,
        data_processor: JSONDataProcessor,
        translated_entries_map: Dict,
        output_path: str,
    ):
        """Save only translated entries."""
        translated_entries = []

        for path, translated_entry in translated_entries_map.items():
            try:
                entry_with_metadata = json.loads(json.dumps(translated_entry))
                entry_with_metadata["_translation_metadata"] = {
                    "original_path": path,
                    "translated_at": datetime.now().isoformat(),
                    "entry_type": type(translated_entry).__name__,
                }
                translated_entries.append(entry_with_metadata)
            except Exception as e:
                print(f"❌ Error processing translated entry for path '{path}': {e}")
                error_entry = {
                    "path": path,
                    "error": str(e),
                    "_translation_metadata": {
                        "original_path": path,
                        "translated_at": datetime.now().isoformat(),
                        "error": str(e),
                    },
                }
                translated_entries.append(error_entry)

        data_processor.save_results(translated_entries, output_path)
        print(f"💾 Saved {len(translated_entries)} translated entries to {output_path}")

    def _save_full_structure(
        self,
        data_processor: JSONDataProcessor,
        translated_entries_map: Dict,
        output_path: str,
    ):
        """Save full structure with translations applied."""
        original_data = data_processor.load_data()
        translated_data = json.loads(json.dumps(original_data))

        for path, translated_entry in translated_entries_map.items():
            try:
                current = translated_data

                if path:
                    parts = self._parse_path(path)

                    for part in parts[:-1]:
                        current = self._navigate_to_part(current, part)

                    final_part = parts[-1]
                    current = self._navigate_to_parent(current, final_part)
                    current[final_part] = translated_entry

            except Exception as e:
                print(f"❌ Error applying translation for path '{path}': {e}")
                continue

        data_processor.save_results(translated_data, output_path)
        print(
            f"💾 Saved full structure with {len(translated_entries_map)} translated entries to {output_path}"
        )

    def _parse_path(self, path: str) -> List[str]:
        """Parse a path string into parts."""
        parts = []
        current_part = ""
        bracket_count = 0

        for char in path:
            if char == "[":
                bracket_count += 1
                current_part += char
            elif char == "]":
                bracket_count -= 1
                current_part += char
            elif char == "." and bracket_count == 0:
                if current_part:
                    parts.append(current_part)
                    current_part = ""
            else:
                current_part += char

        if current_part:
            parts.append(current_part)

        return parts

    def _navigate_to_part(self, current: Any, part: str) -> Any:
        """Navigate to a specific part in the data structure."""
        if part.startswith("[") and part.endswith("]"):
            index = int(part[1:-1])
            return current[index]
        elif "[" in part and "]" in part:
            bracket_pos = part.index("[")
            key = part[:bracket_pos]
            index = int(part[bracket_pos + 1 : part.index("]")])
            return current[key][index]
        else:
            return current[part]

    def _navigate_to_parent(self, current: Any, part: str) -> Any:
        """Navigate to the parent of a specific part."""
        if part.startswith("[") and part.endswith("]"):
            return current
        elif "[" in part and "]" in part:
            bracket_pos = part.index("[")
            key = part[:bracket_pos]
            return current[key]
        else:
            return current


class CSVTranslator(Translator):
    """Specialized translator for CSV files."""

    def translate_dataset(
        self,
        data_processor: CSVDataProcessor,
        target_language: str,
        batch_size: int,
        max_new_tokens: int,
    ) -> str:
        """Translate CSV dataset."""
        df = data_processor.load_data()

        # Prepare new columns
        for col in data_processor.text_columns:
            df[f"{col}_in_{target_language}"] = ""

        # Generate output path
        output_path = data_processor.generate_output_path(
            self.model_manager.model_name,
            target_language,
            text_columns=data_processor.text_columns,
        )

        # Write header first
        df.iloc[:0].to_csv(output_path, index=False)

        # Process rows in batches
        total_rows = len(df)
        for start_idx in tqdm(
            range(0, total_rows, batch_size), desc="Translating rows"
        ):
            end_idx = min(start_idx + batch_size, total_rows)
            batch_df = df.iloc[start_idx:end_idx].copy()

            # Translate all columns for this batch
            for col in data_processor.text_columns:
                texts = batch_df[col].astype(str).tolist()
                translations = self.translate_batch(
                    texts, target_language, max_new_tokens
                )

                for i, translation in enumerate(translations):
                    batch_df.iloc[
                        i, batch_df.columns.get_loc(f"{col}_in_{target_language}")
                    ] = translation

            # Append this batch to the CSV file
            batch_df.to_csv(output_path, mode="a", header=False, index=False)
            del batch_df

        print(f"✅ Translations saved to {output_path}")
        return output_path


def create_data_processor(
    input_file: str, output_folder: str, text_columns: Optional[List[str]] = None
) -> DataProcessor:
    """Factory function to create appropriate data processor."""
    file_ext = os.path.splitext(input_file)[1].lower()

    if file_ext == ".json":
        return JSONDataProcessor(input_file, output_folder)
    elif file_ext == ".csv":
        if not text_columns:
            raise ValueError("text_columns parameter is required for CSV files")
        return CSVDataProcessor(input_file, output_folder, text_columns)
    else:
        raise ValueError(
            f"Unsupported file type: {file_ext}. Only .csv and .json files are supported."
        )


def translate_dataset(
    model_name: str,
    input_file: str,
    text_columns: Optional[List[str]] = None,
    target_language: str = None,
    output_folder: str = "results",
    batch_size: int = 8,
    max_new_tokens: int = 50,
    save_interval: Optional[int] = None,
    save_only_translated: bool = True,
    num_gpus: int = 2,
):
    """Main function that orchestrates the translation process."""

    # Create components
    model_manager = ModelManager(model_name, num_gpus)
    prompt_builder = TranslationPromptBuilder()
    data_processor = create_data_processor(input_file, output_folder, text_columns)

    # Create appropriate translator
    if isinstance(data_processor, JSONDataProcessor):
        translator = JSONTranslator(model_manager, prompt_builder)
        return translator.translate_dataset(
            data_processor=data_processor,
            target_language=target_language,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            save_interval=save_interval,
            save_only_translated=save_only_translated,
        )
    elif isinstance(data_processor, CSVDataProcessor):
        translator = CSVTranslator(model_manager, prompt_builder)
        return translator.translate_dataset(
            data_processor=data_processor,
            target_language=target_language,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
        )
    else:
        raise ValueError(f"Unsupported data processor type: {type(data_processor)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Translate English text in CSV or JSON files to a target language using a HF seq2seq model."
    )
    parser.add_argument(
        "--model_name",
        "-m",
        required=True,
        help="Hugging Face model name (e.g. google/gemma-3-12b-it)",
    )
    parser.add_argument(
        "--input_file",
        "-i",
        required=True,
        help="Path to the input file (CSV or JSON).",
    )
    parser.add_argument(
        "--text_columns",
        "-c",
        nargs="+",
        help="Space-separated list of column names to translate (required for CSV files, ignored for JSON).",
    )
    parser.add_argument(
        "--target_language",
        "-t",
        required=True,
        help="Target language for translation (e.g. 'Ukrainian', 'Italian').",
    )
    parser.add_argument(
        "--output_folder",
        "-o",
        default="results",
        help="Folder to write the translated file (default: 'results').",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=8,
        help="How many texts to send to the model per batch (default: 8).",
    )
    parser.add_argument(
        "--max_new_tokens",
        "-n",
        type=int,
        default=50,
        help="Maximum number of new tokens to generate (default: 50).",
    )
    parser.add_argument(
        "--save_interval",
        "-s",
        type=int,
        default=None,
        help="How often to save progress for JSON files (default: auto, every 100 items or 10%% of total).",
    )
    parser.add_argument(
        "--save_full_structure",
        action="store_true",
        help="Save the full original data structure with translations (default: save only translated entries).",
    )
    parser.add_argument(
        "--num_gpus", type=int, default=2, help="Number of GPUs to use (default: 2)."
    )

    args = parser.parse_args()
    translate_dataset(
        model_name=args.model_name,
        input_file=args.input_file,
        text_columns=args.text_columns,
        target_language=args.target_language,
        output_folder=args.output_folder,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        save_interval=args.save_interval,
        save_only_translated=not args.save_full_structure,
        num_gpus=args.num_gpus,
    )
