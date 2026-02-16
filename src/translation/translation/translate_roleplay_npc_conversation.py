import argparse
import asyncio
import csv
import json
import logging
import multiprocessing as mp
import os
import random
import re
import shutil
import warnings
from multiprocessing import Manager, Process, Queue
from time import sleep
from typing import Any, Dict, List, Optional

import datasets
import pandas as pd
from openai import AsyncOpenAI
from tqdm import tqdm

TRANSLATE_PROMPT_TEMPLATE = """You are a professional translator specialized in English to Ukrainian translation.
You will be given an already translated instruction used to generate a conversation, a conversation between a player and an NPC itself, and a single message from this conversation.
Your task is to translate only this message, using the context of the instruction and conversation to understand the meaning and tone of the message.
Translate the given text naturally and idiomatically into Ukrainian.
Preserve the meaning, tone, and style of the original text, considering the wider context of the conversation.
Return only the translated text, without explanations or notes.

Instruction:

{instruction}


Conversation:

{conversation}


Text to translate:

{message}"""

# Set up logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def load_custom_translations(custom_dir: Optional[str] = None) -> Dict[str, str]:
    """
    Load custom translations from a directory containing translation files.
    Supports JSON, JSONL, CSV, and Parquet formats.

    Expected format for translation files:
    - JSON/JSONL: {"text": "original", "translation": "translated"} or {"english": "...", "ukrainian": "..."}
    - CSV: columns 'text' and 'translation' or 'english' and 'ukrainian'
    - Parquet: same as CSV

    Returns a dictionary mapping original text to translated text.
    """
    translations = {}

    if not custom_dir or not os.path.exists(custom_dir):
        logger.info(f"No custom translations directory found at: {custom_dir}")
        return translations

    logger.info(f"Loading custom translations from: {custom_dir}")

    for filename in os.listdir(custom_dir):
        filepath = os.path.join(custom_dir, filename)

        if not os.path.isfile(filepath):
            continue

        try:
            if filename.endswith(".json"):
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict):
                                text = item.get(
                                    "text", item.get("english", item.get("source", ""))
                                )
                                translation = item.get(
                                    "translation",
                                    item.get("ukrainian", item.get("target", "")),
                                )
                                if text and translation:
                                    translations[text.strip()] = translation.strip()

            elif filename.endswith(".jsonl"):
                with open(filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        item = json.loads(line.strip())
                        text = item.get(
                            "text", item.get("english", item.get("source", ""))
                        )
                        translation = item.get(
                            "translation", item.get("ukrainian", item.get("target", ""))
                        )
                        if text and translation:
                            translations[text.strip()] = translation.strip()

            elif filename.endswith(".csv"):
                with open(filepath, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        text = row.get(
                            "text", row.get("english", row.get("source", ""))
                        )
                        translation = row.get(
                            "translation", row.get("ukrainian", row.get("target", ""))
                        )
                        if text and translation:
                            translations[text.strip()] = translation.strip()

            elif filename.endswith(".parquet"):
                dataset = datasets.load_dataset(
                    "parquet", data_files=filepath, split="train"
                )
                for item in dataset:
                    text = item.get("text", item.get("english", item.get("source", "")))
                    translation = item.get(
                        "translation", item.get("ukrainian", item.get("target", ""))
                    )
                    if text and translation:
                        translations[text.strip()] = translation.strip()

            logger.info(f"Loaded {len(translations)} translations from {filename}")

        except Exception as e:
            logger.warning(f"Failed to load translations from {filename}: {e}")

    logger.info(f"Total custom translations loaded: {len(translations)}")
    return translations


class AsyncTranslator:
    """
    Async version of the translator for better concurrent performance
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        model: str = "meta/llama-2-70b-chat",
        custom_translations: Optional[Dict[str, str]] = None,
    ):
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.system_prompt = TRANSLATE_PROMPT_TEMPLATE
        self.custom_translations = custom_translations or {}
        logger.info(
            f"Initialized translator with {len(self.custom_translations)} custom translations"
        )

    async def translate(
        self, text: str, instruction: str, conversation: str, max_retries: int = 40
    ) -> str:
        """Translate a single text from English to Ukrainian with async retry logic."""
        text = text.strip()

        # Check if we have a custom translation for this exact text
        if text in self.custom_translations:
            logger.debug(f"Using custom translation for: {text[:50]}...")
            return self.custom_translations[text]

        if len(text) <= 5000:
            messages = [
                {
                    "role": "user",
                    "content": self.system_prompt.format(
                        instruction=instruction, conversation=conversation, message=text
                    ),
                },
            ]

            for attempt in range(max_retries):
                try:
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        stream=False,
                        top_p=0.95,
                        max_tokens=16000,
                        temperature=0.5,
                    )
                    result = response.choices[0].message.content.strip()
                    return result.strip()

                except Exception as e:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(30)
                    else:
                        logger.error(
                            f"All translation attempts failed after {max_retries} retries: {str(e)}"
                        )
                        return ""
            return response.choices[0].message.content.strip()

        # For long texts, split at newlines near 5000 characters while preserving empty lines
        chunks = []
        current_chunk = []
        current_length = 0

        # Split by lines but preserve consecutive newlines
        lines = text.split("\n")
        i = 0
        while i < len(lines):
            # Count consecutive empty lines
            empty_lines = 0
            while i + empty_lines < len(lines) and not lines[i + empty_lines].strip():
                empty_lines += 1

            if empty_lines > 0:
                # If we have a current chunk and adding empty lines won't exceed limit
                if current_chunk and current_length + empty_lines <= 5000:
                    current_chunk.extend([""] * empty_lines)
                    current_length += empty_lines
                else:
                    # If we have a current chunk, save it
                    if current_chunk:
                        chunks.append("\n".join(current_chunk))
                    current_chunk = [""] * empty_lines
                    current_length = empty_lines
                i += empty_lines
                continue

            # Handle non-empty line
            line = lines[i]
            if current_length + len(line) + 1 <= 5000:
                current_chunk.append(line)
                current_length += len(line) + 1
            else:
                if current_chunk:
                    chunks.append("\n".join(current_chunk))
                current_chunk = [line]
                current_length = len(line)
            i += 1

        if current_chunk:
            chunks.append("\n".join(current_chunk))

        # Translate each chunk separately
        translated_chunks = []
        for chunk in chunks:
            messages = [
                {
                    "role": "user",
                    "content": self.system_prompt.format(
                        instruction=instruction,
                        conversation=conversation,
                        message=chunk,
                    ),
                },
            ]
            result = ""
            for attempt in range(max_retries):
                try:
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        stream=False,
                        top_p=0.95,
                        max_tokens=16000,
                        temperature=0.5,
                    )
                    result = response.choices[0].message.content.strip()

                except Exception as e:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(30)
                    else:
                        logger.error(
                            f"All translation attempts failed after {max_retries} retries: {str(e)}"
                        )
                        return ""
                else:
                    break

            translated_chunks.append(result)

        return "\n\n".join(translated_chunks)


async def process_batch_async(
    batch: List[Dict[str, Any]],
    translator: AsyncTranslator,
    semaphore: asyncio.Semaphore,
    progress_queue: Queue,
) -> List[Dict[str, Any]]:
    """Process a batch of examples asynchronously with concurrency control."""

    async def process_single_example(example):
        async with semaphore:  # Limit concurrent requests
            try:
                original_conversation = example["messages"]
                instruction = example["instruction"]
                role2name = {"user": "Player", "assistant": "NPC"}
                conversation = "\n\n".join(
                    [
                        f"{role2name[message['role']]}: {message['content']}"
                        for message in original_conversation
                    ]
                )

                translated_conversation = []
                for message in original_conversation:
                    content = message["content"]
                    content = await translator.translate(
                        content, instruction, conversation
                    )
                    translated_conversation.append(
                        {"role": message["role"], "content": content}
                    )

                example["messages"] = translated_conversation
                example["original_messages"] = original_conversation

                progress_queue.put(1)  # Signal progress
                return example
            except Exception as e:
                logger.error(f"Failed to process example: {str(e)}")
                example["translated"] = ""
                progress_queue.put(1)  # Signal progress even on failure
                return example

    # Process all examples in the batch concurrently
    tasks = [process_single_example(example) for example in batch]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle any exceptions that occurred
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Exception in batch processing: {result}")
            batch[i]["translated"] = ""
            processed_results.append(batch[i])
        else:
            processed_results.append(result)

    return processed_results


def worker_process(
    worker_id: int,
    data_chunk: List[Dict[str, Any]],
    config: Dict[str, Any],
    progress_queue: Queue,
    result_queue: Queue,
    max_concurrent_requests: int,
    custom_translations: Optional[Dict[str, str]] = None,
):
    """Worker process that handles a chunk of data with async translation."""

    async def async_worker():
        try:
            # Create translator for this worker
            translator = AsyncTranslator(
                base_url=config["base_url"],
                api_key=config["api_key"],
                model=config["model"],
                custom_translations=custom_translations,
            )

            # Create semaphore to limit concurrent requests per worker
            semaphore = asyncio.Semaphore(max_concurrent_requests)

            # Process the data chunk
            processed_chunk = await process_batch_async(
                data_chunk, translator, semaphore, progress_queue
            )

            # Send results back
            result_queue.put((worker_id, processed_chunk))

        except Exception as e:
            logger.error(f"Worker {worker_id} failed: {str(e)}")
            result_queue.put((worker_id, []))

    # Run the async worker
    asyncio.run(async_worker())


def split_dataset(dataset, num_workers: int) -> List[List[Dict[str, Any]]]:
    """Split dataset into chunks for workers."""
    dataset_list = list(dataset)
    chunk_size = len(dataset_list) // num_workers
    remainder = len(dataset_list) % num_workers

    chunks = []
    start_idx = 0

    for i in range(num_workers):
        # Add one extra item to the first 'remainder' chunks
        current_chunk_size = chunk_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_chunk_size
        chunks.append(dataset_list[start_idx:end_idx])
        start_idx = end_idx

    return chunks


def progress_monitor(progress_queue: Queue, total_items: int):
    """Monitor progress from all workers and update progress bar."""
    with tqdm(total=total_items, desc="Translating", unit="items") as pbar:
        completed = 0
        while completed < total_items:
            try:
                # Wait for progress updates
                progress_queue.get(timeout=1)
                completed += 1
                pbar.update(1)
            except:
                continue  # Timeout is normal, just continue monitoring


def main():
    parser = argparse.ArgumentParser(
        description="Translate dataset using multiprocessing"
    )
    parser.add_argument(
        "--num-workers", type=int, default=16, help="Number of worker processes"
    )
    parser.add_argument(
        "--max-concurrent-per-worker",
        type=int,
        default=500,
        help="Maximum concurrent requests per worker",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://fs-cai-008:8000/v1",
        help="Base URL for the LLM API",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="API KEY (of needed)",
        help="API key for the LLM service",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-3-1b-it",
        help="Model name to use for translation",
    )
    parser.add_argument(
        "--test-mode", action="store_true", help="Run in test mode with limited dataset"
    )
    parser.add_argument(
        "--custom-translations-dir",
        type=str,
        default=None,
        help="Directory containing custom translation files",
    )

    args = parser.parse_args()

    # Set logging level based on test mode
    if args.test_mode:
        logging.getLogger().setLevel(logging.INFO)
        logger.info("Running in test mode with verbose logging enabled")

    # Load custom translations if directory is provided
    custom_translations = load_custom_translations(args.custom_translations_dir)

    # Setup directories
    cur_dir = os.getcwd()
    data_dir = f"{cur_dir}/data/"
    translated_output_data_dir = os.path.join(data_dir, "roleplay_npc")
    os.makedirs(translated_output_data_dir, exist_ok=True)

    # Configuration for workers
    config = {"base_url": args.base_url, "api_key": args.api_key, "model": args.model}

    # Test the translator first
    print("Testing translator...")
    test_translator = AsyncTranslator(**config, custom_translations=custom_translations)

    async def test_translation():
        try:
            result = await test_translator.translate(
                "Hello, how are you?",
                "Привітайся",
                "NPC: Hello, how are you?\n\nPlayer: Good, thank you!",
            )
            print(f"Test translation result: {result}")
            return True
        except Exception as e:
            print(f"Test failed: {e}")
            return False

    if not asyncio.run(test_translation()):
        print("Translator test failed. Please check your configuration.")
        return

    print("Translator test successful!")

    # Load dataset
    print("Loading dataset...")
    dataset = datasets.load_dataset(
        "le-llm/RolePlay-NPC-instructions-uk-truncated", split="train"
    )

    if args.test_mode:
        dataset = dataset.select(range(100))  # Limit for testing
        print("Running in test mode with 100 examples")

    print(f"Dataset loaded with {len(dataset)} examples")

    # Split dataset for workers
    print(f"Splitting dataset for {args.num_workers} workers...")
    data_chunks = split_dataset(dataset, args.num_workers)

    # Create multiprocessing queues
    manager = Manager()
    progress_queue = manager.Queue()
    result_queue = manager.Queue()

    # Start progress monitor in a separate process
    progress_proc = Process(
        target=progress_monitor, args=(progress_queue, len(dataset))
    )
    progress_proc.start()

    # Start worker processes
    print(
        f"Starting {args.num_workers} workers with {args.max_concurrent_per_worker} concurrent requests each..."
    )
    workers = []

    for worker_id in range(args.num_workers):
        worker = Process(
            target=worker_process,
            args=(
                worker_id,
                data_chunks[worker_id],
                config,
                progress_queue,
                result_queue,
                args.max_concurrent_per_worker,
                custom_translations,
            ),
        )
        worker.start()
        workers.append(worker)

    # Collect results from workers
    print("Waiting for workers to complete...")
    all_results = []
    completed_workers = 0

    while completed_workers < args.num_workers:
        try:
            worker_id, worker_results = result_queue.get(timeout=10)
            all_results.extend(worker_results)
            completed_workers += 1
        except:
            continue  # Timeout is normal, just continue waiting

    # Wait for all workers to finish
    for worker in workers:
        worker.join()

    # Stop progress monitor
    progress_proc.terminate()
    progress_proc.join()

    print("All workers completed!")

    # Convert results back to dataset and save
    print("Saving results...")
    if all_results:
        processed_dataset = datasets.Dataset.from_list(all_results)
        output_path = os.path.join(
            translated_output_data_dir, "roleplay_npc_conversation.parquet"
        )
        processed_dataset.to_parquet(output_path)
        print(f"Results saved to: {output_path}")
        print(f"Successfully processed {len(all_results)} examples")
    else:
        print("No results to save. Check for errors in worker processes.")


if __name__ == "__main__":
    # Set multiprocessing start method for compatibility
    mp.set_start_method("spawn", force=True)
    main()
