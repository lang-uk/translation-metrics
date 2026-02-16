import argparse
import asyncio
import csv
import json
import logging
import multiprocessing as mp
import os
from multiprocessing import Manager, Process, Queue
from typing import Any, Dict, List

import datasets
from shared_translation import (
    AsyncTranslator,
    load_custom_translations,
    progress_monitor,
    split_dataset,
)

# Set up logging
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def process_text_batch_async(
    batch: List[Dict[str, Any]],
    translator: AsyncTranslator,
    semaphore: asyncio.Semaphore,
    progress_queue: Queue,
) -> List[Dict[str, Any]]:
    """Process a batch of text examples asynchronously with concurrency control."""

    async def process_single_text(example):
        async with semaphore:
            try:
                text = example.get("text", "")
                if not text:
                    logger.warning(f"Empty text found in example: {example}")
                    example["translation"] = ""
                    progress_queue.put(1)
                    return example

                translation = await translator.translate(text)
                example["translation"] = translation

                progress_queue.put(1)
                return example
            except Exception as e:
                logger.error(f"Failed to process text: {str(e)}")
                example["translation"] = ""
                progress_queue.put(1)
                return example

    tasks = [process_single_text(example) for example in batch]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Exception in batch processing: {result}")
            batch[i]["translation"] = ""
            processed_results.append(batch[i])
        else:
            processed_results.append(result)

    return processed_results


def text_worker_process(
    worker_id: int,
    data_chunk: List[Dict[str, Any]],
    config: Dict[str, Any],
    progress_queue: Queue,
    result_queue: Queue,
    max_concurrent_requests: int,
    custom_translations: Dict[str, str] = None,
):
    """Worker process that handles a chunk of text data with async translation."""

    async def async_worker():
        try:
            translator = AsyncTranslator(
                base_url=config["base_url"],
                api_key=config["api_key"],
                model=config["model"],
                custom_translations=custom_translations,
            )

            semaphore = asyncio.Semaphore(max_concurrent_requests)

            processed_chunk = await process_text_batch_async(
                data_chunk, translator, semaphore, progress_queue
            )

            result_queue.put((worker_id, processed_chunk))

        except Exception as e:
            logger.error(f"Worker {worker_id} failed: {str(e)}")
            result_queue.put((worker_id, []))

    asyncio.run(async_worker())


def save_results(results: List[Dict[str, Any]], output_path: str, output_format: str):
    """Save translation results in the specified format."""

    if output_format == "json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to: {output_path}")

    elif output_format == "jsonl":
        with open(output_path, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(f"Results saved to: {output_path}")

    elif output_format == "csv":
        if not results:
            print("No results to save")
            return

        keys = results[0].keys()
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        print(f"Results saved to: {output_path}")

    elif output_format == "parquet":
        processed_dataset = datasets.Dataset.from_list(results)
        processed_dataset.to_parquet(output_path)
        print(f"Results saved to: {output_path}")

    else:
        raise ValueError(f"Unsupported output format: {output_format}")


def load_input_data(input_path: str, text_column: str = "text") -> List[Dict[str, Any]]:
    """Load input data from various formats."""

    if input_path.endswith(".json"):
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                # Ensure each item has a 'text' field
                processed_data = []
                for item in data:
                    if isinstance(item, str):
                        processed_data.append({"text": item})
                    elif isinstance(item, dict):
                        if text_column not in item and "text" in item:
                            processed_data.append(item)
                        elif text_column in item:
                            processed_data.append({**item, "text": item[text_column]})
                        else:
                            processed_data.append(item)
                    else:
                        logger.warning(f"Unexpected data type in JSON: {type(item)}")
                return processed_data
            else:
                raise ValueError("JSON file must contain a list of texts or objects")

    elif input_path.endswith(".jsonl"):
        data = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                if isinstance(item, str):
                    data.append({"text": item})
                elif isinstance(item, dict):
                    if text_column not in item and "text" in item:
                        data.append(item)
                    elif text_column in item:
                        data.append({**item, "text": item[text_column]})
                    else:
                        data.append(item)
        return data

    elif input_path.endswith(".csv"):
        data = []
        with open(input_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if text_column not in row and "text" in row:
                    data.append(row)
                elif text_column in row:
                    data.append({**row, "text": row[text_column]})
                else:
                    data.append(row)
        return data

    elif input_path.endswith(".txt"):
        with open(input_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            return [{"text": line} for line in lines]

    elif input_path.endswith(".parquet"):
        dataset = datasets.load_dataset("parquet", data_files=input_path, split="train")
        data = []
        for item in dataset:
            if text_column not in item and "text" in item:
                data.append(dict(item))
            elif text_column in item:
                data.append({**dict(item), "text": item[text_column]})
            else:
                data.append(dict(item))
        return data

    else:
        raise ValueError(f"Unsupported input format: {input_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Translate English texts to Ukrainian using multiprocessing"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input file path (json, jsonl, csv, txt, or parquet)",
    )
    parser.add_argument("--output", type=str, required=True, help="Output file path")
    parser.add_argument(
        "--output-format",
        type=str,
        default="json",
        choices=["json", "jsonl", "csv", "parquet"],
        help="Output format (default: json)",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Name of the column containing text to translate (default: text)",
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
        default="http://localhost:8000/v1",
        help="Base URL for the LLM API",
    )
    parser.add_argument(
        "--api-key", type=str, default="EMPTY", help="API key for the LLM service"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="lapa-llm/lapa-v0.1.2-instruct",
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

    # Configuration for workers
    config = {"base_url": args.base_url, "api_key": args.api_key, "model": args.model}

    # Test the translator first
    print("Testing translator...")
    test_translator = AsyncTranslator(**config, custom_translations=custom_translations)

    async def test_translation():
        try:
            text_to_translate = "Hello, how are you?"
            result = await test_translator.translate(text_to_translate)
            print(f"Test translation result: {result}")
            return True
        except Exception as e:
            print(f"Test failed: {e}")
            return False

    if not asyncio.run(test_translation()):
        print("Translator test failed. Please check your configuration.")
        return
    print("Translator test successful!")

    # Load input data
    print(f"Loading data from {args.input}...")
    try:
        data = load_input_data(args.input, args.text_column)
    except Exception as e:
        print(f"Failed to load input data: {e}")
        return

    print(f"Loaded {len(data)} texts to translate")

    if not data:
        print("No data to translate")
        return

    # Print first example
    print("First example:")
    print(json.dumps(data[0], ensure_ascii=False, indent=2))

    if args.test_mode:
        data = data[:100]  # Limit for testing
        print("Running in test mode with 100 examples")

    # Split dataset for workers
    print(f"Splitting dataset for {args.num_workers} workers...")
    data_chunks = split_dataset(data, args.num_workers)

    # Create multiprocessing queues
    manager = Manager()
    progress_queue = manager.Queue()
    result_queue = manager.Queue()

    # Start progress monitor in a separate process
    progress_proc = Process(target=progress_monitor, args=(progress_queue, len(data)))
    progress_proc.start()

    # Start worker processes
    print(
        f"Starting {args.num_workers} workers with {args.max_concurrent_per_worker} concurrent requests each..."
    )
    workers = []

    for worker_id in range(args.num_workers):
        worker = Process(
            target=text_worker_process,
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
    worker_results = {}  # Dict to store results by worker_id
    completed_workers = 0

    while completed_workers < args.num_workers:
        try:
            worker_id, results = result_queue.get(timeout=10)
            worker_results[worker_id] = results
            completed_workers += 1
        except:
            continue  # Timeout is normal, just continue waiting

    # Reassemble results in the correct order
    all_results = []
    for worker_id in range(args.num_workers):
        if worker_id in worker_results:
            all_results.extend(worker_results[worker_id])

    # Wait for all workers to finish
    for worker in workers:
        worker.join()

    # Stop progress monitor
    progress_proc.terminate()
    progress_proc.join()

    print("All workers completed!")

    # Save results
    print("Saving results...")
    if all_results:
        try:
            save_results(all_results, args.output, args.output_format)
            print(f"Successfully processed {len(all_results)} texts")
        except Exception as e:
            print(f"Failed to save results: {e}")
    else:
        print("No results to save. Check for errors in worker processes.")


if __name__ == "__main__":
    # Set multiprocessing start method for compatibility
    mp.set_start_method("spawn", force=True)
    main()
