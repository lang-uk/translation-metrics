import argparse
import asyncio
import logging
import multiprocessing as mp
import os
from multiprocessing import Manager, Process, Queue

import datasets
from shared_translation import (
    AsyncTranslator,
    load_custom_translations,
    progress_monitor,
    split_dataset,
    worker_process,
)
from split_chunks import split_chunks

# Set up logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


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
        default="google/gemma-3-27b-it",
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
    translated_output_data_dir = os.path.join(data_dir, "hermes")
    os.makedirs(translated_output_data_dir, exist_ok=True)

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

    # Load dataset
    print("Loading dataset...")
    dataset = datasets.load_dataset("NousResearch/Hermes-3-Dataset", split="train")

    def prefilter_long_chunks(example):
        max_chunk_length = 5300
        for message in example["conversations"]:
            text = message["value"]
            chunks, _ = split_chunks(text)
            if any(len(chunk) > max_chunk_length for chunk in chunks):
                return False
        return True

    # get number of cores
    num_cores = mp.cpu_count()
    print(f"Number of CPU cores available: {num_cores}")
    print(f"Dataset size before filtering: {len(dataset)}")
    dataset = dataset.map(
        lambda x: {"too_long": prefilter_long_chunks(x)}, num_proc=num_cores
    )
    dataset = dataset.filter(lambda x: x["too_long"])
    dataset = dataset.remove_columns("too_long")
    print(f"Dataset size after filtering: {len(dataset)}")

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
        output_path = os.path.join(translated_output_data_dir, "hermes_3.parquet")
        processed_dataset.to_parquet(output_path)
        print(f"Results saved to: {output_path}")
        print(f"Successfully processed {len(all_results)} examples")
    else:
        print("No results to save. Check for errors in worker processes.")


if __name__ == "__main__":
    # Set multiprocessing start method for compatibility
    mp.set_start_method("spawn", force=True)
    main()
