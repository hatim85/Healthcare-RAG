import sys
from pathlib import Path

# Add project root to Python import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

from app.ingestion.config import MAX_WORKERS
from app.ingestion.pdf_worker import process_pdf


def main():
    pdf_path = Path("data/raw/who_type2_diabetes_2020.pdf").resolve()

    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)

    print("Starting process-pool smoke test...")
    print("Start method: spawn")
    print("Configured workers:", MAX_WORKERS)

    context = get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
        mp_context=context,
    ) as executor:
        future = executor.submit(process_pdf, str(pdf_path))
        result = future.result(timeout=180)

    print("Worker status:", result["status"])
    print("Worker process completed:", result["path"])
    print("Chunks generated:", result["chunk_count"])

    if result["status"] != "success":
        raise RuntimeError(result.get("error", "Worker failed."))

    print("Multiprocessing smoke test: PASSED")


if __name__ == "__main__":
    main()