import os
from dotenv import load_dotenv

load_dotenv()
load_dotenv('.env.example')

def _positive_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(
            f"{name} must be a valid integer."
        ) from exc

    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")

    return value


MAX_WORKERS = _positive_int("INGESTION_MAX_WORKERS", 3)
BATCH_SIZE = _positive_int("INGESTION_BATCH_SIZE", 32)
MAX_RETRIES = _positive_int("INGESTION_MAX_RETRIES", 3)
RETRY_DELAY_SECONDS = _positive_int(
    "INGESTION_RETRY_DELAY_SECONDS", 2
)