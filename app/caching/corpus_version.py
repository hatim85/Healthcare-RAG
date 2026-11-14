import json
import os
import re
import tempfile
from pathlib import Path

DEFAULT_VERSION = "v1"

VERSION_FILE = Path(
    os.getenv(
        "RAG_CORPUS_VERSION_FILE",
        "data/corpus_version.json",
    )
)


def _read_version() -> str:
    """Read the persisted corpus version."""

    if not VERSION_FILE.exists():
        return os.getenv("RAG_CORPUS_VERSION", DEFAULT_VERSION)

    try:
        with VERSION_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)

        version = data.get("version")

        if not isinstance(version, str) or not version.strip():
            raise ValueError("Invalid corpus version manifest.")

        return version

    except Exception as exc:
        raise RuntimeError(
            "Could not read corpus version manifest. "
            "Cache operations must not continue with an uncertain version."
        ) from exc


def get_corpus_version() -> str:
    """Return the current persisted corpus version."""
    return _read_version()


def bump_corpus_version() -> str:
    """
    Increment and persist the corpus version.

    Uses atomic file replacement to avoid partially written manifests.
    """

    current = _read_version()

    match = re.fullmatch(r"v(\d+)", current)

    if match:
        next_version = f"v{int(match.group(1)) + 1}"
    else:
        raise RuntimeError(
            f"Cannot automatically increment corpus version: {current}"
        )

    VERSION_FILE.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "version": next_version,
    }

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=VERSION_FILE.parent,
            prefix=".corpus-version-",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            json.dump(payload, temp_file, indent=2)
            temp_file.write("\n")
            temp_file.flush()
            os.fsync(temp_file.fileno())
            temp_path = Path(temp_file.name)

        os.replace(temp_path, VERSION_FILE)

    except Exception:
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise

    return next_version