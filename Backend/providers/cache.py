"""Persistent JSON search cache; atomic replacement supports concurrent workers."""
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

from logstream import log


SEARCH_CACHE_TTL = 24 * 60 * 60


def cache_path(endpoint: str, params: dict) -> Path:
    public_params = {name: value for name, value in params.items() if name != "key"}
    identity = json.dumps([endpoint, public_params], sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    configured = os.getenv("MEDIA_CACHE_DIR", "").strip()
    directory = Path(configured).expanduser() if configured else Path.home() / ".cache" / "MoneyPrinterProMax" / "media"
    return directory / "pixabay" / f"{digest}.json"


def read_search_cache(path: Path) -> dict | None:
    try:
        for attempt in range(5):
            try:
                entry = json.loads(path.read_text(encoding="utf-8"))
                break
            except PermissionError:
                # Windows may briefly deny opens while another worker replaces the file.
                if attempt == 4:
                    raise
                time.sleep(0.01 * (attempt + 1))
        age = time.time() - float(entry["created_at"])
        payload = entry["response"]
        if 0 <= age < SEARCH_CACHE_TTL and isinstance(payload, dict) and isinstance(payload.get("hits"), list):
            return payload
    except FileNotFoundError:
        return None
    except (OSError, ValueError, TypeError, KeyError):
        log("[Media] Ignoring unreadable Pixabay search cache entry.", "warning")
    return None


def write_search_cache(path: Path, payload: dict) -> None:
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
            json.dump({"created_at": time.time(), "response": payload}, file)
            file.flush()
            os.fsync(file.fileno())
        for attempt in range(5):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                # Readers on Windows can temporarily prevent atomic replacement.
                if attempt == 4:
                    raise
                time.sleep(0.01 * (attempt + 1))
    except (OSError, ValueError, TypeError):
        log("[Media] Could not persist Pixabay search cache.", "warning")
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                log("[Media] Could not remove temporary Pixabay cache entry.", "warning")
