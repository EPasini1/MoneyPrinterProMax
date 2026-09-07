import math
import os
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlparse

from logstream import log


@dataclass
class MediaCandidate:
    provider: str
    media_type: str
    source_id: str
    query: str
    download_url: str
    page_url: str | None
    width: int | None
    height: int | None
    duration: float | None
    title: str | None = None
    description: str | None = None
    score: float = 0.0


class MediaProvider(Protocol):
    name: str

    def search(self, query: str, orientation: str, max_results: int,
               search_images: bool = False) -> list[MediaCandidate]: ...


def timeout_setting(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
        if math.isfinite(value) and value > 0:
            return value
    except ValueError:
        pass
    log(f"[Media] Invalid {name}; using {default} seconds.", "warning")
    return default


def image_fallback_enabled() -> bool:
    return os.getenv("ENABLE_IMAGE_FALLBACK", "true").strip().lower() in {"true", "1", "yes", "on"}


def max_image_clips() -> int:
    try:
        value = int(os.getenv("MAX_IMAGE_CLIPS", "4"))
        if value >= 0:
            return value
    except ValueError:
        pass
    log("[Media] Invalid MAX_IMAGE_CLIPS; using 4.", "warning")
    return 4


def usable_url(value: object) -> bool:
    return isinstance(value, str) and urlparse(value).scheme in {"http", "https"} and bool(urlparse(value).netloc)


_disabled_logged: set[str] = set()


def provider_key(name: str, explicit: str | None = None) -> str:
    key = (os.getenv(f"{name.upper()}_API_KEY", "") if explicit is None else explicit).strip()
    if not key and name not in _disabled_logged:
        log(f"[Media] {name.title()} disabled: API key is missing.", "warning")
        _disabled_logged.add(name)
    return key
