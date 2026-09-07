import json
import math

import requests

from logstream import log
from providers.base import MediaCandidate, image_fallback_enabled, provider_key, timeout_setting, usable_url
from providers.cache import cache_path, read_search_cache, write_search_cache
from providers.ranking import _rendition_rank


class PixabayProvider:
    name = "pixabay"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = provider_key(self.name, api_key)

    def _search_response(self, endpoint: str, params: dict) -> dict:
        path = cache_path(endpoint, params)
        cached = read_search_cache(path)
        if cached is not None:
            return cached
        response = requests.get(endpoint, params={**params, "key": self.api_key},
                                timeout=timeout_setting("MEDIA_SEARCH_TIMEOUT", 30))
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("hits"), list):
            raise ValueError("Malformed Pixabay search response.")
        # Do not persist credentials even if an unexpected response echoes them.
        if self.api_key not in json.dumps(payload, ensure_ascii=False):
            write_search_cache(path, payload)
        return payload

    def search(self, query: str, orientation: str, max_results: int,
               search_images: bool = False) -> list[MediaCandidate]:
        if not self.api_key or (search_images and not image_fallback_enabled()):
            return []
        candidates = []
        kinds = ["image"] if search_images else ["video"]
        for kind in kinds:
            params = {"q": query, "per_page": min(20, max(3, max_results)),
                      "safesearch": "true", "order": "popular"}
            if kind == "image":
                params["image_type"] = "photo"
            endpoint = "https://pixabay.com/api/videos/" if kind == "video" else "https://pixabay.com/api/"
            try:
                hits = self._search_response(endpoint, params)["hits"]
                for hit in hits:
                    try:
                        candidate = self._candidate(hit, query, kind)
                        if candidate:
                            candidates.append(candidate)
                    except (KeyError, ValueError, TypeError, AttributeError):
                        log("[Media] Skipping malformed Pixabay candidate.", "warning")
            except (requests.RequestException, ValueError, TypeError, AttributeError) as err:
                # HTTP exceptions can contain the API key in their request URL.
                log(f"[Media] Pixabay {kind} search failed ({type(err).__name__}).", "warning")
        seen = set()
        unique = []
        for candidate in candidates:
            identity = (candidate.media_type, candidate.source_id)
            if identity not in seen:
                seen.add(identity)
                unique.append(candidate)
        return unique

    def _candidate(self, hit: dict, query: str, kind: str) -> MediaCandidate | None:
        duration = None
        if kind == "video":
            files = [item for item in hit.get("videos", {}).values()
                     if usable_url(item.get("url")) and int(item.get("width", 0)) > 0 and int(item.get("height", 0)) > 0]
            if not files:
                return None
            def rank(item: dict) -> tuple:
                width, height = int(item["width"]), int(item["height"])
                target = (1080, 1920) if width < height else (1920, 1080)
                return _rendition_rank(width, height, *target)
            rendition = min(files, key=rank)
            url, width, height = rendition["url"], int(rendition["width"]), int(rendition["height"])
            if hit.get("duration") is not None:
                duration = float(hit["duration"])
                if not math.isfinite(duration) or duration <= 0:
                    return None
        else:
            url = hit.get("largeImageURL") or hit.get("webformatURL")
            width, height = int(hit.get("imageWidth") or 0), int(hit.get("imageHeight") or 0)
            # largeImageURL is capped at 1280px; do not claim original dimensions.
            if width and height and hit.get("largeImageURL"):
                scale = min(1, 1280 / max(width, height))
                width, height = round(width * scale), round(height * scale)
            elif not hit.get("largeImageURL"):
                width, height = hit.get("webformatWidth"), hit.get("webformatHeight")
        if not usable_url(url):
            return None
        return MediaCandidate(self.name, kind, str(hit["id"]), query, url,
                              hit.get("pageURL"), width or None, height or None, duration,
                              title=hit.get("tags"))
