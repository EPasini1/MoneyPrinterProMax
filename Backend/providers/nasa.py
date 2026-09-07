from urllib.parse import quote, urlparse

import requests

from logstream import log
from providers.base import MediaCandidate, image_fallback_enabled, timeout_setting, usable_url
from providers.ranking import _relevance_score, _subject_tokens


MAX_ASSET_RESOLUTIONS = 5


def _video_asset_rank(url: str) -> tuple[int, str]:
    name = urlparse(url).path.lower()
    # The library exposes named renditions rather than reliable pixel metadata.
    for priority, marker in enumerate(("1080", "720", "~medium", "~large", "~small", "~mobile")):
        if marker in name:
            return priority, name
    return 10, name


def _image_asset_rank(url: str, previews: list[str]) -> tuple[int, str]:
    name = urlparse(url).path.lower()
    if "thumb" in name:
        return 6, name
    if "~orig" in name or "master" in name:
        return 5, name
    if "~large" in name:
        return 0, name
    if "~medium" in name:
        return 1, name
    if "~small" in name:
        return 4, name
    return (2 if url in previews else 3), name


class NasaProvider:
    name = "nasa"

    def _get(self, path: str, params: dict | None = None) -> dict:
        response = requests.get("https://images-api.nasa.gov" + path, params=params,
                                timeout=timeout_setting("MEDIA_SEARCH_TIMEOUT", 30))
        response.raise_for_status()
        return response.json()

    def search(self, query: str, orientation: str, max_results: int,
               search_images: bool = False) -> list[MediaCandidate]:
        candidates = []
        if max_results <= 0 or (search_images and not image_fallback_enabled()):
            return candidates
        try:
            media_type = "image" if search_images else "image,video" if image_fallback_enabled() else "video"
            payload = self._get("/search", {"q": query, "media_type": media_type,
                                            "page_size": min(20, max(1, max_results))})
            ranked = []
            for item in payload.get("collection", {}).get("items", []):
                try:
                    data = item["data"][0]
                    if data["media_type"] not in media_type.split(",") or not data["nasa_id"]:
                        continue
                    metadata = " ".join(filter(None, (data.get("title"), data.get("description"))))
                    score = _relevance_score(_subject_tokens(query), _subject_tokens(metadata))
                    ranked.append((score, item))
                except (KeyError, IndexError, TypeError, ValueError, AttributeError):
                    log("[Media] Skipping malformed NASA candidate.", "warning")
            # Stable sorting retains API order on ties; metadata ranking needs no manifests.
            ranked.sort(key=lambda entry: -entry[0])
            resolved_ids = set()
            for _, item in ranked:
                source_id = str(item["data"][0]["nasa_id"])
                if source_id in resolved_ids:
                    continue
                if len(resolved_ids) >= min(MAX_ASSET_RESOLUTIONS, max_results):
                    break
                resolved_ids.add(source_id)
                try:
                    candidate = self._candidate(item, query)
                    if candidate:
                        candidates.append(candidate)
                except (KeyError, IndexError, TypeError, ValueError, AttributeError):
                    log("[Media] Skipping malformed NASA candidate.", "warning")
        except (requests.RequestException, ValueError, TypeError, AttributeError) as err:
            log(f"[Media] NASA search failed ({type(err).__name__}).", "warning")
        return candidates

    def _candidate(self, item: dict, query: str) -> MediaCandidate | None:
        data = item["data"][0]
        source_id, kind = str(data["nasa_id"]), data["media_type"]
        if kind not in {"image", "video"}:
            return None
        previews = [link["href"] for link in item.get("links", [])
                  if link.get("render") == "image" and usable_url(link.get("href"))]
        images = list(previews)
        url = None
        try:
            manifest = self._get("/asset/" + quote(source_id, safe=""))
            assets = [asset["href"] for asset in manifest.get("collection", {}).get("items", [])
                      if usable_url(asset.get("href"))]
            if kind == "video":
                videos = [asset for asset in assets if urlparse(asset).path.lower().endswith(".mp4")
                          and not any(word in urlparse(asset).path.lower() for word in ("thumb", "~orig", "master", "2160", "4k"))]
                if videos:
                    url = min(videos, key=_video_asset_rank)
            images.extend(asset for asset in assets if urlparse(asset).path.lower().endswith((".jpg", ".jpeg", ".png")))
        except (requests.RequestException, ValueError, TypeError, AttributeError) as err:
            log(f"[Media] NASA asset lookup failed ({type(err).__name__}); trying image preview.", "warning")
        if not url and images and image_fallback_enabled():
            kind, url = "image", min(images, key=lambda asset: _image_asset_rank(asset, previews))
        if not url:
            return None
        return MediaCandidate(self.name, kind, source_id, query, url,
                              "https://images.nasa.gov/details/" + quote(source_id, safe=""),
                              None, None, None, title=data.get("title"), description=data.get("description"))
