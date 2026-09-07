import math
from dataclasses import dataclass
from typing import List
from urllib.parse import unquote, urlparse

import requests

from logstream import log
from providers.base import MediaCandidate, provider_key, timeout_setting
from providers.ranking import _subject_tokens, _relevance_score, _rendition_rank
from utils import get_max_clip_duration


@dataclass(frozen=True)
class StockVideo:
    id: int
    url: str
    duration: float
    width: int
    height: int
    page_url: str
    relevance: float = 0.0


def search_for_stock_videos(
    query: str, api_key: str, it: int, min_dur: float, orientation: str = ""
) -> List[StockVideo]:
    """Return distinct sources ranked by page-slug relevance and duration.

    Missing overlap lowers ranking without excluding possible synonyms.
    API order breaks ties between equally relevant, equally suitable candidates.
    """
    api_key = provider_key("pexels", api_key)
    if not api_key:
        return []
    params = {"query": query, "per_page": it}
    if orientation:
        params["orientation"] = orientation
    try:
        response = requests.get(
            "https://api.pexels.com/videos/search",
            headers={"Authorization": api_key},
            params=params,
            timeout=timeout_setting("MEDIA_SEARCH_TIMEOUT", 30),
        )
        response.raise_for_status()
        payload = response.json()
        videos = payload.get("videos", []) if isinstance(payload, dict) else []
        if not isinstance(videos, list):
            raise ValueError("Pexels videos field is not a list.")
    except (requests.RequestException, ValueError) as err:
        log(f"[Media] Pexels search failed ({type(err).__name__}).", "warning")
        return []

    candidates = []
    seen_ids = set()
    seen_urls = set()
    query_tokens = _subject_tokens(query)
    for video in videos:
        try:
            video_id = int(video["id"])
            duration = float(video["duration"])
            width, height = int(video["width"]), int(video["height"])
            if video_id in seen_ids or not math.isfinite(duration) or duration <= 0:
                continue
            if width <= 0 or height <= 0:
                continue
            if orientation == "portrait" and width >= height:
                continue
            if orientation == "landscape" and width <= height:
                continue
            if orientation == "square" and width != height:
                continue

            page_url = str(video.get("url") or "")
            slug = unquote(urlparse(page_url).path.rstrip("/").rsplit("/", 1)[-1])
            metadata_tokens = _subject_tokens(slug)
            relevance = _relevance_score(query_tokens, metadata_tokens)

            # Infer a standard target from source orientation without changing the API.
            if width < height:
                target_width, target_height = 1080, 1920
            elif width > height:
                target_width, target_height = 1920, 1080
            else:
                target_width, target_height = 1080, 1080
            files = []
            for rendition in video.get("video_files", []):
                if not isinstance(rendition, dict):
                    continue
                link = rendition.get("link")
                if not isinstance(link, str) or urlparse(link).scheme not in {"http", "https"}:
                    continue
                if rendition.get("file_type") != "video/mp4":
                    continue
                file_width = int(rendition.get("width") or 0)
                file_height = int(rendition.get("height") or 0)
                if file_width > 0 and file_height > 0:
                    files.append((
                        _rendition_rank(file_width, file_height, target_width, target_height),
                        link, file_width, file_height,
                    ))
            if not files:
                continue
            _, link, file_width, file_height = min(files, key=lambda item: item[0])
            if link in seen_urls:
                continue
            candidates.append(StockVideo(
                video_id, link, duration, file_width, file_height, page_url, relevance,
            ))
            seen_ids.add(video_id)
            seen_urls.add(link)
        except (KeyError, TypeError, ValueError, AttributeError) as err:
            log("[Media] Skipping malformed Pexels candidate.", "warning")

    # Prefer corroborated subjects and long-enough clips; preserve API order on ties.
    candidates.sort(key=lambda item: (
        -item.relevance, item.duration < min_dur,
    ))
    log(f"\t=> {query!r} found {len(candidates)} usable videos", "info")
    return candidates


class PexelsProvider:
    name = "pexels"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = provider_key(self.name, api_key)

    def search(self, query: str, orientation: str, max_results: int,
               search_images: bool = False) -> list[MediaCandidate]:
        if search_images:
            return []
        videos = search_for_stock_videos(query, self.api_key, max_results, get_max_clip_duration(), orientation)
        return [MediaCandidate(
            self.name, "video", str(item.id), query, item.url, item.page_url,
            item.width, item.height, item.duration,
            title=unquote(urlparse(item.page_url).path.rstrip("/").rsplit("/", 1)[-1]).replace("-", " "),
            score=item.relevance,
        ) for item in videos]
