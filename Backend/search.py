import math
import re
from dataclasses import dataclass
from typing import List
from urllib.parse import unquote, urlparse

import requests

from logstream import log


@dataclass(frozen=True)
class StockVideo:
    id: int
    url: str
    duration: float
    width: int
    height: int
    page_url: str
    relevance: float = 0.0


def _subject_tokens(text: str) -> set[str]:
    ignored = {
        "a", "an", "the", "of", "in", "on", "and", "with", "video", "videos",
        "footage", "stock", "view", "close", "up", "wide", "full", "details", "scene",
    }
    return {
        word.removesuffix("s")
        for word in re.findall(r"[^\W\d_]+", text.casefold())
        if word not in ignored
    }


def _relevance_score(query_tokens: set[str], metadata_tokens: set[str]) -> float:
    overlap = query_tokens & metadata_tokens
    if overlap:
        return 4.0 + 2.0 * len(overlap) / len(query_tokens)
    if not metadata_tokens or not query_tokens:
        return 0.0

    # Missing lexical evidence may just mean synonyms. Penalize, never reject.
    score = -1.0
    suspicious_categories = (
        "food rice bowl cooking kitchen chef meal restaurant",
        "office business meeting desk paperwork corporate",
        "people person man woman crowd portrait",
        "abstract background pattern texture",
    )
    for category in suspicious_categories:
        tokens = _subject_tokens(category)
        if metadata_tokens & tokens and not query_tokens & tokens:
            score -= 3.0
    return score


def _rendition_rank(
    width: int, height: int, target_width: int, target_height: int
) -> tuple[int, float, int]:
    """Favor crop-ready 1080p, then 720p, before oversized or tiny files."""
    pixels = width * height
    scale = min(width / target_width, height / target_height)
    reasonable_size = pixels <= target_width * target_height * 1.5
    if scale >= 1 and reasonable_size:
        return (0, scale, pixels)
    if scale >= 2 / 3 and reasonable_size:
        return (1, -scale, pixels)
    if scale >= 2 / 3:
        return (2, pixels, pixels)
    return (3, -scale, pixels)


def search_for_stock_videos(
    query: str, api_key: str, it: int, min_dur: float, orientation: str = ""
) -> List[StockVideo]:
    """Return distinct sources ranked by page-slug relevance and duration.

    Missing overlap lowers ranking without excluding possible synonyms.
    API order breaks ties between equally relevant, equally suitable candidates.
    """
    params = {"query": query, "per_page": it}
    if orientation:
        params["orientation"] = orientation
    try:
        response = requests.get(
            "https://api.pexels.com/videos/search",
            headers={"Authorization": api_key},
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        videos = payload.get("videos", []) if isinstance(payload, dict) else []
        if not isinstance(videos, list):
            raise ValueError("Pexels videos field is not a list.")
    except (requests.RequestException, ValueError) as err:
        log(f"[!] Pexels search failed for {query!r}: {err}", "warning")
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
                        link,
                    ))
            if not files:
                continue
            _, link = min(files, key=lambda item: item[0])
            if link in seen_urls:
                continue
            candidates.append(StockVideo(
                video_id, link, duration, width, height, page_url, relevance,
            ))
            seen_ids.add(video_id)
            seen_urls.add(link)
        except (KeyError, TypeError, ValueError, AttributeError) as err:
            log(f"[!] Skipping malformed Pexels candidate for {query!r}: {err}", "warning")

    # Prefer corroborated subjects and long-enough clips; preserve API order on ties.
    candidates.sort(key=lambda item: (
        -item.relevance, item.duration < min_dur,
    ))
    log(f"\t=> {query!r} found {len(candidates)} usable videos", "info")
    return candidates
