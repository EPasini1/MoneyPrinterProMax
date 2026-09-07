"""Provider routing and deterministic cross-provider media ranking."""
import os
import re

import requests  # Compatibility for callers mocking the legacy Pexels HTTP boundary.

from logstream import log
from providers.base import MediaCandidate, MediaProvider
from providers.nasa import NasaProvider
from providers.pexels import PexelsProvider, StockVideo, search_for_stock_videos
from providers.pixabay import PixabayProvider
from providers.ranking import _subject_tokens, _relevance_score
from utils import get_max_clip_duration


SPACE_PATTERN = re.compile(
    r"\b(?:space|planets?|galaxy|galaxies|universe|nasa|moon|mars|jupiter|saturn|"
    r"venus|mercury|neptune|uranus|asteroids?|comets?|telescopes?|astronomy|"
    r"solar\s+system|stars?|nebula|black\s+hole)\b", re.IGNORECASE,
)


def is_space_topic(subject: str, query: str = "") -> bool:
    return bool(SPACE_PATTERN.search(subject + " " + query))


def get_enabled_providers(video_subject: str = "", query: str = "") -> list[MediaProvider]:
    names = list(dict.fromkeys(name.strip().lower() for name in
                             os.getenv("MEDIA_PROVIDERS", "pexels,pixabay,nasa").split(",") if name.strip()))
    factories = {"pexels": PexelsProvider, "pixabay": PixabayProvider, "nasa": NasaProvider}
    for name in names:
        if name not in factories:
            log(f"[Media] Unknown provider {name!r}; ignored.", "warning")
    order = ("nasa", "pixabay", "pexels") if is_space_topic(video_subject, query) else ("pexels", "pixabay", "nasa")
    providers = []
    for name in order:
        if name in names:
            provider = factories[name]()
            if name == "nasa" or provider.api_key:
                providers.append(provider)
    if not providers:
        raise RuntimeError("No usable media providers. Enable NASA or configure a Pexels/Pixabay API key.")
    return providers


def rank_candidates(candidates: list[MediaCandidate], orientation: str, video_subject: str) -> list[MediaCandidate]:
    for item in candidates:
        metadata = " ".join(filter(None, (item.title, item.description)))
        query_tokens = _subject_tokens(item.query)
        score = _relevance_score(query_tokens, _subject_tokens(metadata))
        # Exact multiword names/entities provide stronger evidence than one generic word.
        phrase = " ".join(re.findall(r"\w+", item.query.casefold()))
        normalized = " ".join(re.findall(r"\w+", metadata.casefold()))
        if phrase and re.search(r"\b" + re.escape(phrase) + r"\b", normalized):
            score += 3
        if item.width and item.height:
            actual = "portrait" if item.width < item.height else "landscape" if item.width > item.height else "square"
            score += 2 if not orientation or actual == orientation else -2
            score += 1 if min(item.width, item.height) >= 720 else -2
        if item.duration is not None:
            score += 1 if item.duration >= get_max_clip_duration() else -1
        if item.provider == "nasa" and is_space_topic(video_subject, item.query):
            score += 2
        item.score = score
    return sorted(candidates, key=lambda item: -item.score)


def search_media_candidates(query: str, orientation: str = "", video_subject: str = "",
                            max_results: int = 20, search_images: bool = False) -> list[MediaCandidate]:
    if max_results <= 0:
        return []
    providers = get_enabled_providers(video_subject, query)
    log(f"[Media] Query: {query!r}", "info")
    candidates = []
    for provider in providers:
        try:
            results = provider.search(query, orientation, max_results, search_images=search_images)
            log(f"[Media] {provider.name.title()}: {len(results)} candidates", "info")
            candidates.extend(results)
        except Exception as err:
            log(f"[Media] {provider.name.title()} failed ({type(err).__name__}); trying remaining providers.", "warning")
    # Keep alternate renditions available after a failed download; selection deduplicates successes.
    return rank_candidates(candidates, orientation, video_subject)
