"""Provider routing and deterministic cross-provider media ranking."""
import os
import re

import requests  # Compatibility for callers mocking the legacy Pexels HTTP boundary.

from logstream import log
from providers.base import MediaCandidate, MediaProvider, media_min_score
from providers.nasa import NasaProvider
from providers.pexels import PexelsProvider, StockVideo, search_for_stock_videos
from providers.pixabay import PixabayProvider
from providers.ranking import (
    _candidate_subject_lock, _has_curated_feature, _nasa_program_penalty,
    _query_compounds, _subject_anchor_score, _subject_tokens,
    _query_context_evidence, _relevance_score, _title_evidence_score, locked_subject_tokens,
)
from utils import get_max_clip_duration


SPACE_PATTERN = re.compile(
    r"\b(?:space|planets?|galaxy|galaxies|universe|nasa|moon|mars|jupiter|saturn|"
    r"venus|mercury|neptune|uranus|sun|earth|pluto|asteroids?|comets?|telescopes?|astronomy|"
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
    subject_tokens = locked_subject_tokens(video_subject)
    subject_compounds = _query_compounds(video_subject)
    minimum_score = media_min_score()
    for item in candidates:
        metadata = "\n".join(filter(None, (item.title, item.description)))
        query_tokens = _subject_tokens(item.query)
        metadata_tokens = _subject_tokens(metadata)
        metadata_compounds = _query_compounds(metadata)
        # Computed once and reused everywhere a curated feature counts as evidence.
        feature_match = _has_curated_feature(subject_tokens, metadata)
        # Semantic relevance to the subject/query/scene - technical and provider
        # bonuses below are kept out of this total on purpose (see gate below).
        semantic_score = _relevance_score(query_tokens, metadata_tokens)
        semantic_score += _subject_anchor_score(subject_tokens, metadata_tokens, feature_match)
        semantic_score += _title_evidence_score(item.title or "", query_tokens, subject_tokens)
        context_score, sufficient_evidence = _query_context_evidence(item.query, video_subject, metadata, feature_match)
        semantic_score += context_score
        # Candidate Lock: metadata alone must prove the subject; the query it
        # was found under is never accepted as evidence (see providers/ranking.py).
        candidate_lock_ok = _candidate_subject_lock(subject_tokens, subject_compounds,
                                                    metadata_tokens, metadata_compounds, feature_match)
        if item.provider == "nasa" and item.media_type == "video":
            semantic_score -= _nasa_program_penalty(item.title or "")
        # Exact multiword names/entities provide stronger evidence than one generic word.
        phrase = " ".join(re.findall(r"\w+", item.query.casefold()))
        normalized = " ".join(re.findall(r"\w+", metadata.casefold()))
        if phrase and re.search(r"\b" + re.escape(phrase) + r"\b", normalized):
            semantic_score += 3

        # Technical/provider quality only ranks among semantically eligible
        # candidates; it must never be the reason one becomes selectable.
        technical_score = 0.0
        if item.width and item.height:
            actual = "portrait" if item.width < item.height else "landscape" if item.width > item.height else "square"
            technical_score += 2 if not orientation or actual == orientation else -2
            technical_score += 1 if min(item.width, item.height) >= 720 else -2
        if item.duration is not None:
            technical_score += 1 if item.duration >= get_max_clip_duration() else -1
        if (item.provider == "nasa" and is_space_topic(video_subject, item.query)
                and metadata_tokens & (subject_tokens | query_tokens)):
            technical_score += 5

        eligible = sufficient_evidence and candidate_lock_ok
        if eligible and feature_match:
            # A curated feature alone (no genuine token/compound/domain anchor)
            # must still clear real query/scene relevance, not just Candidate
            # Lock, before technical bonuses are allowed to count.
            _, sufficient_without_feature = _query_context_evidence(item.query, video_subject, metadata)
            locked_without_feature = _candidate_subject_lock(subject_tokens, subject_compounds,
                                                              metadata_tokens, metadata_compounds)
            feature_only_rescue = not (sufficient_without_feature and locked_without_feature)
            eligible = not feature_only_rescue or semantic_score >= minimum_score
        # Technical/provider bonuses cannot rescue a semantically ineligible candidate.
        item.score = semantic_score + technical_score if eligible else min(semantic_score, -4.0)
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
