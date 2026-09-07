import re


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
