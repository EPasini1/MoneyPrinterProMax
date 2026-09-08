import re


SPACE_ENTITIES = {
    "sun", "mercury", "venus", "earth", "moon", "mars", "jupiter",
    "saturn", "uranus", "neptune", "pluto",
}


def _normalize_token(word: str) -> str:
    """Small English plural normalization, shared by tokens and compound phrases."""
    if word in SPACE_ENTITIES | {"kubernetes", "series", "species", "physics", "gas", "analysis"}:
        return word
    if word in {"volcanoes", "potatoes", "tomatoes"}:
        return word[:-2]
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith(("ches", "shes", "sses", "xes", "zzes")):
        return word[:-2]
    return word if word.endswith("ss") else word.removesuffix("s")


def _subject_tokens(text: str) -> set[str]:
    ignored = {
        "a", "an", "the", "of", "in", "on", "and", "with", "video", "videos",
        "footage", "stock", "view", "close", "up", "wide", "full", "details", "scene",
        "fact", "facts", "strange", "interesting", "amazing", "surprising",
        "thing", "things", "about", "top", "best", "detail", "scenes", "etc",
        "for", "to", "is", "are", "what", "how", "why", "you", "should", "know",
        "incredible", "information",
        "work", "working", "actually", "explain", "explained", "guide",
        "tip", "trick", "tutorial", "learn", "learning", "easy", "simple",
        "way", "reason",
    }
    tokens = set()
    for word in re.findall(r"[^\W\d_]+", text.casefold()):
        if word in ignored:
            continue
        # Preserve names ending in s, and filter again after singularization:
        # e.g. works -> work must not become a subject anchor.
        normalized = _normalize_token(word)
        if normalized and normalized not in ignored:
            tokens.add(normalized)
    return tokens


# These components have common meanings outside their compound's domain. They
# remain query tokens, but cannot establish a strong subject anchor on their own.
AMBIGUOUS_COMPONENTS = {
    "plate", "chamber", "network", "intelligence", "artificial", "black", "hole", "shield", "ash",
}

# Extensible domain evidence, rather than a subject-specific exclusion list.
# Ambiguous components are deliberately absent from these distinctive markers.
DOMAIN_CONTEXT = {name: _subject_tokens(words) for name, words in {
    "geology": "volcano volcanic eruption lava magma crater tectonic geology geological seismic earthquake",
    "computing": "ai neural algorithm computing software python docker kubernetes database linux javascript",
    "astronomy": "astronomy astronomical galaxy nebula telescope jupiter saturn mars venus neptune uranus pluto lunar solar io europa",
    "cooking": "bread baking bakery kitchen cooking dough food rice chef meal restaurant",
    "business": "office business meeting desk paperwork corporate commerce marketing",
    "music": "music musical musician concert orchestra piano guitar singer",
    "military": "military army soldier battlefield weapon warfare espionage",
}.items()}
DOMAIN_COMPOUNDS = {
    "astronomy": {("black", "hole")},
    "computing": {("artificial", "intelligence")},
}


def _query_compounds(text: str) -> set[tuple[str, str]]:
    """Adjacent meaningful words; punctuation/field boundaries break phrases."""
    compounds = set()
    for segment in re.split(r"[^\w\s-]|[\r\n]", text.casefold()):
        words = re.findall(r"[^\W\d_]+", segment)
        for left, right in zip(words, words[1:]):
            if _subject_tokens(left) and _subject_tokens(right):
                compounds.add((_normalize_token(left), _normalize_token(right)))
    return compounds


def _context_domains(tokens: set[str], compounds: set[tuple[str, str]]) -> set[str]:
    return {name for name, words in DOMAIN_CONTEXT.items()
            if words & tokens or compounds & DOMAIN_COMPOUNDS.get(name, set())}


def _query_context_evidence(query: str, subject: str, metadata: str) -> tuple[float, bool]:
    """Return coverage/context score and whether lexical evidence clears the guard.

    Quality/provider bonuses must never rescue a candidate with weak evidence.
    Shared distinctive domain terms allow related footage without exact wording.
    """
    query_tokens = _subject_tokens(query)
    subject_tokens = _subject_tokens(subject)
    metadata_tokens = _subject_tokens(metadata)
    overlap = query_tokens & metadata_tokens
    coverage = len(overlap) / len(query_tokens) if query_tokens else 0.0
    query_compounds = _query_compounds(query)
    subject_compounds = _query_compounds(subject)
    metadata_compounds = _query_compounds(metadata)
    compound_matches = query_compounds & metadata_compounds
    expected_domains = _context_domains(query_tokens | subject_tokens, query_compounds | subject_compounds)
    candidate_domains = _context_domains(metadata_tokens, metadata_compounds)
    shared_domain = bool(expected_domains & candidate_domains)
    direct_anchor = (bool((subject_tokens & metadata_tokens) - AMBIGUOUS_COMPONENTS)
                     or bool(subject_compounds & metadata_compounds))
    strong_anchor = direct_anchor or shared_domain

    score = 6.0 * coverage + min(12.0, 6.0 * len(compound_matches))
    # Generic query phrases such as "atmosphere storm" must not cancel an
    # explicit subject-entity penalty (e.g. terrestrial storms for Jupiter).
    if subject_tokens & SPACE_ENTITIES and not strong_anchor:
        score = 0.0
    if shared_domain and not direct_anchor:
        score += 8.0
    # Two distinctive terms from an unrelated domain are stronger evidence than
    # one incidental word. Mixed-domain topics can legitimately share either.
    if expected_domains and not shared_domain and any(
        len(DOMAIN_CONTEXT[i] & metadata_tokens) >= 2 for i in candidate_domains
    ):
        score -= 12.0
    if len(query_tokens) >= 3 and len(overlap) < 2 and not strong_anchor:
        score -= 8.0 * (1.0 - coverage)

    required_matches = min(2, len(query_tokens))
    enough_coverage = bool(query_tokens) and len(overlap) >= required_matches
    if len(query_tokens) > 1 and not overlap - AMBIGUOUS_COMPONENTS and not compound_matches:
        enough_coverage = False
    return score, enough_coverage or strong_anchor


def _subject_anchor_score(subject_tokens: set[str], metadata_tokens: set[str]) -> float:
    """Weight the video's subject above query framing and technical quality."""
    if not subject_tokens:
        return 0.0
    overlap = (subject_tokens & metadata_tokens) - AMBIGUOUS_COMPONENTS
    requested_entities = subject_tokens & SPACE_ENTITIES
    score = 8.0 + 4.0 * len(overlap) / len(subject_tokens) if overlap else -3.0
    if requested_entities:
        if requested_entities & metadata_tokens:
            score += 10.0
        else:
            score -= 10.0
            if metadata_tokens & SPACE_ENTITIES:
                score -= 16.0
    return score


def _title_evidence_score(title: str, query_tokens: set[str], subject_tokens: set[str]) -> float:
    """Direct title evidence distinguishes subjects merely mentioned in descriptions."""
    tokens = _subject_tokens(title)
    score = 6.0 * len(tokens & query_tokens) / len(query_tokens) if query_tokens else 0.0
    if subject_tokens:
        score += 4.0 * len(tokens & subject_tokens) / len(subject_tokens)
    return score


def _nasa_program_penalty(title: str) -> float:
    """Program titles suggest presenters/title cards; keep them as softer fallbacks."""
    normalized = " ".join(re.findall(r"\w+", title.casefold()))
    patterns = (r"live event", r"starting soon", r"science live", r"planetary protection",
                r"interviews?", r"podcasts?", r"briefings?", r"press conference", r"episodes?")
    matches = sum(bool(re.search(r"\b" + pattern + r"\b", normalized)) for pattern in patterns)
    return min(20.0, 12.0 + 4.0 * (matches - 1)) if matches else 0.0


def _relevance_score(query_tokens: set[str], metadata_tokens: set[str]) -> float:
    overlap = query_tokens & metadata_tokens
    if not metadata_tokens or not query_tokens:
        return 0.0

    # Missing lexical evidence may just mean synonyms. Penalize, never reject.
    score = 4.0 + 2.0 * len(overlap) / len(query_tokens) if overlap else -1.0
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
