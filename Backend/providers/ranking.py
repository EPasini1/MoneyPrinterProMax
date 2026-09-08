import re


SPACE_ENTITIES = {
    "sun", "mercury", "venus", "earth", "moon", "mars", "jupiter",
    "saturn", "uranus", "neptune", "pluto",
}


# Irregular Latin plurals that show up in trivia topics but don't fit the
# regular "-us"/"-uses" pattern handled in _normalize_token.
_IRREGULAR_PLURALS = {
    "octopi": "octopus", "cacti": "cactus", "fungi": "fungus", "nuclei": "nucleus",
}


def _normalize_token(word: str) -> str:
    """Small English plural normalization, shared by tokens and compound phrases."""
    if word in SPACE_ENTITIES | {"kubernetes", "series", "species", "physics", "gas", "analysis"}:
        return word
    if word in {"volcanoes", "potatoes", "tomatoes"}:
        return word[:-2]
    if word in _IRREGULAR_PLURALS:
        return _IRREGULAR_PLURALS[word]
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith(("ches", "shes", "sses", "xes", "zzes")):
        return word[:-2]
    # "-us" nouns pluralize as "-uses" (octopus/virus/walrus/campus/bus); an
    # already-singular "-us" word must not have its trailing "s" stripped.
    if word.endswith("uses"):
        return word[:-2]
    if word.endswith("us"):
        return word
    return word if word.endswith("ss") else word.removesuffix("s")


# Generic clickbait/framing words that never describe a visual subject. Shared
# with gpt.py so query generation and ranking cannot drift out of sync again
# (e.g. "weird" previously leaked through here and falsely anchored candidates).
GENERIC_FRAMING_WORDS = {
    "a", "an", "the", "of", "in", "on", "and", "with", "video", "videos",
    "footage", "stock", "view", "close", "up", "wide", "full", "details", "scene",
    "fact", "facts", "strange", "strangest", "interesting", "amazing", "surprising",
    "thing", "things", "about", "top", "best", "detail", "scenes", "etc",
    "for", "to", "is", "are", "what", "how", "why", "you", "should", "know",
    "incredible", "information",
    "weird", "weirdest", "odd", "bizarre", "crazy", "wild", "unusual",
    "mysterious", "shocking", "unbelievable",
    "work", "working", "actually", "explain", "explained", "guide",
    "tip", "trick", "tutorial", "learn", "learning", "easy", "simple",
    "way", "reason",
}


def _subject_tokens(text: str) -> set[str]:
    tokens = set()
    for word in re.findall(r"[^\W\d_]+", text.casefold()):
        if word in GENERIC_FRAMING_WORDS:
            continue
        # Preserve names ending in s, and filter again after singularization:
        # e.g. works -> work must not become a subject anchor.
        normalized = _normalize_token(word)
        if normalized and normalized not in GENERIC_FRAMING_WORDS:
            tokens.add(normalized)
    return tokens


# Curated aliases for locked subjects whose common synonym differs from the
# literal subject noun. Extend as new problem topics are discovered.
SUBJECT_ALIASES: dict[str, set[str]] = {
    "octopus": {"cephalopod"},
}

# Curated, narrow named visual sub-features of a locked subject (e.g. a named
# storm system or landmark). Deliberately NOT derived from DOMAIN_CONTEXT or
# arbitrary script text - each entry is a specific, visually identifiable
# phrase, kept small so it cannot weaken Subject Lock into a domain match.
SUBJECT_FEATURES: dict[str, set[str]] = {
    "jupiter": {"great red spot"},
}


def locked_subject_tokens(text: str) -> set[str]:
    """Canonical subject tokens, expanded to full curated alias groups.

    A subject phrased with only the alias (e.g. "cephalopod") must still
    resolve to the same locked group as the canonical word.
    """
    tokens = _subject_tokens(text)
    expanded = set(tokens)
    for canonical, aliases in SUBJECT_ALIASES.items():
        group = {canonical} | aliases
        if tokens & group:
            expanded |= group
    return expanded


# These components have common meanings outside their compound's domain. They
# remain query tokens, but cannot establish a strong subject anchor on their own.
AMBIGUOUS_COMPONENTS = {
    "plate", "chamber", "network", "intelligence", "artificial", "black", "hole", "shield", "ash",
}


def _candidate_is_gated(subject_lock_tokens: set[str]) -> bool:
    """Only curated/space subjects get the hard Candidate Lock; others stay soft."""
    return bool(subject_lock_tokens & (SPACE_ENTITIES | set(SUBJECT_ALIASES)))


def _has_curated_feature(subject_lock_tokens: set[str], text: str) -> bool:
    """A curated named sub-feature (e.g. Jupiter's Great Red Spot) is
    independently sufficient evidence, without weakening the lock to any
    script phrase or DOMAIN_CONTEXT match. Reused by both lock layers so a
    query built from a curated feature and a candidate described by it agree.
    """
    if not text:
        return False
    normalized = " ".join(re.findall(r"[^\W_]+", text.casefold()))
    for canonical in subject_lock_tokens:
        for feature in SUBJECT_FEATURES.get(canonical, ()):
            if re.search(r"\b" + re.escape(feature) + r"\b", normalized):
                return True
    return False


def _candidate_subject_lock(
    subject_lock_tokens: set[str], subject_compounds: set[tuple[str, str]],
    metadata_tokens: set[str], metadata_compounds: set[tuple[str, str]],
    feature_match: bool = False,
) -> bool:
    """A candidate's OWN metadata must prove the subject; its query is never consulted.

    Shared DOMAIN_CONTEXT domain is deliberately NOT a pass condition here: it
    remains a soft signal inside _query_context_evidence only. `feature_match`
    is computed once per candidate via _has_curated_feature() and threaded
    through every consumer, so there is a single canonical feature check.
    """
    if not subject_lock_tokens or not _candidate_is_gated(subject_lock_tokens):
        return True
    if (subject_lock_tokens & metadata_tokens) - AMBIGUOUS_COMPONENTS:
        return True
    if subject_compounds & metadata_compounds:
        return True
    return feature_match

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


def _query_context_evidence(query: str, subject: str, metadata: str, feature_match: bool = False) -> tuple[float, bool]:
    """Return coverage/context score and whether lexical evidence clears the guard.

    Quality/provider bonuses must never rescue a candidate with weak evidence.
    Shared distinctive domain terms allow related footage without exact wording.
    """
    query_tokens = _subject_tokens(query)
    subject_tokens = locked_subject_tokens(subject)
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
                     or bool(subject_compounds & metadata_compounds)
                     or feature_match)
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


def _subject_anchor_score(subject_tokens: set[str], metadata_tokens: set[str], feature_match: bool = False) -> float:
    """Weight the video's subject above query framing and technical quality.

    A curated feature (feature_match) proves the subject is present without
    the literal name, so it clears the usual penalties, but it is treated as
    neutral rather than a bonus: actual query/scene relevance still decides
    whether the candidate clears MEDIA_MIN_SCORE.
    """
    if not subject_tokens:
        return 0.0
    overlap = (subject_tokens & metadata_tokens) - AMBIGUOUS_COMPONENTS
    requested_entities = subject_tokens & SPACE_ENTITIES
    if overlap:
        score = 8.0 + 4.0 * len(overlap) / len(subject_tokens)
    elif feature_match:
        score = 0.0
    else:
        score = -3.0
    if requested_entities:
        if requested_entities & metadata_tokens:
            score += 10.0
        elif not feature_match:
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
