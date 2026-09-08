"""Phase 3.3 — Entity / Subject Lock: Query Lock (gpt.py) + Candidate Lock (ranking/search)."""
import pytest

import gpt
import search
from providers.base import MediaCandidate, media_min_score
from providers.ranking import (
    SPACE_ENTITIES, SUBJECT_ALIASES, SUBJECT_FEATURES, _candidate_is_gated,
    _candidate_subject_lock, _has_curated_feature, _query_compounds,
    _query_context_evidence, _subject_tokens, locked_subject_tokens,
)


def media(source_id: str, title: str | None = "Octopus", description: str | None = None,
          score: float = 5, kind: str = "video", provider: str = "pexels",
          query: str = "octopus") -> MediaCandidate:
    return MediaCandidate(provider, kind, source_id, query, f"https://example.com/{source_id}",
                          None, 1080, 1920, 10 if kind == "video" else None,
                          title=title, description=description, score=score)


# --- Bug fixes: pluralization + generic framing words -----------------------

@pytest.mark.parametrize("singular, plural", [
    ("octopus", "octopuses"), ("virus", "viruses"), ("walrus", "walruses"), ("cactus", "cactuses"),
])
def test_us_noun_pluralization_normalizes_consistently(singular: str, plural: str) -> None:
    assert _subject_tokens(singular) == _subject_tokens(plural) == {singular}


@pytest.mark.parametrize("irregular, singular", [
    ("octopi", "octopus"), ("cacti", "cactus"), ("fungi", "fungus"),
])
def test_irregular_plural_normalizes_to_singular(irregular: str, singular: str) -> None:
    assert _subject_tokens(irregular) == {singular}


def test_weird_is_filtered_as_generic_framing_word() -> None:
    assert _subject_tokens("3 Weird Facts About Octopuses") == {"octopus"}


# --- locked_subject_tokens: curated alias expansion --------------------------

def test_locked_subject_tokens_expands_curated_alias_bidirectionally() -> None:
    assert locked_subject_tokens("Octopus facts") == {"octopus", "cephalopod"}
    assert locked_subject_tokens("Cephalopod facts") == {"octopus", "cephalopod"}


# --- Candidate Lock: query must never satisfy the lock -----------------------

def test_query_cannot_rescue_candidate_lacking_subject_metadata() -> None:
    subject = "3 Weird Facts About Octopuses"
    query = "octopus red heart necklace"
    # The pre-existing per-query overlap guard alone WOULD clear this candidate
    # (3 shared non-subject words), proving the loophole Candidate Lock closes.
    _, sufficient = _query_context_evidence(query, subject, "red heart necklace jewelry gift")
    assert sufficient

    unrelated = media("necklace", title="Red heart necklace", description="jewelry gift", query=query)
    genuine = media("real", title="Giant Pacific octopus swimming underwater", query=query)
    assert search.rank_candidates([unrelated, genuine], "portrait", subject) == [genuine, unrelated]
    assert unrelated.score < media_min_score()
    assert genuine.score >= media_min_score()


@pytest.mark.parametrize("title", [
    "Giant Pacific octopus swimming underwater", "Octopus crawling reef",
    "Cephalopod camouflage underwater",
])
def test_alias_aware_query_context_evidence_accepts_curated_alias(title: str) -> None:
    # The alias must be recognized both by Candidate Lock and by the
    # pre-existing _query_context_evidence guard (single source of truth).
    candidate = media("anchored", title=title, query="octopus giant pacific")
    search.rank_candidates([candidate], "portrait", "3 Weird Facts About Octopuses")
    assert candidate.score > media_min_score()


@pytest.mark.parametrize("title", [
    "Jellyfish swimming aquarium", "Fish swimming underwater",
    "Coral reef ocean", "Generic marine life",
])
def test_shared_domain_alone_does_not_satisfy_candidate_lock_for_octopus(title: str) -> None:
    candidate = media("marine", title=title, query="octopus habitat")
    search.rank_candidates([candidate], "portrait", "3 Weird Facts About Octopuses")
    assert candidate.score < media_min_score()


def test_shared_domain_alone_does_not_satisfy_candidate_lock_for_jupiter() -> None:
    candidate = media("stars", title="Stars in space", query="Jupiter atmosphere", provider="nasa")
    search.rank_candidates([candidate], "portrait", "Jupiter")
    assert candidate.score < media_min_score()


@pytest.mark.parametrize("provider", ["pexels", "nasa", "pixabay"])
@pytest.mark.parametrize("title", ["Great Red Spot", "Great Red Spot storm"])
def test_curated_subject_feature_satisfies_jupiter_candidate_lock(title: str, provider: str) -> None:
    # No literal "Jupiter" anywhere in the metadata - only the curated feature.
    # An exact feature-query match must clear the threshold on ANY provider;
    # provider bonuses may rank results but must not decide pass/fail.
    candidate = media("feature", title=title, query="Jupiter Great Red Spot", provider=provider)
    search.rank_candidates([candidate], "portrait", "Jupiter")
    assert candidate.score > media_min_score()


@pytest.mark.parametrize("title", [
    "Stars in space", "Galaxy background", "Generic planet", "Space telescope", "Saturn rings",
])
def test_curated_subject_feature_does_not_rescue_unrelated_jupiter_candidates(title: str) -> None:
    candidate = media("unrelated", title=title, query="Jupiter Great Red Spot", provider="nasa")
    search.rank_candidates([candidate], "portrait", "Jupiter")
    assert candidate.score < media_min_score()


def test_subject_features_registry_is_curated_and_narrow() -> None:
    assert SUBJECT_FEATURES["jupiter"] == {"great red spot"}
    assert _has_curated_feature({"jupiter"}, "A view of the Great Red Spot storm")
    assert not _has_curated_feature({"jupiter"}, "A view of distant stars in space")


# --- Scenario A-E: curated feature proves the subject, but query/scene ------
# --- relevance is a separate, still-enforced concern (item 6) --------------

@pytest.mark.parametrize("provider", ["pexels", "nasa"])
def test_scenario_ab_exact_feature_query_clears_threshold_on_any_provider(provider: str) -> None:
    candidate = media("exact", title="Great Red Spot", query="Great Red Spot", provider=provider)
    search.rank_candidates([candidate], "portrait", "Jupiter")
    assert candidate.score > media_min_score()


def test_scenario_c_feature_query_with_extra_context_is_valid() -> None:
    candidate = media("context", title="Great Red Spot storm",
                      query="Jupiter atmosphere Great Red Spot")
    search.rank_candidates([candidate], "portrait", "Jupiter")
    assert candidate.score > media_min_score()


def test_scenario_d_curated_feature_does_not_force_unrelated_query_above_threshold() -> None:
    tokens = locked_subject_tokens("Jupiter")
    subject_compounds = _query_compounds("Jupiter")
    metadata_tokens = _subject_tokens("Great Red Spot")
    feature_match = _has_curated_feature(tokens, "Great Red Spot")
    # Candidate Lock passes purely on the curated feature ...
    assert _candidate_subject_lock(tokens, subject_compounds, metadata_tokens, set(), feature_match)
    # ... but an unrelated query must not be rescued to the same strength as
    # an exact feature-query match: real query/scene relevance still matters.
    exact = media("exact", title="Great Red Spot", query="Great Red Spot")
    mismatched = MediaCandidate("pexels", "video", "mismatched", "Jupiter core hydrogen interior",
                                "https://example.com/mismatched", None, None, None, None,
                                title="Great Red Spot")
    search.rank_candidates([exact, mismatched], "portrait", "Jupiter")
    assert mismatched.score < media_min_score()
    assert exact.score > mismatched.score + 10


def test_technical_quality_alone_cannot_rescue_semantically_weak_candidate() -> None:
    """Same weak semantic match (Jupiter subject, mismatched scene query); only
    technical properties differ. Neither low nor ideal quality may cross
    MEDIA_MIN_SCORE - only genuine query/scene relevance does."""
    subject, weak_query = "Jupiter", "Jupiter core hydrogen interior"
    low_quality = MediaCandidate("pexels", "video", "low", weak_query, "https://example.com/low",
                                 None, 480, 270, 1, title="Great Red Spot")
    ideal_quality = MediaCandidate("nasa", "video", "ideal", weak_query, "https://example.com/ideal",
                                   None, 1080, 1920, 10, title="Great Red Spot")
    search.rank_candidates([low_quality, ideal_quality], "portrait", subject)
    assert low_quality.score < media_min_score()
    assert ideal_quality.score < media_min_score()

    # Same ideal technical properties, but a genuinely relevant query/scene.
    relevant = MediaCandidate("nasa", "video", "relevant", "Great Red Spot",
                              "https://example.com/relevant", None, 1080, 1920, 10, title="Great Red Spot")
    search.rank_candidates([relevant], "portrait", subject)
    assert relevant.score >= media_min_score()


@pytest.mark.parametrize("title", ["Stars in space", "Space telescope", "Saturn rings"])
def test_scenario_e_unrelated_jupiter_candidates_fail_candidate_lock(title: str) -> None:
    tokens = locked_subject_tokens("Jupiter")
    subject_compounds = _query_compounds("Jupiter")
    metadata_tokens = _subject_tokens(title)
    feature_match = _has_curated_feature(tokens, title)
    assert not feature_match
    assert not _candidate_subject_lock(tokens, subject_compounds, metadata_tokens, set(), feature_match)


def test_sparse_metadata_fails_closed_for_gated_subject() -> None:
    candidate = media("empty", title=None, description=None, query="octopus tentacles")
    search.rank_candidates([candidate], "portrait", "Octopus facts")
    assert candidate.score < media_min_score()


@pytest.mark.parametrize("subject, expected", [
    ("Octopus facts", True), ("Cephalopod facts", True), ("Jupiter", True),
    ("Interesting facts about a puppy", False), ("3 Strange Facts About Volcanoes", False),
    ("How AI works", False),
])
def test_candidate_lock_activation_is_scoped_to_curated_entries(subject: str, expected: bool) -> None:
    assert _candidate_is_gated(locked_subject_tokens(subject)) is expected


def test_candidate_subject_lock_direct_and_compound_evidence() -> None:
    tokens = locked_subject_tokens("Octopus facts")
    subject_compounds = _query_compounds("Octopus facts")
    # Direct alias-token overlap passes even without a literal "octopus".
    assert _candidate_subject_lock(tokens, subject_compounds, {"cephalopod", "ink"}, set())
    # No token, compound, or curated-feature evidence at all fails.
    assert not _candidate_subject_lock(tokens, subject_compounds, {"jellyfish"}, set())
    # A curated feature match alone (no direct token) also passes.
    jupiter_tokens = locked_subject_tokens("Jupiter")
    assert _candidate_subject_lock(jupiter_tokens, set(), {"great", "red", "spot"}, set(),
                                   feature_match=True)
    # An ungated subject (no curated entry) always passes.
    assert _candidate_subject_lock(locked_subject_tokens("puppy"), set(), {"dog"}, set())


# --- Existing regressions stay governed by the untouched soft path ----------

def test_dog_puppy_synonym_softness_is_unaffected(monkeypatch: pytest.MonkeyPatch) -> None:
    matching = media("match", "Puppy playing", kind="image", query="dog playing")
    synonym = media("synonym", "Dog playing", query="dog playing")
    ranked = search.rank_candidates([synonym, matching], "portrait", "Interesting facts about a puppy")
    assert ranked[0] is matching
    assert synonym.score >= 3


def test_volcano_bread_regression_is_unaffected() -> None:
    subject = "3 Strange Facts About Volcanoes"
    kitchen = media("254753", "Woman baking bread in kitchen", provider="pixabay",
                    query="tectonic plate boundaries")
    kitchen.description = "Plate, baking tray, bread, dough, kitchen, cooking, bakery"
    geology = media("tectonics", "Tectonic plates and volcanic geology", query="tectonic plate boundaries")
    geology.description = "Plate boundary animation showing tectonic plate boundaries"
    assert search.rank_candidates([kitchen, geology], "portrait", subject) == [geology, kitchen]
    assert kitchen.score < media_min_score()
    assert geology.score >= media_min_score()


# --- Query Lock: get_search_terms() hard rejection ---------------------------

def test_get_search_terms_hard_rejects_off_subject_terms(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = iter([
        '["octopus tentacles close up", "laboratory beaker chemistry", '
        '"heart necklace jewelry", "octopus camouflage skin texture"]',
        '["octopus underwater habitat", "giant pacific octopus swimming"]',
    ])
    monkeypatch.setattr(gpt, "generate_response", lambda prompt, model: next(responses))
    terms = gpt.get_search_terms("Octopus facts", 4, "The octopus has three hearts.", "test")
    assert len(terms) == 4
    assert all("octopus" in term.casefold() for term in terms)
    assert len(set(term.casefold() for term in terms)) == 4


def test_get_search_terms_skips_anchor_check_for_abstract_subject(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpt, "generate_response", lambda prompt, model:
                        '["calm morning routine", "daily time management tips"]')
    terms = gpt.get_search_terms("Interesting Amazing Surprising Things", 2, "script", "test")
    assert len(terms) == 2


def test_fallback_search_terms_remain_subject_anchored_for_octopus() -> None:
    terms = gpt._fallback_search_terms("3 Weird Facts About Octopuses", "The octopus has three hearts.")
    assert terms
    assert all("octopus" in term.casefold() for term in terms)


def test_query_lock_accepts_curated_feature_consistent_with_candidate_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    # No script grounding for "Great Red Spot" here - only the curated registry
    # should admit it, matching what Candidate Lock independently accepts.
    monkeypatch.setattr(gpt, "generate_response", lambda prompt, model:
                        '["Great Red Spot", "Jupiter clouds close up"]')
    terms = gpt.get_search_terms("Jupiter", 2, "Jupiter is a gas giant.", "test")
    assert "Great Red Spot" in terms
    assert len(terms) == 2
