"""Phase 3.3 — Entity / Subject Lock: Query Lock (gpt.py) + Candidate Lock (ranking/search)."""
import pytest

import gpt
import search
from providers.base import MediaCandidate, media_min_score
from providers.ranking import (
    SPACE_ENTITIES, SUBJECT_ALIASES, _candidate_is_gated, _candidate_subject_lock,
    _query_compounds, _query_context_evidence, _subject_tokens, locked_subject_tokens,
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
    # No token or compound evidence at all fails.
    assert not _candidate_subject_lock(tokens, subject_compounds, {"jellyfish"}, set())
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
