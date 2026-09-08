from unittest.mock import Mock

import pytest

import gpt
import pipeline
import search
from fact_sources import FactSource
from providers.base import MediaCandidate, media_min_score
from providers.ranking import (
    SPACE_ENTITIES, _query_compounds, _query_context_evidence,
    _subject_anchor_score, _subject_tokens,
)


@pytest.fixture(autouse=True)
def mocked_references(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gpt, "retrieve_fact_sources", Mock(return_value=[
        FactSource("NASA", "Jupiter Facts", "https://science.nasa.gov/jupiter/jupiter-facts/",
                   "Jupiter has cloud bands. The cause remains unknown to scientists."),
    ]))


def media(source_id: str, title: str = "Jupiter", score: float = 5,
          kind: str = "video", provider: str = "pexels") -> MediaCandidate:
    return MediaCandidate(provider, kind, source_id, "Jupiter atmosphere storm",
                          f"https://example.com/{source_id}", None, 1080, 1920,
                          10 if kind == "video" else None, title=title, score=score)


@pytest.mark.parametrize("notes", [
    "CLAIMS:", "CLAIM 1: Jupiter is solid", "VERDICT: FALSE", "FACT CHECK:",
    "FACT-CHECK: Review", "MISLEADING:", "EXAGGERATION:", "UNKNOWN:",
    "TRUE:", "FALSE:", "UNKNOWN", "## CLAIMS", "- **VERDICT:** False",
    "1. **CLAIM 1:** text", "> - **MISLEADING**: text", "| Claim | Verdict |",
    "Narration.\n### FACT-CHECK\n- UNKNOWN: uncertain",
])
def test_detects_structural_notes(notes: str) -> None:
    assert gpt._looks_like_fact_check_notes(notes)


@pytest.mark.parametrize("narration", [
    "The cause remains unknown to scientists.", "Unknown forces may play a role.",
    "It is true that Jupiter has clouds.", "True colors emerge in sunlight.",
    "Claims about Jupiter need careful study.",
])
def test_clean_narration_needs_no_retry(monkeypatch: pytest.MonkeyPatch, narration: str) -> None:
    respond = Mock(return_value=narration)
    monkeypatch.setattr(gpt, "generate_response", respond)
    assert gpt.verify_and_rewrite_script("Jupiter", narration, "test", "en_us_001") == narration
    assert respond.call_count == 1


def test_editorial_response_retries_once_and_accepts_clean_narration(monkeypatch: pytest.MonkeyPatch) -> None:
    respond = Mock(side_effect=["CLAIMS:\n1. VERDICT: FALSE", "Jupiter has cloud bands."])
    warning = Mock()
    monkeypatch.setattr(gpt, "generate_response", respond)
    monkeypatch.setattr(gpt, "log", warning)
    assert gpt.verify_and_rewrite_script("Jupiter", "Original.", "test", "en_us_001") == "Jupiter has cloud bands."
    assert respond.call_count == 2
    retry = respond.call_args.args[0]
    for requirement in ("ONLY the finished narration", "text-to-speech", "No claims list",
                        "No verdicts", "No labels", "No analysis", "No markdown", "No explanations"):
        assert requirement in retry
    assert any(call.args[1] == "warning" for call in warning.call_args_list)


@pytest.mark.parametrize("responses, message", [
    (["CLAIMS:", "VERDICT: UNKNOWN"], "editorial fact-check notes after one retry"),
    ([""], "empty narration"), ([None], "empty narration"),
    (["CLAIMS:", "  "], "empty narration"),
])
def test_verification_fails_closed(monkeypatch: pytest.MonkeyPatch, responses: list, message: str) -> None:
    respond = Mock(side_effect=responses)
    monkeypatch.setattr(gpt, "generate_response", respond)
    with pytest.raises(RuntimeError, match=message):
        gpt.verify_and_rewrite_script("Jupiter", "Original.", "test", "en_us_001")
    assert respond.call_count == len(responses)


def test_subject_anchor_ignores_generic_title_words() -> None:
    assert _subject_tokens("3 Strange Facts About Jupiter") == {"jupiter"}
    assert _subject_tokens("Interesting amazing surprising things top best video footage stock view scene details") == set()
    assert _subject_tokens(" ".join(SPACE_ENTITIES)) == SPACE_ENTITIES


def test_jupiter_still_outweighs_perfect_storm_and_conflicting_saturn() -> None:
    jupiter = media("jupiter", "Jupiter atmosphere storm", kind="image", provider="nasa")
    jupiter.width, jupiter.height = 1920, 1080
    saturn = media("saturn", "Saturn planet with rings")
    storm = media("storm", "Terrestrial atmosphere storm tornado")
    ranked = search.rank_candidates([storm, saturn, jupiter], "portrait", "3 Strange Facts About Jupiter")
    assert ranked[0] is jupiter
    assert jupiter.score > storm.score + 20
    assert saturn.score < storm.score - 10
    assert storm.score < 3 and saturn.score < 3


@pytest.mark.parametrize("entity", sorted(SPACE_ENTITIES))
def test_named_space_objects_match_and_conflict(entity: str) -> None:
    other = "saturn" if entity != "saturn" else "jupiter"
    matching = media("match", entity)
    conflicting = media("conflict", other)
    both = media("both", f"{entity} and {other}")
    for item in (matching, conflicting, both):
        item.query = entity
    search.rank_candidates([matching, conflicting, both], "portrait", entity)
    assert matching.score == both.score
    assert matching.score > conflicting.score + 20
    assert search.is_space_topic(entity)


def test_normal_subject_anchor_and_synonyms_are_soft_scores() -> None:
    matching = media("match", "Puppy playing", kind="image")
    synonym = media("synonym", "Dog playing")
    for item in (matching, synonym):
        item.query = "dog playing"
    ranked = search.rank_candidates([synonym, matching], "portrait", "Interesting facts about a puppy")
    assert ranked[0] is matching
    assert synonym.score >= 3


def test_nasa_bonus_requires_relevance() -> None:
    nasa = media("nasa", "Office meeting", provider="nasa")
    stock = media("stock", "Office meeting")
    search.rank_candidates([nasa, stock], "portrait", "Jupiter")
    assert nasa.score == stock.score


@pytest.mark.parametrize("value, expected", [
    (None, 3), ("", 3), ("bad", 3), ("nan", 3), ("inf", 3), ("-inf", 3),
    ("4.5", 4.5), ("0", 0), ("-2", -2),
])
def test_min_score_configuration(monkeypatch: pytest.MonkeyPatch, value: str | None, expected: float) -> None:
    monkeypatch.delenv("MEDIA_MIN_SCORE", raising=False)
    if value is not None:
        monkeypatch.setenv("MEDIA_MIN_SCORE", value)
    assert media_min_score() == expected


def test_quality_floor_across_all_passes_and_shortfall_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEDIA_PROVIDERS", "nasa")
    monkeypatch.setenv("MEDIA_MIN_SCORE", "4")
    monkeypatch.setenv("ENABLE_IMAGE_FALLBACK", "true")
    monkeypatch.setenv("MAX_IMAGE_CLIPS", "1")
    calls, logs = [], []
    def find(query: str, *args: object, search_images: bool = False) -> list[MediaCandidate]:
        calls.append((query, search_images))
        if search_images:
            return [media("low-image", score=3.99, kind="image"),
                    media("best-image", score=12, kind="image", provider="nasa")]
        if query == "Jupiter":
            return [media("low-broad", score=2)]
        return [media("low-normal", score=-1), media("good", score=4), media("nan", score=float("nan"))]
    download = Mock(side_effect=lambda item: item.download_url)
    monkeypatch.setattr(pipeline, "search_media_candidates", find)
    monkeypatch.setattr(pipeline, "download_media", download)
    paths = pipeline.select_media(["a", "b"], "Jupiter", "portrait", 5, lambda: None,
                                  lambda message, level: logs.append((message, level)))
    assert len(paths) == 2
    assert [call.args[0].source_id for call in download.call_args_list] == ["good", "best-image"]
    assert calls == [("a", False), ("b", False), ("Jupiter", False),
                     ("a", True), ("b", True), ("Jupiter", True)]
    assert any("Only 2 of 5" in message and level == "warning" for message, level in logs)
    messages = [message for message, _ in logs]
    assert "[Media] Candidates rejected below threshold (4): 4" in messages
    assert "[Media] Videos: 1" in messages and "[Media] Images: 1" in messages
    assert any("nasa / image / best-image / score=12.00 / query='Jupiter atmosphere storm'" in message for message in messages)


def test_cancellation_during_download_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEDIA_PROVIDERS", "nasa")
    monkeypatch.setenv("MEDIA_MIN_SCORE", "3")
    monkeypatch.setattr(pipeline, "search_media_candidates", Mock(return_value=[media("1")]))
    monkeypatch.setattr(pipeline, "download_media", Mock(side_effect=pipeline.PipelineCancelled()))
    with pytest.raises(pipeline.PipelineCancelled):
        pipeline.select_media(["a"], "Jupiter", "portrait", 2, lambda: None, lambda *args: None)


@pytest.mark.parametrize("subject, expected", [
    ("How AI works", {"ai"}),
    ("How Docker actually works", {"docker"}),
])
def test_technical_subject_ignores_action_framing(subject: str, expected: set[str]) -> None:
    assert _subject_tokens(subject) == expected


def test_normalized_generic_words_are_filtered_and_technical_terms_preserved() -> None:
    framing = ("work works working actually explained explain guide guides tips tricks "
               "tutorial tutorials learn learning easy simple ways reasons")
    technical = "python docker kubernetes ai neural network database linux javascript"
    assert _subject_tokens(framing) == set()
    assert _subject_tokens(framing + " " + technical) == set(technical.split())
    assert _subject_tokens("neural networks databases") == {"neural", "network", "database"}


def test_ai_media_strongly_outranks_office_work_without_false_anchor() -> None:
    subject = "How AI works"
    office = media("office", "People at work in office")
    ai = media("ai", "AI neural network visualization")
    for item in (office, ai):
        item.query = subject
    assert _subject_anchor_score(_subject_tokens(subject), _subject_tokens(office.title)) < 0
    assert _subject_anchor_score(_subject_tokens(subject), _subject_tokens(ai.title)) >= 10
    assert search.rank_candidates([office, ai], "portrait", subject) == [ai, office]
    assert ai.score > office.score + 20
    assert office.score < 3


@pytest.mark.parametrize("b_score, expected", [
    (8, ["a1", "a2", "b1"]),
    (10, ["a1", "b1", "a2"]),
])
def test_query_diversity_only_breaks_score_ties_deterministically(
    monkeypatch: pytest.MonkeyPatch, b_score: float, expected: list[str],
) -> None:
    monkeypatch.setenv("MEDIA_PROVIDERS", "nasa")
    monkeypatch.setenv("MEDIA_MIN_SCORE", "3")
    groups = {"A": [media("a1", score=10), media("a2", score=10)],
              "B": [media("b1", score=b_score)]}
    for query, candidates in groups.items():
        for candidate in candidates:
            candidate.query = query
    find = Mock(side_effect=lambda query, *args: groups[query])
    download = Mock(side_effect=lambda item: item.source_id)
    monkeypatch.setattr(pipeline, "search_media_candidates", find)
    monkeypatch.setattr(pipeline, "download_media", download)
    for _ in range(3):
        assert pipeline.select_media(["A", "B"], "AI", "portrait", 3,
                                     lambda: None, lambda *args: None) == expected


def test_failed_download_does_not_mark_query_represented(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEDIA_PROVIDERS", "nasa")
    monkeypatch.setenv("MEDIA_MIN_SCORE", "3")
    groups = {"A": [media("failed", score=10), media("a1", score=10), media("a2", score=10)],
              "B": [media("b1", score=10)]}
    for query, candidates in groups.items():
        for candidate in candidates:
            candidate.query = query
    def download(item: MediaCandidate) -> str:
        if item.source_id == "failed":
            raise RuntimeError("Download failed")
        return item.source_id
    monkeypatch.setattr(pipeline, "search_media_candidates", lambda query, *args: groups[query])
    monkeypatch.setattr(pipeline, "download_media", download)
    assert pipeline.select_media(["A", "B"], "AI", "portrait", 3,
                                 lambda: None, lambda *args: None) == ["a1", "b1", "a2"]


@pytest.mark.parametrize("text", [
    "Jupiter's north pole storm is not confirmed by the provided sources.",
    "Jupiter's color change is not supported by the available information.",
    "According to the supplied sources, Jupiter has clouds.",
    "The available sources describe a storm.", "The reference material describes Jupiter.",
    "The references do not confirm that location.", "This is not supported by the sources.",
    "This is not confirmed by the sources.", "The sources do not confirm the claim.",
    "There is insufficient information about this claim.", "This cannot be verified from this text.",
    "References", "The **provided\n sources** do not confirm this.",
])
def test_verification_meta_language_is_rejected(text: str) -> None:
    assert gpt._looks_like_fact_check_notes(text)


@pytest.mark.parametrize("text", [
    "The Sun is Earth's main source of energy.", "The river's source is a spring.",
    "Open source software makes its source code available.",
    "Astronomers study sources of radio waves.", "Python keeps references to objects.",
])
def test_factual_source_language_is_allowed(text: str) -> None:
    assert not gpt._looks_like_fact_check_notes(text)


@pytest.mark.parametrize("second, raises", [
    ("Jupiter has cloud bands.", False),
    ("The claim is not supported by the available information.", True),
])
def test_jupiter_meta_narration_retries_once(monkeypatch: pytest.MonkeyPatch, second: str, raises: bool) -> None:
    respond = Mock(side_effect=["The location is not confirmed by the provided sources.", second])
    monkeypatch.setattr(gpt, "generate_response", respond)
    if raises:
        with pytest.raises(RuntimeError, match="after one retry"):
            gpt.verify_and_rewrite_script("Jupiter", "Original narration.", "test", "en_us_001")
    else:
        assert gpt.verify_and_rewrite_script("Jupiter", "Original narration.", "test", "en_us_001") == second
    assert respond.call_count == 2
    for call in respond.call_args_list:
        assert "REMOVE the claim" in call.args[0]
        assert "Never tell the viewer that a claim was unsupported" in call.args[0]
        assert "Never mention the verification process or reference material" in call.args[0]


@pytest.mark.parametrize("subject, query, direct_title, program_title, description", [
    ("Jupiter", "Jupiter Great Red Spot", "Jupiter Great Red Spot", "NASA Science Live: Jupiter Great Red Spot",
     "Jupiter Great Red Spot is a storm in Jupiter's atmosphere."),
    ("Jupiter", "Io volcanoes", "Io volcanoes seen by Juno", "Juno mission episode",
     "Juno observes Io volcanoes around Jupiter."),
    ("Jupiter", "Europa ice", "Europa ice surface", "Planetary Protection program briefing",
     "Europa ice and Jupiter's moons are discussed in this program."),
])
def test_direct_imagery_beats_nasa_program_material(subject: str, query: str, direct_title: str,
                                                  program_title: str, description: str) -> None:
    direct = media("direct", direct_title, kind="image", provider="nasa")
    direct.width, direct.height = 1920, 1080  # Relevant landscape still vs perfect portrait program video.
    program = media("program", program_title, provider="nasa")
    for item in (direct, program):
        item.query, item.description = query, description
    assert search.rank_candidates([program, direct], "portrait", subject) == [direct, program]
    assert direct.score > program.score + 5
    assert program.score >= 3  # Penalties do not categorically exclude relevant programs.


@pytest.mark.parametrize("marker", ["live event", "starting soon", "science-live", "planetary protection",
                                    "interview", "podcast", "briefing", "press conference", "episode"])
def test_nasa_program_title_penalty(marker: str) -> None:
    direct = media("direct", "Jupiter", provider="nasa")
    program = media("program", "Jupiter " + marker, provider="nasa")
    search.rank_candidates([direct, program], "portrait", "Jupiter")
    assert direct.score >= program.score + 12


def test_pixabay_baking_plate_regression(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEDIA_MIN_SCORE", "3")
    subject = "3 Strange Facts About Volcanoes"
    kitchen = media("254753", "Woman baking bread in kitchen", provider="pixabay")
    kitchen.description = "Plate, baking tray, bread, dough, kitchen, cooking, bakery"
    geology = media("tectonics", "Tectonic plates and volcanic geology")
    geology.description = "Plate boundary animation showing tectonic plate boundaries"
    for item in (kitchen, geology):
        item.query = "tectonic plate boundaries"
    assert search.rank_candidates([kitchen, geology], "portrait", subject) == [geology, kitchen]
    assert kitchen.score < media_min_score()
    assert geology.score >= media_min_score()
    assert geology.score > kitchen.score + 20
    assert _subject_tokens(subject) == {"volcano"}
    assert _subject_tokens(kitchen.description) & _subject_tokens(kitchen.query) == {"plate"}


@pytest.mark.parametrize("title", [
    "Plate tectonics animation", "Tectonic plates at a boundary", "Lava eruption", "Volcanic ash",
])
def test_geological_context_or_reordered_compound_remains_eligible(title: str) -> None:
    candidate = media("geology", title)
    candidate.query = "tectonic plate boundaries"
    search.rank_candidates([candidate], "portrait", "3 Strange Facts About Volcanoes")
    assert candidate.score >= 3


@pytest.mark.parametrize("query", ["amber cobalt quartz", "amber cobalt quartz crystal", "amber cobalt quartz crystal mineral"])
@pytest.mark.parametrize("provider", ["pexels", "pixabay", "nasa"])
def test_single_overlap_in_long_query_cannot_pass_on_technical_quality(query: str, provider: str) -> None:
    candidate = media("weak", "Amber", provider=provider)
    candidate.query = query
    search.rank_candidates([candidate], "portrait", "")
    assert candidate.score < 3


@pytest.mark.parametrize("query, unrelated", [
    ("tectonic plate", "Plate of bread in a kitchen"),
    ("volcanic ash", "Ash wood furniture"),
    ("magma chamber", "Chamber music orchestra concert"),
    ("shield volcano", "Shield carried by a soldier"),
    ("neural network", "Corporate business network meeting"),
    ("artificial intelligence", "Military intelligence army briefing"),
    ("black hole", "Black abstract background with a hole"),
])
def test_compound_beats_ambiguous_component_even_in_subject(query: str, unrelated: str) -> None:
    direct = media("direct", query)
    ambiguous = media("ambiguous", unrelated)
    for item in (direct, ambiguous):
        item.query = query
    assert search.rank_candidates([ambiguous, direct], "portrait", query) == [direct, ambiguous]
    assert direct.score >= 3
    assert ambiguous.score < 3
    assert direct.score > ambiguous.score + 15


def test_compounds_use_normalized_adjacent_words_without_crossing_boundaries() -> None:
    assert _subject_tokens("volcanoes boundaries plates tectonics") == {"volcano", "boundary", "plate", "tectonic"}
    assert ("tectonic", "plate") in _query_compounds("tectonic plates")
    assert ("neural", "network") in _query_compounds("neural-networks")
    for text in ("neural. Network", "neural, network", "neural\nnetwork", "neural and network"):
        assert ("neural", "network") not in _query_compounds(text)
    assert "" not in _subject_tokens("Jupiter's clouds")


def test_adjacent_compound_adds_stronger_evidence_than_separate_tokens() -> None:
    adjacent, eligible = _query_context_evidence("tectonic plate boundaries", "", "tectonic plates")
    separate, also_eligible = _query_context_evidence("tectonic plate boundaries", "", "plates and tectonics")
    assert eligible and also_eligible
    assert adjacent == separate + 6


def test_increasing_query_coverage_improves_score() -> None:
    candidates = [media(str(i), title) for i, title in enumerate([
        "Amber", "Amber cobalt", "Amber cobalt quartz",
    ])]
    for candidate in candidates:
        candidate.query = "amber cobalt quartz"
    assert search.rank_candidates(candidates, "portrait", "") == list(reversed(candidates))
    assert candidates[0].score < 3 <= candidates[1].score < candidates[2].score


@pytest.mark.parametrize("subject, query, conflicting", [
    ("Volcanoes", "tectonic plate boundaries", "plate baking bread kitchen dough"),
    ("How AI works", "neural network architecture", "network business office meeting"),
    ("Chamber music", "orchestra chamber concert", "chamber magma lava eruption"),
])
def test_context_mismatch_penalty_is_reusable(subject: str, query: str, conflicting: str) -> None:
    shared_words = " ".join(sorted(_subject_tokens(query) & _subject_tokens(conflicting)))
    neutral_score, _ = _query_context_evidence(query, subject, shared_words)
    conflict_score, eligible = _query_context_evidence(query, subject, conflicting)
    assert not eligible
    assert conflict_score <= neutral_score - 12


def test_cooking_metadata_is_valid_for_cooking_subject() -> None:
    candidate = media("bread", "Woman baking bread in kitchen", provider="pixabay")
    candidate.description = "Dough on a plate and baking tray"
    candidate.query = "bread baking techniques"
    search.rank_candidates([candidate], "portrait", "How bread is made")
    assert candidate.score >= 3


def test_nasa_bonus_cannot_rescue_ambiguous_space_query() -> None:
    candidate = media("background", "Black abstract background", provider="nasa")
    candidate.query = "black hole formation"
    search.rank_candidates([candidate], "portrait", "Black holes")
    assert candidate.score < 3


@pytest.mark.parametrize("subject, query, title", [
    ("Black holes", "accretion disk simulation", "Black hole telescope imagery"),
    ("Artificial intelligence", "neural network architecture", "Artificial intelligence AI visualization"),
    ("Golden retrievers", "dog playing outdoors", "Golden retriever playing"),
])
def test_strong_subject_anchor_can_support_low_query_coverage(subject: str, query: str, title: str) -> None:
    candidate = media("anchored", title)
    candidate.query = query
    search.rank_candidates([candidate], "portrait", subject)
    assert candidate.score >= 3
