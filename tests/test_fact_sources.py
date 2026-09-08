import json
from unittest.mock import Mock

import pytest
import requests

import fact_sources as facts
import gpt
import pipeline


REFERENCE = (
    "Jupiter is a gas giant with an atmosphere composed mainly of hydrogen and helium. "
    "The Great Red Spot is a storm in Jupiter's southern hemisphere. "
    "Jupiter rotates in the same direction as Earth. "
    "Jupiter has bands of clouds in its atmosphere. Its interior is under intense pressure."
)
BAD_NARRATION = (
    "The Great Red Spot is at Jupiter's north pole. Jupiter rotates opposite Earth. "
    "Jupiter recently turned grey-blue because it lacks enough atmosphere. "
    "Jupiter's core could float in water."
)
CLEAN_NARRATION = (
    "The Great Red Spot is a storm in Jupiter's southern hemisphere. "
    "Jupiter rotates in the same direction as Earth. "
    "Jupiter is a gas giant with an atmosphere composed mainly of hydrogen and helium."
)


@pytest.fixture(autouse=True)
def isolated_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FACT_CHECK_REQUIRE_SOURCES", "true")
    monkeypatch.setattr(requests, "get", Mock(side_effect=AssertionError("Unexpected HTTP request")))


def source(text: str = REFERENCE) -> facts.FactSource:
    return facts.FactSource("NASA", "Jupiter Facts", facts.JUPITER_FACTS_URL, text)


def test_jupiter_prefers_nasa_page_and_excludes_navigation(monkeypatch: pytest.MonkeyPatch) -> None:
    fetch = Mock(return_value=f"<nav>Wrong facts</nav><main><script>bad instructions</script><p>{REFERENCE}</p></main>")
    monkeypatch.setattr(facts, "_fetch", fetch)
    sources = facts.retrieve_fact_sources("3 Strange Facts About Jupiter", "")
    assert len(sources) == 1 and sources[0].provider == "NASA"
    assert sources[0].url == facts.JUPITER_FACTS_URL
    assert sources[0].text == REFERENCE
    fetch.assert_called_once_with(facts.JUPITER_FACTS_URL)


def test_space_search_uses_nasa_api_descriptions(monkeypatch: pytest.MonkeyPatch) -> None:
    text = "The Orion nebula contains gas and dust where stars form. It is a region studied by astronomical telescopes."
    fetch = Mock(return_value=json.dumps({"collection": {"items": [{"data": [
        {"nasa_id": "orion", "title": "Orion nebula", "description": text},
    ]}]}}))
    monkeypatch.setattr(facts, "_fetch", fetch)
    sources = facts.retrieve_fact_sources("Orion nebula", "")
    assert sources[0].provider == "NASA" and sources[0].text == text
    assert fetch.call_args.args[0] == "https://images-api.nasa.gov/search"


def test_general_subject_uses_wikipedia_api(monkeypatch: pytest.MonkeyPatch) -> None:
    text = "Python is a programming language. Python supports multiple programming paradigms and has a large standard library."
    fetch = Mock(return_value=json.dumps({"query": {"pages": [
        {"title": "Python (programming language)", "index": 1, "extract": text},
    ]}}))
    monkeypatch.setattr(facts, "_fetch", fetch)
    sources = facts.retrieve_fact_sources("How Python works", "")
    assert sources[0].provider == "Wikipedia" and sources[0].text == text
    assert fetch.call_args.args[0] == "https://en.wikipedia.org/w/api.php"
    assert fetch.call_args.args[1]["gsrlimit"] == facts.RESULTS_PER_QUERY
    assert fetch.call_args.args[1]["explaintext"] == 1
    assert fetch.call_args.args[1]["exchars"] <= 1200


@pytest.mark.parametrize("failure", [requests.Timeout("secret-url"), requests.ConnectionError("secret-url"), ValueError("bad JSON")])
def test_nasa_failure_falls_back_to_wikipedia_without_leaking_error(
    monkeypatch: pytest.MonkeyPatch, failure: Exception,
) -> None:
    fetch = Mock(side_effect=[failure, json.dumps({"query": {"pages": [
        {"title": "Jupiter", "extract": REFERENCE},
    ]}})])
    logs = Mock()
    monkeypatch.setattr(facts, "_fetch", fetch)
    sources = facts.retrieve_fact_sources("Jupiter", "", logs)
    assert sources[0].provider == "Wikipedia"
    assert fetch.call_count == 2
    assert "secret-url" not in str(logs.call_args_list)


def test_queries_use_subject_and_claims_with_hard_bound() -> None:
    queries = facts.factual_queries("3 Strange Facts About Jupiter", BAD_NARRATION * 10)
    assert queries[0] == "jupiter"
    assert "great red spot" in queries[1]
    assert len(queries) == len(set(queries)) == facts.MAX_QUERIES
    assert all(len(query) <= 160 and "jupiter" in query for query in queries)


def test_empty_sources_bounded_queries_and_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    fetch = Mock(return_value='{"query": {"pages": []}}')
    monkeypatch.setattr(facts, "_fetch", fetch)
    assert facts.retrieve_fact_sources("Python", "Functions accept arguments. Classes define behavior. Modules contain code. More facts.") == []
    assert fetch.call_count == facts.MAX_QUERIES


def test_source_count_excerpt_size_and_deduplication(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = Mock(name="provider")
    provider.name = "NASA"
    def search(query: str) -> list[facts.FactSource]:
        return [facts.FactSource("NASA", "Jupiter " + query, "https://example.com/" + query,
                                 REFERENCE * 100 + query + "."), source()]
    provider.search.side_effect = search
    monkeypatch.setattr(facts, "get_fact_providers", lambda subject: [provider])
    sources = facts.retrieve_fact_sources("Jupiter", BAD_NARRATION)
    assert 1 <= len(sources) <= facts.MAX_SOURCES
    assert len({item.url for item in sources}) == len(sources)
    assert len({item.text for item in sources}) == len(sources)
    assert all(80 <= len(item.text) <= facts.MAX_EXCERPT_CHARS for item in sources)


def test_http_timeout_user_agent_and_response_size_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    response = Mock()
    response.iter_content.return_value = iter([b"x" * (facts.MAX_RESPONSE_BYTES + 1)])
    context = Mock()
    context.__enter__ = Mock(return_value=response)
    context.__exit__ = Mock(return_value=False)
    get = Mock(return_value=context)
    monkeypatch.setattr(requests, "get", get)
    with pytest.raises(ValueError, match="size limit"):
        facts._fetch(facts.JUPITER_FACTS_URL)
    assert get.call_args.kwargs["timeout"] == facts.HTTP_TIMEOUT
    assert get.call_args.kwargs["headers"]["User-Agent"] == facts.USER_AGENT
    assert get.call_args.kwargs["stream"] is True
    context.__exit__.assert_called_once()


def test_retrieval_stops_at_source_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = Mock()
    provider.name = "NASA"
    provider.search.side_effect = lambda query: [
        facts.FactSource("NASA", "Jupiter", f"https://example.com/{query}/{index}",
                         f"Jupiter reference {query}, item {index}. " + REFERENCE)
        for index in range(facts.RESULTS_PER_QUERY)
    ]
    monkeypatch.setattr(facts, "get_fact_providers", lambda subject: [provider])
    assert len(facts.retrieve_fact_sources("Jupiter", BAD_NARRATION)) == facts.MAX_SOURCES
    assert provider.search.call_count == 2


def test_retrieval_deadline_stops_new_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = Mock()
    provider.name = "NASA"
    monkeypatch.setattr(facts, "get_fact_providers", lambda subject: [provider])
    monkeypatch.setattr(facts.time, "monotonic", Mock(side_effect=[0, facts.RETRIEVAL_SECONDS]))
    assert facts.retrieve_fact_sources("Jupiter", BAD_NARRATION) == []
    provider.search.assert_not_called()


def test_unrelated_or_empty_reference_is_not_usable(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = Mock()
    provider.name = "Wikipedia"
    provider.search.return_value = [facts.FactSource("Wikipedia", "Cooking", "https://example.com/", "Cooking food in a kitchen. " * 10)]
    monkeypatch.setattr(facts, "get_fact_providers", lambda subject: [provider])
    assert facts.retrieve_fact_sources("Jupiter", "") == []


@pytest.mark.parametrize("setting", [None, "true", "bad", ""])
def test_required_sources_fail_before_llm(monkeypatch: pytest.MonkeyPatch, setting: str | None) -> None:
    monkeypatch.delenv("FACT_CHECK_REQUIRE_SOURCES", raising=False)
    if setting is not None:
        monkeypatch.setenv("FACT_CHECK_REQUIRE_SOURCES", setting)
    monkeypatch.setattr(gpt, "retrieve_fact_sources", Mock(return_value=[]))
    generate = Mock()
    monkeypatch.setattr(gpt, "generate_response", generate)
    with pytest.raises(RuntimeError, match="could not retrieve sufficient reference material"):
        gpt.verify_and_rewrite_script("Jupiter", BAD_NARRATION, "test", "en_us_001")
    generate.assert_not_called()


def test_optional_sources_allow_llm_only_with_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FACT_CHECK_REQUIRE_SOURCES", "false")
    monkeypatch.setattr(gpt, "retrieve_fact_sources", Mock(return_value=[]))
    monkeypatch.setattr(gpt, "generate_response", Mock(return_value=CLEAN_NARRATION))
    logs = Mock()
    assert gpt.verify_and_rewrite_script("Jupiter", BAD_NARRATION, "test", "en_us_001", logs) == CLEAN_NARRATION
    assert any("LLM-only" in call.args[0] and call.args[1] == "warning" for call in logs.call_args_list)
    assert "Fact-checked narration" not in str(logs.call_args_list)


def test_grounded_jupiter_regression_with_deterministic_model(monkeypatch: pytest.MonkeyPatch) -> None:
    retrieve = Mock(return_value=[source()])
    monkeypatch.setattr(gpt, "retrieve_fact_sources", retrieve)
    def rewrite(prompt: str, model: str) -> str:
        # This fixture verifies grounding orchestration, not a live model's accuracy.
        assert REFERENCE in prompt and BAD_NARRATION in prompt
        for constraint in ("ONLY the supplied reference material", "not supported by the references",
                           "Do not preserve unsupported specificity", "Do not invent analogies",
                           "No citations", "No source list", "No claims", "No verdicts", "No analysis", "No markdown"):
            assert constraint in prompt
        assert all(sentence in REFERENCE for sentence in CLEAN_NARRATION.split(". "))
        return CLEAN_NARRATION
    monkeypatch.setattr(gpt, "generate_response", rewrite)
    logs = Mock()
    result = gpt.verify_and_rewrite_script("Jupiter", BAD_NARRATION, "test", "en_us_001", logs)
    assert result == CLEAN_NARRATION
    for unsupported in ("north pole", "opposite", "grey-blue", "lacks", "float", "marble", "compressed"):
        assert unsupported not in result
    retrieve.assert_called_once()
    messages = [call.args[0] for call in logs.call_args_list]
    assert "[FactCheck] Sources retrieved: 1" in messages
    assert "[FactCheck] Source: NASA / Jupiter Facts" in messages
    assert f"[+] Fact-checked narration:\n{CLEAN_NARRATION}" in messages
    assert not any(REFERENCE in message for message in messages)


@pytest.mark.parametrize("second, fails", [(CLEAN_NARRATION, False), ("VERDICT: FALSE", True)])
def test_grounded_contamination_retries_once_with_same_sources(
    monkeypatch: pytest.MonkeyPatch, second: str, fails: bool,
) -> None:
    retrieve = Mock(return_value=[source()])
    generate = Mock(side_effect=["CLAIMS: MISLEADING", second])
    monkeypatch.setattr(gpt, "retrieve_fact_sources", retrieve)
    monkeypatch.setattr(gpt, "generate_response", generate)
    if fails:
        with pytest.raises(RuntimeError, match="after one retry"):
            gpt.verify_and_rewrite_script("Jupiter", BAD_NARRATION, "test", "en_us_001")
    else:
        assert gpt.verify_and_rewrite_script("Jupiter", BAD_NARRATION, "test", "en_us_001") == second
    assert generate.call_count == 2 and retrieve.call_count == 1
    assert all(REFERENCE in call.args[0] for call in generate.call_args_list)


def test_pipeline_no_sources_fails_before_search_and_tts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCRIPT_FACT_CHECK", "true")
    monkeypatch.setattr(pipeline, "generate_script", Mock(return_value=BAD_NARRATION))
    monkeypatch.setattr(gpt, "retrieve_fact_sources", Mock(return_value=[]))
    search, tts, logs = Mock(), Mock(), Mock()
    monkeypatch.setattr(pipeline, "get_search_terms", search)
    monkeypatch.setattr(pipeline, "tts", tts)
    with pytest.raises(RuntimeError, match="reference material"):
        pipeline.run_generation_pipeline({"videoSubject": "Jupiter", "voice": "en_us_001", "customPrompt": ""}, None, logs)
    search.assert_not_called()
    tts.assert_not_called()
    assert any("Sources retrieved: 0" in call.args[0] for call in logs.call_args_list)
