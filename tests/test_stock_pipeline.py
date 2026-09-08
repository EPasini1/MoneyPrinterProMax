from pathlib import Path
from typing import Iterator
from unittest.mock import Mock

import pytest
import requests
from moviepy import ColorClip

import gpt
import pipeline
import search
import video
from fact_sources import FactSource
from utils import get_max_clip_duration, get_stock_video_count
from providers.base import MediaCandidate


@pytest.mark.parametrize("value, expected", [
    (None, 10), ("", 10), ("bad", 10), ("0", 10), ("-1", 10), ("3.5", 10), ("12", 12),
])
def test_stock_count_configuration(monkeypatch, value, expected) -> None:
    monkeypatch.delenv("STOCK_VIDEO_COUNT", raising=False)
    if value is not None:
        monkeypatch.setenv("STOCK_VIDEO_COUNT", value)
    assert get_stock_video_count() == expected


def test_stock_count_caller_override_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STOCK_VIDEO_COUNT", "12")
    assert get_stock_video_count(8) == 8


@pytest.mark.parametrize("value, expected", [
    (None, 6), ("", 6), ("bad", 6), ("nan", 6), ("inf", 6),
    ("1", 2), ("11", 10), ("4.5", 4.5),
])
def test_clip_duration_configuration(monkeypatch, value, expected) -> None:
    monkeypatch.delenv("MAX_CLIP_DURATION", raising=False)
    if value is not None:
        monkeypatch.setenv("MAX_CLIP_DURATION", value)
    assert get_max_clip_duration() == expected


def test_search_terms_retry_only_deficit_and_validate(monkeypatch) -> None:
    prompts = []
    responses = iter([
        '["Jupiter planet", "jupiter PLANET", "facts information", "stars", 7]',
        '```json\n["Jupiter planet", "Great Red Spot"]\n```',
        '{"terms": ["not an array"]}',
        '["deep space stars", "space telescope", "extra valid term"]',
    ])

    def respond(prompt: str, model: str) -> str:
        prompts.append(prompt)
        return next(responses)

    monkeypatch.setattr(gpt, "generate_response", respond)
    terms = gpt.get_search_terms("Jupiter", 4, "Jupiter has a Great Red Spot.", "test")
    assert terms == ["Jupiter planet", "Great Red Spot", "deep space stars", "space telescope"]
    assert len(prompts) == 4
    assert "exactly 3 new" in prompts[1]
    assert '["Jupiter planet"]' in prompts[1]
    assert "exactly 2 new" in prompts[2]
    assert "Great Red Spot" in prompts[3]


@pytest.mark.parametrize("subject", ["Jupiter", "Interesting facts about Jupiter", "Great Barrier Reef marine life"])
def test_search_terms_fallback_is_unique_and_subject_derived(monkeypatch, subject) -> None:
    calls = []

    def respond(prompt: str, model: str) -> str:
        calls.append(prompt)
        return '[null, {}, "one", "vague concept"]'

    monkeypatch.setattr(gpt, "generate_response", respond)
    terms = gpt.get_search_terms(subject, 8, "", "test")
    assert len(calls) == 4
    assert len(terms) == len({term.casefold() for term in terms}) == 8
    assert all(2 <= len(term.split()) <= 5 for term in terms)
    assert all("Jupiter" in term if "Jupiter" in subject else "Great Barrier Reef" in term for term in terms)


def test_search_terms_connection_failure_uses_fallback(monkeypatch) -> None:
    def fail(prompt: str, model: str) -> str:
        raise RuntimeError("offline")

    monkeypatch.setattr(gpt, "generate_response", fail)
    assert len(gpt.get_search_terms("Jupiter", 8, "", "test")) == 8
    assert gpt.get_search_terms("Jupiter", 0, "", "test") == []


def test_verification_preserves_raw_corrected_narration(monkeypatch) -> None:
    monkeypatch.setattr(gpt, "retrieve_fact_sources", lambda *args: [
        FactSource("NASA", "Jupiter Facts", "https://science.nasa.gov/jupiter/", "Jupiter has clouds."),
    ])
    def respond(prompt: str, model: str) -> str:
        assert "every factual claim" in prompt
        assert "Do not invent replacement statistics" in prompt
        assert "br_001" in prompt
        assert "Incorrect narration" in prompt
        return "Narração corrigida."

    monkeypatch.setattr(gpt, "generate_response", respond)
    assert gpt.verify_and_rewrite_script("Jupiter", "Incorrect narration", "test", "br_001") == "Narração corrigida."


def _pexels_video(video_id: int, duration: float = 10, slug: str = "jupiter-planet",
                  width: int = 1080, height: int = 1920) -> dict:
    return {
        "id": video_id, "duration": duration, "width": width, "height": height,
        "url": f"https://www.pexels.com/video/{slug}-{video_id}/",
        "video_files": [
            {"file_type": "video/mp4", "width": width, "height": height,
             "link": f"https://videos.pexels.com/video-files/{video_id}/hd.mp4"},
            {"file_type": "video/mp4", "width": 360, "height": 640,
             "link": f"https://videos.pexels.com/video-files/{video_id}/sd.mp4"},
        ],
    }


def test_pexels_keeps_multiple_ids_and_downranks_irrelevant_footage(monkeypatch) -> None:
    class Response:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {"videos": [
                _pexels_video(1, 3), _pexels_video(2), _pexels_video(2),
                _pexels_video(3, slug="bowl-of-rice"),
                _pexels_video(4, width=1920, height=1080),
                _pexels_video(5, width=720, height=1280),
                {"id": 6},
            ]}

    def get(url: str, **kwargs) -> Response:
        assert url == "https://api.pexels.com/videos/search"
        assert kwargs["params"] == {"query": "Jupiter & planet", "per_page": 15, "orientation": "portrait"}
        assert kwargs["timeout"] == 30
        return Response()

    monkeypatch.setattr(search.requests, "get", get)
    candidates = search.search_for_stock_videos("Jupiter & planet", "test", 15, 6, "portrait")
    assert [candidate.id for candidate in candidates] == [2, 5, 1, 3]
    assert all(candidate.url.endswith("hd.mp4") for candidate in candidates)


@pytest.fixture
def pexels_results(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    videos = []

    class Response:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {"videos": videos}

    monkeypatch.setattr(search.requests, "get", lambda *args, **kwargs: Response())
    return videos


def test_pexels_preserves_zero_overlap_candidates_and_api_ties(pexels_results: list[dict]) -> None:
    pexels_results.extend([
        _pexels_video(9, slug="bowl-of-rice"),
        _pexels_video(7, slug="distant-celestial-body"),
        _pexels_video(2, slug="distant-celestial-body"),
        _pexels_video(5, slug="jupiter-planet"),
    ])
    candidates = search.search_for_stock_videos("Jupiter planet", "test", 15, 6, "portrait")
    assert [candidate.id for candidate in candidates] == [5, 7, 2, 9]
    assert candidates[0].relevance > 0 > candidates[1].relevance
    assert candidates[1].relevance == candidates[2].relevance
    assert candidates[-1].relevance < candidates[1].relevance


def test_pexels_deduplicates_download_urls(pexels_results: list[dict]) -> None:
    first = _pexels_video(1)
    second = _pexels_video(2)
    second["video_files"] = first["video_files"]
    pexels_results.extend([first, second])
    assert len(search.search_for_stock_videos("Jupiter planet", "test", 15, 6)) == 1


@pytest.mark.parametrize("width, height, orientation", [
    (1080, 1920, "portrait"), (1920, 1080, "landscape"), (1080, 1080, "square"),
])
@pytest.mark.parametrize("scales, expected", [
    ([2, 1, 2 / 3, 1 / 3], 1),  # 1080p wins over available 4K.
    ([2 / 3, 1 / 3], 2 / 3),  # 720p wins over lower quality.
    ([2, 2 / 3, 1 / 3], 2 / 3),  # 720p also avoids unnecessary 4K.
    ([2, 1 / 3], 2),  # Prefer 4K to an unusably small rendition.
])
def test_pexels_prefers_reasonable_renditions(
    pexels_results: list[dict], width: int, height: int, orientation: str,
    scales: list[float], expected: float,
) -> None:
    source = _pexels_video(1, width=width * 2, height=height * 2)
    source["video_files"] = [
        {"file_type": "video/mp4", "width": round(width * scale),
         "height": round(height * scale), "link": f"https://example.com/{scale}.mp4"}
        for scale in scales
    ]
    source["video_files"].insert(0, {
        "file_type": "video/webm", "width": width, "height": height,
        "link": "https://example.com/ideal-size.webm",
    })
    pexels_results.append(source)
    candidates = search.search_for_stock_videos("Jupiter planet", "test", 15, 6, orientation)
    assert len(candidates) == 1
    assert candidates[0].url == f"https://example.com/{expected}.mp4"


def test_pexels_http_failure_returns_no_candidates(monkeypatch) -> None:
    def fail(*args, **kwargs) -> None:
        raise requests.HTTPError("429")

    monkeypatch.setattr(search.requests, "get", fail)
    assert search.search_for_stock_videos("Jupiter planet", "test", 15, 6) == []


def _mock_narration_audio(monkeypatch: pytest.MonkeyPatch, duration: float = 1.0) -> Mock:
    factory = Mock(side_effect=lambda *args: Mock(duration=duration))
    monkeypatch.setattr(pipeline, "AudioFileClip", factory)
    monkeypatch.setattr(pipeline, "concatenate_audioclips", Mock(return_value=Mock(duration=duration)))
    monkeypatch.setattr(pipeline, "tts", Mock())
    return factory


def test_pipeline_reports_empty_candidate_pool_after_audio(monkeypatch) -> None:
    _mock_narration_audio(monkeypatch)
    monkeypatch.setenv("SCRIPT_FACT_CHECK", "false")
    monkeypatch.delenv("STOCK_VIDEO_COUNT", raising=False)
    monkeypatch.setattr(pipeline, "generate_script", lambda *args: "Jupiter narration.")

    def terms(subject: str, amount: int, script: str, model: str) -> list[str]:
        assert amount == 10
        return ["Jupiter planet"]

    monkeypatch.setattr(pipeline, "get_search_terms", terms)
    monkeypatch.setattr(pipeline, "search_media_candidates", lambda *args, **kwargs: [])
    monkeypatch.setenv("MEDIA_PROVIDERS", "nasa")
    logs = []
    with pytest.raises(RuntimeError, match="No usable media"):
        pipeline.run_generation_pipeline(
            {"videoSubject": "Jupiter", "customPrompt": ""}, None,
            lambda message, level: logs.append((message, level)),
        )
    assert any("Selected 0 unique media items" in message for message, _ in logs)
    assert any("Only 0 of 10" in message and level == "warning" for message, level in logs)


def test_failed_download_removes_partial_file(monkeypatch, tmp_path) -> None:
    class Response:
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args) -> None:
            pass

        def raise_for_status(self) -> None:
            pass

        def iter_content(self, chunk_size: int) -> Iterator[bytes]:
            yield b"partial media"
            raise requests.ConnectionError("interrupted")

    monkeypatch.setattr(video.requests, "get", lambda *args, **kwargs: Response())
    with pytest.raises(requests.ConnectionError, match="interrupted"):
        video.save_video("https://example.com/stock.mp4", str(tmp_path))
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("enabled", [True, False])
def test_pipeline_uses_spares_deduplicates_ids_and_checks_script_before_search(monkeypatch, enabled) -> None:
    _mock_narration_audio(monkeypatch)
    monkeypatch.setenv("STOCK_VIDEO_COUNT", "3")
    monkeypatch.setenv("MAX_CLIP_DURATION", "7")
    monkeypatch.setenv("SCRIPT_FACT_CHECK", str(enabled))
    monkeypatch.setenv("MEDIA_MIN_SCORE", "3.0")
    monkeypatch.setattr(pipeline, "generate_script", lambda *args: "Original script.")
    verification_calls = []

    def verify(*args) -> str:
        verification_calls.append(args)
        return "Corrected script."

    def terms(subject: str, amount: int, script: str, model: str) -> list[str]:
        assert amount == 3
        assert script == ("Corrected script." if enabled else "Original script.")
        return ["query one", "query two", "query three"]

    def candidate(video_id: int, url: str = "") -> MediaCandidate:
        return MediaCandidate("pexels", "video", str(video_id), "query", url or f"url-{video_id}", "", 1080, 1920, 8, score=5)

    groups = {
        "query one": [candidate(1), candidate(3), candidate(3, "alternate-after-failure"), candidate(4)],
        "query two": [candidate(1, "alternate-rendition"), candidate(2)],
        "query three": [],
    }

    def stock(query: str, orientation: str, subject: str, count: int) -> list:
        assert get_max_clip_duration() == 7
        assert orientation == "portrait"
        return groups[query]

    downloads = []

    def save(candidate: MediaCandidate) -> str:
        url = candidate.download_url
        downloads.append(url)
        if url == "url-3":
            raise RuntimeError("download failed")
        return url

    def tts(sentence: str, voice: str, filename: str) -> None:
        assert sentence == ("Corrected script." if enabled else "Original script.")

    monkeypatch.setattr(pipeline, "verify_and_rewrite_script", verify)
    monkeypatch.setattr(pipeline, "get_search_terms", terms)
    monkeypatch.setattr(pipeline, "search_media_candidates", stock)
    monkeypatch.setenv("MEDIA_PROVIDERS", "nasa")
    monkeypatch.setattr(pipeline, "download_media", save)
    monkeypatch.setattr(pipeline, "tts", tts)
    monkeypatch.setattr(pipeline, "generate_subtitles", Mock(return_value=None))
    logs = []
    with pytest.raises(RuntimeError, match="Could not generate subtitles"):
        pipeline.run_generation_pipeline(
            {"videoSubject": "Jupiter", "customPrompt": "", "voice": "en_us_001"},
            lambda: False, lambda message, level: logs.append(message),
        )
    assert bool(verification_calls) == enabled
    assert downloads == ["url-1", "url-3", "alternate-after-failure", "url-4"]
    assert any("Selected 3 unique media items" in message for message in logs)
    if enabled:
        assert callable(verification_calls[0][-1])  # Verification logs use the persisted job callback.


@pytest.mark.parametrize("duration, configured, expected", [
    (65, 10, (11, 11)), (65.1, 10, (11, 11)), (60, 10, (10, 10)),
    (60.001, 10, (11, 11)), (6, 10, (1, 10)), (65.1, 20, (11, 20)),
    (6, 1, (1, 2)), (1000, 10, (167, 120)), (6, 200, (1, 120)),
])
def test_dynamic_media_requirement(duration: float, configured: int, expected: tuple) -> None:
    assert pipeline.media_requirement(duration, 6, configured) == expected


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf")])
def test_invalid_audio_duration_fails(duration: float) -> None:
    with pytest.raises(RuntimeError, match="positive finite duration"):
        pipeline.media_requirement(duration, 6, 10)


@pytest.mark.parametrize("quality_count", [8, 12])
def test_pipeline_uses_measured_audio_and_cycles_only_quality_sources(monkeypatch: pytest.MonkeyPatch,
                                                                    quality_count: int) -> None:
    _mock_narration_audio(monkeypatch, 65.1)
    monkeypatch.setattr(pipeline, "concatenate_audioclips", Mock(return_value=Mock(duration=60)))
    monkeypatch.setenv("SCRIPT_FACT_CHECK", "false")
    monkeypatch.setenv("STOCK_VIDEO_COUNT", "10")
    monkeypatch.setenv("MAX_CLIP_DURATION", "6")
    monkeypatch.setenv("MEDIA_MIN_SCORE", "3")
    monkeypatch.setenv("MEDIA_PROVIDERS", "nasa")
    monkeypatch.setenv("ENABLE_IMAGE_FALLBACK", "false")
    monkeypatch.setattr(pipeline, "generate_script", Mock(return_value="Jupiter narration."))
    terms = Mock(return_value=["Jupiter"])
    monkeypatch.setattr(pipeline, "get_search_terms", terms)
    candidates = [MediaCandidate("nasa", "video", str(i), "Jupiter", f"url-{i}", None,
                                 1080, 1920, 6, score=5 if i < quality_count else 2) for i in range(15)]
    monkeypatch.setattr(pipeline, "search_media_candidates", Mock(return_value=candidates))
    download = Mock(side_effect=lambda candidate: candidate.download_url)
    monkeypatch.setattr(pipeline, "download_media", download)
    monkeypatch.setattr(pipeline, "generate_subtitles", Mock(return_value="subtitles.srt"))
    class ReachedCombine(Exception):
        pass
    combine = Mock(side_effect=ReachedCombine)
    monkeypatch.setattr(pipeline, "combine_videos", combine)
    logs = []
    with pytest.raises(ReachedCombine):
        pipeline.run_generation_pipeline({"videoSubject": "Jupiter", "customPrompt": ""}, None,
                                         lambda message, level: logs.append(message))
    selected_count = min(quality_count, 11)
    assert combine.call_args.args[:3] == ([f"url-{i}" for i in range(selected_count)], 65.1, 6)
    assert download.call_count == selected_count
    assert terms.call_args.args[1] == 10  # More scenes do not require more LLM queries.
    assert "[Media] Audio duration: 65.1s" in logs
    assert "[Media] Unique scenes required at 6s max: 11" in logs
    assert "[Media] Target scenes: 11" in logs
    assert any("Only 8 of 11" in message for message in logs) == (quality_count == 8)


def test_tts_failure_closes_previous_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    clip = Mock(duration=2)
    monkeypatch.setenv("SCRIPT_FACT_CHECK", "false")
    monkeypatch.setattr(pipeline, "generate_script", Mock(return_value="First sentence. Second sentence."))
    monkeypatch.setattr(pipeline, "AudioFileClip", Mock(return_value=clip))
    monkeypatch.setattr(pipeline, "tts", Mock(side_effect=[None, RuntimeError("TTS unavailable")]))
    with pytest.raises(RuntimeError, match="TTS unavailable"):
        pipeline.run_generation_pipeline({"videoSubject": "Jupiter", "customPrompt": ""}, None, None)
    clip.close.assert_called_once()


@pytest.mark.parametrize("size", [(90, 160), (160, 90), (100, 100), (80, 100)])
def test_combine_cycles_unique_sources_at_configured_duration(monkeypatch, tmp_path, size) -> None:
    monkeypatch.setattr(video, "TEMP_DIR", tmp_path)

    def source(path: str) -> ColorClip:
        color = (255, 0, 0) if Path(path).name == "a.mp4" else (0, 0, 255)
        return ColorClip((160, 90), color=color, duration=12)

    observed = []

    def concatenate(clips: list, method: str) -> ColorClip:
        for clip in clips:
            observed.append((tuple(clip.get_frame(0)[0, 0]), clip.duration))
            assert tuple(clip.size) == size
            assert clip.fps == 30
        return ColorClip(size, color=(0, 0, 0), duration=sum(clip.duration for clip in clips))

    def write(clip: ColorClip, path: str, **kwargs) -> None:
        assert clip.duration == 20
        assert kwargs["fps"] == 30

    monkeypatch.setattr(video, "VideoFileClip", source)
    monkeypatch.setattr(video, "concatenate_videoclips", concatenate)
    monkeypatch.setattr(ColorClip, "write_videofile", write)
    video.combine_videos(["a.mp4", "a.mp4", "b.mp4"], 20, 6, 1, *size)
    assert observed == [((255, 0, 0), 6), ((0, 0, 255), 6), ((255, 0, 0), 6), ((0, 0, 255), 2)]
