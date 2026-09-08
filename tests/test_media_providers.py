import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import requests
from PIL import Image
from moviepy import ColorClip, VideoFileClip

import pipeline
import search
import video
from providers import base, cache
from providers.base import MediaCandidate
from providers.nasa import NasaProvider
from providers.pixabay import PixabayProvider


@pytest.fixture(autouse=True)
def isolated_media(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MEDIA_CACHE_DIR", str(tmp_path / "cache"))
    for name in ("PEXELS_API_KEY", "PIXABAY_API_KEY", "NASA_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MEDIA_PROVIDERS", "pexels,pixabay,nasa")
    monkeypatch.setenv("ENABLE_IMAGE_FALLBACK", "true")
    monkeypatch.setenv("MAX_IMAGE_CLIPS", "4")
    monkeypatch.setenv("MEDIA_MIN_SCORE", "3.0")
    monkeypatch.setenv("MEDIA_SEARCH_TIMEOUT", "30")
    monkeypatch.setenv("MEDIA_DOWNLOAD_TIMEOUT", "60")
    monkeypatch.setattr(base, "_disabled_logged", set())
    # A missing mock must never result in a real external request.
    monkeypatch.setattr(requests, "get", Mock(side_effect=AssertionError("Unexpected HTTP request")))


def candidate(source_id: str, kind: str = "video", provider: str = "pexels",
              url: str | None = None, score: float = 5) -> MediaCandidate:
    return MediaCandidate(provider, kind, source_id, "Great Red Spot",
                          url or f"https://example.com/{provider}/{source_id}", None,
                          1080, 1920, 8 if kind == "video" else None,
                          title="Great Red Spot on Jupiter", score=score)


def response(payload: dict) -> Mock:
    return Mock(json=Mock(return_value=payload), raise_for_status=Mock())


@pytest.mark.parametrize("subject,expected", [
    ("Jupiter", ["nasa", "pixabay", "pexels"]),
    ("Cooking rice", ["pexels", "pixabay", "nasa"]),
])
def test_provider_order_and_parsing(monkeypatch: pytest.MonkeyPatch, subject: str, expected: list) -> None:
    monkeypatch.setenv("PEXELS_API_KEY", "test")
    monkeypatch.setenv("PIXABAY_API_KEY", "test")
    monkeypatch.setenv("MEDIA_PROVIDERS", " NASA, PEXELS,pixabay,nasa,unknown, ")
    logs = []
    monkeypatch.setattr(search, "log", lambda message, level: logs.append(message))
    assert [provider.name for provider in search.get_enabled_providers(subject)] == expected
    assert any("Unknown provider" in message for message in logs)


def test_missing_keys_leave_nasa_usable_and_log_once(monkeypatch: pytest.MonkeyPatch) -> None:
    logs = []
    monkeypatch.setattr(base, "log", lambda message, level: logs.append(message))
    for _ in range(2):
        assert [provider.name for provider in search.get_enabled_providers()] == ["nasa"]
        assert PixabayProvider().search("test", "", 20) == []
    assert len(logs) == 2


@pytest.mark.parametrize("config", ["", "unknown", "pexels,pixabay"])
def test_no_usable_provider_is_clear(monkeypatch: pytest.MonkeyPatch, config: str) -> None:
    monkeypatch.setenv("MEDIA_PROVIDERS", config)
    with pytest.raises(RuntimeError, match="No usable media providers"):
        search.get_enabled_providers()


@pytest.mark.parametrize("term", ["space", "planet", "galaxy", "universe", "nasa", "moon", "mars",
    "jupiter", "saturn", "venus", "mercury", "neptune", "uranus", "asteroid", "comet", "telescope",
    "astronomy", "solar system", "star", "stars", "nebula", "black hole"])
def test_space_routing_keywords(term: str) -> None:
    assert search.is_space_topic("", term.upper())
    assert not search.is_space_topic("office workspace starter kit")


def test_provider_exception_falls_through_without_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    providers = [Mock(name="unused"), Mock(name="unused")]
    providers[0].name, providers[1].name = "nasa", "pixabay"
    providers[0].search.side_effect = requests.HTTPError("?key=secret")
    providers[1].search.return_value = [candidate("1", provider="pixabay")]
    monkeypatch.setattr(search, "get_enabled_providers", lambda *args: providers)
    logs = []
    monkeypatch.setattr(search, "log", lambda message, level: logs.append(message))
    assert len(search.search_media_candidates("Jupiter")) == 1
    assert "secret" not in " ".join(logs)


@pytest.mark.parametrize("medium_width,expected", [(1920, "medium"), (1280, "medium")])
def test_pixabay_video_and_image_parsing(monkeypatch: pytest.MonkeyPatch, medium_width: int, expected: str) -> None:
    calls = []
    hit = {"id": 42, "duration": 9, "pageURL": "https://pixabay.com/videos/42/", "tags": "Jupiter",
           "videos": {name: {"url": f"https://example.com/{name}.mp4", "width": width, "height": width * 9 // 16}
                      for name, width in [("large", 3840), ("medium", medium_width), ("small", 960)]}}
    def get(url: str, **kwargs: dict) -> Mock:
        calls.append((url, kwargs))
        if "videos/" in url:
            return response({"hits": [hit, hit]})
        return response({"hits": [{"id": 8, "largeImageURL": "https://example.com/photo.jpg",
                                   "imageWidth": 2400, "imageHeight": 1600, "tags": "Jupiter",
                                   "pageURL": "https://pixabay.com/photos/8/"}]})
    monkeypatch.setattr(requests, "get", get)
    results = PixabayProvider("test").search("Jupiter & moons", "portrait", 20)
    results += PixabayProvider("test").search("Jupiter & moons", "portrait", 20, search_images=True)
    assert len(results) == 2
    assert results[0].download_url.endswith(f"{expected}.mp4")
    assert results[0].duration == 9
    assert results[1].media_type == "image" and results[1].width == 1280
    assert results[1].page_url == "https://pixabay.com/photos/8/"
    assert all(call[1]["params"]["q"] == "Jupiter & moons" for call in calls)
    assert calls[1][1]["params"]["image_type"] == "photo"
    assert all(call[1]["params"]["safesearch"] == "true" and call[1]["timeout"] == 30 for call in calls)


def nasa_item(kind: str) -> dict:
    return {"data": [{"nasa_id": "PIA123", "media_type": kind, "title": "Great Red Spot",
                      "description": "Jupiter storm"}],
            "links": [{"render": "image", "href": "https://images-assets.nasa.gov/PIA123~medium.jpg"}]}


@pytest.mark.parametrize("kind,assets,expected", [
    ("image", [], ".jpg"),
    ("video", ["~orig.mp4", "~large.mp4", "~medium.mp4", "~thumb.jpg"], "~medium.mp4"),
    ("video", ["~orig.mp4", "~thumb.jpg", ".mov"], ".jpg"),
])
def test_nasa_parses_and_resolves_assets_without_key(monkeypatch: pytest.MonkeyPatch, kind: str,
                                                   assets: list, expected: str) -> None:
    calls = []
    def get(url: str, **kwargs: dict) -> Mock:
        calls.append((url, kwargs))
        if url.endswith("/search"):
            return response({"collection": {"items": [nasa_item(kind)]}})
        assert url.endswith("/asset/PIA123")
        return response({"collection": {"items": [{"href": "https://images-assets.nasa.gov/PIA123" + suffix}
                                                   for suffix in assets]}})
    monkeypatch.setattr(requests, "get", get)
    results = NasaProvider().search("Great Red Spot", "portrait", 20)
    assert len(results) == 1 and results[0].download_url.endswith(expected)
    assert results[0].media_type == ("video" if expected.endswith("mp4") else "image")
    assert results[0].source_id == "PIA123" and results[0].description == "Jupiter storm"
    assert calls[0][1]["params"] == {"q": "Great Red Spot", "media_type": "image,video", "page_size": 20}
    assert "key" not in str(calls)


def test_nasa_failed_asset_uses_preview(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(requests, "get", Mock(side_effect=[
        response({"collection": {"items": [nasa_item("video")]}}), requests.Timeout("timeout")]))
    assert NasaProvider().search("Jupiter", "", 20)[0].media_type == "image"


def test_ranking_prefers_relevant_nasa_but_retains_synonyms() -> None:
    unrelated = candidate("1")
    unrelated.title = "office meeting bowl of rice"
    relevant = candidate("2", "image", "nasa")
    synonym = candidate("3")
    synonym.title = "distant celestial body"
    results = search.rank_candidates([unrelated, synonym, relevant], "portrait", "Jupiter")
    assert results == [relevant, synonym, unrelated]


def select(monkeypatch: pytest.MonkeyPatch, groups: dict, target: int) -> tuple[list, list]:
    calls, downloads = [], []
    def find(query: str, *args: object, **kwargs: object) -> list:
        calls.append(query)
        return list(groups.get(query, []))
    def download(item: MediaCandidate) -> str:
        downloads.append(item)
        return item.download_url
    monkeypatch.setattr(pipeline, "search_media_candidates", find)
    monkeypatch.setattr(pipeline, "download_media", download)
    pipeline.select_media(["a", "b"], "Jupiter", "portrait", target, lambda: None, lambda *args: None)
    return calls, downloads


def test_selection_global_id_url_deduplication_and_broader_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    one = candidate("1")
    calls, downloaded = select(monkeypatch, {
        "a": [one, candidate("1", url="https://example.com/alternate"), candidate("2", url=one.download_url)],
        "b": [candidate("1", provider="pixabay")], "Jupiter": [candidate("3")],
    }, 3)
    assert [(item.provider, item.source_id) for item in downloaded] == [("pexels", "1"), ("pixabay", "1"), ("pexels", "3")]
    assert calls == ["a", "b", "Jupiter"]


@pytest.mark.parametrize("enabled,limit,expected", [("true", "2", 2), ("false", "4", 0), ("true", "0", 0)])
def test_image_fallback_cap_and_enable_setting(monkeypatch: pytest.MonkeyPatch, enabled: str,
                                                limit: str, expected: int) -> None:
    monkeypatch.setenv("ENABLE_IMAGE_FALLBACK", enabled)
    monkeypatch.setenv("MAX_IMAGE_CLIPS", limit)
    _, downloaded = select(monkeypatch, {
        "a": [candidate(str(i), "image", "pixabay") for i in range(6)] + [candidate("v1")],
        "b": [candidate("v2")],
    }, 10)
    assert sum(item.media_type == "video" for item in downloaded) == 2
    assert sum(item.media_type == "image" for item in downloaded) == expected


def test_strong_nasa_image_beats_poor_video(monkeypatch: pytest.MonkeyPatch) -> None:
    calls, downloaded = select(monkeypatch, {"a": [candidate("mediocre", score=4), candidate("spare", score=3)],
                                           "b": [candidate("good"), candidate("image", "image", "nasa", score=10)]}, 2)
    assert [item.source_id for item in downloaded] == ["image", "good"]
    assert calls == ["a", "b"]


def test_single_item_fails_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(RuntimeError, match="At least 2 unique"):
        select(monkeypatch, {"a": [candidate("1")]}, 10)


def test_selection_cancellation_is_not_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    def cancel() -> None:
        raise pipeline.PipelineCancelled()
    with pytest.raises(pipeline.PipelineCancelled):
        pipeline.select_media(["a"], "Jupiter", "", 10, cancel, lambda *args: None)


@pytest.mark.parametrize("size", [(90, 160), (160, 90), (100, 100), (80, 100)])
def test_image_clip_crop_zoom_and_duration(tmp_path: Path, size: tuple) -> None:
    path = tmp_path / "still.png"
    pixels = np.zeros((180, 320, 3), dtype=np.uint8)
    pixels[:, :, 0] = np.arange(320, dtype=np.uint16) % 256
    Image.fromarray(pixels).save(path)
    clip = video.create_image_clip(str(path), 3, *size)
    try:
        assert tuple(clip.size) == size and clip.fps == 30 and clip.duration == 3
        assert clip.get_frame(0).shape == (size[1], size[0], 3)
        assert not np.array_equal(clip.get_frame(0), clip.get_frame(2.9))
    finally:
        clip.close()


def test_real_mixed_render_duration_and_crop(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(video, "TEMP_DIR", tmp_path)
    still = tmp_path / "image.png"
    Image.new("RGB", (160, 90), "red").save(still)
    movie = tmp_path / "source.mp4"
    with ColorClip((160, 90), color=(0, 0, 255), duration=3) as clip:
        clip.write_videofile(str(movie), fps=30, codec="libx264", logger=None)
    result = video.combine_videos([str(still), str(movie)], 5, 2, 1, 90, 160)
    with VideoFileClip(result) as clip:
        assert tuple(clip.size) == (90, 160) and clip.fps == 30
        assert abs(clip.duration - 5) <= 1 / 30
        assert clip.get_frame(0.5)[0, 0, 0] > 200
        assert clip.get_frame(2.5)[0, 0, 2] > 200
        assert clip.get_frame(4.5)[0, 0, 0] > 200


def test_download_streams_image_with_extension_and_timeout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from io import BytesIO
    content = BytesIO()
    Image.new("RGB", (40, 60), "red").save(content, format="PNG")
    mocked = Mock()
    mocked.__enter__ = Mock(return_value=mocked)
    mocked.__exit__ = Mock(return_value=False)
    mocked.iter_content.return_value = [content.getvalue()[:20], b"", content.getvalue()[20:]]
    get = Mock(return_value=mocked)
    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setenv("MEDIA_DOWNLOAD_TIMEOUT", "12")
    result = video.download_media(candidate("1", "image", url="https://example.com/download?id=1"), str(tmp_path))
    assert Path(result).suffix == ".png"
    assert get.call_args.kwargs["stream"] is True and get.call_args.kwargs["timeout"] == 12
    assert "User-Agent" in get.call_args.kwargs["headers"]
    mocked.raise_for_status.assert_called_once()


def test_invalid_image_download_is_removed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    mocked = Mock()
    mocked.__enter__ = Mock(return_value=mocked)
    mocked.__exit__ = Mock(return_value=False)
    mocked.iter_content.return_value = [b"<html>not media</html>"]
    monkeypatch.setattr(requests, "get", Mock(return_value=mocked))
    with pytest.raises(OSError):
        video.download_media(candidate("1", "image"), str(tmp_path))
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("kind", ["image", "video"])
def test_nasa_twenty_results_resolve_only_five_strongest(monkeypatch: pytest.MonkeyPatch, kind: str) -> None:
    items = []
    for index in range(20):
        item = nasa_item(kind)
        item["data"][0].update(nasa_id=f"ID{index}", title="Great Red Spot" if index >= 15 else "office meeting",
                               description="")
        items.append(item)
    calls = []
    def get(url: str, **kwargs: dict) -> Mock:
        calls.append(url)
        if url.endswith("/search"):
            return response({"collection": {"items": items}})
        assert "/asset/" in url  # Search never downloads any media bytes.
        return response({"collection": {"items": [{"href": "https://example.com/scene~large.jpg"}]}})
    monkeypatch.setattr(requests, "get", get)
    results = NasaProvider().search("Great Red Spot", "", 20)
    assert [item.source_id for item in results] == [f"ID{index}" for index in range(15, 20)]
    assert len(calls) == 6
    assert [url.rsplit("/", 1)[-1] for url in calls[1:]] == [f"ID{index}" for index in range(15, 20)]


@pytest.mark.parametrize("assets,preview,expected", [
    (["~orig.jpg", "~thumb.jpg", "~medium.jpg", "~large.jpg"], "preview.jpg", "~large.jpg"),
    (["~orig.jpg", "~small.jpg", "~medium.jpg"], "preview.jpg", "~medium.jpg"),
    (["~orig.jpg", "~small.jpg", "~thumb.jpg"], "preview.jpg", "preview.jpg"),
    (["~orig.jpg", "~small.jpg"], "~thumb.jpg", "~small.jpg"),
    (["~small.jpg"], None, "~small.jpg"),
])
def test_nasa_image_rendition_priority(monkeypatch: pytest.MonkeyPatch, assets: list,
                                      preview: str | None, expected: str) -> None:
    item = nasa_item("image")
    item["links"] = [{"render": "image", "href": "https://example.com/" + preview}] if preview else []
    monkeypatch.setattr(requests, "get", Mock(side_effect=[
        response({"collection": {"items": [item]}}),
        response({"collection": {"items": [{"href": "https://example.com/" + asset} for asset in assets]}}),
    ]))
    assert NasaProvider().search("Great Red Spot", "", 20)[0].download_url.endswith(expected)


def test_nasa_image_lookup_failure_preserves_preview_and_other_results(monkeypatch: pytest.MonkeyPatch) -> None:
    first, second = nasa_item("image"), nasa_item("image")
    second["data"][0]["nasa_id"] = "PIA456"
    monkeypatch.setattr(requests, "get", Mock(side_effect=[
        response({"collection": {"items": [first, second]}}), requests.Timeout(),
        response({"collection": {"items": [{"href": "https://example.com/~large.jpg"}]}}),
    ]))
    results = NasaProvider().search("Great Red Spot", "", 20)
    assert len(results) == 2
    assert results[0].download_url == first["links"][0]["href"]
    assert results[1].download_url.endswith("~large.jpg")


def test_pixabay_normal_pass_never_calls_image_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEDIA_PROVIDERS", "pixabay")
    monkeypatch.setenv("PIXABAY_API_KEY", "private-api-key")
    get = Mock(return_value=response({"hits": []}))
    monkeypatch.setattr(requests, "get", get)
    search.search_media_candidates("Jupiter", "portrait", "Jupiter")
    assert [call.args[0] for call in get.call_args_list] == ["https://pixabay.com/api/videos/"]
    search.search_media_candidates("Jupiter", "portrait", "Jupiter", search_images=True)
    assert [call.args[0] for call in get.call_args_list] == ["https://pixabay.com/api/videos/", "https://pixabay.com/api/"]


@pytest.mark.parametrize("enough_videos", [True, False])
def test_pipeline_requests_images_only_after_video_shortfall(monkeypatch: pytest.MonkeyPatch, enough_videos: bool) -> None:
    calls, downloaded = [], []
    def find(query: str, *args: object, search_images: bool = False) -> list:
        calls.append((query, search_images))
        if search_images:
            assert calls[:3] == [("a", False), ("b", False), ("Jupiter", False)]
            return [candidate("duplicate", "image", "pixabay", url="https://example.com/pexels/v1"),
                    candidate("image", "image", "pixabay")]
        return [candidate("v1")] + ([candidate("v2")] if enough_videos else [])
    monkeypatch.setattr(pipeline, "search_media_candidates", find)
    monkeypatch.setattr(pipeline, "download_media", lambda item: downloaded.append(item) or item.download_url)
    paths = pipeline.select_media(["a", "b"], "Jupiter", "portrait", 2, lambda: None, lambda *args: None)
    assert len(paths) == len(set(paths)) == 2
    assert sum(image_pass for _, image_pass in calls) == (0 if enough_videos else 3)
    assert [item.media_type for item in downloaded] == ["video", "video" if enough_videos else "image"]


def test_pixabay_search_cache_persists_without_api_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    key = "private-api-key-do-not-persist"
    get = Mock(return_value=response({"hits": []}))
    monkeypatch.setattr(requests, "get", get)
    PixabayProvider(key).search("Jupiter", "", 20)
    # A fresh provider instance (or worker) shares the persistent cache.
    PixabayProvider("replacement-api-key").search("Jupiter", "", 20)
    assert get.call_count == 1
    files = list((tmp_path / "cache").rglob("*.json"))
    assert len(files) == 1
    assert key not in str(files[0]) + files[0].read_text()
    assert '"key"' not in files[0].read_text()
    params = {"q": "Jupiter"}
    assert cache.cache_path("video", {**params, "key": key}) == cache.cache_path("video", params)
    assert cache.cache_path("video", params) != cache.cache_path("image", params)
    assert cache.cache_path("video", params) != cache.cache_path("video", {"q": "Mars"})


@pytest.mark.parametrize("age,expected_calls", [(86399, 1), (86400, 2), (86401, 2)])
def test_pixabay_cache_expires_after_24_hours(monkeypatch: pytest.MonkeyPatch, age: int, expected_calls: int) -> None:
    now = 1_000_000
    monkeypatch.setattr(cache.time, "time", lambda: now)
    get = Mock(return_value=response({"hits": []}))
    monkeypatch.setattr(requests, "get", get)
    provider = PixabayProvider("private-api-key")
    provider.search("Jupiter", "", 20)
    now += age
    provider.search("Jupiter", "", 20)
    assert get.call_count == expected_calls


@pytest.mark.parametrize("corrupt", ["broken JSON", "[]", '{"created_at": "bad", "response": {}}'])
def test_pixabay_corrupt_cache_is_refetched(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, corrupt: str) -> None:
    get = Mock(return_value=response({"hits": []}))
    monkeypatch.setattr(requests, "get", get)
    provider = PixabayProvider("private-api-key")
    provider.search("Jupiter", "", 20)
    path = next((tmp_path / "cache").rglob("*.json"))
    path.write_text(corrupt)
    provider.search("Jupiter", "", 20)
    assert get.call_count == 2 and cache.read_search_cache(path) == {"hits": []}


def test_pixabay_http_failures_are_not_cached(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    failed = response({"hits": []})
    failed.raise_for_status.side_effect = requests.HTTPError("secret credential URL")
    get = Mock(side_effect=[failed, response({"hits": []})])
    monkeypatch.setattr(requests, "get", get)
    provider = PixabayProvider("private-api-key")
    assert provider.search("Jupiter", "", 20) == []
    assert not list((tmp_path / "cache").rglob("*.json"))
    provider.search("Jupiter", "", 20)
    assert get.call_count == 2


def test_pixabay_does_not_cache_echoed_credentials(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    key = "private-api-key"
    monkeypatch.setattr(requests, "get", Mock(return_value=response({"hits": [], "unexpected": key})))
    PixabayProvider(key).search("Jupiter", "", 20)
    assert not list((tmp_path / "cache").rglob("*.json"))


def test_pixabay_cache_concurrent_writes_are_atomic(tmp_path: Path) -> None:
    path = tmp_path / "shared.json"
    def write(index: int) -> None:
        cache.write_search_cache(path, {"hits": [{"id": index, "tags": "x" * 10000}]})
        assert cache.read_search_cache(path) is not None
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(write, range(20)))
    assert len(json.loads(path.read_text())["response"]["hits"]) == 1
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("size", [(90, 160), (160, 90), (100, 100), (80, 100)])
def test_combine_does_not_recrop_or_resize_target_sized_images(monkeypatch: pytest.MonkeyPatch,
                                                             tmp_path: Path, size: tuple) -> None:
    paths = []
    for name in ("a", "b"):
        path = tmp_path / f"{name}.png"
        Image.new("RGB", (160, 90), "red").save(path)
        paths.append(str(path))
    monkeypatch.setattr(video, "TEMP_DIR", tmp_path)
    def redundant(*args: object, **kwargs: object) -> None:
        pytest.fail("Image-derived clip received an unnecessary crop/resize")
    def write(clip: video.VideoClip, *args: object, **kwargs: object) -> None:
        assert tuple(clip.size) == size and clip.duration == 5 and clip.fps == 30
    monkeypatch.setattr(video.VideoClip, "cropped", redundant)
    monkeypatch.setattr(video.VideoClip, "resized", redundant)
    monkeypatch.setattr(video.VideoClip, "write_videofile", write)
    video.combine_videos(paths, 5, 2, 1, *size)
