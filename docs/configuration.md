# Configuration

MoneyPrinter reads configuration from `.env` (project root).

Use `.env.example` as your template.

## Required

| Variable | Description |
|---|---|
| `TIKTOK_SESSION_ID` | TikTok session cookie (`sessionid`) used for TTS voice endpoint calls. |

## Optional

| Variable | Description | Default |
|---|---|---|
| `PEXELS_API_KEY` | Enables Pexels video search; blank disables it with one warning per process. | empty |
| `PIXABAY_API_KEY` | Enables Pixabay video/photo search; blank disables it with one warning per process. | empty |
| `NASA_API_KEY` | Reserved for other NASA APIs; unused by the Image and Video Library. | empty |
| `MEDIA_PROVIDERS` | Comma-separated enabled names, case-insensitive. Unknown names warn and are ignored. Empty/no usable providers fails generation clearly. | `pexels,pixabay,nasa` |
| `ENABLE_IMAGE_FALLBACK` | Allow stills, including strongly relevant NASA imagery; true/1/yes/on enable. | `true` |
| `MAX_IMAGE_CLIPS` | Maximum unique still sources per job; zero disables stills. Invalid/negative values use 4. Repeated scenes do not consume additional source slots. | `4` |
| `MEDIA_DOWNLOAD_TIMEOUT` | Positive finite HTTP download timeout in seconds; invalid values use 60. This bounds socket inactivity, not total download time. | `60` |
| `MEDIA_CACHE_DIR` | Persistent Pixabay JSON search cache root; blank uses the default below. Compose uses a named volume. | `~/.cache/MoneyPrinterProMax/media` |
| `MEDIA_SEARCH_TIMEOUT` | Positive finite timeout for each provider search/asset request; invalid values use 30. | `30` |
| `IMAGEMAGICK_BINARY` | Absolute path to ImageMagick executable. If empty, auto-detected from `PATH`. | auto-detect |
| `OLLAMA_BASE_URL` | Ollama server base URL used for model listing and chat generation. | `http://localhost:11434` |
| `OLLAMA_MODEL` | Fallback model if frontend does not send a model value. | `llama3.1:8b` |
| `STOCK_VIDEO_COUNT` | Target number of unique media items (videos plus stills) per output. Missing, non-integer, or non-positive values use `10`. Caller-provided overrides still take precedence. | `10` |
| `MAX_CLIP_DURATION` | Maximum seconds per stock segment; finite numeric values are clamped to 2–10. Invalid values use `6`. | `6` |
| `SCRIPT_FACT_CHECK` | Review narration with the selected Ollama model before stock search and TTS. Enabled by `true`, `1`, `yes`, or `on` (case-insensitive); use `false` to disable. | `true` |
| `ASSEMBLY_AI_API_KEY` | If set, subtitles are generated with AssemblyAI; otherwise local subtitle generation is used. | empty |
| `POSTGRES_DB` | Database name for Docker Postgres service. | `moneyprinter` |
| `POSTGRES_USER` | Database user for Docker Postgres service. | `moneyprinter` |
| `POSTGRES_PASSWORD` | Database password for Docker Postgres service. | `moneyprinter` |
| `DATABASE_URL` | SQLAlchemy DSN used by API and worker (`postgresql+psycopg://...` or `sqlite:///...`). | `sqlite:///moneyprinter.db` |

## Notes

- Ollama models shown in the frontend are fetched from backend endpoint `/api/models`, which queries `OLLAMA_BASE_URL/api/tags`.
- Pull models before use, for example:

```bash
ollama pull llama3.1:8b
```

- If ImageMagick is not discovered automatically, set `IMAGEMAGICK_BINARY` explicitly.
- New architecture uses a database-backed job queue. In Docker, use Postgres via `DATABASE_URL`.
- Docker Compose forwards media provider and generation settings to the backend and worker. Recreate those services after changing their environment.
- Search generation requests exactly the configured count of 2–5 word queries. It deduplicates case-insensitively and makes up to three additional attempts for missing terms, then uses subject/script-derived fallback queries. Fallback queries can describe the same subject with different framing; they do not guarantee different search results.
- Pexels candidates retain video IDs so alternate renditions cannot count as different sources. Lexical page-URL overlap strongly boosts ranking; no overlap incurs a penalty without excluding possible synonyms. Unmatched food, office, people, or abstract-background metadata receives an additional penalty when that category is absent from the query. Numeric-only URLs have neutral relevance. Clips meeting the duration target win relevance ties, with API order preserved on remaining ties. This is lexical scoring, not visual verification, and can miss misleading metadata.
- Rendition selection prefers MP4 around 1080x1920 (portrait), 1920x1080 (landscape), or 1080x1080 (square). It considers both dimensions for crop quality, prefers 720p over oversized downloads when 1080p is unavailable, and uses larger files before sub-720p footage if no reasonable rendition exists.
- The default 10 clips at 6 seconds each provide approximately 60 seconds of unique footage before cycling, subject to available source durations.
- Downloads use spare candidates across queries to reach the requested count where possible. Shorter clips remain usable as a fallback. If fewer sources are available, the job warns and cycles through every unique clip before reusing it; at least two unique usable items are required, and zero or one usable item fails with an explicit error.
- Segments last up to `MAX_CLIP_DURATION` (shorter for short sources or the final audio remainder). Existing aspect-ratio cropping and 30 fps rendering apply to every segment.
- Script verification adds one model request and reviews claims using the model's knowledge, without independent source retrieval. It is not a guarantee of factual accuracy. The corrected narration is used for search, TTS, subtitles, and metadata; approximate length/fact count are prompt constraints, with accuracy taking priority. Model failures propagate as job errors rather than silently skipping enabled verification.

## Media provider pipeline

`Backend/providers/base.py` defines `MediaCandidate`. Providers normalize external JSON;
`Backend/search.py` routes and ranks it, `pipeline.select_media` downloads unique sources,
and `video.combine_videos` accepts both video and image paths. Legacy Pexels search and
`save_video` interfaces remain available.

General topics call Pexels, Pixabay, then NASA. Space keywords in either subject or query
switch priority to NASA, Pixabay, then Pexels. Routing is deterministic and uses word
boundaries, without an LLM. Provider order breaks score ties; all enabled providers are
queried in the normal pass, with Pixabay searching videos only. Failures fall through without printing credential-bearing exception URLs.
NASA requires no key: see the [NASA library API reference](https://images.nasa.gov/docs/images.nasa.gov_api_docs.pdf).
Pixabay uses the official [video and photo APIs](https://pixabay.com/api/docs/).

Selection takes one usable candidate per query per round, then uses spare candidates.
Successful sources deduplicate globally by `(provider, source_id)` and download URL.
Failed URLs are not retried, but another rendition of the same failed ID may be tried.
Videos are preferred. A NASA image scoring at least 8 may precede videos only when that
query has no video scoring at least 2. Other images fill remaining slots after the video
pool and one broader subject search are exhausted, subject to `MAX_IMAGE_CLIPS`.
The fallback pass reuses resolved NASA stills, then requests Pixabay photos and NASA
image-only results only if slots and image allowance remain. Successful IDs/URLs stay
deduplicated across passes; Pexels is not queried again for images.
Searches are limited to the first page (up to 20 items per provider/media search).

Cross-provider scores combine lexical overlap, exact query phrases, orientation,
resolution, duration, and NASA's space-topic bonus. Missing lexical overlap is penalized,
never a hard rejection. This does not inspect visual content. NASA manifests often omit
dimensions and duration, so those fields remain unknown rather than being guessed.
NASA ranks search-result title/description relevance before resolving manifests for at
most five distinct IDs per search call (normal or image-only pass), with API order breaking
ties. Lower-ranked results are not resolved. Video assets still prefer HD/named medium
MP4s, avoiding original/master/4K files. Image manifests use separate ranking: large,
medium, usable preview, then small. Originals/masters and thumbnails are last resorts.
An asset lookup failure preserves the search preview and does not stop other results.
Video-to-image fallback follows the same image ranking. Search resolves URLs only; it
does not download images. Preview resolution can be limited.

Downloads stream to `temp/`, validate decodable media, and delete partial/invalid files.
Image scenes center-crop to the chosen aspect ratio, animate a subtle 100-106% zoom,
and produce fixed-size frames at 30 fps. Image-derived clips already at the target size
skip the compositor's additional crop/resize. Only a target-sized still is retained for each
source, and video readers are reused across cycles. Composition uses each unique source
before repeating, never repeats immediately, and caps scenes at `MAX_CLIP_DURATION`.
The composition duration matches narration; encoded media is quantized to 30 fps frames.

Unit tests mock HTTP and need no API keys. Run `uv run python -m pytest tests -v`;
`tests/test_media_providers.py` also renders a tiny mixed scene using MoviePy's FFmpeg.

## Pixabay search cache

Pixabay requires [24-hour request caching](https://pixabay.com/api/docs/). Both video and
photo search responses are cached as JSON under `MEDIA_CACHE_DIR/pixabay/`, keyed by a
SHA-256 digest of the endpoint and non-secret parameters. API keys are excluded from
cache identity, filenames, and metadata; responses unexpectedly echoing the key are
not persisted. Different API keys share the same public search cache.

Blank `MEDIA_CACHE_DIR` defaults to `~/.cache/MoneyPrinterProMax/media`, outside the
per-job `temp/` cleanup. Docker backend and worker share the `media_cache` named volume
at `/app/media-cache` by default; setting `MEDIA_CACHE_DIR` changes its container mount
location. Restarts/recreates preserve it; `docker compose down -v` removes named volumes.

Entries expire after 24 hours. Expired/corrupt entries are ignored and replaced after a
successful request. HTTP failures are never cached. Atomic replacement from unique
same-directory temporary files prevents concurrent writers from exposing partial JSON.
Concurrent cache misses may still issue duplicate requests. Cache I/O failures warn
and allow generation to continue. Downloaded media is not cached here, and old entries
for queries never requested again are not automatically pruned.
