# Configuration

MoneyPrinter reads configuration from `.env` (project root).

Use `.env.example` as your template.

## Frontend dashboard

At viewport widths of 1080px and above, generation settings and a sticky Live Output
panel appear side by side; smaller screens stack logs below the form. Generation
Settings are always visible, with no collapsible or inert settings container.
Model/voice, aspect ratio/subtitle position, color/threads, and paragraphs/duration
share paired rows; songs and custom prompt use the full form width. Live Output
retains logs, the final status, and any full failure message until a new generation
or an explicit Clear action (Clear removes log entries, retaining the status summary).
Logs are kept only in the current page session, not across page reloads.
Auto-scroll starts enabled and switches off when scrolling away from the bottom.
The toolbar can copy timestamped plain text or expand the same viewer; Escape closes
expanded mode and returns keyboard focus. The elapsed timer stops at a terminal state.

Job polling remains every 1.2 seconds, with overlapping requests suppressed. Temporary
failures warn at most once per 30 seconds during an outage; recovery is logged separately.
Failed refreshes do not mark the generation failed. Terminal jobs drain their remaining
events, restore Generate, and keep the output visible.

Generation preferences are saved immediately in the browser's single localStorage
object `moneyPrinter.settings.v1`: model, voice, aspect ratio, subtitle position/color,
threads, paragraphs, minimum duration, Upload to YouTube, Use Music, Reuse Choices,
and Custom Prompt. Restoration is independent of the Reuse Choices toggle. Video
Subject, songs/file selections, uploaded paths, and backend credentials are never
stored by this preference feature. Previous per-field storage keys are no longer read
or written. Preferences are local to the browser and frontend origin, with no backend
or database persistence.

Saved values are validated against current options, numeric limits, and expected types.
Models are restored only after the current Ollama model list loads. Missing/stale values
keep normal defaults; the default voice is English (US) → Male 2 (`en_us_007`). Styled
native selects display the restored option label and the subtitle color swatch updates.
Reset settings removes the storage object and restores defaults without changing the
subject, file selection, or Live Output. File inputs are never restored programmatically
and remain empty after reload. Storage failures leave the form usable and show a notice.
The script reference includes a cache version (`app.js?v=settings-v1`); bump this value
when shipping subsequent frontend script changes.

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
| `MEDIA_MIN_SCORE` | Minimum candidate score for every selection pass. Invalid/blank/non-finite values use 3.0; any finite value is accepted. Never reduced automatically to fill the target. | `3.0` |
| `MEDIA_DOWNLOAD_TIMEOUT` | Positive finite HTTP download timeout in seconds; invalid values use 60. This bounds socket inactivity, not total download time. | `60` |
| `MEDIA_CACHE_DIR` | Persistent Pixabay JSON search cache root; blank uses the default below. Compose uses a named volume. | `~/.cache/MoneyPrinterProMax/media` |
| `MEDIA_SEARCH_TIMEOUT` | Positive finite timeout for each provider search/asset request; invalid values use 30. | `30` |
| `IMAGEMAGICK_BINARY` | Absolute path to ImageMagick executable. If empty, auto-detected from `PATH`. | auto-detect |
| `OLLAMA_BASE_URL` | Ollama server base URL used for model listing and chat generation. | `http://localhost:11434` |
| `OLLAMA_MODEL` | Fallback model if frontend does not send a model value. | `llama3.1:8b` |
| `OLLAMA_THINK` | Thinking is explicitly disabled for chat and generate. Values requesting thinking warn and are ignored. Legacy clients/servers that reject the argument retry without it. | `false` |
| `STOCK_VIDEO_COUNT` | Baseline number of unique media items (videos plus stills). Missing, non-integer, or non-positive values use `10`. Caller overrides take precedence as the baseline; actual audio can raise the target, subject to the 120-source cap. | `10` |
| `MAX_CLIP_DURATION` | Maximum seconds per stock segment; finite numeric values are clamped to 2–10. Invalid values use `6`. | `6` |
| `SCRIPT_FACT_CHECK` | Review narration with the selected Ollama model before stock search and TTS. Enabled by `true`, `1`, `yes`, or `on` (case-insensitive); use `false` to disable. | `true` |
| `FACT_CHECK_REQUIRE_SOURCES` | Fail before rewriting/TTS if no usable references are retrieved. Explicit false/0/no/off permits LLM-only review with a warning; empty/invalid values remain fail-safe. | `true` |
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
- After TTS, the pipeline measures the written audio file and selects `max(STOCK_VIDEO_COUNT, ceil(audio_duration / MAX_CLIP_DURATION))` sources, bounded to 2–120. For 65.1 seconds at 6 seconds per clip, it requests 11 unique sources. The duration, required count, and final target are logged; exceeding the cap warns. Media queries use the configured baseline (also bounded by the final target), rather than growing with audio duration. As a result, a media shortfall is now discovered after TTS. Short source durations or insufficient quality sources can still require cycling.
- Downloads use spare candidates across queries to reach the requested count where possible. Shorter clips remain usable as a fallback. If fewer sources are available, the job warns and cycles through every unique clip before reusing it; at least two unique usable items are required, and zero or one usable item fails with an explicit error.
- Segments last up to `MAX_CLIP_DURATION` (shorter for short sources or the final audio remainder). Existing aspect-ratio cropping and 30 fps rendering apply to every segment.
- Script verification retrieves references before making one model rewrite request. Unsupported claims must be removed without telling viewers they lacked support. Structural editorial labels (such as CLAIMS or VERDICT) and verification language (such as "provided sources", "available information", or "reference material") trigger exactly one narration-only retry using the same references; empty responses or repeated contamination fail before TTS. Normal factual uses such as "source of energy" and "source code" remain allowed. The final narration is logged and used for search, TTS, subtitles, and metadata. Model failures propagate as job errors rather than silently skipping enabled verification.

## Source-grounded factual verification

`Backend/fact_sources.py` defines a `FactSourceProvider` interface for future providers.
Space topics prefer NASA: Jupiter uses its public
[facts page](https://science.nasa.gov/jupiter/jupiter-facts/); subsequent claim searches
and other astronomy topics use textual descriptions from the NASA Image and Video Library API.
General topics use English Wikipedia's [TextExtracts API](https://www.mediawiki.org/wiki/Extension:TextExtracts),
also used when a NASA query returns no usable text or fails. No keys or Google scraping are involved.
The English baseline can have limited coverage for queries in other languages.

Retrieval derives at most four distinct queries from the subject and original narration,
without extra LLM calls. Each provider returns at most two results per query; the final
context contains at most four unique sources, each with at most 2,500 characters.
Wikipedia requests respect its API's 1,200-character extract parameter limit.
Excerpts prioritize sentences overlapping the subject/claims, preserving source order.
Empty/short (under 80 characters) and lexically unrelated excerpts are discarded.
Responses use a User-Agent, a 5-second socket timeout, streamed body/time limits, and a
512,000-byte response cap. A 20-second retrieval budget is checked before each request;
an in-flight request can finish after that budget. Failed providers are skipped for the
rest of that retrieval. There are at most eight provider requests and no automatic HTTP retries.

The rewrite prompt permits factual claims only from the supplied excerpts, correcting
contradictions and removing unsupported specificity, analogies, and replacement facts.
References are delimited as JSON data; they are not instructions. The sources' provider/title
and final narration are logged to the job, without logging excerpts. No usable sources
causes `Fact-check could not retrieve sufficient reference material.` by default.
With `FACT_CHECK_REQUIRE_SOURCES=false`, an empty retrieval instead logs an explicit
LLM-only warning and labels the output as unverified by sources.

This is source-grounded rewriting, not a deterministic proof of each claim: retrieval can
miss context, pages can be stale, and models can still make mistakes. Mocked Jupiter tests
verify that references and constraints reach the model; they do not measure live model accuracy.
Ollama requests explicitly send `think=False` as described in its
[thinking API documentation](https://docs.ollama.com/capabilities/thinking). Compatibility
fallback omits unsupported arguments; older clients and models that cannot disable thinking
cannot guarantee the same latency. Thinking is never explicitly enabled by this application.

## Media provider pipeline

`Backend/providers/base.py` defines `MediaCandidate`. Providers normalize external JSON;
`Backend/search.py` routes and ranks it, `pipeline.select_media` downloads unique sources,
and `video.combine_videos` accepts both video and image paths. Legacy Pexels search and
`save_video` interfaces remain available.

General topics call Pexels, Pixabay, then NASA. Space keywords in either subject or query
switch priority to NASA, Pixabay, then Pexels. Routing is deterministic and uses word
boundaries, without an LLM. Provider order breaks remaining ties after query diversity; all enabled providers are
queried in the normal pass, with Pixabay searching videos only. Failures fall through without printing credential-bearing exception URLs.
NASA requires no key: see the [NASA library API reference](https://images.nasa.gov/docs/images.nasa.gov_api_docs.pdf).
Pixabay uses the official [video and photo APIs](https://pixabay.com/api/docs/).

Selection gathers all normal queries and orders candidates globally by score, with
images competing immediately alongside videos when `ENABLE_IMAGE_FALLBACK` is enabled.
Only exact score ties favor queries without a successfully selected source yet, then
preserve provider/source input order. Any higher score wins over query diversity.
Every pass enforces `MEDIA_MIN_SCORE` and the job-wide `MAX_IMAGE_CLIPS` cap.
Successful sources deduplicate globally by `(provider, source_id)` and download URL.
Failed URLs are not retried, but another rendition of the same failed ID may be tried.
If below target, exactly one broader subject search runs. If slots and image allowance
still remain, the fallback pass gathers Pixabay photos and NASA image-only results
across queries and ranks them globally. Successful IDs/URLs stay
deduplicated across passes; Pexels is not queried again for images.
Below-floor candidates are never downloaded just to fill slots. A shortfall warns and
cycles the relevant sources; fewer than two unique usable items fails. Selection logs
include provider, type, source ID, score, and query, plus video/image and rejected counts.
Searches are limited to the first page (up to 20 items per provider/media search).

Cross-provider scores prioritize meaningful subject tokens above query overlap and
technical bonuses. Generic title/action words and numbers are ignored before and after
singularization (for example, "How AI works" anchors on "ai", not "work"). Technical
terms such as Docker, Kubernetes, and neural network remain meaningful. Named space objects
receive stronger matching boosts, missing-entity penalties, and conflict penalties
when metadata names another object without the requested one. NASA receives a stronger
space bonus only with lexical relevance evidence. Orientation, resolution, and duration
still contribute. Direct query/subject matches in titles add up to 10 points, separating
direct imagery from incidental description mentions. NASA video titles suggesting
program material (live event, starting soon, science live, planetary protection,
interview, podcast, briefing, press conference, episode) lose 12 points for one marker,
with additional markers increasing the penalty up to 20 points. These are soft penalties;
strongly relevant programs remain eligible above the unchanged quality floor. This uses
metadata only and does not detect actual title-card frames.

Query coverage (matched meaningful tokens divided by total meaningful tokens) also
contributes to relevance. Multiword queries require at least two matching tokens or
a strong subject/domain anchor; weak coverage in queries of three or more tokens
receives an additional penalty. Candidates lacking that evidence have their final
score capped at -4, below the default quality floor, regardless of provider or technical
bonuses. Adjacent meaningful query phrases receive a 6-point boost per matching pair
(up to 12), with shared plural normalization: "tectonic plates" matches "tectonic plate".
Punctuation and title/description boundaries break phrases. Ambiguous components such
as "plate", "chamber", "network", and "intelligence" cannot anchor the subject alone.
Reordered terms such as "plate tectonics animation" still qualify through token coverage.

An extensible contextual vocabulary covers geology, computing, astronomy, cooking,
business, music, and military subjects. Shared distinctive context can support related
footage with different wording; two distinctive metadata terms in an unrelated domain
incur a 12-point penalty when there is no shared domain. Thus kitchen/bread footage
loses relevance for a tectonic query, while remaining valid for a cooking subject.
These heuristics depend on metadata quality and a limited vocabulary, so synonyms
outside that vocabulary may be missed. The quality floor is never lowered to fill a target.
This does not inspect visual content. NASA manifests often omit
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
