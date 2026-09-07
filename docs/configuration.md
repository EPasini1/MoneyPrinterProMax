# Configuration

MoneyPrinter reads configuration from `.env` (project root).

Use `.env.example` as your template.

## Required

| Variable | Description |
|---|---|
| `TIKTOK_SESSION_ID` | TikTok session cookie (`sessionid`) used for TTS voice endpoint calls. |
| `PEXELS_API_KEY` | API key used to fetch stock video clips. |

## Optional

| Variable | Description | Default |
|---|---|---|
| `IMAGEMAGICK_BINARY` | Absolute path to ImageMagick executable. If empty, auto-detected from `PATH`. | auto-detect |
| `OLLAMA_BASE_URL` | Ollama server base URL used for model listing and chat generation. | `http://localhost:11434` |
| `OLLAMA_MODEL` | Fallback model if frontend does not send a model value. | `llama3.1:8b` |
| `STOCK_VIDEO_COUNT` | Target number of unique stock clips per video. Missing, non-integer, or non-positive values use `10`. Caller-provided overrides still take precedence. | `10` |
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
- Docker Compose forwards all three generation settings to the backend and worker. Recreate those services after changing their environment.
- Search generation requests exactly the configured count of 2–5 word queries. It deduplicates case-insensitively and makes up to three additional attempts for missing terms, then uses subject/script-derived fallback queries. Fallback queries can describe the same subject with different framing; they do not guarantee different search results.
- Pexels candidates retain video IDs so alternate renditions cannot count as different sources. Lexical page-URL overlap strongly boosts ranking; no overlap incurs a penalty without excluding possible synonyms. Unmatched food, office, people, or abstract-background metadata receives an additional penalty when that category is absent from the query. Numeric-only URLs have neutral relevance. Clips meeting the duration target win relevance ties, with API order preserved on remaining ties. This is lexical scoring, not visual verification, and can miss misleading metadata.
- Rendition selection prefers MP4 around 1080x1920 (portrait), 1920x1080 (landscape), or 1080x1080 (square). It considers both dimensions for crop quality, prefers 720p over oversized downloads when 1080p is unavailable, and uses larger files before sub-720p footage if no reasonable rendition exists.
- The default 10 clips at 6 seconds each provide approximately 60 seconds of unique footage before cycling, subject to available source durations.
- Downloads use spare candidates across queries to reach the requested count where possible. Shorter clips remain usable as a fallback. If fewer sources are available, the job warns and cycles through every unique clip before reusing it; with one source, repetition is unavoidable. Zero usable sources fails with an explicit error.
- Segments last up to `MAX_CLIP_DURATION` (shorter for short sources or the final audio remainder). Existing aspect-ratio cropping and 30 fps rendering apply to every segment.
- Script verification adds one model request and reviews claims using the model's knowledge, without independent source retrieval. It is not a guarantee of factual accuracy. The corrected narration is used for search, TTS, subtitles, and metadata; approximate length/fact count are prompt constraints, with accuracy taking priority. Model failures propagate as job errors rather than silently skipping enabled verification.
