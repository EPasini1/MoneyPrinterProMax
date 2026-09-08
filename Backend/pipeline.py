import math
import os
import re
import shutil
import subprocess
from itertools import groupby
from typing import Callable, Optional

from apiclient.errors import HttpError
from moviepy import (
    AudioFileClip,
    CompositeAudioClip,
    VideoFileClip,
    afx,
    concatenate_audioclips,
)
from uuid import uuid4

from gpt import (
    generate_hashtags,
    generate_metadata,
    generate_script,
    get_search_terms,
    verify_and_rewrite_script,
)
from logstream import log
from providers.base import MediaCandidate, image_fallback_enabled, max_image_clips, media_min_score
from search import get_enabled_providers, search_media_candidates
from tiktokvoice import tts
from utils import (
    BASE_DIR,
    OUTPUT_DIR,
    PROJECT_ROOT,
    SONGS_DIR,
    SUBTITLES_DIR,
    TEMP_DIR,
    choose_random_song,
    get_max_clip_duration,
    get_stock_video_count,
)
from video import combine_videos, download_media, generate_subtitles, generate_video
from youtube import upload_video


class PipelineCancelled(Exception):
    pass


MAX_MEDIA_SCENES = 120


def media_requirement(audio_duration: float, max_clip_duration: float, configured_count: int) -> tuple[int, int]:
    if not math.isfinite(audio_duration) or audio_duration <= 0:
        raise RuntimeError("Narration audio must have a positive finite duration.")
    if not math.isfinite(max_clip_duration) or max_clip_duration <= 0:
        raise RuntimeError("Maximum clip duration must be positive and finite.")
    required = math.ceil(audio_duration / max_clip_duration)
    return required, min(MAX_MEDIA_SCENES, max(2, configured_count, required))


def select_media(
    search_terms: list[str], video_subject: str, orientation: str, target: int,
    guard_cancelled: Callable[[], None], emit: Callable[[str, str], None],
) -> list[str]:
    providers = get_enabled_providers(video_subject)
    emit("[Media] Providers enabled: " + ", ".join(provider.name for provider in providers), "info")
    emit(f"[Media] Target scenes: {target}", "info")
    image_limit = max_image_clips() if image_fallback_enabled() else 0
    minimum_score = media_min_score()
    paths: list[str] = []
    seen_ids: set[tuple[str, str]] = set()
    seen_urls: set[str] = set()
    attempted_urls: set[str] = set()
    rejected: set[tuple[str, str, str]] = set()
    represented_queries: set[str] = set()
    images = 0

    def take(candidate: MediaCandidate) -> bool:
        nonlocal images
        guard_cancelled()
        identity = (candidate.provider, candidate.source_id)
        if identity in seen_ids or candidate.download_url in seen_urls or candidate.download_url in attempted_urls:
            return False
        if candidate.media_type == "image" and images >= image_limit:
            return False
        attempted_urls.add(candidate.download_url)
        try:
            path = download_media(candidate)
        except PipelineCancelled:
            raise
        except Exception as err:
            emit(f"[Media] Download failed: {candidate.provider} / {candidate.source_id} ({type(err).__name__}).", "warning")
            return False
        paths.append(path)
        seen_ids.add(identity)
        seen_urls.add(candidate.download_url)
        images += candidate.media_type == "image"
        represented_queries.add(candidate.query)
        emit(f"[Media] Selected: {candidate.provider} / {candidate.media_type} / {candidate.source_id} / "
             f"score={candidate.score:.2f} / query={candidate.query!r}", "info")
        return True

    def fill(candidates: list[MediaCandidate]) -> None:
        eligible = []
        for candidate in candidates:
            guard_cancelled()
            if not math.isfinite(candidate.score) or candidate.score < minimum_score:
                rejected.add((candidate.provider, candidate.source_id, candidate.download_url))
            else:
                eligible.append(candidate)
        # Diversity only breaks exact score ties; a higher score always wins.
        # Stable input order breaks remaining ties, preserving provider/source
        # order. Only successful downloads count as represented queries.
        ordered = sorted(eligible, key=lambda item: -item.score)
        for _, group in groupby(ordered, key=lambda item: item.score):
            if len(paths) >= target:
                break
            tied = list(group)
            while tied and len(paths) < target:
                index = next((i for i, item in enumerate(tied)
                              if item.query not in represented_queries), 0)
                take(tied.pop(index))

    candidates = []
    for term in search_terms:
        guard_cancelled()
        candidates.extend(search_media_candidates(term, orientation, video_subject, 20))
    fill(candidates)
    if len(paths) < target:
        guard_cancelled()
        # Exactly one broader search, including when generated terms found nothing.
        fill(search_media_candidates(video_subject, orientation, video_subject, 20))
    if len(paths) < target and images < image_limit:
        # Pixabay photos are requested only after the normal and broad passes.
        candidates = []
        for term in dict.fromkeys([*search_terms, video_subject]):
            guard_cancelled()
            candidates.extend(search_media_candidates(term, orientation, video_subject, 20, search_images=True))
        fill(candidates)
    emit(f"[Media] Selected {len(paths)} unique media items", "info")
    emit(f"[Media] Videos: {len(paths) - images}", "info")
    emit(f"[Media] Images: {images}", "info")
    emit(f"[Media] Candidates rejected below threshold ({minimum_score:g}): {len(rejected)}", "info")
    if len(paths) < target:
        emit(f"[Media] Only {len(paths)} of {target} media items available; sources will cycle if needed.", "warning")
    if not paths:
        raise RuntimeError("No usable media could be downloaded from enabled providers.")
    if len(paths) < 2:
        raise RuntimeError("At least 2 unique media items are required; only one usable item was found.")
    return paths


def run_generation_pipeline(
    data: dict,
    is_cancelled: Optional[Callable[[], bool]],
    on_log: Optional[Callable[[str, str], None]],
    amount_of_stock_videos: Optional[int] = None,
) -> str:
    def emit(message: str, level: str = "info") -> None:
        log(message, level)
        if on_log:
            on_log(message, level)

    def guard_cancelled() -> None:
        if is_cancelled and is_cancelled():
            raise PipelineCancelled("Video generation was cancelled.")

    paragraph_number = int(data.get("paragraphNumber", 1))
    ai_model = data.get("aiModel")
    n_threads = data.get("threads") or 2
    subtitles_position = data.get("subtitlesPosition") or "center,bottom"
    text_color = data.get("color") or "#FFFF00"
    amount_of_stock_videos = get_stock_video_count(amount_of_stock_videos)
    max_clip_duration = get_max_clip_duration()
    emit(f"[+] Requested stock clip count: {amount_of_stock_videos}")

    # Aspect ratio -> (width, height, Pexels orientation). Default 9:16 vertical.
    aspect_presets = {
        "9:16": (1080, 1920, "portrait"),
        "16:9": (1920, 1080, "landscape"),
        "1:1": (1080, 1080, "square"),
        "4:5": (1080, 1350, "portrait"),
    }
    target_width, target_height, orientation = aspect_presets.get(
        data.get("aspectRatio") or "9:16", aspect_presets["9:16"]
    )
    use_music = data.get("useMusic", False)
    automate_youtube_upload = data.get("automateYoutubeUpload", False)

    emit("[Video to be generated]", "info")
    emit("   Subject: " + data["videoSubject"], "info")
    emit("   AI Model: " + str(ai_model), "info")
    emit("   Custom Prompt: " + data["customPrompt"], "info")

    guard_cancelled()

    voice = data.get("voice", "")
    voice_prefix = voice[:2]

    if not voice:
        emit('[!] No voice was selected. Defaulting to "en_us_001"', "warning")
        voice = "en_us_001"
        voice_prefix = voice[:2]

    # Minimum video length (seconds). TikTok TTS speaks ~2.4 words/sec, so we
    # target a word count with headroom and regenerate if the script is short.
    min_duration = int(data.get("minDuration") or 0)
    custom_prompt = data["customPrompt"]

    if min_duration > 0 and not custom_prompt:
        words_per_second = 2.4
        accept_words = int(min_duration * words_per_second)
        target_words = int(accept_words * 1.2)  # ask for ~20% extra as buffer
        # Cap so a chatty model doesn't produce a 3-minute video for a 60s floor.
        max_words = int(accept_words * 1.6)
        script = None
        for attempt in range(1, 4):
            guard_cancelled()
            script = generate_script(
                data["videoSubject"],
                paragraph_number,
                ai_model,
                voice,
                custom_prompt,
                min_words=target_words,
            )
            word_count = len((script or "").split())
            if script and word_count >= accept_words:
                emit(
                    f"[+] Script is {word_count} words (~{round(word_count / words_per_second)}s).",
                    "success",
                )
                break
            emit(
                f"[i] Script was {word_count} words; need ~{accept_words} for "
                f"{min_duration}s. Regenerating longer (attempt {attempt}/3)...",
                "info",
            )
            target_words = int(target_words * 1.5)

        # Trim an over-long script back to max_words at a sentence boundary.
        if script and len(script.split()) > max_words:
            kept, used = [], 0
            for sentence in re.split(r"(?<=[.!?])\s+", script.strip()):
                words_in = len(sentence.split())
                if used + words_in > max_words and kept:
                    break
                kept.append(sentence)
                used += words_in
            script = " ".join(kept)
            emit(
                f"[i] Trimmed script to {used} words "
                f"(~{round(used / words_per_second)}s) to cap length.",
                "info",
            )
    else:
        script = generate_script(
            data["videoSubject"],
            paragraph_number,
            ai_model,
            voice,
            custom_prompt,
        )

    if not script:
        raise RuntimeError(
            "Could not generate a script. Try a different model or prompt."
        )

    guard_cancelled()
    if os.getenv("SCRIPT_FACT_CHECK", "true").strip().lower() in {"true", "1", "yes", "on"}:
        emit("[+] Reviewing script factual claims with the selected model...")
        script = verify_and_rewrite_script(data["videoSubject"], script, ai_model, voice, emit)
        guard_cancelled()

    emit("[+] Script generated!", "success")

    guard_cancelled()

    sentences = script.split(". ")
    sentences = list(filter(lambda x: x != "", sentences))
    paths = []

    final_audio = None
    tts_path = str(TEMP_DIR / f"{uuid4()}.mp3")
    try:
        for sentence in sentences:
            guard_cancelled()
            current_tts_path = str(TEMP_DIR / f"{uuid4()}.mp3")
            tts(sentence, voice, filename=current_tts_path)
            paths.append(AudioFileClip(current_tts_path))
        final_audio = concatenate_audioclips(paths)
        final_audio.write_audiofile(tts_path)
    finally:
        if final_audio is not None:
            final_audio.close()
        for audio_clip in paths:
            audio_clip.close()

    guard_cancelled()
    temp_audio = AudioFileClip(tts_path)
    try:
        audio_duration = temp_audio.duration
    finally:
        temp_audio.close()
    required, media_target = media_requirement(audio_duration, max_clip_duration, amount_of_stock_videos)
    emit(f"[Media] Audio duration: {audio_duration:.1f}s")
    emit(f"[Media] Unique scenes required at {max_clip_duration:g}s max: {required}")
    if max(required, amount_of_stock_videos) > MAX_MEDIA_SCENES:
        emit(f"[Media] Target capped at {MAX_MEDIA_SCENES} unique sources; sources may cycle.", "warning")
    search_terms = get_search_terms(
        data["videoSubject"], min(amount_of_stock_videos, media_target), script, ai_model
    )
    emit(f"[+] Generated search terms: {search_terms}")
    video_paths = select_media(
        search_terms, data["videoSubject"], orientation, media_target,
        guard_cancelled, emit,
    )

    try:
        subtitles_path = generate_subtitles(
            audio_path=tts_path,
            sentences=sentences,
            audio_clips=paths,
            voice=voice_prefix,
        )
    except Exception as err:
        emit(f"[-] Error generating subtitles: {err}", "error")
        subtitles_path = None

    if not subtitles_path:
        raise RuntimeError(
            "Could not generate subtitles. Check AssemblyAI key or local subtitle settings."
        )

    guard_cancelled()
    combined_video_path = combine_videos(
        video_paths,
        audio_duration,
        max_clip_duration,
        n_threads or 2,
        target_width,
        target_height,
    )

    try:
        final_video_path = generate_video(
            combined_video_path,
            tts_path,
            subtitles_path,
            n_threads or 2,
            subtitles_position,
            text_color or "#FFFF00",
            target_width,
        )
    except Exception as err:
        raise RuntimeError(
            f"Could not render final video. Check subtitle/font/ImageMagick setup. ({err})"
        ) from err

    title, description, keywords = generate_metadata(
        data["videoSubject"], script, ai_model
    )
    hashtags = generate_hashtags(data["videoSubject"], script, ai_model)

    emit("[-] Metadata for YouTube upload:", "info")
    emit("   Title:", "info")
    emit(f"   {title}", "info")
    emit("   Description:", "info")
    emit(f"   {description}", "info")
    emit("   Hashtags:", "info")
    emit(f"   {' '.join(hashtags)}", "info")
    emit("   Keywords:", "info")
    emit(f"  {', '.join(keywords)}", "info")

    if automate_youtube_upload:
        client_secrets_file = str((BASE_DIR / "client_secret.json").resolve())
        skip_yt_upload = False
        if not os.path.exists(client_secrets_file):
            skip_yt_upload = True
            emit(
                "[-] Client secrets file missing. YouTube upload will be skipped.",
                "warning",
            )
            emit(
                "[-] Please download the client_secret.json from Google Cloud Platform and store this inside the /Backend directory.",
                "error",
            )

        if not skip_yt_upload:
            video_category_id = "28"
            privacy_status = "private"
            video_metadata = {
                "video_path": str((TEMP_DIR / final_video_path).resolve()),
                "title": title,
                "description": description,
                "category": video_category_id,
                "keywords": ",".join(keywords),
                "privacyStatus": privacy_status,
            }

            try:
                video_response = upload_video(
                    video_path=video_metadata["video_path"],
                    title=video_metadata["title"],
                    description=video_metadata["description"],
                    category=video_metadata["category"],
                    keywords=video_metadata["keywords"],
                    privacy_status=video_metadata["privacyStatus"],
                )
                emit(f"Uploaded video ID: {video_response.get('id')}", "success")
            except HttpError as err:
                emit(
                    f"An HTTP error {err.resp.status} occurred:\n{err.content}", "error"
                )

    final_output_path = str(PROJECT_ROOT / final_video_path)
    rendered_video_path = str(TEMP_DIR / final_video_path)
    render_threads = n_threads or (os.cpu_count() or 2)

    guard_cancelled()

    if use_music:
        song_path = choose_random_song()

        if not song_path:
            emit(
                "[-] Could not find songs in Songs/. Continuing without background music.",
                "warning",
            )
            use_music = False

        if use_music:
            video_clip = VideoFileClip(rendered_video_path)
            song_clip = None
            mixed_audio = None
            mixed_audio_path = str(TEMP_DIR / f"{uuid4()}_mixed_audio.m4a")
            try:
                original_duration = video_clip.duration
                original_audio = video_clip.audio
                song_clip = AudioFileClip(song_path).with_fps(44100)
                song_clip = song_clip.with_effects(
                    [afx.AudioLoop(duration=original_duration)]
                )
                song_clip = song_clip.with_volume_scaled(0.1).with_fps(44100)

                mixed_audio = CompositeAudioClip(
                    [original_audio, song_clip]
                ).with_duration(original_duration)
                mixed_audio.write_audiofile(
                    mixed_audio_path,
                    fps=44100,
                    codec="aac",
                    bitrate="192k",
                )
            finally:
                video_clip.close()
                if mixed_audio is not None:
                    mixed_audio.close()
                if song_clip is not None:
                    song_clip.close()

            try:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        rendered_video_path,
                        "-i",
                        mixed_audio_path,
                        "-map",
                        "0:v:0",
                        "-map",
                        "1:a:0",
                        "-c:v",
                        "copy",
                        "-c:a",
                        "aac",
                        "-b:a",
                        "192k",
                        "-shortest",
                        final_output_path,
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except Exception:
                emit(
                    "[!] ffmpeg remux failed. Falling back to MoviePy render for music mix.",
                    "warning",
                )
                video_clip = VideoFileClip(rendered_video_path)
                song_clip = None
                try:
                    original_duration = video_clip.duration
                    original_audio = video_clip.audio
                    song_clip = AudioFileClip(song_path).with_fps(44100)
                    song_clip = song_clip.with_effects(
                        [afx.AudioLoop(duration=original_duration)]
                    )
                    song_clip = song_clip.with_volume_scaled(0.1).with_fps(44100)
                    comp_audio = CompositeAudioClip(
                        [original_audio, song_clip]
                    ).with_duration(original_duration)
                    video_clip = (
                        video_clip.with_audio(comp_audio)
                        .with_fps(30)
                        .with_duration(original_duration)
                    )
                    video_clip.write_videofile(
                        final_output_path,
                        threads=render_threads,
                        fps=30,
                        codec="libx264",
                        audio_codec="aac",
                        preset="medium",
                    )
                finally:
                    video_clip.close()
                    if song_clip is not None:
                        song_clip.close()
            finally:
                if os.path.exists(mixed_audio_path):
                    os.remove(mixed_audio_path)

    if not use_music:
        shutil.copy2(rendered_video_path, final_output_path)

    emit(f"[+] Video generated: {final_video_path}!", "success")

    # Persist a uniquely-named copy in the host-mounted output/ folder so
    # back-to-back jobs don't overwrite each other (temp/ is wiped per run).
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^a-z0-9]+", "-", data["videoSubject"].lower()).strip("-")[:60]
        saved_name = f"{slug or 'video'}-{uuid4().hex[:8]}.mp4"
        saved_path = OUTPUT_DIR / saved_name
        shutil.copy2(final_output_path, saved_path)
        emit(f"[+] Saved to output/{saved_name}", "success")
        final_video_path = f"output/{saved_name}"

        # Write a copy-paste-ready metadata sidecar next to the video.
        hashtags_line = " ".join(hashtags)
        sidecar_name = saved_name.rsplit(".", 1)[0] + ".txt"
        sidecar_text = (
            f"TITLE\n{title}\n\n"
            f"DESCRIPTION\n{description}\n\n{hashtags_line}\n\n"
            f"HASHTAGS\n{hashtags_line}\n\n"
            f"KEYWORDS / TAGS\n{', '.join(keywords)}\n"
        )
        (OUTPUT_DIR / sidecar_name).write_text(sidecar_text, encoding="utf-8")
        emit(f"[+] Saved metadata to output/{sidecar_name}", "success")
    except Exception as err:
        emit(f"[!] Could not copy to output/ folder: {err}", "warning")

    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/f", "/im", "ffmpeg.exe"],
            check=False,
            capture_output=True,
            text=True,
        )
    elif shutil.which("pkill"):
        subprocess.run(
            ["pkill", "-f", "ffmpeg"],
            check=False,
            capture_output=True,
            text=True,
        )

    return final_video_path
