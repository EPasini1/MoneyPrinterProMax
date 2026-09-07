import os
import math
import uuid

import requests
import srt_equalizer
import assemblyai as aai

from typing import Callable, List
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from PIL import Image, ImageOps
from providers.base import MediaCandidate, timeout_setting
from moviepy import (
    AudioFileClip,
    CompositeVideoClip,
    ImageClip,
    TextClip,
    VideoClip,
    VideoFileClip,
    concatenate_videoclips,
)
from dotenv import load_dotenv
from logstream import log
from moviepy.video.tools.subtitles import SubtitlesClip
from utils import ENV_FILE, TEMP_DIR, SUBTITLES_DIR, FONTS_DIR, get_max_clip_duration

load_dotenv(ENV_FILE)

ASSEMBLY_AI_API_KEY = os.getenv("ASSEMBLY_AI_API_KEY")
FRAME_EPSILON = 1 / 120


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def download_media(candidate: MediaCandidate, directory: str | None = None) -> str:
    """Stream and validate a source before it consumes a selection slot."""
    if candidate.media_type not in {"video", "image"}:
        raise ValueError("Unsupported media type.")
    destination = Path(directory) if directory is not None else TEMP_DIR
    destination.mkdir(parents=True, exist_ok=True)
    extension = Path(urlparse(candidate.download_url).path).suffix.lower()
    if candidate.media_type == "video":
        extension = ".mp4"
    elif extension not in IMAGE_EXTENSIONS:
        extension = ".jpg"
    path = destination / f"{uuid.uuid4()}{extension}"
    try:
        with requests.get(candidate.download_url, stream=True,
                          timeout=timeout_setting("MEDIA_DOWNLOAD_TIMEOUT", 60),
                          headers={"User-Agent": "MoneyPrinterProMax/2.0 (media downloader)"}) as response:
            response.raise_for_status()
            with path.open("wb") as file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file.write(chunk)
        if candidate.media_type == "video":
            with VideoFileClip(str(path)) as clip:
                if not math.isfinite(clip.duration) or clip.duration <= FRAME_EPSILON:
                    raise ValueError("Downloaded stock video has no usable duration.")
        else:
            with Image.open(path) as image:
                image.verify()
            # Decode once to reject truncated images before counting the source.
            with Image.open(path) as image:
                image.load()
                actual_extension = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "BMP": ".bmp"}.get(image.format)
            if not actual_extension:
                raise ValueError("Unsupported downloaded image format.")
            if path.suffix != actual_extension:
                path = path.rename(path.with_suffix(actual_extension))
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return str(path)


def save_video(video_url: str, directory: str = str(TEMP_DIR)) -> str:
    """Compatibility wrapper for Phase 1 video-only callers."""
    return download_media(MediaCandidate("pexels", "video", "", "", video_url,
                                         None, None, None, None), directory)


def create_image_clip(image_path: str, duration: float, target_width: int,
                      target_height: int) -> VideoClip:
    """Center crop once, then zoom 6% using a constant-sized output frame."""
    if not math.isfinite(duration) or duration <= 0 or min(target_width, target_height) <= 0:
        raise ValueError("Image scene duration and dimensions must be positive.")
    with Image.open(image_path) as original:
        # JPEG decoding can downsample before allocating the full image.
        original.draft("RGB", (target_width, target_height))
        fitted = ImageOps.fit(ImageOps.exif_transpose(original).convert("RGB"),
                              (target_width, target_height), method=Image.Resampling.LANCZOS)
    base = ImageClip(np.asarray(fitted)).with_duration(duration).with_fps(30)

    def zoom(get_frame: Callable, time: float) -> np.ndarray:
        factor = 1 + 0.06 * min(1, max(0, time / duration))
        width, height = target_width / factor, target_height / factor
        left, top = (target_width - width) / 2, (target_height - height) / 2
        frame = Image.fromarray(get_frame(0))
        return np.asarray(frame.resize((target_width, target_height), Image.Resampling.BICUBIC,
                                       box=(left, top, left + width, top + height)))

    return base.transform(zoom).with_duration(duration).with_fps(30)


def __generate_subtitles_assemblyai(audio_path: str, voice: str) -> str:
    """
    Generates subtitles from a given audio file and returns the path to the subtitles.

    Args:
        audio_path (str): The path to the audio file to generate subtitles from.

    Returns:
        str: The generated subtitles
    """

    language_mapping = {
        "br": "pt",
        "id": "en",  # AssemblyAI doesn't have Indonesian
        "jp": "ja",
        "kr": "ko",
    }

    if voice in language_mapping:
        lang_code = language_mapping[voice]
    else:
        lang_code = voice

    aai.settings.api_key = ASSEMBLY_AI_API_KEY
    config = aai.TranscriptionConfig(language_code=lang_code)
    transcriber = aai.Transcriber(config=config)
    transcript = transcriber.transcribe(audio_path)
    subtitles = transcript.export_subtitles_srt()

    return subtitles


def __generate_subtitles_locally(
    sentences: List[str], audio_clips: List[AudioFileClip]
) -> str:
    """
    Generates subtitles from a given audio file and returns the path to the subtitles.

    Args:
        sentences (List[str]): all the sentences said out loud in the audio clips
        audio_clips (List[AudioFileClip]): all the individual audio clips which will make up the final audio track
    Returns:
        str: The generated subtitles
    """

    def convert_to_srt_time_format(total_seconds: float) -> str:
        # Convert total seconds to the SRT time format: HH:MM:SS,mmm
        milliseconds_total = int(round(total_seconds * 1000))
        hours, remainder = divmod(milliseconds_total, 3_600_000)
        minutes, remainder = divmod(remainder, 60_000)
        seconds, milliseconds = divmod(remainder, 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    start_time = 0
    subtitles = []

    for i, (sentence, audio_clip) in enumerate(zip(sentences, audio_clips), start=1):
        duration = audio_clip.duration
        end_time = start_time + duration

        # Format: subtitle index, start time --> end time, sentence
        subtitle_entry = f"{i}\n{convert_to_srt_time_format(start_time)} --> {convert_to_srt_time_format(end_time)}\n{sentence}\n"
        subtitles.append(subtitle_entry)

        start_time += duration  # Update start time for the next subtitle

    return "\n".join(subtitles)


def generate_subtitles(
    audio_path: str, sentences: List[str], audio_clips: List[AudioFileClip], voice: str
) -> str:
    """
    Generates subtitles from a given audio file and returns the path to the subtitles.

    Args:
        audio_path (str): The path to the audio file to generate subtitles from.
        sentences (List[str]): all the sentences said out loud in the audio clips
        audio_clips (List[AudioFileClip]): all the individual audio clips which will make up the final audio track

    Returns:
        str: The path to the generated subtitles.
    """

    def equalize_subtitles(srt_path: str, max_chars: int = 10) -> None:
        # Equalize subtitles
        srt_equalizer.equalize_srt_file(srt_path, srt_path, max_chars)

    # Save subtitles
    SUBTITLES_DIR.mkdir(parents=True, exist_ok=True)
    subtitles_path = SUBTITLES_DIR / f"{uuid.uuid4()}.srt"

    if ASSEMBLY_AI_API_KEY is not None and ASSEMBLY_AI_API_KEY != "":
        log("[+] Creating subtitles using AssemblyAI", "info")
        subtitles = __generate_subtitles_assemblyai(audio_path, voice)
    else:
        log("[+] Creating subtitles locally", "info")
        subtitles = __generate_subtitles_locally(sentences, audio_clips)
        # print(colored("[-] Local subtitle generation has been disabled for the time being.", "red"))
        # print(colored("[-] Exiting.", "red"))
        # sys.exit(1)

    with open(subtitles_path, "w", encoding="utf-8") as file:
        file.write(subtitles)

    # Equalize subtitles
    equalize_subtitles(str(subtitles_path))

    log("[+] Subtitles generated.", "success")

    return str(subtitles_path)


def combine_videos(
    video_paths: List[str],
    max_duration: float,
    max_clip_duration: float,
    threads: int,
    target_width: int = 1080,
    target_height: int = 1920,
) -> str:
    """
    Combines a list of videos into one video and returns the path to the combined video.

    Args:
        video_paths (List): A list of paths to the videos to combine.
        max_duration (float): The maximum duration of the combined video.
        max_clip_duration (float): The maximum duration of each clip.
        threads (int): The number of threads to use for the video processing.

    Returns:
        str: The path to the combined video.
    """
    video_id = uuid.uuid4()
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    combined_video_path = TEMP_DIR / f"{video_id}.mp4"

    # Normalize duplicate paths so every unique source is used before cycling.
    video_paths = list(dict.fromkeys(str(Path(path).resolve()) for path in video_paths))
    if not video_paths:
        raise ValueError("No source videos were provided for concatenation.")

    max_duration = float(max_duration)
    if not math.isfinite(max_duration) or max_duration <= FRAME_EPSILON:
        raise ValueError("Target video duration must be positive and finite.")
    max_clip_duration = get_max_clip_duration(max_clip_duration)

    log("[+] Combining videos...", "info")
    log(f"[+] Each clip will be maximum {max_clip_duration} seconds long.", "info")

    clips = []
    sources = {}
    final_clip = None
    tot_dur = 0.0
    last_path = None
    try:
        while tot_dur < max_duration - 1e-9:
            progressed = False
            for media_path in video_paths:
                remaining = max_duration - tot_dur
                if remaining <= 1e-9:
                    break
                if media_path == last_path:
                    continue
                if media_path not in sources:
                    if Path(media_path).suffix.lower() in IMAGE_EXTENSIONS:
                        sources[media_path] = create_image_clip(media_path, max_clip_duration, target_width, target_height)
                    else:
                        sources[media_path] = VideoFileClip(media_path)
                source = sources[media_path]
                is_image = Path(media_path).suffix.lower() in IMAGE_EXTENSIONS
                safe_duration = source.duration if is_image else source.duration - FRAME_EPSILON
                if not math.isfinite(safe_duration) or safe_duration <= 0:
                    continue
                target_duration = min(max_clip_duration, remaining, safe_duration)
                clip = source.without_audio().subclipped(0, target_duration).with_fps(30)
                if not is_image or tuple(clip.size) != (target_width, target_height):
                    target_ratio = target_width / target_height
                    if clip.w / clip.h < target_ratio:
                        clip = clip.cropped(width=clip.w, height=round(clip.w / target_ratio),
                                            x_center=clip.w / 2, y_center=clip.h / 2)
                    else:
                        clip = clip.cropped(width=round(target_ratio * clip.h), height=clip.h,
                                            x_center=clip.w / 2, y_center=clip.h / 2)
                    clip = clip.resized(new_size=(target_width, target_height))
                clips.append(clip)
                tot_dur += target_duration
                last_path = media_path
                progressed = True
            if not progressed:
                raise RuntimeError("Could not reach target duration without immediate source repetition.")
        final_clip = concatenate_videoclips(clips, method="chain")
        final_clip = final_clip.with_fps(30).with_duration(max_duration)
        final_clip.write_videofile(str(combined_video_path), threads=threads, fps=30,
                                   codec="libx264", preset="medium", audio=False)
    finally:
        if final_clip is not None:
            final_clip.close()
        for clip in clips:
            clip.close()
        for source in sources.values():
            source.close()

    return str(combined_video_path)


def generate_video(
    combined_video_path: str,
    tts_path: str,
    subtitles_path: str,
    threads: int,
    subtitles_position: str,
    text_color: str,
    target_width: int = 1080,
) -> str:
    """
    This function creates the final video, with subtitles and audio.

    Args:
        combined_video_path (str): The path to the combined video.
        tts_path (str): The path to the text-to-speech audio.
        subtitles_path (str): The path to the subtitles.
        threads (int): The number of threads to use for the video processing.
        subtitles_position (str): The position of the subtitles.

    Returns:
        str: The path to the final video.
    """
    # Make a generator that returns a TextClip when called with consecutive
    # Scale caption size with the frame width (100px is tuned for 1080-wide
    # video) so wider formats like 16:9 get proportionally larger subtitles.
    font_path = str((FONTS_DIR / "bold_font.ttf").resolve())
    scale = target_width / 1080
    font_size = round(100 * scale)
    stroke_width = max(1, round(5 * scale))
    generator = lambda txt: TextClip(
        font=font_path,
        text=txt,
        font_size=font_size,
        color=text_color,
        stroke_color="black",
        stroke_width=stroke_width,
    )

    # Split the subtitles position into horizontal and vertical
    horizontal_subtitles_position, vertical_subtitles_position = (
        subtitles_position.split(",")
    )

    # Burn the subtitles into the video
    subtitles = SubtitlesClip(subtitles_path, make_textclip=generator)
    subtitle_vertical_position = vertical_subtitles_position
    if vertical_subtitles_position == "top":
        subtitle_vertical_position = 80

    base_video = VideoFileClip(str(combined_video_path))
    audio = AudioFileClip(tts_path)
    target_duration = min(base_video.duration, audio.duration)

    result = CompositeVideoClip(
        [
            base_video.subclipped(0, target_duration),
            subtitles.with_position(
                (horizontal_subtitles_position, subtitle_vertical_position)
            ).with_duration(target_duration),
        ]
    )

    # Clamp audio/video to exactly the same duration to avoid end-frame overreads.
    result = result.with_audio(audio.subclipped(0, target_duration)).with_duration(
        target_duration
    )

    output_path = TEMP_DIR / "output.mp4"
    try:
        result.write_videofile(
            str(output_path),
            threads=threads or 2,
            fps=30,
            codec="libx264",
            audio_codec="aac",
            preset="medium",
        )
    finally:
        result.close()
        subtitles.close()
        audio.close()
        base_video.close()

    return "output.mp4"
