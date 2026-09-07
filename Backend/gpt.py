import re
import os
import json
from ollama import Client, ResponseError

from dotenv import load_dotenv
from logstream import log
from typing import Tuple, List, Optional
from utils import ENV_FILE

# Load environment variables
load_dotenv(ENV_FILE)

# Set environment variables
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "180"))


def _ollama_client() -> Client:
    return Client(host=OLLAMA_BASE_URL, timeout=OLLAMA_TIMEOUT)


def _extract_model_name(model_obj) -> str:
    if hasattr(model_obj, "model") and getattr(model_obj, "model"):
        return str(getattr(model_obj, "model")).strip()
    if hasattr(model_obj, "name") and getattr(model_obj, "name"):
        return str(getattr(model_obj, "name")).strip()
    if isinstance(model_obj, dict):
        return str(model_obj.get("model") or model_obj.get("name") or "").strip()
    return ""


def list_ollama_models() -> Tuple[List[str], str]:
    """
    Returns available Ollama model names and configured default model.

    Returns:
        Tuple[List[str], str]: (available model names, default model)
    """
    try:
        response = _ollama_client().list()
    except Exception as err:
        raise RuntimeError(f"Failed to fetch Ollama models: {err}") from err

    models = []
    if hasattr(response, "models") and getattr(response, "models") is not None:
        models = list(getattr(response, "models"))
    elif isinstance(response, dict):
        models = response.get("models") or []

    model_names = [_extract_model_name(model) for model in models]
    model_names = [name for name in model_names if name]

    unique_names = list(dict.fromkeys(model_names))

    if OLLAMA_MODEL and OLLAMA_MODEL in unique_names:
        default_model = OLLAMA_MODEL
    elif unique_names:
        default_model = unique_names[0]
    else:
        default_model = OLLAMA_MODEL if OLLAMA_MODEL else ""

    return unique_names, default_model


def generate_response(prompt: str, ai_model: str) -> str:
    """
    Generate a script for a video, depending on the subject of the video.

    Args:
        video_subject (str): The subject of the video.
        ai_model (str): The AI model to use for generation.


    Returns:

        str: The response from the AI model.

    """

    model_name = (ai_model or "").strip() or OLLAMA_MODEL

    try:
        client = _ollama_client()
        try:
            response = client.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
            )
        except ResponseError as err:
            if err.status_code == 404:
                try:
                    response = client.generate(
                        model=model_name, prompt=prompt, stream=False
                    )
                except ResponseError as fallback_err:
                    if (
                        fallback_err.status_code == 404
                        and "not found" in str(fallback_err).lower()
                    ):
                        available_models, _ = list_ollama_models()
                        available = (
                            ", ".join(available_models) if available_models else "none"
                        )
                        raise RuntimeError(
                            f"Ollama model '{model_name}' is not installed. Available models: {available}. "
                            f"Install it with: ollama pull {model_name}"
                        ) from fallback_err
                    raise
            else:
                raise
    except RuntimeError:
        raise
    except Exception as err:
        raise RuntimeError(f"Failed to connect to Ollama: {err}") from err

    content = ""
    if hasattr(response, "message") and getattr(response, "message") is not None:
        message = getattr(response, "message")
        if hasattr(message, "content") and getattr(message, "content"):
            content = str(getattr(message, "content")).strip()
        elif isinstance(message, dict):
            content = str(message.get("content") or "").strip()

    if not content:
        if hasattr(response, "response") and getattr(response, "response"):
            content = str(getattr(response, "response")).strip()
        elif isinstance(response, dict):
            content = (
                str(response.get("message", {}).get("content") or "")
                or str(response.get("response") or "")
            ).strip()

    if not content:
        raise RuntimeError("Ollama returned an empty response.")

    return content


def generate_script(
    video_subject: str,
    paragraph_number: int,
    ai_model: str,
    voice: str,
    customPrompt: str,
    min_words: int = 0,
) -> Optional[str]:
    """
    Generate a script for a video, depending on the subject of the video, the number of paragraphs, and the AI model.



    Args:

        video_subject (str): The subject of the video.

        paragraph_number (int): The number of paragraphs to generate.

        ai_model (str): The AI model to use for generation.



    Returns:

        str: The script for the video.

    """

    # Build prompt

    if customPrompt:
        prompt = customPrompt
    else:
        prompt = """
            Generate a script for a video, depending on the subject of the video.

            The script is to be returned as a string with the specified number of paragraphs.

            Here is an example of a string:
            "This is an example string."

            Do not under any circumstance reference this prompt in your response.

            Get straight to the point, don't start with unnecessary things like, "welcome to this video".

            Obviously, the script should be related to the subject of the video.

            YOU MUST NOT INCLUDE ANY TYPE OF MARKDOWN OR FORMATTING IN THE SCRIPT, NEVER USE A TITLE.
            YOU MUST WRITE THE SCRIPT IN THE LANGUAGE SPECIFIED IN [LANGUAGE].
            ONLY RETURN THE RAW CONTENT OF THE SCRIPT. DO NOT INCLUDE "VOICEOVER", "NARRATOR" OR SIMILAR INDICATORS OF WHAT SHOULD BE SPOKEN AT THE BEGINNING OF EACH PARAGRAPH OR LINE. YOU MUST NOT MENTION THE PROMPT, OR ANYTHING ABOUT THE SCRIPT ITSELF. ALSO, NEVER TALK ABOUT THE AMOUNT OF PARAGRAPHS OR LINES. JUST WRITE THE SCRIPT.

        """

    prompt += f"""

    Subject: {video_subject}
    Number of paragraphs: {paragraph_number}
    Language: {voice}

    """

    # When a minimum length is requested, instruct the model to keep writing
    # until it reaches the target word count (drives the final video duration).
    if min_words and min_words > 0:
        prompt += f"""
    The script MUST be at least {min_words} words long. Keep writing engaging,
    on-topic narration until you reach that length. Do not stop early.
    """

    # Generate script
    response = generate_response(prompt, ai_model)

    log(response, "info")

    # Return the generated script
    if response:
        # Clean the script
        # Remove asterisks, hashes
        response = response.replace("*", "")
        response = response.replace("#", "")

        # Remove markdown syntax
        response = re.sub(r"\[.*\]", "", response)
        response = re.sub(r"\(.*\)", "", response)

        # Split the script into paragraphs
        paragraphs = response.split("\n\n")

        # Select the specified number of paragraphs. When a minimum length is
        # requested, keep all paragraphs so the script isn't truncated short.
        if min_words and min_words > 0:
            selected_paragraphs = paragraphs
        else:
            selected_paragraphs = paragraphs[:paragraph_number]

        # Join the selected paragraphs into a single string
        final_script = "\n\n".join(selected_paragraphs)

        # Print to console the number of paragraphs used
        log(f"Number of paragraphs used: {len(selected_paragraphs)}", "success")

        return final_script
    else:
        log("[-] GPT returned an empty response.", "error")
        return None


def verify_and_rewrite_script(
    video_subject: str, script: str, ai_model: str, voice: str
) -> str:
    """Review narration with the configured model before it reaches TTS/search."""
    prompt = f"""
    Inspect every factual claim in the narration below, including all numbers,
    comparisons, dates, and causal claims. Remove or rewrite claims that are
    uncertain, misleading, numerically dubious, or factually incorrect.
    Do not invent replacement statistics or new facts. Use cautious, accurate
    wording when a precise claim cannot be supported by your knowledge.
    Preserve the same approximate length, narration style, language, and number
    of facts where possible; accuracy takes priority over retaining a false fact.
    If everything is correct, return the original narration unchanged or an
    equivalent corrected narration. Output ONLY the corrected narration.
    No markdown, titles, explanations, fact-check notes, or narrator labels.

    Subject: {video_subject}
    Voice/language: {voice} (preserve the narration's language)
    Narration:
    {script}
    """
    corrected = generate_response(prompt, ai_model).strip()
    if not corrected:
        raise RuntimeError("Script verification returned empty narration.")
    return corrected


_VAGUE_SEARCH_WORDS = {
    "fact", "facts", "information", "concept", "concepts", "movement",
    "idea", "ideas", "knowledge",
}


def _valid_search_term(value: object) -> str:
    if not isinstance(value, str):
        return ""
    term = " ".join(value.split()).strip('"')
    words = re.findall(r"[^\W_]+(?:[-'][^\W_]+)*", term, re.UNICODE)
    if not 2 <= len(term.split()) <= 5 or not 2 <= len(words) <= 5:
        return ""
    if _VAGUE_SEARCH_WORDS.intersection(word.casefold() for word in words):
        return ""
    return term


def _fallback_search_terms(video_subject: str, script: str) -> List[str]:
    """Use subject words and script entities, without inventing unrelated visuals."""
    stop_words = _VAGUE_SEARCH_WORDS | {
        "a", "an", "the", "of", "about", "and", "in", "on", "for", "to",
        "is", "are", "what", "how", "why", "top", "interesting", "amazing",
        "surprising", "incredible", "you", "should", "know",
    }
    subject_words = [
        word for word in re.findall(r"[^\W_]+(?:[-'][^\W_]+)*", video_subject)
        if word.casefold() not in stop_words and not word.isdigit()
    ]
    # Named phrases from narration are safer than arbitrary sentence fragments.
    entities = re.findall(r"\b[A-Z][\w'-]*(?:\s+[A-Z][\w'-]*){1,4}\b", script)
    bases = [" ".join(subject_words[:4])] if subject_words else []
    bases.extend(entities)
    if not bases:
        bases = [
            word for word in re.findall(r"[^\W_]+", script)
            if word.casefold() not in stop_words and not word.isdigit()
        ][:8]

    terms = list(bases)
    # These vary the search framing, never add food, people, or other subjects.
    for base in bases:
        for framing in ("footage", "view", "close up", "wide view", "details",
                        "video", "scene", "full view", "stock footage"):
            terms.append(f"{base} {framing}")
        for framing in ("footage", "video", "view"):
            terms.append(f"{framing} {base}")
        terms.append(f"footage of {base}")
    return terms


def get_search_terms(
    video_subject: str, amount: int, script: str, ai_model: str
) -> List[str]:
    """
    Generate a JSON-Array of search terms for stock videos,
    depending on the subject of a video.

    Args:
        video_subject (str): The subject of the video.
        amount (int): The amount of search terms to generate.
        script (str): The script of the video.
        ai_model (str): The AI model to use for generation.

    Returns:
        List[str]: The search terms for the video subject.
    """

    if amount <= 0:
        return []
    search_terms = []
    seen = set()

    def accept(values: list) -> None:
        for value in values:
            term = _valid_search_term(value)
            if term and term.casefold() not in seen and len(search_terms) < amount:
                seen.add(term.casefold())
                search_terms.append(term)

    # Initial generation plus at most three retries, requesting only the deficit.
    for attempt in range(4):
        remaining = amount - len(search_terms)
        prompt = f"""
        Generate exactly {remaining} new unique search terms for stock-video search.
        We need {amount} terms in total and still require {remaining} more terms.
        Already accepted terms (do not repeat, even with different capitalization):
        {json.dumps(search_terms, ensure_ascii=False)}

        Each term MUST contain 2 to 5 words and describe concrete, visually
        searchable subjects using specific nouns/entities from the script.
        Use queries suitable for stock-video search, in English where possible.
        Avoid vague terms: facts, information, concept, movement, idea, knowledge.
        Avoid metaphorical visuals. Avoid unrelated food, office scenes, generic
        people, and abstract backgrounds unless actually mentioned in the script.
        Prefer actual subjects, for example for a Jupiter script: "Jupiter planet",
        "Great Red Spot", "deep space stars", "space telescope".
        These are examples only; do not use them for unrelated subjects.
        Return ONLY a JSON array of strings. No markdown or explanations.

        Subject: {video_subject}
        Script:
        {script}
        """
        try:
            response = generate_response(prompt, ai_model)
            try:
                values = json.loads(response)
            except json.JSONDecodeError:
                match = re.search(r"\[[\s\S]*\]", response)
                values = json.loads(match.group()) if match else []
            if isinstance(values, list):
                accept(values)
        except (json.JSONDecodeError, RuntimeError) as err:
            log(f"[!] Search-term generation attempt {attempt + 1}/4 failed: {err}", "warning")
        if len(search_terms) == amount:
            break
        log(
            f"[!] Accepted {len(search_terms)}/{amount} unique search terms "
            f"after attempt {attempt + 1}/4.", "warning",
        )

    if len(search_terms) < amount:
        log("[!] Filling missing search terms with subject/script-derived queries.", "warning")
        accept(_fallback_search_terms(video_subject, script))
    if len(search_terms) < amount:
        log(f"[!] Only {len(search_terms)}/{amount} safe unique queries available.", "warning")
    log(f"\nGenerated {len(search_terms)} search terms: {', '.join(search_terms)}", "info")
    return search_terms


def _clean_metadata_text(text: str) -> str:
    """
    Strip common LLM fluff (preambles, markdown, surrounding quotes) from a
    short metadata field so titles/descriptions are ready to paste.
    """
    text = text.strip()

    # Drop a leading conversational preamble line (e.g. "Here are some options:")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if lines and re.match(r"^(here\b|sure\b|certainly\b)", lines[0].strip(), re.I):
        lines = lines[1:]
    text = "\n".join(lines).strip()

    # Remove markdown emphasis/backticks and leading list markers
    text = re.sub(r"[*`#]+", "", text)
    text = re.sub(r"^\s*\d+[.)]\s*", "", text)
    # Strip a single layer of surrounding quotes
    text = text.strip().strip('"').strip("'").strip()
    return text


def generate_metadata(
    video_subject: str, script: str, ai_model: str
) -> Tuple[str, str, List[str]]:
    """
    Generate metadata for a YouTube video, including the title, description, and keywords.

    Args:
        video_subject (str): The subject of the video.
        script (str): The script of the video.
        ai_model (str): The AI model to use for generation.

    Returns:
        Tuple[str, str, List[str]]: The title, description, and keywords for the video.
    """

    # Build prompt for title
    title_prompt = f"""
    Generate ONE catchy, SEO-friendly title for a YouTube Shorts video about {video_subject}.

    Return ONLY the title text on a single line.
    Do NOT provide multiple options, numbering, quotes, markdown, labels, or any
    explanation. Output the title and nothing else.
    """

    # Generate title
    title = _clean_metadata_text(generate_response(title_prompt, ai_model))

    # Build prompt for description
    description_prompt = f"""
    Write ONE brief, engaging description (2-4 sentences) for a YouTube Shorts
    video about {video_subject}, based on the script below.

    Return ONLY the description text.
    Do NOT include a preamble, options, quotes, markdown, hashtags, or labels.
    Output the description and nothing else.

    Script:
    {script}
    """

    # Generate description
    description = _clean_metadata_text(generate_response(description_prompt, ai_model))

    # Generate keywords
    keywords = get_search_terms(video_subject, 6, script, ai_model)

    return title, description, keywords


def generate_hashtags(
    video_subject: str, script: str, ai_model: str, amount: int = 10
) -> List[str]:
    """
    Generate discovery-oriented social hashtags for a video topic.

    Args:
        video_subject (str): The subject of the video.
        script (str): The script of the video.
        ai_model (str): The AI model to use for generation.
        amount (int): How many hashtags to generate.

    Returns:
        List[str]: Hashtags, each prefixed with '#'.
    """

    prompt = f"""
    Generate {amount} relevant, popular hashtags to help a short-form video
    about "{video_subject}" get discovered on YouTube, TikTok and Instagram.

    Each hashtag must be 1-2 words, lowercase, no spaces, and WITHOUT a
    leading '#'. Mix specific topic tags with a few broad-reach tags.

    YOU MUST ONLY RETURN A JSON-ARRAY OF STRINGS. NOTHING ELSE.
    Example: ["oceanfacts", "deepsea", "didyouknow", "shorts", "viral"]

    For context, here is the full script:
    {script}
    """

    response = generate_response(prompt, ai_model)

    tags = []
    try:
        tags = json.loads(response)
        if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
            raise ValueError("Response is not a list of strings.")
    except (json.JSONDecodeError, ValueError):
        match = re.search(r"\[[\s\S]*\]", response)
        if match:
            try:
                tags = json.loads(match.group())
            except json.JSONDecodeError:
                tags = []
        if not tags:
            tags = re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', response)

    # Normalise: strip '#'/spaces/punctuation and re-prefix with a single '#'
    hashtags = []
    for tag in tags:
        cleaned = re.sub(r"[^A-Za-z0-9]", "", str(tag))
        if cleaned:
            hashtags.append("#" + cleaned.lower())

    log(f"[+] Generated {len(hashtags)} hashtags: {' '.join(hashtags)}", "info")
    return hashtags
