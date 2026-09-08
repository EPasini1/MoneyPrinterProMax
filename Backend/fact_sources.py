"""Bounded, keyless factual references. Providers return text, never instructions."""
import html
import json
import re
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Callable, Protocol
from urllib.parse import quote

import requests

from logstream import log
from providers.ranking import _subject_tokens
from search import is_space_topic


MAX_QUERIES = 4
MAX_SOURCES = 4
RESULTS_PER_QUERY = 2
MAX_EXCERPT_CHARS = 2500
MAX_RESPONSE_BYTES = 512_000
HTTP_TIMEOUT = 5.0
RETRIEVAL_SECONDS = 20.0
USER_AGENT = "MoneyPrinterProMax/1.0 (factual-reference-retrieval)"
JUPITER_FACTS_URL = "https://science.nasa.gov/jupiter/jupiter-facts/"


@dataclass(frozen=True)
class FactSource:
    provider: str
    title: str
    url: str
    text: str


class FactSourceProvider(Protocol):
    name: str

    def search(self, query: str) -> list[FactSource]: ...


def factual_queries(subject: str, narration: str) -> list[str]:
    """Subject plus up to three claim-focused queries; no extra model calls."""
    def keywords(text: str) -> list[str]:
        return list(dict.fromkeys(word for word in re.findall(r"[^\W\d_]+", text.casefold())
                                 if len(word) > 1 and _subject_tokens(word)))

    anchor = keywords(subject)[:6]
    if not anchor:
        anchor = keywords(narration)[:6]
    if not anchor:
        return []
    queries = [" ".join(anchor)[:160]]
    for sentence in re.split(r"[.!?\n]+", narration[:12000]):
        words = list(dict.fromkeys([*anchor[:3], *keywords(sentence)]))[:10]
        query = " ".join(words)[:160]
        if query and query not in queries:
            queries.append(query)
        if len(queries) >= MAX_QUERIES:
            break
    return queries


def _fetch(url: str, params: dict | None = None) -> str:
    # Only provider-owned fixed endpoints are fetched; source links are not followed.
    with requests.get(url, params=params, headers={"User-Agent": USER_AGENT},
                      timeout=HTTP_TIMEOUT, stream=True) as response:
        response.raise_for_status()
        body = bytearray()
        started = time.monotonic()
        for chunk in response.iter_content(chunk_size=8192):
            if time.monotonic() - started > HTTP_TIMEOUT:
                raise requests.Timeout("Reference download exceeded time budget")
            body.extend(chunk)
            if len(body) > MAX_RESPONSE_BYTES:
                raise ValueError("Reference response exceeded size limit")
        return body.decode("utf-8", errors="replace")


class _ArticleText(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_main = False
        self.hidden = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "main":
            self.in_main = True
        if tag in {"script", "style", "nav", "header", "footer"}:
            self.hidden += 1

    def handle_endtag(self, tag: str) -> None:
        if tag == "main":
            self.in_main = False
        if tag in {"script", "style", "nav", "header", "footer"}:
            self.hidden = max(0, self.hidden - 1)
        if tag in {"p", "h1", "h2", "li"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self.in_main and not self.hidden:
            self.parts.append(data)


class NasaFactProvider:
    name = "NASA"

    def __init__(self) -> None:
        self.fetched_jupiter = False

    def search(self, query: str) -> list[FactSource]:
        if "jupiter" in _subject_tokens(query) and not self.fetched_jupiter:
            self.fetched_jupiter = True
            parser = _ArticleText()
            parser.feed(_fetch(JUPITER_FACTS_URL))
            return [FactSource(self.name, "Jupiter Facts", JUPITER_FACTS_URL, " ".join(parser.parts))]
        payload = json.loads(_fetch("https://images-api.nasa.gov/search", {
            "q": query, "page_size": RESULTS_PER_QUERY, "media_type": "image",
        }))
        sources = []
        for item in payload.get("collection", {}).get("items", [])[:RESULTS_PER_QUERY]:
            data = (item.get("data") or [{}])[0]
            if data.get("nasa_id"):
                sources.append(FactSource(self.name, data.get("title", ""),
                                          "https://images.nasa.gov/details/" + quote(str(data["nasa_id"]), safe=""),
                                          data.get("description", "")))
        return sources


class WikipediaFactProvider:
    name = "Wikipedia"

    def search(self, query: str) -> list[FactSource]:
        payload = json.loads(_fetch("https://en.wikipedia.org/w/api.php", {
            "action": "query", "format": "json", "formatversion": 2,
            "generator": "search", "gsrsearch": query, "gsrnamespace": 0,
            "gsrlimit": RESULTS_PER_QUERY, "prop": "extracts", "exintro": 1,
            "explaintext": 1, "exlimit": RESULTS_PER_QUERY, "exchars": min(1200, MAX_EXCERPT_CHARS),
        }))
        pages = payload.get("query", {}).get("pages", [])
        return [FactSource(self.name, page.get("title", ""),
                           "https://en.wikipedia.org/wiki/" + quote(page.get("title", "").replace(" ", "_"), safe=""),
                           page.get("extract", ""))
                for page in sorted(pages, key=lambda page: page.get("index", 0))[:RESULTS_PER_QUERY]]


def get_fact_providers(subject: str) -> list[FactSourceProvider]:
    return [NasaFactProvider(), WikipediaFactProvider()] if is_space_topic(subject) else [WikipediaFactProvider()]


def _excerpt(text: str, context: str) -> str:
    text = html.unescape(re.sub(r"<[^>]+>", " ", text))
    sentences = [" ".join(part.split()) for part in re.split(r"(?<=[.!?])\s+|\n+", text)]
    tokens = _subject_tokens(context)
    ordered = sorted(enumerate(sentences), key=lambda pair: -len(tokens & _subject_tokens(pair[1])))
    chosen = []
    length = 0
    for index, sentence in ordered:
        if sentence and length + len(sentence) + 1 <= MAX_EXCERPT_CHARS:
            chosen.append((index, sentence))
            length += len(sentence) + 1
    return " ".join(sentence for _, sentence in sorted(chosen))


def retrieve_fact_sources(subject: str, narration: str,
                          emit: Callable[[str, str], None] = log) -> list[FactSource]:
    providers = get_fact_providers(subject)
    sources: list[FactSource] = []
    seen_urls: set[str] = set()
    seen_text: set[str] = set()
    failed: set[str] = set()
    started = time.monotonic()
    for query in factual_queries(subject, narration):
        for provider in providers:
            if provider.name in failed:
                continue
            if time.monotonic() - started >= RETRIEVAL_SECONDS:
                return sources
            found = False
            try:
                for source in provider.search(query)[:RESULTS_PER_QUERY]:
                    title = " ".join(source.title.split())[:160]
                    excerpt = _excerpt(source.text, subject + " " + narration[:12000])
                    if len(excerpt) < 80 or not _subject_tokens(subject) & _subject_tokens(title + " " + excerpt):
                        continue
                    found = True
                    if source.url not in seen_urls and excerpt not in seen_text:
                        sources.append(FactSource(provider.name, title, source.url, excerpt))
                        seen_urls.add(source.url)
                        seen_text.add(excerpt)
                    if len(sources) >= MAX_SOURCES:
                        return sources
            except (requests.RequestException, ValueError, TypeError, KeyError, AttributeError) as err:
                emit(f"[FactCheck] {provider.name} retrieval failed ({type(err).__name__}).", "warning")
                failed.add(provider.name)
            if found:
                break  # Prefer NASA; Wikipedia supplies queries NASA cannot answer.
    return sources
