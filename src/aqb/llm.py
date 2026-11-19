"""Provider-agnostic LLM clients, retry/backoff wrapper, and robust JSON handling."""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Callable

from aqb.config import Settings

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Raised when the LLM fails permanently (after retries)."""


class LLMClient(ABC):
    """Abstract base class for text/JSON generation."""

    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.15, json_mode: bool = False) -> str:
        """Generate a raw text response for the given prompt."""


# --------------------------------------------------------------------------------------
# Real providers
# --------------------------------------------------------------------------------------
class GeminiClient(LLMClient):
    """Google Gemini client using the google-genai SDK."""

    def __init__(self, model: str, api_key: str | None = None) -> None:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "google-genai is required for the Gemini provider. Install with: pip install google-genai"
            ) from exc

        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set (add it to .env or the environment).")
        self._client = genai.Client(api_key=key)
        self._types = types
        self._model = model

    def generate(self, prompt: str, temperature: float = 0.15, json_mode: bool = False) -> str:
        config = self._types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="application/json" if json_mode else None,
        )
        response = self._client.models.generate_content(
            model=self._model, contents=prompt, config=config
        )
        return response.text or ""


class OpenAIClient(LLMClient):
    """OpenAI (or any OpenAI-compatible endpoint via OPENAI_BASE_URL, e.g. Ollama/Groq/OpenRouter)."""

    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "openai is required for the OpenAI provider. Install with: pip install openai"
            ) from exc

        key = api_key or os.getenv("OPENAI_API_KEY")
        url = base_url or os.getenv("OPENAI_BASE_URL") or None
        if not key and not url:
            raise RuntimeError("OPENAI_API_KEY is not set (add it to .env or the environment).")
        self._client = OpenAI(api_key=key or "not-needed", base_url=url)
        self._model = model

    def generate(self, prompt: str, temperature: float = 0.15, json_mode: bool = False) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.choices[0].message.content or ""


# --------------------------------------------------------------------------------------
# Retry wrapper
# --------------------------------------------------------------------------------------
_NON_RETRYABLE_HINTS = ("api key", "api_key", "permission", "unauthorized", "invalid argument", "not found")


class RetryingLLMClient(LLMClient):
    """Wraps any client with exponential backoff + jitter for transient errors (429/5xx/timeouts)."""

    def __init__(
        self,
        inner: LLMClient,
        max_retries: int = 3,
        base_delay: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.inner = inner
        self.max_retries = max(0, max_retries)
        self.base_delay = base_delay
        self._sleep = sleep

    def generate(self, prompt: str, temperature: float = 0.15, json_mode: bool = False) -> str:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                text = self.inner.generate(prompt, temperature=temperature, json_mode=json_mode)
                if not text or not text.strip():
                    raise LLMError("Empty response from model")
                return text
            except Exception as exc:  # noqa: BLE001 - providers raise many types
                last_exc = exc
                msg = str(exc).lower()
                if any(h in msg for h in _NON_RETRYABLE_HINTS) and "429" not in msg:
                    raise LLMError(f"Non-retryable LLM error: {exc}") from exc
                if attempt >= self.max_retries:
                    break
                delay = self.base_delay * (2**attempt) + random.uniform(0, self.base_delay / 2)
                log.warning("LLM call failed (%s). Retrying in %.1fs [%d/%d]", exc, delay, attempt + 1, self.max_retries)
                self._sleep(delay)
        raise LLMError(f"LLM call failed after {self.max_retries + 1} attempts: {last_exc}") from last_exc


# --------------------------------------------------------------------------------------
# JSON parsing / repair
# --------------------------------------------------------------------------------------
def _extract_json_block(text: str) -> str:
    cleaned = text.strip()
    # Only unwrap a fence that encloses the whole reply; answers themselves may contain ``` blocks.
    fence = re.match(r"^```[a-zA-Z]*\s*\n?(.*?)\n?```\s*$", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    elif cleaned[:1] not in "[{":
        inner = re.search(r"```(?:json)?\s*\n(\s*[\[{].*?)\n```", cleaned, re.DOTALL | re.IGNORECASE)
        if inner:
            cleaned = inner.group(1).strip()
    starts = [i for i in (cleaned.find("["), cleaned.find("{")) if i != -1]
    if not starts:
        return cleaned
    start = min(starts)
    closer = "]" if cleaned[start] == "[" else "}"
    end = cleaned.rfind(closer)
    return cleaned[start : end + 1] if end > start else cleaned[start:]


def _escape_control_chars_in_strings(text: str) -> str:
    """Escape raw newlines/tabs that models sometimes emit inside JSON strings."""
    out: list[str] = []
    in_str = False
    escaped = False
    for ch in text:
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            elif ch == "\n":
                out.append("\\n")
                continue
            elif ch == "\r":
                continue
            elif ch == "\t":
                out.append("\\t")
                continue
        elif ch == '"':
            in_str = True
        out.append(ch)
    return "".join(out)


def parse_json_response(raw: str) -> list | dict:
    """Parse JSON from LLM output, repairing common issues (fences, prose, trailing commas, raw newlines)."""
    block = _extract_json_block(raw)
    attempts = [
        raw.strip(),
        block,
        re.sub(r",\s*([\]}])", r"\1", block),
    ]
    attempts.append(_escape_control_chars_in_strings(attempts[-1]))
    last_err: Exception | None = None
    for candidate in attempts:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_err = exc
    raise ValueError(f"Could not parse JSON from model output: {last_err}") from last_err


def _coerce_list(parsed: list | dict) -> list:
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        for value in parsed.values():
            if isinstance(value, list):
                return value
        return [parsed]
    raise ValueError("Expected a JSON array")


def generate_json_list(
    client: LLMClient,
    prompt: str,
    temperature: float = 0.15,
    repair_attempts: int = 1,
) -> list:
    """Generate and parse a JSON array; on parse failure, re-ask the model to fix its output."""
    raw = client.generate(prompt, temperature=temperature, json_mode=True)
    for attempt in range(repair_attempts + 1):
        try:
            return _coerce_list(parse_json_response(raw))
        except ValueError as exc:
            if attempt >= repair_attempts:
                raise LLMError(f"Model returned invalid JSON: {exc}") from exc
            log.warning("Invalid JSON from model, asking it to repair (%s)", exc)
            raw = client.generate(
                prompt
                + "\n\nIMPORTANT: Your previous reply was not valid JSON. Reply again with ONLY the JSON array, "
                "no prose and no code fences.",
                temperature=0.0,
                json_mode=True,
            )
    return []


# --------------------------------------------------------------------------------------
# Offline mock provider
# --------------------------------------------------------------------------------------
class MockLLMClient(LLMClient):
    """Deterministic offline LLM for demos, tests and CI. Grounds 'answers' in retrieved chunk text."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate(self, prompt: str, temperature: float = 0.15, json_mode: bool = False) -> str:
        self.calls.append(prompt)
        if "QUESTION BANK:" in prompt:
            return json.dumps(self._parse_questions(prompt))
        if "TOPICS TO ANALYZE:" in prompt:
            return json.dumps(self._topic_focus(prompt))
        if "ANSWERS TO VERIFY:" in prompt:
            return json.dumps(self._verify(prompt))
        if "QUESTIONS TO SOLVE IN THIS BATCH:" in prompt:
            return json.dumps(self._solve_batch(prompt))
        return "[]"

    @staticmethod
    def _parse_questions(prompt: str) -> list[dict]:
        from aqb.question_parser import heuristic_parse_questions

        raw = prompt.split("QUESTION BANK:", 1)[1]
        context = re.search(r'most recent section/unit header before this slice was: "([^"]*)"', prompt)
        if context and context.group(1):
            raw = context.group(1) + "\n" + raw
        return [q.to_dict() for q in heuristic_parse_questions(raw)]

    @staticmethod
    def _topic_focus(prompt: str) -> list[dict]:
        section = prompt.split("TOPICS TO ANALYZE:", 1)[1].split("NOTES PREVIEW:", 1)[0]
        result = []
        for match in re.finditer(r"^### (.+?) — ", section, re.MULTILINE):
            topic = match.group(1).strip()
            result.append(
                {
                    "topic": topic,
                    "focus_points": [f"Core definitions in {topic}", f"Frequently repeated {topic} questions"],
                    "rationale": f"{topic} recurs across the question bank.",
                }
            )
        return result

    @staticmethod
    def _verify(prompt: str) -> list[dict]:
        payload = json.loads(prompt.split("ANSWERS TO VERIFY:", 1)[1].strip())
        return [
            {
                "question_id": item["question_id"],
                "supported": True,
                "unsupported_claims": [],
                "revised_answer_markdown": item["answer_markdown"],
            }
            for item in payload
        ]

    @staticmethod
    def _solve_batch(prompt: str) -> list[dict]:
        from aqb.ingestion import tokenize

        notes_part = prompt.split("PROVIDED STUDY NOTES:", 1)[1].split("QUESTIONS TO SOLVE IN THIS BATCH:", 1)[0]
        chunk_re = re.compile(r"^\[([^\]\n]+#[^\]\n]+)\]\n(.*?)(?=^\[[^\]\n]+#[^\]\n]+\]\n|\Z)", re.M | re.S)
        sentences: list[tuple[str, str]] = []  # (chunk_id, sentence)
        for chunk_id, body in chunk_re.findall(notes_part):
            for sent in re.split(r"(?<=[.!?])\s+|\n+", body):
                sent = sent.strip(" -*#`")
                if len(sent) > 25:
                    sentences.append((chunk_id, sent))

        q_part = prompt.split("QUESTIONS TO SOLVE IN THIS BATCH:", 1)[1]
        answers = []
        for line in q_part.splitlines():
            match = re.match(r"-\s*(Q\d+)\s*\((\d+)\s*Marks[^)]*\):\s*(.+)", line.strip())
            if not match:
                continue
            qid, marks_str, qtext = match.groups()
            q_terms = set(tokenize(qtext))
            scored = sorted(
                ((len(q_terms & set(tokenize(s))), i) for i, (_, s) in enumerate(sentences)),
                reverse=True,
            )
            best = [i for score, i in scored if score >= max(1, min(2, len(q_terms)))]
            if not best:
                answers.append({"question_id": qid, "answer_markdown": "[Not found in provided notes]", "citations": []})
                continue
            short = int(marks_str) <= 5
            picked = sorted(best[: 2 if short else 7])
            cites = sorted({sentences[i][0] for i in picked})
            if short:
                body = " ".join(sentences[i][1] for i in picked)
            else:
                bullets = "\n".join(f"- {sentences[i][1]}" for i in picked[1:])
                body = (
                    f"### Introduction\n{sentences[picked[0]][1]}\n\n### Explanation\n{bullets}\n\n"
                    "### Conclusion\nThe above points, taken from the notes, answer the question."
                )
            answers.append({"question_id": qid, "answer_markdown": body, "citations": cites})
        return answers


# --------------------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------------------
def create_llm_client(settings: Settings, api_key: str | None = None, with_retries: bool = True) -> LLMClient:
    """Create the configured LLM client (wrapped with retry/backoff for real providers)."""
    if settings.provider == "gemini":
        client: LLMClient = GeminiClient(model=settings.model, api_key=api_key)
    elif settings.provider == "openai":
        client = OpenAIClient(model=settings.model, api_key=api_key)
    elif settings.provider == "mock":
        return MockLLMClient()
    else:
        raise ValueError(f"Unsupported provider: {settings.provider!r} (use gemini, openai or mock)")
    if with_retries:
        client = RetryingLLMClient(client, max_retries=settings.max_retries, base_delay=settings.retry_base_delay)
    return client
