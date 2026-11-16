"""Provider-agnostic LLM client interface with Gemini, OpenAI, and Mock implementations."""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod

from aqb.config import Settings


class LLMClient(ABC):
    """Abstract base class for text/JSON generation."""

    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.15) -> str:
        """Generate a raw text response for the given prompt."""


class GeminiClient(LLMClient):
    """Google Gemini client using the google-genai SDK."""

    def __init__(self, model: str, api_key: str | None = None) -> None:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "google-genai is required for Gemini provider. Install with: pip install google-genai"
            ) from exc

        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY environment variable is not set.")
        self._client = genai.Client(api_key=key)
        self._types = types
        self._model = model

    def generate(self, prompt: str, temperature: float = 0.15) -> str:
        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=self._types.GenerateContentConfig(temperature=temperature),
        )
        return response.text or ""


class OpenAIClient(LLMClient):
    """OpenAI-compatible chat completions client."""

    def __init__(self, model: str, api_key: str | None = None) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "openai is required for OpenAI provider. Install with: pip install openai"
            ) from exc

        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
        self._client = OpenAI(api_key=key)
        self._model = model

    def generate(self, prompt: str, temperature: float = 0.15) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.choices[0].message.content or ""


class MockLLMClient(LLMClient):
    """Deterministic offline LLM client for local testing and CI."""

    def generate(self, prompt: str, temperature: float = 0.15) -> str:
        if "QUESTION BANK:" in prompt:
            return json.dumps(self._mock_parse_questions(prompt))
        if "PARSED QUESTIONS:" in prompt:
            return json.dumps(self._mock_analyze_topics(prompt))
        if "QUESTIONS TO SOLVE IN THIS BATCH:" in prompt:
            return json.dumps(self._mock_solve_batch(prompt))
        return "[]"

    @staticmethod
    def _mock_parse_questions(prompt: str) -> list[dict]:
        raw = prompt.split("QUESTION BANK:", 1)[1].strip()
        items: list[dict] = []
        current_marks = 2
        idx = 1
        for line in raw.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            lower = stripped.lower()
            if "13 mark" in lower or "part b" in lower:
                current_marks = 13
                if ":" not in stripped and "?" not in stripped:
                    continue
            elif "2 mark" in lower or "part a" in lower:
                current_marks = 2
                if ":" not in stripped and "?" not in stripped:
                    continue

            marks = current_marks
            mark_match = re.search(r"\[(\d+)\s*m(?:arks?)?\]|\((\d+)\s*marks?\)", lower)
            if mark_match:
                marks = int(mark_match.group(1) or mark_match.group(2))

            topic = "General"
            topic_match = re.search(r"topic:\s*([^\]\)]+)", stripped, re.IGNORECASE)
            if topic_match:
                topic = topic_match.group(1).strip()

            items.append(
                {
                    "id": f"Q{idx}",
                    "marks": marks,
                    "topic": topic,
                    "text": stripped,
                }
            )
            idx += 1
        return items

    @staticmethod
    def _mock_analyze_topics(prompt: str) -> list[dict]:
        section = prompt.split("PARSED QUESTIONS:", 1)[1].split("NOTES PREVIEW:", 1)[0]
        counts: dict[str, list[int]] = {}
        for line in section.splitlines():
            match = re.match(r"-\s*\[[^|]+\|\s*(\d+)M\s*\|\s*([^\]]+)\]", line.strip())
            if match:
                marks = int(match.group(1))
                topic = match.group(2).strip()
                counts.setdefault(topic, []).append(marks)

        ranked = sorted(counts.items(), key=lambda kv: sum(kv[1]), reverse=True)
        result = []
        for topic, mark_list in ranked:
            total = sum(mark_list)
            priority = "High" if total >= 15 else ("Medium" if total >= 6 else "Low")
            result.append(
                {
                    "topic": topic,
                    "question_count": len(mark_list),
                    "total_marks": total,
                    "priority": priority,
                    "rationale": f"Appears in {len(mark_list)} question(s) totaling {total} marks.",
                }
            )
        return result

    @staticmethod
    def _mock_solve_batch(prompt: str) -> list[dict]:
        notes_part = prompt.split("PROVIDED STUDY NOTES:", 1)[1].split(
            "QUESTIONS TO SOLVE IN THIS BATCH:", 1
        )[0]
        chunk_ids = re.findall(r"\[([^\]]+#C\d+)\]", notes_part)
        citation = [chunk_ids[0]] if chunk_ids else []

        q_part = prompt.split("QUESTIONS TO SOLVE IN THIS BATCH:", 1)[1]
        answers = []
        for line in q_part.splitlines():
            match = re.match(r"-\s*(Q\d+)\s*\((\d+)\s*Marks[^)]*\):\s*(.+)", line.strip())
            if match:
                qid, marks_str, qtext = match.groups()
                marks = int(marks_str)
                snippet = notes_part.strip().splitlines()[1] if len(notes_part.strip().splitlines()) > 1 else ""
                if marks <= 2:
                    body = f"Concise grounded answer for **{qtext}**. {snippet[:160]}".strip()
                else:
                    body = (
                        f"### Introduction\nGrounded overview for **{qtext}**.\n\n"
                        f"### Detailed Explanation\n{snippet[:300]}\n\n"
                        f"### Conclusion\nSummary aligned with the provided study notes."
                    )
                answers.append(
                    {
                        "question_id": qid,
                        "answer_markdown": body,
                        "citations": citation,
                    }
                )
        return answers


def create_llm_client(settings: Settings) -> LLMClient:
    """Factory for creating the configured LLM client."""
    if settings.provider == "gemini":
        return GeminiClient(model=settings.model)
    if settings.provider == "openai":
        return OpenAIClient(model=settings.model)
    if settings.provider == "mock":
        return MockLLMClient()
    raise ValueError(f"Unsupported provider: {settings.provider}")


def parse_json_response(raw: str) -> list | dict:
    """Parse JSON from LLM output, stripping optional markdown code fences."""
    cleaned = raw.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    return json.loads(cleaned)
