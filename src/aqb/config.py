"""Configuration management for Autonomous Question Bank."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

# Hard limits requested by design: small, focused replies to avoid truncation/hallucination.
BATCH_2M_RANGE = (10, 12)
BATCH_13M_RANGE = (5, 6)


def _clamp(value: int, bounds: tuple[int, int]) -> int:
    return max(bounds[0], min(bounds[1], value))


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


DEFAULT_MODELS = {
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-4o-mini",
    "mock": "mock",
}


@dataclass(frozen=True)
class Settings:
    """Runtime settings for the autonomous solver loop."""

    provider: str = "mock"
    model: str = "mock"
    batch_size_2m: int = 10
    batch_size_13m: int = 5
    temperature: float = 0.15

    # Retrieval
    top_k_per_question: int = 3
    max_context_chunks: int = 14
    full_context_char_limit: int = 60_000  # notes smaller than this are sent in full
    chunk_size_chars: int = 1800
    chunk_overlap_chars: int = 250

    # Question parsing
    parse_mode: str = "auto"  # auto | llm | heuristic
    parse_chunk_chars: int = 6000

    # Reliability
    max_retries: int = 3
    retry_base_delay: float = 2.0
    verify_long_answers: bool = False

    output_dir: Path = Path("outputs")

    def __post_init__(self) -> None:
        object.__setattr__(self, "batch_size_2m", _clamp(self.batch_size_2m, BATCH_2M_RANGE))
        object.__setattr__(self, "batch_size_13m", _clamp(self.batch_size_13m, BATCH_13M_RANGE))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.parse_mode not in {"auto", "llm", "heuristic"}:
            object.__setattr__(self, "parse_mode", "auto")

    def with_overrides(self, **changes) -> "Settings":
        """Return a copy with non-None overrides applied."""
        clean = {k: v for k, v in changes.items() if v is not None}
        if "provider" in clean and "model" not in clean:
            clean["model"] = DEFAULT_MODELS.get(clean["provider"], self.model)
        return replace(self, **clean)

    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from environment variables with safe defaults."""
        provider = os.getenv("AQB_PROVIDER", "mock").strip().lower() or "mock"
        return cls(
            provider=provider,
            model=os.getenv("AQB_MODEL", "").strip() or DEFAULT_MODELS.get(provider, "mock"),
            batch_size_2m=_env_int("AQB_BATCH_SIZE_2M", 10),
            batch_size_13m=_env_int("AQB_BATCH_SIZE_13M", 5),
            temperature=_env_float("AQB_TEMPERATURE", 0.15),
            top_k_per_question=_env_int("AQB_TOP_K_PER_QUESTION", 3),
            max_context_chunks=_env_int("AQB_MAX_CONTEXT_CHUNKS", 14),
            full_context_char_limit=_env_int("AQB_FULL_CONTEXT_CHAR_LIMIT", 60_000),
            parse_mode=os.getenv("AQB_PARSE_MODE", "auto").strip().lower(),
            max_retries=_env_int("AQB_MAX_RETRIES", 3),
            retry_base_delay=_env_float("AQB_RETRY_BASE_DELAY", 2.0),
            verify_long_answers=_env_bool("AQB_VERIFY_LONG_ANSWERS", False),
            output_dir=Path(os.getenv("AQB_OUTPUT_DIR", "outputs")),
        )
