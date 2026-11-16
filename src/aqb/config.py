"""Configuration management for Autonomous Question Bank."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _bounded_int(env_key: str, default: int, min_val: int, max_val: int) -> int:
    """Read an integer from environment and clamp it to [min_val, max_val]."""
    raw = os.getenv(env_key)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(min_val, min(max_val, value))


@dataclass(frozen=True)
class Settings:
    """Runtime settings for the autonomous solver loop."""

    provider: str = "mock"
    model: str = "gemini-2.5-flash"
    batch_size_2m: int = 10
    batch_size_13m: int = 5
    temperature: float = 0.15
    top_k_chunks: int = 8
    chunk_size_chars: int = 1800
    chunk_overlap_chars: int = 250
    output_dir: Path = Path("outputs")

    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from environment variables with safe defaults."""
        return cls(
            provider=os.getenv("AQB_PROVIDER", "mock").strip().lower(),
            model=os.getenv("AQB_MODEL", "gemini-2.5-flash").strip(),
            batch_size_2m=_bounded_int("AQB_BATCH_SIZE_2M", default=10, min_val=10, max_val=12),
            batch_size_13m=_bounded_int("AQB_BATCH_SIZE_13M", default=5, min_val=5, max_val=6),
            temperature=float(os.getenv("AQB_TEMPERATURE", "0.15")),
            top_k_chunks=int(os.getenv("AQB_TOP_K_CHUNKS", "8")),
            output_dir=Path(os.getenv("AQB_OUTPUT_DIR", "outputs")),
        )
