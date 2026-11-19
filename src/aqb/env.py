"""Zero-dependency `.env` loader."""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | Path = ".env", override: bool = False) -> dict[str, str]:
    """Load KEY=VALUE pairs from a .env file into os.environ.

    - Ignores blank lines and `#` comments.
    - Supports optional `export ` prefix and single/double-quoted values.
    - Existing environment variables win unless `override=True`.

    Returns the parsed key/value pairs (whether or not they were applied).
    """
    env_path = Path(path)
    if not env_path.is_file():
        return {}

    parsed: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        if not key:
            continue
        parsed[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return parsed
