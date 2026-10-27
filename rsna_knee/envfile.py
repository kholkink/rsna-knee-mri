"""Minimal ``.env`` loader (KEY=VALUE per line). Values already in the environment win."""
from __future__ import annotations

import os
from pathlib import Path


def load_env_file(path: str | os.PathLike = ".env") -> list[str]:
    p = Path(path)
    loaded: list[str] = []
    if not p.exists():
        return loaded
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip().removeprefix("export ").strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded
