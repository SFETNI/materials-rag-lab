"""Utilities for deterministic parsing and output writing."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def sha256_for_file(path: Path) -> str:
    """Return lowercase SHA256 digest for a file path."""

    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_text(value: Any) -> str:
    """Normalize text in a conservative JSON-safe way."""

    if value is None:
        return ""
    return str(value).replace("\x00", "").strip()


def write_jsonl(records: Iterable[BaseModel], output_path: Path) -> int:
    """Write deterministic JSONL and return number of lines."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = 0
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            payload = record.model_dump(mode="json", exclude_none=True)
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            lines += 1
    return lines


def normalized_id(*parts: str) -> str:
    """Canonical component cleanup for deterministic IDs."""

    return "|".join(part.strip() if isinstance(part, str) else str(part) for part in parts)


def _as_directory(path: Path) -> Path:
    return path if path.is_dir() else path.parent


def _looks_like_repo_root(path: Path) -> bool:
    return (path / "pyproject.toml").is_file() and (path / "src" / "materials_rag").is_dir()


def resolve_repo_root(start: Path | str | None = None) -> Path:
    """Resolve the project root from Git first, then fall back to local markers."""

    start_path = _as_directory(Path(start).resolve()) if start is not None else Path.cwd().resolve()
    candidates = [start_path, Path.cwd().resolve(), _as_directory(Path(__file__).resolve())]

    for candidate in candidates:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=candidate,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            continue
        root = Path(result.stdout.strip()).resolve()
        if _looks_like_repo_root(root):
            return root

    for candidate in candidates:
        for path in (candidate, *candidate.parents):
            if _looks_like_repo_root(path):
                return path

    raise RuntimeError(f"Could not resolve repository root from {start_path}.")
