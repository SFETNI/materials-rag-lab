"""Locate a checked-out Materials RAG Lab project without machine-specific paths."""

from __future__ import annotations

from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """Return the nearest checkout containing public manifests and prompts."""

    starts = [start.resolve() if start else Path.cwd().resolve(), Path(__file__).resolve()]
    seen: set[Path] = set()
    for origin in starts:
        for candidate in [origin, *origin.parents]:
            if candidate in seen:
                continue
            seen.add(candidate)
            if (
                (candidate / "data/manifests/agentic_rag_v0.1.0_experiment_freeze.json").exists()
                and (candidate / "docs/generation/agentic_orchestrator_v1.md").exists()
            ):
                return candidate
    raise FileNotFoundError(
        "Run from a Materials RAG Lab checkout containing data/manifests and docs/generation."
    )


__all__ = ["find_project_root"]
