"""Model and local Agentic runtime status for public operators."""

from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
EMBEDDING_REVISION: str | None = None
RERANKER_MODEL = "BAAI/bge-reranker-base"
RERANKER_REVISION = "2cfc18c9415c912f9d8155881c133215df768a70"
ROLE_MODEL = "gpt-5.5"
ROLE_REASONING_EFFORT = "low"


@dataclass(frozen=True)
class ModelStatus:
    outcome: str
    embedding_available: bool
    reranker_available: bool
    codex_available: bool
    details: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _cached(model_id: str, revision: str | None) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    cached = try_to_load_from_cache(model_id, "config.json", revision=revision)
    return isinstance(cached, str) and Path(cached).exists()


def models_status() -> ModelStatus:
    embedding = _cached(EMBEDDING_MODEL, EMBEDDING_REVISION)
    reranker = _cached(RERANKER_MODEL, RERANKER_REVISION)
    codex = shutil.which("codex") is not None
    details = (
        f"embedding={EMBEDDING_MODEL} revision={EMBEDDING_REVISION or 'not recovered'}",
        f"reranker={RERANKER_MODEL} revision={RERANKER_REVISION}",
        f"isolated roles={ROLE_MODEL} reasoning_effort={ROLE_REASONING_EFFORT}",
    )
    outcome = "PASS" if embedding and reranker and codex else "WARN"
    return ModelStatus(outcome, embedding, reranker, codex, details)


def prepare_models(*, include_reranker: bool = True) -> dict[str, Any]:
    """Download the declared public model snapshots into the normal HF cache."""

    from huggingface_hub import snapshot_download

    embedding_path = snapshot_download(repo_id=EMBEDDING_MODEL)
    reranker_path = None
    if include_reranker:
        reranker_path = snapshot_download(repo_id=RERANKER_MODEL, revision=RERANKER_REVISION)
    return {
        "outcome": "PASS",
        "embedding": {"model": EMBEDDING_MODEL, "path": embedding_path},
        "reranker": {"model": RERANKER_MODEL, "revision": RERANKER_REVISION, "path": reranker_path},
        "note": "A different model or revision is a new replication, not frozen v0.1.0.",
    }


__all__ = [
    "EMBEDDING_MODEL",
    "RERANKER_MODEL",
    "RERANKER_REVISION",
    "ROLE_MODEL",
    "ROLE_REASONING_EFFORT",
    "models_status",
    "prepare_models",
]
