"""Utilities for Phase 3.5B controlled synthetic Markdown corpus metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from materials_rag.ingestion.utils import sha256_for_file

EXPECTED_SYNTHETIC_DOCS = [
    "MAT-SPEC-IN718-001.md",
    "PROC-LPBF-001.md",
    "BUILD-B017.md",
    "BUILD-B021.md",
    "HT-B017.md",
    "HT-B021.md",
    "FAT-B017.md",
    "FAT-B021.md",
    "FRACT-B017.md",
    "NCR-B017-004.md",
    "RCA-B017-004.md",
    "RCA-B017-004-rev2.md",
]


def parse_front_matter(path: Path) -> dict[str, Any]:
    """Parse YAML front matter from a synthetic corpus Markdown document."""

    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError(f"Missing front matter in {path}")

    end_index = next((index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---"), None)
    if end_index is not None:
        metadata = yaml.safe_load("\n".join(lines[1:end_index])) or {}
        if not isinstance(metadata, dict):
            raise ValueError(f"Front matter must be a mapping in {path}")
        return metadata
    raise ValueError(f"Unclosed front matter in {path}")


def build_phase35b_manifest(repo_root: Path) -> dict[str, Any]:
    """Return the installed frozen Phase 3.5B manifest when present."""

    manifest_path = (
        repo_root / "data" / "processed" / "manifests" / "phase_3_5b_synthetic_corpus.json"
    )
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    corpus_root = repo_root / "data" / "raw" / "internal_synthetic"
    documents = []
    for filename in EXPECTED_SYNTHETIC_DOCS:
        path = corpus_root / filename
        metadata = parse_front_matter(path)
        relative_path = path.relative_to(repo_root).as_posix()
        documents.append(
            {
                "path": relative_path,
                "filename": filename,
                "document_id": metadata["document_id"],
                "title": metadata.get("title"),
                "document_type": metadata.get("document_type"),
                "material": metadata.get("material"),
                "process": metadata.get("process"),
                "build_id": metadata.get("build_id"),
                "revision": metadata.get("revision"),
                "status": metadata.get("status"),
                "authority": metadata.get("authority"),
                "supersedes": metadata.get("supersedes"),
                "superseded_by": metadata.get("superseded_by"),
                "synthetic": metadata.get("synthetic"),
                "sha256": sha256_for_file(path),
            }
        )

    relationships = []
    for document in documents:
        if document.get("build_id"):
            relationships.append(
                {
                    "type": "document_describes_build",
                    "document_id": document["document_id"],
                    "build_id": document["build_id"],
                }
            )
        if document.get("supersedes"):
            relationships.append(
                {
                    "type": "supersedes",
                    "document_id": document["document_id"],
                    "target_document_id": document["supersedes"],
                }
            )
        if document.get("superseded_by"):
            relationships.append(
                {
                    "type": "superseded_by",
                    "document_id": document["document_id"],
                    "target_document_id": document["superseded_by"],
                }
            )

    build_relationships = {
        build_id: sorted(
            document["document_id"]
            for document in documents
            if document.get("build_id") == build_id
        )
        for build_id in ["B017", "B021"]
    }
    revision_relationships = [
        {
            "superseded_document_id": "RCA-B017-004",
            "superseding_document_id": "RCA-B017-004-rev2",
        }
    ]

    return {
        "phase": "3.5B",
        "synthetic": True,
        "corpus_root": corpus_root.relative_to(repo_root).as_posix(),
        "document_count": len(documents),
        "document_ids": [document["document_id"] for document in documents],
        "documents": documents,
        "relationships": sorted(
            relationships,
            key=lambda item: (
                item["type"],
                item["document_id"],
                item.get("target_document_id", item.get("build_id", "")),
            ),
        ),
        "revision_relationships": revision_relationships,
        "build_relationships": build_relationships,
    }


def write_phase35b_manifest(repo_root: Path) -> Path:
    """Return the frozen Phase 3.5B manifest path.

    The Astra package owns this manifest. Keeping this function as a no-op
    preserves older local callers without rewriting package-controlled metadata.
    """

    manifest_path = (
        repo_root / "data" / "processed" / "manifests" / "phase_3_5b_synthetic_corpus.json"
    )
    if not manifest_path.exists():
        manifest = build_phase35b_manifest(repo_root)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_path
