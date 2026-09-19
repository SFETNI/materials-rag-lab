"""Phase 3.5C ingestion for the frozen internal synthetic Markdown corpus."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from materials_rag.ingestion.chunker import ChunkConfig
from materials_rag.ingestion.utils import resolve_repo_root, sha256_for_file, write_jsonl
from materials_rag.schemas import Chunk, Document, SourceRef, make_human_readable_id

ANCHOR_RE = re.compile(r'<a id="([a-z0-9-]+)"></a>')
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
TOKEN_RE = re.compile(r"\S+")
SENTENCE_END_RE = re.compile(r"[.!?:;]$")

EXCLUDED_PATH_PREFIXES = [
    "authoring/phase35b/",
    "data/eval/phase35b/",
    "docs/phase35b/",
    "schemas/synthetic_benchmark/",
    "scripts/phase35b/",
    "tests/phase35b_package/",
    "data/processed/manifests/",
]


@dataclass(frozen=True)
class MarkdownSection:
    document_id: str
    section_id: str
    evidence_id: str
    heading: str
    text: str
    raw_segment: str
    source_path: str
    line_start: int
    line_end: int
    source_sha256: str
    section_sha256: str
    metadata: dict[str, Any]


@dataclass
class Phase35CResult:
    documents: list[Document]
    chunks: list[Chunk]
    sections: list[MarkdownSection]
    evidence_map: list[dict[str, Any]]
    warnings: list[dict[str, str]]
    outputs: list[Path]
    manifest_path: Path


class UniqueSafeLoader(yaml.SafeLoader):
    """YAML loader that rejects duplicate mapping keys."""


def _unique_mapping(loader: UniqueSafeLoader, node: Any, deep: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"Duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _unique_mapping,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _token_count(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _front_matter_and_body(text: str, path: Path) -> tuple[dict[str, Any], str]:
    match = re.match(r"\A---\n(.*?)\n---\n(.*)\Z", text, re.DOTALL)
    if not match:
        raise ValueError(f"Missing YAML front matter or non-LF source: {path}")
    metadata = yaml.load(match.group(1), Loader=UniqueSafeLoader)
    if not isinstance(metadata, dict):
        raise TypeError(f"Front matter must be a mapping: {path}")
    return metadata, match.group(2)


def _required_metadata(metadata: dict[str, Any], path: Path) -> None:
    required = [
        "corpus_id",
        "corpus_version",
        "synthetic",
        "source_kind",
        "document_id",
        "revision_family_id",
        "title",
        "document_type",
        "material",
        "process",
        "study_id",
        "build_ids",
        "revision",
        "status",
        "authority",
        "is_current",
        "issued_on",
        "effective_from",
        "snapshot_date",
        "related_documents",
        "language",
        "supersedes",
        "superseded_by",
    ]
    missing = [key for key in required if key not in metadata]
    if missing:
        raise ValueError(f"Missing required front-matter keys in {path}: {missing}")
    if metadata.get("synthetic") is not True:
        raise ValueError(f"Markdown source must explicitly be synthetic: {path}")


def _section_text_without_anchor(raw_segment: str) -> tuple[str, str]:
    lines = raw_segment.splitlines()
    if lines and ANCHOR_RE.fullmatch(lines[0].strip()):
        lines = lines[1:]
    text = "\n".join(lines).strip()
    heading_match = HEADING_RE.search(text)
    heading = heading_match.group(2).strip() if heading_match else ""
    return heading, text


def _parse_sections(
    full_text: str,
    source_path: str,
    metadata: dict[str, Any],
    source_sha256: str,
) -> list[MarkdownSection]:
    matches = list(ANCHOR_RE.finditer(full_text))
    sections: list[MarkdownSection] = []
    seen: set[str] = set()
    for index, match in enumerate(matches):
        section_id = match.group(1)
        if section_id in seen:
            raise ValueError(f"Duplicate section anchor in {source_path}: {section_id}")
        seen.add(section_id)
        end = matches[index + 1].start() if index + 1 < len(matches) else len(full_text)
        raw_segment = full_text[match.start() : end].strip()
        heading, section_text = _section_text_without_anchor(raw_segment)
        document_id = str(metadata["document_id"])
        sections.append(
            MarkdownSection(
                document_id=document_id,
                section_id=section_id,
                evidence_id=f"{document_id}#{section_id}",
                heading=heading,
                text=section_text,
                raw_segment=raw_segment,
                source_path=source_path,
                line_start=full_text[: match.start()].count("\n") + 1,
                line_end=full_text[:end].count("\n"),
                source_sha256=source_sha256,
                section_sha256=hashlib.sha256(raw_segment.encode()).hexdigest(),
                metadata=dict(metadata),
            )
        )
    return sections


def _load_manifest(repo_root: Path) -> dict[str, Any]:
    path = repo_root / "data" / "processed" / "manifests" / "phase_3_5b_synthetic_corpus.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_source_paths(manifest: dict[str, Any]) -> list[str]:
    paths = [document["relative_path"] for document in manifest["documents"]]
    if len(paths) != 12 or len(set(paths)) != 12:
        raise ValueError("Phase 3.5B manifest must list exactly 12 unique source paths.")
    for path in paths:
        if not path.startswith("data/raw/internal_synthetic/") or not path.endswith(".md"):
            raise ValueError(f"Invalid Phase 3.5B index source path: {path}")
    return sorted(paths)


def parse_internal_markdown_corpus(repo_root: Path) -> tuple[list[Document], list[MarkdownSection]]:
    """Parse only the Phase 3.5B manifest-listed Markdown sources."""

    manifest = _load_manifest(repo_root)
    documents: list[Document] = []
    sections: list[MarkdownSection] = []
    for relative_path in _manifest_source_paths(manifest):
        path = repo_root / relative_path
        source_hash = sha256_for_file(path)
        text = path.read_text(encoding="utf-8")
        metadata, body = _front_matter_and_body(text, path)
        _required_metadata(metadata, path)
        if metadata["document_id"] != path.stem:
            raise ValueError(f"Document ID / filename mismatch: {relative_path}")

        document_sections = _parse_sections(text, relative_path, metadata, source_hash)
        document_id = make_human_readable_id(
            "document",
            str(metadata["corpus_id"]),
            str(metadata["document_id"]),
        )
        source = SourceRef(
            dataset=str(metadata["corpus_id"]),
            source_path=relative_path,
            authority=str(metadata["authority"]),
            source_dataset="internal_synthetic",
            source_revision=str(metadata.get("corpus_version") or ""),
            file_hash=source_hash,
        )
        retrieval_text = (
            f"Document: {metadata['title']}\n"
            f"Document type: {metadata['document_type']}\n"
            f"Status: {metadata['status']}\n"
            f"Synthetic internal source."
        )
        documents.append(
            Document(
                id=document_id,
                title=str(metadata["title"]),
                document_type=str(metadata["document_type"]),
                language=str(metadata.get("language") or "en"),
                retrieval_text=retrieval_text,
                source=source,
                metadata={
                    "front_matter": metadata,
                    "body": body,
                    "source_path": relative_path,
                    "source_sha256": source_hash,
                    "document_id": metadata["document_id"],
                    "corpus_id": metadata["corpus_id"],
                    "corpus_version": metadata["corpus_version"],
                    "synthetic": metadata["synthetic"],
                    "source_kind": metadata["source_kind"],
                    "revision_family_id": metadata["revision_family_id"],
                    "material": metadata["material"],
                    "process": metadata["process"],
                    "study_id": metadata["study_id"],
                    "build_ids": metadata["build_ids"],
                    "revision": metadata["revision"],
                    "status": metadata["status"],
                    "authority": metadata["authority"],
                    "is_current": metadata["is_current"],
                    "issued_on": metadata["issued_on"],
                    "effective_from": metadata["effective_from"],
                    "effective_to": metadata.get("effective_to"),
                    "snapshot_date": metadata["snapshot_date"],
                    "supersedes": metadata.get("supersedes"),
                    "superseded_by": metadata.get("superseded_by"),
                    "related_documents": metadata["related_documents"],
                    "section_ids": [section.section_id for section in document_sections],
                },
            )
        )
        sections.extend(document_sections)
    return documents, sections


def _split_tokens_for_section(text: str, config: ChunkConfig) -> list[str]:
    tokens = TOKEN_RE.findall(text)
    if len(tokens) <= config.max_tokens:
        return [" ".join(tokens)]

    chunks: list[list[str]] = []
    cursor = 0
    content_limit = max(1, config.max_tokens - config.overlap_tokens)
    while cursor < len(tokens):
        remaining = len(tokens) - cursor
        if remaining <= config.max_tokens:
            chunks.append(tokens[cursor:])
            break

        min_end = min(len(tokens) - 1, cursor + config.target_tokens - 1)
        max_end = min(len(tokens) - 1, cursor + content_limit - 1)
        split_end = max_end
        for index in range(max_end, min_end - 1, -1):
            if SENTENCE_END_RE.search(tokens[index]):
                split_end = index
                break
        chunks.append(tokens[cursor : split_end + 1])
        cursor = split_end + 1

    result: list[str] = []
    for index, chunk_tokens in enumerate(chunks):
        if index > 0:
            overlap = chunks[index - 1][-config.overlap_tokens :]
            chunk_tokens = overlap + chunk_tokens
        if len(chunk_tokens) > config.max_tokens:
            chunk_tokens = chunk_tokens[-config.max_tokens :]
        result.append(" ".join(chunk_tokens))
    return result


def _retrieval_text(section: MarkdownSection, text: str) -> str:
    metadata = section.metadata
    build_ids = metadata.get("build_ids") or []
    build_line = f"Build: {', '.join(build_ids)}\n" if build_ids else ""
    return (
        f"Document: {metadata.get('title')}\n"
        f"Document type: {metadata.get('document_type')}\n"
        f"{build_line}"
        f"Section: {section.heading}\n"
        f"{text}"
    )


def chunk_markdown_sections(
    documents: list[Document],
    sections: list[MarkdownSection],
    config: ChunkConfig | None = None,
) -> list[Chunk]:
    if config is None:
        config = ChunkConfig()
    document_by_source_id = {
        str(document.metadata["document_id"]): document for document in documents
    }
    chunk_indexes: dict[str, int] = defaultdict(int)
    chunks: list[Chunk] = []

    for section in sections:
        document = document_by_source_id[section.document_id]
        split_texts = _split_tokens_for_section(section.text, config)
        for section_chunk_index, text in enumerate(split_texts):
            chunk_index = chunk_indexes[document.id]
            chunk_indexes[document.id] += 1
            chunk_id = make_human_readable_id(
                "chunk",
                document.id,
                section.section_id,
                str(section_chunk_index),
            )
            metadata = {
                "corpus_id": section.metadata.get("corpus_id"),
                "corpus_version": section.metadata.get("corpus_version"),
                "synthetic": section.metadata.get("synthetic"),
                "source_kind": section.metadata.get("source_kind"),
                "source_path": section.source_path,
                "source_sha256": section.source_sha256,
                "document_source_id": section.document_id,
                "section_id": section.section_id,
                "evidence_id": section.evidence_id,
                "heading": section.heading,
                "section_chunk_index": section_chunk_index,
                "section_chunk_count": len(split_texts),
                "line_start": section.line_start,
                "line_end": section.line_end,
                "section_sha256": section.section_sha256,
                "revision_family_id": section.metadata.get("revision_family_id"),
                "revision": section.metadata.get("revision"),
                "status": section.metadata.get("status"),
                "authority": section.metadata.get("authority"),
                "is_current": section.metadata.get("is_current"),
                "issued_on": section.metadata.get("issued_on"),
                "effective_from": section.metadata.get("effective_from"),
                "effective_to": section.metadata.get("effective_to"),
                "supersedes": section.metadata.get("supersedes"),
                "superseded_by": section.metadata.get("superseded_by"),
                "build_ids": section.metadata.get("build_ids"),
                "related_documents": section.metadata.get("related_documents"),
                "overlap_tokens_with_previous": config.overlap_tokens
                if section_chunk_index > 0
                else 0,
            }
            chunks.append(
                Chunk(
                    id=chunk_id,
                    retrieval_text=_retrieval_text(section, text),
                    source=document.source,
                    metadata=metadata,
                    document_id=document.id,
                    chunk_index=chunk_index,
                    start_char=None,
                    end_char=None,
                    text=text,
                    token_count=_token_count(text),
                )
            )
    return chunks


def validate_evidence_registry(
    repo_root: Path,
    sections: list[MarkdownSection],
    chunks: list[Chunk],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    registry_path = repo_root / "authoring" / "phase35b" / "evidence_registry.jsonl"
    registry = _read_jsonl(registry_path)
    section_by_evidence = {section.evidence_id: section for section in sections}
    chunks_by_evidence: dict[str, list[str]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_evidence[str(chunk.metadata["evidence_id"])].append(chunk.id)

    errors: list[dict[str, str]] = []
    records: list[dict[str, Any]] = []
    for row in registry:
        evidence_id = str(row["evidence_id"])
        expected_evidence_id = f"{row['document_id']}#{row['section_id']}"
        section = section_by_evidence.get(evidence_id)
        if evidence_id != expected_evidence_id:
            errors.append({"type": "evidence_id_mismatch", "evidence_id": evidence_id})
            continue
        if section is None:
            errors.append({"type": "missing_section", "evidence_id": evidence_id})
            continue
        comparisons = {
            "source_path": section.source_path,
            "source_sha256": section.source_sha256,
            "section_sha256": section.section_sha256,
            "line_start": section.line_start,
            "line_end": section.line_end,
        }
        for key, value in comparisons.items():
            if row.get(key) != value:
                errors.append({
                    "type": f"{key}_mismatch",
                    "evidence_id": evidence_id,
                    "expected": str(row.get(key)),
                    "actual": str(value),
                })
        chunk_ids = chunks_by_evidence.get(evidence_id, [])
        if not chunk_ids:
            errors.append({"type": "unmapped_evidence", "evidence_id": evidence_id})
        records.append(
            {
                "evidence_id": evidence_id,
                "document_id": row["document_id"],
                "section_id": row["section_id"],
                "chunk_ids": chunk_ids,
            }
        )
    return records, errors


def _write_evidence_map(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _manifest(
    repo_root: Path,
    documents: list[Document],
    sections: list[MarkdownSection],
    chunks: list[Chunk],
    evidence_map: list[dict[str, Any]],
    evidence_errors: list[dict[str, str]],
    config: ChunkConfig,
    output_paths: list[Path],
) -> dict[str, Any]:
    token_counts = sorted(chunk.token_count for chunk in chunks)
    chunks_by_document = Counter(chunk.document_id for chunk in chunks)
    chunks_by_section = Counter(str(chunk.metadata["evidence_id"]) for chunk in chunks)
    revision_status_counts = Counter(
        f"rev{section.metadata.get('revision')}:{section.metadata.get('status')}"
        for section in sections
    )
    index_sources = [document.source.source_path for document in documents]
    return {
        "phase": "3.5C",
        "version": "1.0.0",
        "source_documents": len(documents),
        "parsed_sections_count": len(sections),
        "generated_chunks_count": len(chunks),
        "chunks_per_document": dict(sorted(chunks_by_document.items())),
        "chunks_per_section_distribution": dict(Counter(chunks_by_section.values())),
        "token_statistics": {
            "min": token_counts[0] if token_counts else 0,
            "median": token_counts[len(token_counts) // 2] if token_counts else 0,
            "mean": float(sum(token_counts) / len(token_counts)) if token_counts else 0.0,
            "max": token_counts[-1] if token_counts else 0,
        },
        "evidence_registry_entries": len(evidence_map),
        "evidence_ids_successfully_mapped": sum(1 for record in evidence_map if record["chunk_ids"]),
        "unmapped_evidence_ids": [
            record["evidence_id"] for record in evidence_map if not record["chunk_ids"]
        ],
        "evidence_validation_errors": evidence_errors,
        "revision_status_counts": dict(sorted(revision_status_counts.items())),
        "source_hashes": {
            document.source.source_path: document.source.file_hash for document in documents
        },
        "chunking_configuration": {
            "strategy": "anchored_markdown_section_first",
            "target_tokens": config.target_tokens,
            "max_tokens": config.max_tokens,
            "overlap_tokens_within_section": config.overlap_tokens,
            "cross_section_overlap": False,
        },
        "index_eligible_source_paths": sorted(index_sources),
        "excluded_paths": EXCLUDED_PATH_PREFIXES,
        "output_artifacts": [
            {"path": path.relative_to(repo_root).as_posix(), "type": path.suffix.lstrip(".")}
            for path in output_paths
        ],
        "warnings": [],
    }


def run_phase35c(
    repo_root: Path | None = None,
    output_root: Path | None = None,
    config: ChunkConfig | None = None,
) -> Phase35CResult:
    """Run Phase 3.5C Markdown ingestion and evidence-to-chunk mapping."""

    if repo_root is None:
        repo_root = resolve_repo_root(Path(__file__))
    if output_root is None:
        output_root = repo_root / "data" / "processed"
    if config is None:
        config = ChunkConfig()

    documents, sections = parse_internal_markdown_corpus(repo_root)
    chunks = chunk_markdown_sections(documents, sections, config=config)
    evidence_map, evidence_errors = validate_evidence_registry(repo_root, sections, chunks)
    if evidence_errors:
        raise ValueError(f"Evidence registry validation failed: {evidence_errors[:3]}")

    canonical_root = output_root / "phase3_5c" / "canonical"
    documents_path = canonical_root / "documents.jsonl"
    chunks_path = canonical_root / "chunks.jsonl"
    evidence_map_path = output_root / "phase3_5c" / "evidence_chunk_map.jsonl"
    manifest_path = output_root / "manifests" / "phase_3_5c_manifest.json"

    write_jsonl(documents, documents_path)
    write_jsonl(chunks, chunks_path)
    _write_evidence_map(evidence_map, evidence_map_path)

    output_paths = [documents_path, chunks_path, evidence_map_path]
    manifest = _manifest(
        repo_root,
        documents,
        sections,
        chunks,
        evidence_map,
        evidence_errors,
        config,
        output_paths,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    return Phase35CResult(
        documents=documents,
        chunks=chunks,
        sections=sections,
        evidence_map=evidence_map,
        warnings=[],
        outputs=output_paths + [manifest_path],
        manifest_path=manifest_path,
    )
