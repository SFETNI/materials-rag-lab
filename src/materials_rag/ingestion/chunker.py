"""Chunking utilities for narrative Document objects."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from materials_rag.ingestion.text_quality import find_suspicious_text
from materials_rag.schemas import Chunk, Document, infer_chunk_id

_WHITESPACE_RE = re.compile(r"[ \t]+")
_TOKEN_RE = re.compile(r"\S+")
_SENTENCE_END_RE = re.compile(r"[.!?:;]$")
_END_WITH_ELLIPSIS_RE = re.compile(r"\.\.\.$")


@dataclass
class ChunkConfig:
    target_tokens: int = 500
    max_tokens: int = 650
    overlap_tokens: int = 75
    tiny_chunk_threshold: int = 90


@dataclass
class _TokenSpan:
    token: str
    page: int


def _token_count(text: str) -> int:
    return len(_TOKEN_RE.findall(text))


def _clean_for_chunking(text: str) -> str:
    text = text.replace("\r", "\n")
    text = text.replace("\u0000", "")
    return _WHITESPACE_RE.sub(" ", text).strip()


def _to_token_stream(document: Document) -> list[_TokenSpan]:
    pages = document.metadata.get("page_text_normalized")
    if not isinstance(pages, list):
        return []

    spans: list[_TokenSpan] = []
    for page_entry in pages:
        if not isinstance(page_entry, dict):
            continue
        page = int(page_entry.get("page", 0)) if page_entry.get("page") else 0
        if page <= 0:
            continue
        raw_text = str(page_entry.get("text", ""))
        normalized = _clean_for_chunking(raw_text)
        if not normalized:
            continue
        for token in _TOKEN_RE.findall(normalized):
            spans.append(_TokenSpan(token=token, page=page))
    return spans


def _is_sentence_boundary(token: str) -> bool:
    return bool(_SENTENCE_END_RE.search(token)) and not bool(_END_WITH_ELLIPSIS_RE.search(token))


def _last_sentence_boundary(spans: list[_TokenSpan], min_index: int = 0) -> int | None:
    for index in range(len(spans) - 1, max(min_index - 1, 0) - 1, -1):
        if _is_sentence_boundary(spans[index].token):
            return index
    return None


def _chunk_from_token_spans(chunk_index: int, document: Document, spans: list[_TokenSpan]) -> Chunk:
    token_texts = [span.token for span in spans]
    token_text = " ".join(token_texts)
    suspicious_findings = find_suspicious_text(token_text)
    source_pages = sorted({span.page for span in spans})
    page_start = min(source_pages) if source_pages else None
    page_end = max(source_pages) if source_pages else None
    retrieval_title = (
        f"Document: {document.title}" if document.title else f"Document: {document.id}"
    )

    return Chunk(
        id=infer_chunk_id(document.id, chunk_index),
        kind="chunk",
        retrieval_text=f"{retrieval_title} | {token_text}",
        source=document.source,
        text=token_text,
        token_count=len(token_texts),
        document_id=document.id,
        chunk_index=chunk_index,
        page_start=page_start,
        page_end=page_end,
        metadata={
            "source_dataset": document.source.source_path.split("/", 1)[0],
            "source_path": document.source.source_path,
            "page_start": page_start,
            "page_end": page_end,
            "page_count": len(document.metadata.get("page_text_normalized") or []),
            "source_text_pages": source_pages,
            "contains_unknown_encoding": "?" in token_text,
            "overlap_tokens_with_previous": 0,
            "contains_suspicious_glyph": bool(suspicious_findings),
            "suspicious_text_findings": [
                finding.as_warning_fields() for finding in suspicious_findings
            ],
        },
    )


def _select_seed_chunks(spans: list[_TokenSpan], config: ChunkConfig) -> list[list[_TokenSpan]]:
    seed_chunks: list[list[_TokenSpan]] = []
    cursor = 0
    content_limit = max(1, config.max_tokens - config.overlap_tokens)

    while cursor < len(spans):
        remaining = len(spans) - cursor
        if remaining <= config.max_tokens and not seed_chunks:
            seed_chunks.append(spans[cursor:])
            break
        if remaining <= content_limit:
            seed_chunks.append(spans[cursor:])
            break

        min_end = min(len(spans) - 1, cursor + config.target_tokens - 1)
        max_end = min(len(spans) - 1, cursor + content_limit - 1)
        window = spans[cursor : max_end + 1]
        min_index = max(0, min_end - cursor)
        split_index = _last_sentence_boundary(window, min_index=min_index)
        if split_index is None:
            split_index = len(window) - 1

        end = cursor + split_index
        seed_chunks.append(spans[cursor : end + 1])
        cursor = end + 1

    if len(seed_chunks) >= 2:
        tail = seed_chunks[-1]
        previous = seed_chunks[-2]
        if len(tail) <= config.tiny_chunk_threshold and len(previous) + len(tail) <= content_limit:
            seed_chunks[-2] = previous + tail
            seed_chunks.pop()

    return seed_chunks


def chunk_document(
    document: Document,
    config: ChunkConfig | None = None,
) -> tuple[list[Chunk], list[dict[str, str]]]:
    if config is None:
        config = ChunkConfig()

    warnings: list[dict[str, str]] = []
    spans = _to_token_stream(document)
    if not spans:
        return [], [{"type": "no_text", "document_id": document.id}]

    chunks: list[Chunk] = []
    seed_chunks = _select_seed_chunks(spans, config)
    for chunk_index, seed in enumerate(seed_chunks):
        overlap = seed_chunks[chunk_index - 1][-config.overlap_tokens :] if chunk_index > 0 else []
        combined = overlap + seed
        if len(combined) > config.max_tokens:
            combined = combined[len(combined) - config.max_tokens :]

        chunk = _chunk_from_token_spans(chunk_index, document, combined)
        chunk.metadata["overlap_tokens_with_previous"] = len(overlap)
        chunks.append(chunk)

        if chunk.metadata.get("contains_unknown_encoding"):
            warnings.append(
                {
                    "type": "encoding_artifact",
                    "document_id": document.id,
                    "source_path": document.source.source_path,
                    "chunk_index": str(chunk.chunk_index),
                    "page_start": str(chunk.page_start),
                    "page_end": str(chunk.page_end),
                }
            )
        if chunk.metadata.get("contains_suspicious_glyph"):
            for finding in chunk.metadata.get("suspicious_text_findings", []):
                warnings.append({
                    "type": "suspicious_glyph_in_chunk",
                    "document_id": document.id,
                    "source_path": document.source.source_path,
                    "chunk_index": str(chunk.chunk_index),
                    "page_start": str(chunk.page_start),
                    "page_end": str(chunk.page_end),
                    **finding,
                })

    return chunks, warnings


def chunk_documents(
    documents: list[Document],
    config: ChunkConfig | None = None,
) -> tuple[list[Chunk], list[dict[str, str]]]:
    all_chunks: list[Chunk] = []
    warnings: list[dict[str, str]] = []
    for document in documents:
        doc_chunks, doc_warnings = chunk_document(document, config=config)
        all_chunks.extend(doc_chunks)
        warnings.extend(doc_warnings)
    return all_chunks, warnings


def write_chunks(chunks: list[Chunk], output_root: Path) -> list[Path]:
    output_path = output_root / "phase3" / "canonical" / "chunks.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(chunk.model_dump_json() for chunk in chunks) + "\n", encoding="utf-8")
    return [output_path]
