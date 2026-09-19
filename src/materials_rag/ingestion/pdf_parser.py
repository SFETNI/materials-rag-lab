"""PDF parser for generating Document objects with raw and normalized page text."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from pathlib import Path

import pymupdf

from materials_rag.ingestion.text_quality import find_suspicious_text
from materials_rag.schemas import Document, SourceRef, make_human_readable_id


def normalize_pdf_page_text(raw: str) -> str:
    """Normalize raw page text conservatively for deterministic downstream use."""

    normalized = unicodedata.normalize("NFC", raw)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\x00", "")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _warn_if_compatibility_transforms(
    raw: str,
    page_no: int,
    document_id: str,
    source_path: str,
    source_dataset: str,
) -> list[dict[str, str]]:
    warnings: list[dict[str, str]] = []
    nfc_text = unicodedata.normalize("NFC", raw)
    nfkc_text = unicodedata.normalize("NFKC", raw)
    suspicious_findings = find_suspicious_text(raw)

    if nfc_text != nfkc_text and suspicious_findings:
        warnings.append(
            {
                "type": "compatibility_normalization_difference",
                "document_id": document_id,
                "source_path": source_path,
                "page": str(page_no),
                "source_dataset": source_dataset,
                "category": "compatibility_normalization_difference",
                "nfc_sample": normalize_pdf_page_text(nfc_text)[:300],
                "nfkc_sample": normalize_pdf_page_text(nfkc_text)[:300],
            }
        )

    for finding in suspicious_findings:
        warning = {
            "type": finding.category,
            "document_id": document_id,
            "source_path": source_path,
            "page": str(page_no),
            "source_dataset": source_dataset,
        }
        warning.update(finding.as_warning_fields())
        warnings.append(warning)

    return warnings


def parse_pdf_documents(
    pdf_paths: Iterable[tuple[Path, SourceRef]],
) -> tuple[list[Document], list[dict[str, str]]]:
    """Parse PDFs into Document objects without chunking."""

    documents: list[Document] = []
    warnings: list[dict[str, str]] = []

    for path, source in sorted(pdf_paths, key=lambda item: item[0].name):
        if not path.exists():
            warnings.append({"type": "missing_file", "path": path.as_posix()})
            continue

        doc = pymupdf.open(path)
        pdf_metadata = dict(doc.metadata or {})
        document_id = make_human_readable_id("document", source.dataset, path.stem)
        raw_pages: list[dict[str, str]] = []
        norm_pages: list[dict[str, str]] = []
        for page_no in range(doc.page_count):
            page = doc[page_no]
            raw = page.get_text("text") or ""
            norm = normalize_pdf_page_text(raw)
            raw_pages.append({"page": page_no + 1, "text": raw})
            norm_pages.append({"page": page_no + 1, "text": norm})
            warnings.extend(
                _warn_if_compatibility_transforms(
                    raw,
                    page_no=page_no + 1,
                    document_id=document_id,
                    source_path=source.source_path,
                    source_dataset=source.dataset,
                )
            )

        page_count_text = f"Document with {doc.page_count} pages."
        if raw_pages:
            first_norm = normalize_pdf_page_text(raw_pages[0]["text"])
            retrieval_text = f"{path.name}: {first_norm[:500]}" if first_norm else page_count_text
        else:
            retrieval_text = page_count_text

        doc_obj = Document(
            id=document_id,
            title=path.name,
            retrieval_text=retrieval_text,
            source=source,
            language="en",
            metadata={
                "page_count": doc.page_count,
                "page_text_raw": raw_pages,
                "page_text_normalized": norm_pages,
                "pdf_metadata": pdf_metadata,
                "pdf_metadata_title": str(pdf_metadata.get("title") or "").strip(),
                "source_path": str(path),
            },
            document_type="pdf",
        )
        documents.append(doc_obj)
        doc.close()

        if not raw_pages:
            warnings.append({"type": "empty_pdf", "path": str(path)})

    return documents, warnings
