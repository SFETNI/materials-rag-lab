"""Show the public data-preparation rung on one safe synthetic record."""

from __future__ import annotations

import json

from materials_rag.ingestion.normalization import normalize_specimen_identifier
from materials_rag.schemas import Chunk, Document, SourceRef, infer_chunk_id, infer_document_id


def main() -> None:
    raw = {
        "source_path": "samples/fatigue_note.md",
        "title": "Synthetic fatigue note",
        "specimen": "IN718_30um_6p2_test2.csv",
        "text": "Specimen 6p2 test2 reached runout; retain the stated stress definition.",
    }
    source = SourceRef(
        dataset="public_quick_demo",
        source_path=raw["source_path"],
        authority="project_authored_sample",
    )
    document_id = infer_document_id(source)
    normalized = normalize_specimen_identifier(raw["specimen"])
    document = Document(
        id=document_id,
        title=raw["title"],
        source=source,
        retrieval_text=f"Document: {raw['title']}\n{raw['text']}",
        metadata={"specimen_id": normalized.canonical_specimen_id},
    )
    chunk = Chunk(
        id=infer_chunk_id(document_id, 0),
        document_id=document_id,
        chunk_index=0,
        text=raw["text"],
        token_count=len(raw["text"].split()),
        retrieval_text=(
            f"Document: {raw['title']}\n"
            f"Specimen: {normalized.canonical_specimen_id}\n"
            f"Run: {normalized.run_id}\n{raw['text']}"
        ),
        source=source,
        metadata={"raw_specimen_id": raw["specimen"]},
    )
    output = {
        "raw_input": raw,
        "normalized_identifier": normalized.__dict__,
        "canonical_document": document.model_dump(mode="json"),
        "knowledge_unit": chunk.model_dump(mode="json"),
        "provenance": source.model_dump(mode="json"),
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
