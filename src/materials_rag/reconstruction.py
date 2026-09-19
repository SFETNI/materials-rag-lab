"""Rights-aware reconstruction of the 277-unit experimental retrieval corpus."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from materials_rag.ingestion.chunker import chunk_documents
from materials_rag.ingestion.parsers import _enrich_publication_document
from materials_rag.ingestion.pdf_parser import parse_pdf_documents
from materials_rag.schemas import SourceRef


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _compact_hash(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _reference_specs(project_root: Path) -> list[dict[str, Any]]:
    rows = _read_jsonl(project_root / "data/reconstruction/reference_only_sources.jsonl")
    return [row for row in rows if str(row.get("source_id", "")).startswith("reference_only:")]


def _resolve_source(source_root: Path, spec: dict[str, Any]) -> Path:
    expected = Path(spec["expected_local_path"])
    external_relative = Path(*expected.parts[3:]) if expected.parts[:3] == ("data", "raw", "external") else expected
    candidates = [
        source_root / expected,
        source_root / external_relative,
        source_root / spec["expected_filename"],
    ]
    matches = [path.resolve() for path in candidates if path.is_file()]
    if not matches:
        raise FileNotFoundError(
            f"Missing authorized source {spec['expected_filename']}; expected one of: "
            + ", ".join(str(path) for path in candidates)
        )
    path = matches[0]
    actual = _sha256(path)
    if actual != spec["sha256"]:
        raise ValueError(
            f"Source hash mismatch for {spec['expected_filename']}: "
            f"expected {spec['sha256']}, got {actual}"
        )
    return path


def _source_ref(spec: dict[str, Any]) -> SourceRef:
    expected = Path(spec["expected_local_path"])
    relative = Path(*expected.parts[3:]).as_posix()
    dataset = relative.split("/", 1)[0]
    return SourceRef(
        dataset=dataset,
        source_path=relative,
        source_dataset="publications",
        file_hash=spec["sha256"],
    )


def _missing_units(project_root: Path, source_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    specs = _reference_specs(project_root)
    inputs = [(_resolve_source(source_root, spec), _source_ref(spec)) for spec in specs]
    documents, parse_warnings = parse_pdf_documents(inputs)
    for document in documents:
        _enrich_publication_document(document)
    chunks, chunk_warnings = chunk_documents(documents)
    units: list[dict[str, Any]] = []
    for chunk in chunks:
        record = chunk.model_dump(mode="json")
        source = {key: value for key, value in record["source"].items() if value is not None}
        metadata = dict(record["metadata"])
        metadata.update(
            {
                "retrieval_unit_kind": "chunk_public",
                "canonical_kind": record["kind"],
                "input_artifact": "data/processed/phase3_5a/canonical/chunks.jsonl",
            }
        )
        units.append(
            {
                "id": record["id"],
                "kind": "chunk_public",
                "retrieval_text": record["retrieval_text"],
                "source": source,
                "metadata": metadata,
            }
        )
    expected_ids = [unit_id for spec in specs for unit_id in spec["retrieval_unit_ids_after_ingestion"]]
    actual_ids = [unit["id"] for unit in units]
    if len(actual_ids) != len(expected_ids) or set(actual_ids) != set(expected_ids):
        raise ValueError(
            "Reference-only parsing did not reproduce the declared 25 retrieval-unit IDs"
        )
    return units, [*parse_warnings, *chunk_warnings]


def reconstruct_full_corpus(
    project_root: Path,
    source_root: Path,
    output_root: Path,
    *,
    check_only: bool = False,
) -> dict[str, Any]:
    """Combine verified reference-only chunks with the installed 252-unit projection."""

    project_root = project_root.resolve()
    source_root = source_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    public_path = project_root / "data/runtime/dataset/canonical/retrieval_units.jsonl"
    if not public_path.exists():
        raise FileNotFoundError(
            "Install the 252-unit companion dataset before full-corpus reconstruction"
        )
    public_lines = [line for line in public_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    public_units = [json.loads(line) for line in public_lines]
    if len(public_units) != 252:
        raise ValueError(f"Expected installed 252-unit projection, found {len(public_units)}")

    missing_units, warnings = _missing_units(project_root, source_root)
    public_chunks = [unit for unit in public_units if unit["kind"] == "chunk_public"]
    combined_public = [*public_chunks, *missing_units]
    combined_public.sort(
        key=lambda unit: (
            Path(unit["source"]["source_path"]).name,
            int(unit.get("metadata", {}).get("chunk_index", 0)),
        )
    )
    combined = [
        *combined_public,
        *[unit for unit in public_units if unit["kind"] != "chunk_public"],
    ]
    ids = [unit["id"] for unit in combined]
    if len(combined) != 277 or len(ids) != len(set(ids)):
        raise ValueError("Reconstructed corpus must contain exactly 277 unique retrieval units")

    identity = json.loads(
        (project_root / "data/reconstruction/full_corpus_identity.json").read_text(encoding="utf-8")
    )
    ids_hash = _compact_hash(ids)
    id_text_hash = _compact_hash(
        [{"id": unit["id"], "retrieval_text": unit["retrieval_text"]} for unit in combined]
    )
    if ids_hash != identity["ordered_ids_sha256"]:
        raise ValueError(f"Ordered retrieval-unit identity mismatch: {ids_hash}")
    if id_text_hash != identity["ordered_id_and_retrieval_text_sha256"]:
        raise ValueError(f"Ordered retrieval text identity mismatch: {id_text_hash}")

    payload = "".join(
        json.dumps(unit, ensure_ascii=False, sort_keys=True) + "\n" for unit in combined
    )
    output_path = output_root / "retrieval_units.jsonl"
    report = {
        "outcome": "PASS",
        "check_only": check_only,
        "redistributable_units": len(public_units),
        "reference_only_units_generated": len(missing_units),
        "retrieval_unit_count": len(combined),
        "ordered_ids_sha256": ids_hash,
        "ordered_id_and_retrieval_text_sha256": id_text_hash,
        "retrieval_identity_matches_frozen": True,
        "frozen_jsonl_sha256": identity["frozen_jsonl_sha256"],
        "release_normalized_jsonl_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "frozen_byte_hash_expected": False,
        "warning_count": len(warnings),
        "output": str(output_path),
        "source_files_retained": False,
    }
    if check_only:
        return report
    output_root.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        existing_count = len(_read_jsonl(output_path))
        if existing_count not in {252, 277}:
            raise ValueError(f"Refusing to replace unrecognized runtime corpus: {output_path}")
        if existing_count == 277 and _sha256(output_path) != report["release_normalized_jsonl_sha256"]:
            raise ValueError(f"Refusing to overwrite a different reconstructed corpus: {output_path}")
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="\n",
        dir=output_root,
        delete=False,
    ) as stream:
        stream.write(payload)
        temporary = Path(stream.name)
    temporary.replace(output_path)
    report_path = output_root / "reconstruction_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


__all__ = ["reconstruct_full_corpus"]
