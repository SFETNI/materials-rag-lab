"""Phase 4.5 unified retrieval-corpus assembly."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from materials_rag.ingestion.markdown_parser import EXCLUDED_PATH_PREFIXES
from materials_rag.ingestion.utils import resolve_repo_root, sha256_for_file

WORD_RE = re.compile(r"\w+")


@dataclass
class Phase45Result:
    units: list[dict[str, Any]]
    manifest: dict[str, Any]
    outputs: list[Path]
    manifest_path: Path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def _source_dataset(source: dict[str, Any]) -> str:
    return str(source.get("dataset") or source.get("source_dataset") or "unknown")


def _quality_warning(unit: dict[str, Any]) -> dict[str, Any] | None:
    text = str(unit.get("retrieval_text") or "")
    words = _word_count(text)
    if not text.strip():
        return {"id": unit.get("id"), "type": "empty_retrieval_text"}
    if words < 5:
        return {"id": unit.get("id"), "type": "very_short_retrieval_text", "word_count": words}
    if text.count("=") >= 3 and text.count("|") >= 3:
        return {"id": unit.get("id"), "type": "machine_like_retrieval_text", "word_count": words}
    return None


def _project_test_retrieval_text(record: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    original = str(record.get("retrieval_text") or "")
    warning = _quality_warning({"id": record.get("id"), "retrieval_text": original})
    if not warning:
        return original, None

    stats = record.get("statistics") if isinstance(record.get("statistics"), dict) else {}
    load_stats = stats.get("Load (N)", {}) if isinstance(stats.get("Load (N)"), dict) else {}
    stress_stats = stats.get("Stress (MPa)", {}) if isinstance(stats.get("Stress (MPa)"), dict) else {}
    parts = [
        f"Fatigue test log for specimen {record.get('specimen_id')}.",
        f"Raw identifier: {record.get('specimen_id_raw')}.",
        f"Run: {record.get('run_id') or 'standard'}; file variant: {record.get('file_variant')}.",
        f"CSV artifact: {record.get('csv_artifact_path')}.",
    ]
    if record.get("test_metadata_path"):
        parts.append(f"Paired test metadata: {record.get('test_metadata_path')}.")
    if record.get("cycle_count") is not None:
        parts.append(f"Recorded rows or cycles: {record.get('cycle_count')}.")
    if load_stats:
        parts.append(
            "Load channel statistics in newtons: "
            f"mean {load_stats.get('mean')}, min {load_stats.get('min')}, max {load_stats.get('max')}."
        )
    if stress_stats:
        parts.append(
            "Stress channel statistics in MPa: "
            f"mean {stress_stats.get('mean')}, min {stress_stats.get('min')}, max {stress_stats.get('max')}."
        )
    if record.get("experiment_record_id"):
        parts.append(f"Linked experiment record: {record.get('experiment_record_id')}.")
    parts.append(f"Link status: {record.get('link_status')}.")
    warning["projection_applied"] = True
    return " ".join(parts), warning


def _make_unit(record: dict[str, Any], kind: str, input_path: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
    retrieval_text = str(record.get("retrieval_text") or "")
    warning: dict[str, Any] | None = None
    if kind == "test":
        retrieval_text, warning = _project_test_retrieval_text(record)
    else:
        warning = _quality_warning({"id": record.get("id"), "retrieval_text": retrieval_text})

    source = record.get("source")
    if not isinstance(source, dict) or not source.get("source_path"):
        raise ValueError(f"Missing source provenance for {record.get('id')}")

    metadata = dict(record.get("metadata") or {})
    metadata.update(
        {
            "retrieval_unit_kind": kind,
            "canonical_kind": record.get("kind"),
            "input_artifact": input_path,
        }
    )
    if kind == "test" and retrieval_text != record.get("retrieval_text"):
        metadata["canonical_retrieval_text"] = record.get("retrieval_text")

    return (
        {
            "id": record["id"],
            "kind": kind,
            "retrieval_text": retrieval_text,
            "source": source,
            "metadata": metadata,
        },
        warning,
    )


def _load_units_from_jsonl(path: Path, kind: str, repo_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    units: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    input_path = path.relative_to(repo_root).as_posix()
    for record in _read_jsonl(path):
        unit, warning = _make_unit(record, kind, input_path)
        units.append(unit)
        if warning:
            warnings.append(warning)
    return units, warnings


def assemble_retrieval_corpus(repo_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    inputs = {
        "chunk_public": repo_root / "data/processed/phase3_5a/canonical/chunks.jsonl",
        "chunk_internal": repo_root / "data/processed/phase3_5c/canonical/chunks.jsonl",
        "experiment": repo_root / "data/processed/phase2a/canonical/experiment_records.jsonl",
        "process": repo_root / "data/processed/phase2a/canonical/process_records.jsonl",
        "analysis": repo_root / "data/processed/phase2a/canonical/analysis_records.jsonl",
        "test": repo_root / "data/processed/phase2b/canonical/test_records.jsonl",
    }
    all_units: list[dict[str, Any]] = []
    quality_warnings: list[dict[str, Any]] = []
    for kind, path in inputs.items():
        units, warnings = _load_units_from_jsonl(path, kind, repo_root)
        all_units.extend(units)
        quality_warnings.extend(warnings)

    ids = [unit["id"] for unit in all_units]
    duplicate_ids = sorted([unit_id for unit_id, count in Counter(ids).items() if count > 1])
    if duplicate_ids:
        raise ValueError(f"Duplicate retrieval unit IDs: {duplicate_ids[:5]}")

    excluded_sources = sorted(
        {
            unit["source"]["source_path"]
            for unit in all_units
            if str(unit["source"]["source_path"]).startswith(tuple(EXCLUDED_PATH_PREFIXES))
        }
    )
    if excluded_sources:
        raise ValueError(f"Excluded source paths entered retrieval corpus: {excluded_sources}")

    phase4_rows = _read_jsonl(repo_root / "data/processed/phase4/retrieval_eval.jsonl")
    unit_ids = set(ids)
    missing_gold = sorted(
        {
            chunk_id
            for row in phase4_rows
            for chunk_id in row.get("gold_chunk_ids", [])
            if chunk_id not in unit_ids
        }
    )
    missing_hard_negatives = sorted(
        {
            chunk_id
            for row in phase4_rows
            for chunk_id in row.get("hard_negative_chunk_ids", [])
            if chunk_id not in unit_ids
        }
    )
    if missing_gold or missing_hard_negatives:
        raise ValueError(
            "Phase 4 benchmark chunks missing from retrieval corpus: "
            f"gold={missing_gold[:5]}, hard_negatives={missing_hard_negatives[:5]}"
        )

    counts_by_kind = Counter(unit["kind"] for unit in all_units)
    counts_by_dataset = Counter(_source_dataset(unit["source"]) for unit in all_units)
    input_hashes = {
        path.relative_to(repo_root).as_posix(): sha256_for_file(path)
        for path in inputs.values()
    }
    input_hashes["data/processed/phase4/retrieval_eval.jsonl"] = sha256_for_file(
        repo_root / "data/processed/phase4/retrieval_eval.jsonl"
    )

    empty_count = sum(1 for unit in all_units if not str(unit["retrieval_text"]).strip())
    manifest = {
        "phase": "4.5",
        "version": "1.0.0",
        "total_retrieval_units": len(all_units),
        "counts_by_kind": dict(sorted(counts_by_kind.items())),
        "counts_by_source_dataset": dict(sorted(counts_by_dataset.items())),
        "public_chunk_count": counts_by_kind.get("chunk_public", 0),
        "internal_synthetic_chunk_count": counts_by_kind.get("chunk_internal", 0),
        "experiment_record_count": counts_by_kind.get("experiment", 0),
        "process_record_count": counts_by_kind.get("process", 0),
        "analysis_record_count": counts_by_kind.get("analysis", 0),
        "test_record_count": counts_by_kind.get("test", 0),
        "duplicate_id_count": len(duplicate_ids),
        "duplicate_ids": duplicate_ids,
        "empty_retrieval_text_count": empty_count,
        "retrieval_text_quality_warnings": quality_warnings,
        "excluded_path_validation": {
            "excluded_prefixes": EXCLUDED_PATH_PREFIXES,
            "violations": excluded_sources,
            "passed": not excluded_sources,
        },
        "phase4_benchmark_compatibility": {
            "missing_gold_chunk_ids": missing_gold,
            "missing_hard_negative_chunk_ids": missing_hard_negatives,
            "passed": not missing_gold and not missing_hard_negatives,
            "candidate_corpus_is_not_gold_corpus": True,
        },
        "source_hashes": input_hashes,
    }
    return all_units, manifest


def run_phase45(repo_root: Path | None = None, output_root: Path | None = None) -> Phase45Result:
    """Assemble the deterministic Phase 4.5 unified retrieval corpus."""

    if repo_root is None:
        repo_root = resolve_repo_root(Path(__file__))
    if output_root is None:
        output_root = repo_root / "data" / "processed"
    units, manifest = assemble_retrieval_corpus(repo_root)
    output_path = output_root / "phase4_5" / "retrieval_units.jsonl"
    manifest_path = output_root / "manifests" / "phase_4_5_manifest.json"
    _write_jsonl(units, output_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return Phase45Result(
        units=units,
        manifest=manifest,
        outputs=[output_path, manifest_path],
        manifest_path=manifest_path,
    )
