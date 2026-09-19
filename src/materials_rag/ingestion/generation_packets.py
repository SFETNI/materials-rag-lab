"""Phase 5C-A Vanilla RAG generation packet construction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from materials_rag.ingestion.markdown_parser import EXCLUDED_PATH_PREFIXES
from materials_rag.ingestion.utils import resolve_repo_root, sha256_for_file

TOP_K = 5
SYSTEM_PROMPT_VERSION = "vanilla_generator_v1"
PROMPT_PATH = Path("docs/generation/vanilla_generator_v1.md")
LEAKAGE_TERMS = [
    "gold_chunk_ids",
    "gold_evidence_ids",
    "hard_negative",
    "reference answer",
    "reference_answer",
    "answerability",
    "label_status",
    "evidence_requirements",
    "required_claims",
    "forbidden_claims",
    "numeric_checks",
    "intent_id",
    "split",
    "score:",
    "similarity score",
    "similarity_score",
    "source_path",
    "file_hash",
]
MANUAL_CASES = [
    {
        "case": 1,
        "case_type": "easy single-evidence factual question",
        "question_id": "SYNQ-003-A",
    },
    {
        "case": 2,
        "case_type": "B017-F03 multi-evidence question",
        "question_id": "SYNQ-001-A",
    },
    {
        "case": 3,
        "case_type": "exact identifier question",
        "question_id": "SYNQ-006-A",
    },
    {
        "case": 4,
        "case_type": "revision-sensitive current-answer question",
        "question_id": "SYNQ-012-A",
    },
    {
        "case": 5,
        "case_type": "historical revision question",
        "question_id": "SYNQ-013-A",
    },
    {
        "case": 6,
        "case_type": "runout/right-censoring question",
        "question_id": "SYNQ-002-A",
    },
    {
        "case": 7,
        "case_type": "multi-document comparison question",
        "question_id": "SYNQ-015-A",
    },
    {
        "case": 8,
        "case_type": "unsupported question",
        "question_id": "SYNQ-021-A",
    },
]


@dataclass
class Phase5CResult:
    packets: list[dict[str, Any]]
    manual_cases: list[dict[str, Any]]
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


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _source_for_context(source: dict[str, Any]) -> dict[str, Any]:
    hidden_fields = {"source_path", "file_hash"}
    return {
        key: value
        for key, value in source.items()
        if value is not None and key not in hidden_fields
    }


def _context_item(unit: dict[str, Any]) -> dict[str, Any]:
    metadata = unit.get("metadata", {})
    item = {
        "id": unit["id"],
        "kind": unit["kind"],
        "source": _source_for_context(unit["source"]),
        "retrieval_text": unit["retrieval_text"],
    }
    evidence_id = metadata.get("evidence_id")
    document_id = metadata.get("document_source_id") or metadata.get("document_id")
    if evidence_id:
        item["evidence_id"] = evidence_id
    if document_id:
        item["document_id"] = document_id
    return item


def format_generator_input(query: str, context_items: list[dict[str, Any]]) -> str:
    lines = ["QUESTION:", query, "", "CONTEXT:"]
    for item in context_items:
        lines.extend(
            [
                "",
                f"id: {item['id']}",
                f"kind: {item['kind']}",
            ]
        )
        if "evidence_id" in item:
            lines.append(f"evidence_id: {item['evidence_id']}")
        if "document_id" in item:
            lines.append(f"document_id: {item['document_id']}")
        lines.append("source:")
        for key, value in item["source"].items():
            lines.append(f"  {key}: {value}")
        lines.extend(["retrieval_text:", item["retrieval_text"]])
    return "\n".join(lines) + "\n"


def _packet_hash(generator_input: str) -> str:
    return _sha256_text(generator_input)


def _packet_for_question(
    question: dict[str, Any],
    qdrant_row: dict[str, Any],
    units_by_id: dict[str, dict[str, Any]],
    system_prompt_hash: str,
) -> dict[str, Any]:
    top_results = qdrant_row["results"][:TOP_K]
    retrieval_unit_ids = [row["retrieval_unit_id"] for row in top_results]
    context_items = [_context_item(units_by_id[unit_id]) for unit_id in retrieval_unit_ids]
    generator_input = format_generator_input(question["query"], context_items)
    return {
        "question_id": question["question_id"],
        "query": question["query"],
        "top_k": TOP_K,
        "retrieval_unit_ids": retrieval_unit_ids,
        "system_prompt_version": SYSTEM_PROMPT_VERSION,
        "system_prompt_hash": system_prompt_hash,
        "generation_packet_hash": _packet_hash(generator_input),
        "generator_input": generator_input,
    }


def _source_path_violations(packets: list[dict[str, Any]], units_by_id: dict[str, dict[str, Any]]) -> list[str]:
    violations: list[str] = []
    for packet in packets:
        for unit_id in packet["retrieval_unit_ids"]:
            source_path = units_by_id[unit_id].get("source", {}).get("source_path", "")
            if any(source_path.startswith(prefix) for prefix in EXCLUDED_PATH_PREFIXES):
                violations.append(f"{packet['question_id']}:{unit_id}:{source_path}")
    return violations


def validate_packets(packets: list[dict[str, Any]], units_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    leakage_hits: list[dict[str, str]] = []
    for packet in packets:
        lowered = packet["generator_input"].lower()
        for term in LEAKAGE_TERMS:
            if term.lower() in lowered:
                leakage_hits.append({"question_id": packet["question_id"], "term": term})
    source_violations = _source_path_violations(packets, units_by_id)
    return {
        "passed": not leakage_hits and not source_violations,
        "leakage_hits": leakage_hits,
        "excluded_source_path_violations": source_violations,
        "checked_terms": LEAKAGE_TERMS,
        "excluded_path_prefixes": EXCLUDED_PATH_PREFIXES,
    }


def build_generation_packets(repo_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    prompt_text = (repo_root / PROMPT_PATH).read_text(encoding="utf-8")
    prompt_hash = _sha256_text(prompt_text)

    questions_path = repo_root / "data/processed/phase4/retrieval_eval.jsonl"
    units_path = repo_root / "data/processed/phase4_5/retrieval_units.jsonl"
    qdrant_results_path = repo_root / "data/processed/phase5b/qdrant_results.jsonl"
    phase5b_manifest_path = repo_root / "data/processed/manifests/phase_5b_manifest.json"

    questions = _read_jsonl(questions_path)
    units = _read_jsonl(units_path)
    qdrant_results = _read_jsonl(qdrant_results_path)
    units_by_id = {unit["id"]: unit for unit in units}
    qdrant_by_question = {row["question_id"]: row for row in qdrant_results}

    packets = [
        _packet_for_question(question, qdrant_by_question[question["question_id"]], units_by_id, prompt_hash)
        for question in questions
    ]
    packet_by_question = {packet["question_id"]: packet for packet in packets}
    manual_cases = [
        {
            **case,
            "query": packet_by_question[case["question_id"]]["query"],
            "generation_packet_hash": packet_by_question[case["question_id"]][
                "generation_packet_hash"
            ],
            "retrieval_unit_ids": packet_by_question[case["question_id"]]["retrieval_unit_ids"],
        }
        for case in MANUAL_CASES
    ]
    validation = validate_packets(packets, units_by_id)
    manifest = {
        "phase": "5C-A",
        "version": "1.0.0",
        "objective": "Vanilla RAG context construction and isolated generation packets",
        "packet_count": len(packets),
        "top_k": TOP_K,
        "retrieval_backend": "Phase 5B Qdrant",
        "context_policy": {
            "preserve_phase5b_order": True,
            "reranking": False,
            "neighbor_chunks": False,
            "metadata_filters": False,
            "missing_gold_added": False,
        },
        "system_prompt": {
            "version": SYSTEM_PROMPT_VERSION,
            "path": PROMPT_PATH.as_posix(),
            "sha256": prompt_hash,
        },
        "representative_manual_case_ids": [case["question_id"] for case in manual_cases],
        "input_artifact_hashes": {
            path.relative_to(repo_root).as_posix(): sha256_for_file(path)
            for path in [
                questions_path,
                units_path,
                qdrant_results_path,
                phase5b_manifest_path,
                repo_root / PROMPT_PATH,
            ]
        },
        "leakage_validation": validation,
    }
    return packets, manual_cases, manifest


def run_phase5c(repo_root: Path | None = None) -> Phase5CResult:
    """Create Vanilla RAG generation packets without calling a generator."""

    if repo_root is None:
        repo_root = resolve_repo_root(Path(__file__))
    packets, manual_cases, manifest = build_generation_packets(repo_root)
    if not manifest["leakage_validation"]["passed"]:
        raise ValueError(f"Generation packet leakage validation failed: {manifest['leakage_validation']}")

    output_root = repo_root / "data" / "processed" / "phase5c"
    packets_path = output_root / "generation_packets.jsonl"
    manual_cases_path = output_root / "manual_cases.jsonl"
    manifest_path = repo_root / "data" / "processed" / "manifests" / "phase_5c_context_manifest.json"

    _write_jsonl(packets, packets_path)
    _write_jsonl(manual_cases, manual_cases_path)
    _write_json(manifest, manifest_path)
    return Phase5CResult(
        packets=packets,
        manual_cases=manual_cases,
        manifest=manifest,
        outputs=[packets_path, manual_cases_path, manifest_path],
        manifest_path=manifest_path,
    )
