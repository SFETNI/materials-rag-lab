"""Phase 4 retrieval-evaluation dataset compilation."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from materials_rag.ingestion.markdown_parser import EXCLUDED_PATH_PREFIXES
from materials_rag.ingestion.utils import resolve_repo_root, sha256_for_file
from materials_rag.schemas import Chunk


@dataclass
class Phase4Result:
    records: list[dict[str, Any]]
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


def _unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _evidence_ids_from_requirements(requirements: list[dict[str, Any]]) -> list[str]:
    evidence_ids: list[str] = []
    for group in requirements:
        evidence_ids.extend(str(evidence_id) for evidence_id in group.get("any_of", []))
    return _unique_preserve_order(evidence_ids)


def _chunk_ids_for_evidence(
    evidence_ids: list[str],
    evidence_to_chunks: dict[str, list[str]],
) -> list[str]:
    chunk_ids: list[str] = []
    for evidence_id in evidence_ids:
        chunk_ids.extend(evidence_to_chunks.get(evidence_id, []))
    return _unique_preserve_order(chunk_ids)


def _load_chunks(path: Path) -> dict[str, Chunk]:
    chunks: dict[str, Chunk] = {}
    for row in _read_jsonl(path):
        chunk = Chunk.model_validate(row)
        chunks[chunk.id] = chunk
    return chunks


def compile_retrieval_eval(repo_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    questions_path = repo_root / "data" / "eval" / "phase35b" / "questions.jsonl"
    gold_path = repo_root / "data" / "eval" / "phase35b" / "gold.jsonl"
    hard_negatives_path = repo_root / "data" / "eval" / "phase35b" / "hard_negatives.jsonl"
    evidence_map_path = repo_root / "data" / "processed" / "phase3_5c" / "evidence_chunk_map.jsonl"
    chunks_path = repo_root / "data" / "processed" / "phase3_5c" / "canonical" / "chunks.jsonl"
    evidence_registry_path = repo_root / "authoring" / "phase35b" / "evidence_registry.jsonl"

    questions = _read_jsonl(questions_path)
    gold_rows = _read_jsonl(gold_path)
    hard_negative_rows = _read_jsonl(hard_negatives_path)
    evidence_map_rows = _read_jsonl(evidence_map_path)
    evidence_registry_rows = _read_jsonl(evidence_registry_path)
    chunks = _load_chunks(chunks_path)

    question_by_id = {row["question_id"]: row for row in questions}
    gold_by_question = {row["question_id"]: row for row in gold_rows}
    if len(question_by_id) != len(questions):
        raise ValueError("Question IDs must be unique.")
    if set(question_by_id) != set(gold_by_question):
        raise ValueError("Question and gold files must contain the same question IDs.")

    evidence_to_chunks = {
        row["evidence_id"]: list(row.get("chunk_ids", [])) for row in evidence_map_rows
    }
    evidence_registry_ids = {row["evidence_id"] for row in evidence_registry_rows}
    hard_negatives_by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in hard_negative_rows:
        hard_negatives_by_question[row["question_id"]].append(row)

    unresolved_evidence: set[str] = set()
    unresolved_hard_negatives: set[str] = set()
    records: list[dict[str, Any]] = []

    for question_id in sorted(question_by_id):
        question = question_by_id[question_id]
        gold = gold_by_question[question_id]
        evidence_requirements = list(gold.get("evidence_requirements", []))
        gold_evidence_ids = _evidence_ids_from_requirements(evidence_requirements)
        for evidence_id in gold_evidence_ids:
            if evidence_id not in evidence_registry_ids or not evidence_to_chunks.get(evidence_id):
                unresolved_evidence.add(evidence_id)
        gold_chunk_ids = _chunk_ids_for_evidence(gold_evidence_ids, evidence_to_chunks)

        hard_rows = hard_negatives_by_question.get(question_id, [])
        hard_negative_evidence_ids = _unique_preserve_order(
            [str(row["candidate_evidence_id"]) for row in hard_rows]
        )
        for evidence_id in hard_negative_evidence_ids:
            if evidence_id not in evidence_registry_ids or not evidence_to_chunks.get(evidence_id):
                unresolved_hard_negatives.add(evidence_id)
        hard_negative_chunk_ids = _chunk_ids_for_evidence(
            hard_negative_evidence_ids,
            evidence_to_chunks,
        )
        overlap = set(gold_chunk_ids) & set(hard_negative_chunk_ids)
        if overlap:
            raise ValueError(f"Hard-negative chunks overlap gold chunks for {question_id}: {overlap}")

        records.append(
            {
                "question_id": question_id,
                "intent_id": question.get("intent_id"),
                "query": question.get("question"),
                "split": question.get("split"),
                "task_family": question.get("task_family"),
                "as_of": question.get("as_of"),
                "corpus_id": question.get("corpus_id"),
                "study_id": question.get("study_id"),
                "synthetic": question.get("synthetic"),
                "answerability": gold.get("response_mode"),
                "response_mode": gold.get("response_mode"),
                "label_status": gold.get("label_status"),
                "evidence_requirements": evidence_requirements,
                "gold_evidence_ids": gold_evidence_ids,
                "gold_chunk_ids": gold_chunk_ids,
                "acceptable_supporting_chunk_ids": [
                    _chunk_ids_for_evidence(list(group.get("any_of", [])), evidence_to_chunks)
                    for group in evidence_requirements
                ],
                "hard_negative_evidence_ids": hard_negative_evidence_ids,
                "hard_negative_chunk_ids": hard_negative_chunk_ids,
                "hard_negatives": hard_rows,
                "required_claims": gold.get("required_claims", []),
                "forbidden_claims": gold.get("forbidden_claims", []),
                "numeric_checks": gold.get("numeric_checks", []),
            }
        )

    label_source_paths = [
        questions_path,
        gold_path,
        hard_negatives_path,
        evidence_map_path,
        chunks_path,
        evidence_registry_path,
    ]
    chunk_source_paths = {chunk.source.source_path for chunk in chunks.values()}
    leaked_sources = sorted(
        source for source in chunk_source_paths if source.startswith(tuple(EXCLUDED_PATH_PREFIXES))
    )
    if leaked_sources:
        raise ValueError(f"Retrievable chunks include excluded paths: {leaked_sources}")

    split_counts = Counter(str(row["split"]) for row in records)
    answerability_counts = Counter(str(row["answerability"]) for row in records)
    intent_splits: dict[str, set[str]] = defaultdict(set)
    for row in records:
        intent_splits[str(row["intent_id"])].add(str(row["split"]))
    split_violations = {
        intent_id: sorted(splits)
        for intent_id, splits in intent_splits.items()
        if len(splits) > 1
    }
    if split_violations:
        raise ValueError(f"Intent split preservation failed: {split_violations}")

    gold_evidence_sets = [row["gold_evidence_ids"] for row in records]
    manifest = {
        "phase": "4",
        "version": "1.0.0",
        "total_questions": len(records),
        "total_intents": len(intent_splits),
        "split_counts": dict(sorted(split_counts.items())),
        "answerability_counts": dict(sorted(answerability_counts.items())),
        "answerable_count": sum(1 for row in records if row["answerability"] != "abstain"),
        "unsupported_unanswerable_count": sum(
            1 for row in records if row["answerability"] == "abstain"
        ),
        "questions_with_1_gold_evidence_section": sum(
            1 for evidence_ids in gold_evidence_sets if len(evidence_ids) == 1
        ),
        "questions_with_multiple_gold_evidence_sections": sum(
            1 for evidence_ids in gold_evidence_sets if len(evidence_ids) > 1
        ),
        "total_unique_gold_evidence_ids": len(
            {evidence_id for row in records for evidence_id in row["gold_evidence_ids"]}
        ),
        "total_unique_gold_chunk_ids": len(
            {chunk_id for row in records for chunk_id in row["gold_chunk_ids"]}
        ),
        "hard_negative_pair_count": len(hard_negative_rows),
        "unresolved_evidence_references": sorted(unresolved_evidence),
        "unresolved_hard_negative_references": sorted(unresolved_hard_negatives),
        "source_hashes": {
            path.relative_to(repo_root).as_posix(): sha256_for_file(path)
            for path in label_source_paths
        },
        "retrievable_chunk_count": len(chunks),
        "label_leakage_check": {
            "excluded_paths": EXCLUDED_PATH_PREFIXES,
            "leaked_sources": leaked_sources,
            "passed": not leaked_sources,
        },
    }
    return records, manifest


def run_phase4(repo_root: Path | None = None, output_root: Path | None = None) -> Phase4Result:
    """Compile the deterministic Phase 4 retrieval-evaluation dataset."""

    if repo_root is None:
        repo_root = resolve_repo_root(Path(__file__))
    if output_root is None:
        output_root = repo_root / "data" / "processed"

    records, manifest = compile_retrieval_eval(repo_root)
    output_path = output_root / "phase4" / "retrieval_eval.jsonl"
    manifest_path = output_root / "manifests" / "phase_4_manifest.json"
    _write_jsonl(records, output_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return Phase4Result(
        records=records,
        manifest=manifest,
        outputs=[output_path, manifest_path],
        manifest_path=manifest_path,
    )
