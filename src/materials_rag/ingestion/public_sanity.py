"""Phase 5C-B public-corpus RAG generation sanity packets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sentence_transformers import SentenceTransformer

from materials_rag.ingestion.dense_retrieval import MODEL_NAME, QUERY_PREFIX, _encode_texts
from materials_rag.ingestion.generation_packets import (
    PROMPT_PATH,
    SYSTEM_PROMPT_VERSION,
    TOP_K,
    _context_item,
    _packet_hash,
    _sha256_text,
    _write_jsonl,
    format_generator_input,
    validate_packets,
)
from materials_rag.ingestion.qdrant_retrieval import (
    COLLECTION_NAME,
    open_qdrant,
    qdrant_search,
    run_phase5b,
)
from materials_rag.ingestion.utils import resolve_repo_root, sha256_for_file

PUBLIC_SANITY_QUESTIONS = [
    {
        "case": 1,
        "question_id": "PUBSAN-001",
        "query": "What stress ratio was used for the reported NASA high-cycle fatigue tests comparing as-built and low-stress-ground IN718 specimens?",
        "source_type": "NASA PDF",
        "expected_retrieval_unit_ids": ["chunk|document_nasa_20160006998|2"],
        "reference_facts": [
            "The NASA surface-finish HCF tests were conducted at stress ratio R = 0.12.",
            "The same passage defines R as minimum stress divided by maximum stress.",
        ],
        "notes": "Grounded in the NASA 20160006998 surface-finish HCF report.",
    },
    {
        "case": 2,
        "question_id": "PUBSAN-002",
        "query": "Which surface conditions were compared in the NASA high-cycle fatigue surface-finish study?",
        "source_type": "NASA PDF",
        "expected_retrieval_unit_ids": ["chunk|document_nasa_20160006998|2"],
        "reference_facts": [
            "The study compared as-built (AB) and low-stress-ground (LSG) surface conditions.",
            "The LSG specimens were machined to final dimensions to meet a surface-finish requirement.",
        ],
        "notes": "Uses the HCF specimen and surface-condition description.",
    },
    {
        "case": 3,
        "question_id": "PUBSAN-003",
        "query": "How did the NASA report describe surface-processed high-cycle fatigue specimens compared with as-built behavior and published IN718 references?",
        "source_type": "NASA PDF",
        "expected_retrieval_unit_ids": ["chunk|document_nasa_20150016245|4"],
        "reference_facts": [
            "As-built HCF results were below the MMPDS-08 reference curves.",
            "Some low-stress-ground or bead-blasted surface-processed specimens performed as well as or better than some published references, depending on vendor and heat treatment.",
        ],
        "notes": "Grounded in the NASA IN718 properties report discussion of Figures 13 and 14.",
    },
    {
        "case": 4,
        "question_id": "PUBSAN-004",
        "query": "What did the NASA report observe about the 0.045 mm build-layer thickness as laser power increased?",
        "source_type": "NASA PDF",
        "expected_retrieval_unit_ids": ["chunk|document_nasa_20150016245|2"],
        "reference_facts": [
            "For 0.045 mm layer thickness, ultimate tensile strength increased as input power increased.",
            "Gauge elongation also increased, and the 0.045 mm layer thickness appeared to converge toward the 0.030 mm layer-thickness values as laser power increased.",
        ],
        "notes": "Grounded in the build-parameter characterization section.",
    },
    {
        "case": 5,
        "question_id": "PUBSAN-005",
        "query": "What pressure, temperature, hold time, pressure medium, and cooling type were recorded for the public NIST IN718 HIP cycle?",
        "source_type": "NIST HIP record plus NIST publication",
        "expected_retrieval_unit_ids": [
            "process_record|nist_in718|HIP_cycles|cycle_1",
            "chunk|document_nist_in718_Kafka_2023_contour_fatigue|6",
        ],
        "reference_facts": [
            "The canonical NIST HIP record gives 100 MPa, 1150 C, 240 min, and Argon 99,995%.",
            "The Kafka contour-fatigue publication describes HIP as 1150 C for 4 h at 100 MPa in Ar, furnace cool to room temperature.",
        ],
        "notes": "Uses the canonical process record for medium and the publication chunk for furnace cooling.",
    },
    {
        "case": 6,
        "question_id": "PUBSAN-006",
        "query": "What polishing or surface-finish requirement was specified for the Kafka rotating-bending specimen drawing?",
        "source_type": "Kafka drawing",
        "expected_retrieval_unit_ids": ["chunk|document_nist_in718_Kafka_AM_cylinder_to_RBF|0"],
        "reference_facts": [
            "The drawing says to follow ISO 1143:2010 polishing steps.",
            "The final 0.025 mm is removed by longitudinal polishing to a maximum surface roughness of 0.2 micrometer Ra, with no circumferential machining visible at about 20x magnification.",
        ],
        "notes": "Grounded in the Kafka AM cylinder-to-RBF drawing.",
    },
    {
        "case": 7,
        "question_id": "PUBSAN-007",
        "query": "Identify one NIST recorded runout and distinguish it from an observed fatigue failure.",
        "source_type": "NIST fatigue publication and NIST fatigue record",
        "expected_retrieval_unit_ids": [
            "chunk|document_nist_in718_Kafka_2023_contour_fatigue|7",
            "experiment_record|nist_in718|RBF_Results_30um|6.2|21",
        ],
        "reference_facts": [
            "The publication defines runout as reaching 1e7 cycles or more rather than satisfying the load-drop failure condition.",
            "Specimen 6.2 is reported with 304,730,636 cycles and was stopped, making it a runout example rather than an observed fracture failure.",
        ],
        "notes": "Combines the publication definition with the canonical fatigue result row.",
    },
    {
        "case": 8,
        "question_id": "PUBSAN-008",
        "query": "What did the NIST contour-pass fatigue publication conclude about contour passes, surface roughness, and fatigue life?",
        "source_type": "NIST contour/fatigue publication",
        "expected_retrieval_unit_ids": ["chunk|document_nist_in718_Kafka_2023_contour_fatigue|16"],
        "reference_facts": [
            "Rough surfaces decreased fatigue life and fatigue performance compared with polished surfaces.",
            "No statistically significant fatigue-life differences were identified among the different contour-pass surface conditions.",
            "The authors noted possible lower-confidence differences involving the 0-contour condition above 1e6 cycles and that adding a second contour pass may have marginal gain over one contour pass.",
        ],
        "notes": "Grounded in the conclusion section of the Kafka contour-pass fatigue publication.",
    },
]


@dataclass
class PublicSanityResult:
    questions: list[dict[str, Any]]
    references: list[dict[str, Any]]
    vanilla_packets: list[dict[str, Any]]
    source_packets: list[dict[str, Any]]
    inspection_rows: list[dict[str, Any]]
    outputs: list[Path]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _packet(
    question: dict[str, Any],
    context_unit_ids: list[str],
    units_by_id: dict[str, dict[str, Any]],
    prompt_hash: str,
) -> dict[str, Any]:
    context_items = [_context_item(units_by_id[unit_id]) for unit_id in context_unit_ids]
    generator_input = format_generator_input(question["query"], context_items)
    return {
        "question_id": question["question_id"],
        "query": question["query"],
        "context_unit_ids": context_unit_ids,
        "system_prompt_version": SYSTEM_PROMPT_VERSION,
        "system_prompt_hash": prompt_hash,
        "generation_packet_hash": _packet_hash(generator_input),
        "generator_input": generator_input,
    }


def _question_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "case": row["case"],
        "question_id": row["question_id"],
        "query": row["query"],
        "public_synthetic": "public",
    }


def _reference_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "case": row["case"],
        "question_id": row["question_id"],
        "query": row["query"],
        "expected_retrieval_unit_ids": row["expected_retrieval_unit_ids"],
        "reference_facts": row["reference_facts"],
        "public_synthetic": "public",
        "source_type": row["source_type"],
        "notes": row["notes"],
    }


def _source_is_public(unit: dict[str, Any]) -> bool:
    source_dataset = unit.get("source", {}).get("source_dataset")
    return source_dataset != "internal_synthetic"


def _inspection_row(
    question: dict[str, Any],
    vanilla_packet: dict[str, Any],
) -> dict[str, Any]:
    expected = question["expected_retrieval_unit_ids"]
    vanilla_ids = vanilla_packet["context_unit_ids"]
    return {
        "case": question["case"],
        "question_id": question["question_id"],
        "query": question["query"],
        "source_type": question["source_type"],
        "expected_retrieval_unit_ids": expected,
        "vanilla_top5_ids": vanilla_ids,
        "expected_source_in_top1": bool(vanilla_ids and vanilla_ids[0] in expected),
        "expected_source_in_top5": bool(set(vanilla_ids) & set(expected)),
    }


def _inspection_text(rows: list[dict[str, Any]]) -> str:
    lines = ["Phase 5C-B public sanity inspection", "=" * 38, ""]
    for row in rows:
        lines.extend(
            [
                f"Case {row['case']}: {row['question_id']}",
                f"question: {row['query']}",
                f"source_type: {row['source_type']}",
                "expected public source IDs:",
                *[f"  - {unit_id}" for unit_id in row["expected_retrieval_unit_ids"]],
                "Vanilla Top-5 IDs:",
                *[f"  - {unit_id}" for unit_id in row["vanilla_top5_ids"]],
                f"expected_source_in_top1: {row['expected_source_in_top1']}",
                f"expected_source_in_top5: {row['expected_source_in_top5']}",
                "",
            ]
        )
    return "\n".join(lines)


def build_public_sanity(repo_root: Path) -> PublicSanityResult:
    units = _read_jsonl(repo_root / "data/processed/phase4_5/retrieval_units.jsonl")
    units_by_id = {unit["id"]: unit for unit in units}
    prompt_text = (repo_root / PROMPT_PATH).read_text(encoding="utf-8")
    prompt_hash = _sha256_text(prompt_text)

    for question in PUBLIC_SANITY_QUESTIONS:
        for unit_id in question["expected_retrieval_unit_ids"]:
            if unit_id not in units_by_id:
                raise ValueError(f"Missing public sanity source unit: {unit_id}")
            if not _source_is_public(units_by_id[unit_id]):
                raise ValueError(f"Public sanity source is synthetic: {unit_id}")

    run_phase5b(repo_root)
    model = SentenceTransformer(MODEL_NAME, local_files_only=True)
    model.eval()
    query_matrix = _encode_texts(
        model,
        [QUERY_PREFIX + question["query"] for question in PUBLIC_SANITY_QUESTIONS],
    )
    client = open_qdrant(repo_root / "data/processed/phase5b/qdrant_storage")
    try:
        vanilla_unit_ids = [
            [
                row["retrieval_unit_id"]
                for row in qdrant_search(client, COLLECTION_NAME, query_matrix[index], limit=TOP_K)
            ]
            for index, _ in enumerate(PUBLIC_SANITY_QUESTIONS)
        ]
    finally:
        client.close()

    questions = [_question_record(row) for row in PUBLIC_SANITY_QUESTIONS]
    references = [_reference_record(row) for row in PUBLIC_SANITY_QUESTIONS]
    vanilla_packets = [
        _packet(question, vanilla_unit_ids[index], units_by_id, prompt_hash)
        for index, question in enumerate(PUBLIC_SANITY_QUESTIONS)
    ]
    source_packets = [
        _packet(question, question["expected_retrieval_unit_ids"], units_by_id, prompt_hash)
        for question in PUBLIC_SANITY_QUESTIONS
    ]

    validation_units = units_by_id
    vanilla_validation = validate_packets(
        [
            {
                "question_id": packet["question_id"],
                "generator_input": packet["generator_input"],
                "retrieval_unit_ids": packet["context_unit_ids"],
            }
            for packet in vanilla_packets
        ],
        validation_units,
    )
    source_validation = validate_packets(
        [
            {
                "question_id": packet["question_id"],
                "generator_input": packet["generator_input"],
                "retrieval_unit_ids": packet["context_unit_ids"],
            }
            for packet in source_packets
        ],
        validation_units,
    )
    if not vanilla_validation["passed"] or not source_validation["passed"]:
        raise ValueError(
            "Public sanity packet leakage validation failed: "
            f"vanilla={vanilla_validation}; source={source_validation}"
        )

    inspection_rows = [
        _inspection_row(question, vanilla_packets[index])
        for index, question in enumerate(PUBLIC_SANITY_QUESTIONS)
    ]
    return PublicSanityResult(
        questions=questions,
        references=references,
        vanilla_packets=vanilla_packets,
        source_packets=source_packets,
        inspection_rows=inspection_rows,
        outputs=[],
    )


def run_public_sanity(repo_root: Path | None = None) -> PublicSanityResult:
    """Build public-corpus sanity questions and generation packets without LLM calls."""

    if repo_root is None:
        repo_root = resolve_repo_root(Path(__file__))
    result = build_public_sanity(repo_root)

    questions_path = repo_root / "data/eval/public_sanity/questions.jsonl"
    reference_path = repo_root / "data/eval/public_sanity/reference.jsonl"
    vanilla_path = repo_root / "data/processed/phase5c/public_sanity_packets.jsonl"
    source_path = repo_root / "data/processed/phase5c/public_sanity_gold_packets.jsonl"
    report_path = repo_root / "experiments/13_public_sanity_inspection.txt"

    _write_jsonl(result.questions, questions_path)
    _write_jsonl(result.references, reference_path)
    _write_jsonl(result.vanilla_packets, vanilla_path)
    _write_jsonl(result.source_packets, source_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_inspection_text(result.inspection_rows), encoding="utf-8")

    result.outputs.extend([questions_path, reference_path, vanilla_path, source_path, report_path])
    manifest_path = repo_root / "data/processed/manifests/phase_5c_public_sanity_manifest.json"
    manifest = {
        "phase": "5C-B",
        "version": "1.0.0",
        "question_count": len(result.questions),
        "top_k": TOP_K,
        "retrieval_backend": "Phase 5B Qdrant",
        "system_prompt": {
            "version": SYSTEM_PROMPT_VERSION,
            "path": PROMPT_PATH.as_posix(),
            "sha256": _sha256_text((repo_root / PROMPT_PATH).read_text(encoding="utf-8")),
        },
        "input_artifact_hashes": {
            path.relative_to(repo_root).as_posix(): sha256_for_file(path)
            for path in [
                repo_root / "data/processed/phase4_5/retrieval_units.jsonl",
                repo_root / "data/processed/phase5b/qdrant_results.jsonl",
                repo_root / PROMPT_PATH,
            ]
        },
        "output_artifact_hashes": {
            path.relative_to(repo_root).as_posix(): sha256_for_file(path)
            for path in result.outputs
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    result.outputs.append(manifest_path)
    return result
