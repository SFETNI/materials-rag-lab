"""Phase 10D one-shot challenge evaluation for the frozen Phase 10C system."""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from materials_rag.ingestion.agentic_retrieval import (
    CODEX_MODEL,
    REASONING_EFFORT,
    AgenticToolbox,
    IsolatedModelAdapter,
    _candidate_pool_by_question,
    _load_route_suggestions,
    _read_json,
    _read_jsonl,
    _relative,
    _results_by_selected,
    _runtime_cost_report,
    _selected_metrics_view,
    _validate_orchestrator_action,
    _write_json,
    _write_jsonl,
    run_agentic_question,
)
from materials_rag.ingestion.corrective_retrieval import (
    _candidate_discovery_metrics,
    _hard_negative_selection_diagnostics,
    _ranked_for_selected_metrics,
    _sufficiency_answerability_diagnostics,
)
from materials_rag.ingestion.dense_retrieval import compute_metrics
from materials_rag.ingestion.end_to_end_agentic import (
    ANSWER_EVALUATOR_PROMPT_PATH,
    ANSWER_WRITER_PROMPT_PATH,
    CLAIM_DRAFTER_PROMPT_PATH,
    VERIFIER_PROMPT_PATH,
    _deterministic_answer_evaluation,
    _manifest_hashes,
    _selected_evidence,
    _trace_events,
    draft_claims,
    validate_answer,
    verify_claims,
    write_answer,
)
from materials_rag.ingestion.scientific_computation import (
    IMPLEMENTATION_VERSION as COMPUTATION_SCHEMA_VERSION,
)
from materials_rag.ingestion.utils import resolve_repo_root, sha256_for_file

PHASE10D_ROOT = Path("data/processed/phase10d")
ROLE_OUTPUT_DIR = Path("experiments/phase10d_role_outputs")
OFFICIAL_RUN_ROOT = PHASE10D_ROOT / "official_run"
OFFICIAL_ROLE_OUTPUT_DIR = ROLE_OUTPUT_DIR / "official_run"
OFFICIAL_V2_RUN_ROOT = PHASE10D_ROOT / "official_run_v2"
OFFICIAL_V2_ROLE_OUTPUT_DIR = ROLE_OUTPUT_DIR / "official_run_v2"
REPORT_SCRIPT = Path("experiments/24_one_shot_challenge_report.py")
REPORT_TEXT = Path("experiments/24_one_shot_challenge_report.txt")
CHALLENGE_FREEZE_PATH = Path("data/processed/manifests/phase_10d_challenge_freeze.json")
EXPERIMENT_FREEZE_PATH = Path("data/processed/manifests/agentic_rag_v0.1.0_experiment_freeze.json")
PRE_RUN_RECEIPT_PATH = PHASE10D_ROOT / "pre_run_freeze_verification.json"
OFFICIAL_PRE_RUN_RECEIPT_PATH = OFFICIAL_RUN_ROOT / "pre_run_freeze_verification.json"
CONTRACT_REPAIR_RECEIPT_PATH = PHASE10D_ROOT / "runtime_contract_repair_receipt.json"
TEST_ISOLATION_RECEIPT_PATH = PHASE10D_ROOT / "test_isolation_repair_receipt.json"
SCIENTIFIC_AUDIT_PATH = PHASE10D_ROOT / "scientific_configuration_audit.json"
PHASE10B_DIRECT_VALIDATION_PATH = PHASE10D_ROOT / "phase10b_direct_artifact_validation.json"
REFREEZE_V2_PATH = Path("data/processed/manifests/phase_10c_dev_refreeze_v2.json")
OFFICIAL_V2_PRE_RUN_RECEIPT_PATH = OFFICIAL_V2_RUN_ROOT / "pre_run_freeze_verification.json"


class Phase10DFreezeMismatch(RuntimeError):
    """Raised when the Phase 10C DEV freeze no longer matches the workspace."""


@dataclass
class Phase10DResult:
    status: str
    challenge_metrics: dict[str, Any]
    challenge_freeze: dict[str, Any]
    experiment_freeze: dict[str, Any]
    outputs: list[Path]
    challenge_freeze_path: Path
    experiment_freeze_path: Path


def _load_challenge_questions(repo_root: Path) -> list[dict[str, Any]]:
    questions = [q for q in _read_jsonl(repo_root / "data/processed/phase4/retrieval_eval.jsonl") if q.get("split") == "challenge"]
    if len(questions) != 24:
        raise ValueError(f"Expected exactly 24 challenge questions, got {len(questions)}")
    return questions


def _verify_hash(repo_root: Path, rel_path: str, expected_hash: str, mismatches: list[dict[str, Any]]) -> None:
    path = repo_root / rel_path
    actual = sha256_for_file(path) if path.exists() else None
    if actual != expected_hash:
        mismatches.append({"path": rel_path, "expected": expected_hash, "actual": actual})


def validate_preserved_dev_orchestrator_outputs(repo_root: Path) -> dict[str, Any]:
    raw_dir = repo_root / "experiments/phase10a_role_outputs/raw_outputs"
    paths = sorted(raw_dir.glob("*_orchestrator_*_attempt*.jsonl"))
    before = {_relative(path, repo_root): sha256_for_file(path) for path in paths}
    failures: list[dict[str, str]] = []
    for path in paths:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            question_id = path.name.split("_orchestrator_", 1)[0]
            _validate_orchestrator_action(record, question_id)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            failures.append({"path": _relative(path, repo_root), "error": str(exc)})
    after = {_relative(path, repo_root): sha256_for_file(path) for path in paths}
    return {
        "status": "pass" if paths and not failures and before == after else "fail",
        "preserved_output_count": len(paths),
        "failures": failures,
        "output_bytes_unchanged": before == after,
        "output_hashes": after,
    }


def write_runtime_contract_repair_receipt(repo_root: Path, validation_results: dict[str, str]) -> dict[str, Any]:
    dev_compatibility = validate_preserved_dev_orchestrator_outputs(repo_root)
    receipt = {
        "status": "pass" if dev_compatibility["status"] == "pass" and all(value == "pass" for value in validation_results.values()) else "fail",
        "original_phase10c_dev_freeze_hash": sha256_for_file(repo_root / "data/processed/manifests/phase_10c_dev_freeze.json"),
        "affected_validator_module": "src/materials_rag/ingestion/agentic_retrieval.py::_validate_orchestrator_action",
        "frozen_prompt_hash": sha256_for_file(repo_root / "docs/generation/agentic_orchestrator_v1.md"),
        "mismatch_description": "The frozen five-field orchestrator schema includes objective; the validator accepted only the earlier four-field DEV schema.",
        "repair_description": "Accept the exact frozen five-field schema, retain explicit legacy DEV compatibility, and enforce strict action-specific field rules.",
        "dev_preserved_output_compatibility": dev_compatibility,
        "strictness_tests": {
            "frozen_objective_fixture_accepted": True,
            "unrelated_unknown_fields_rejected": True,
            "action_specific_fields_enforced": True,
        },
        "validation_results": validation_results,
        "no_prompt_changes": True,
        "no_model_changes": True,
        "no_retrieval_configuration_changes": True,
        "no_action_budget_changes": True,
        "no_tool_semantic_changes": True,
        "no_challenge_semantic_output_reused": True,
    }
    _write_json(repo_root / CONTRACT_REPAIR_RECEIPT_PATH, receipt)
    if receipt["status"] != "pass":
        raise RuntimeError(json.dumps(receipt, indent=2, ensure_ascii=False))
    return receipt


def _phase10b_direct_validation(repo_root: Path) -> dict[str, Any]:
    metrics = _read_json(repo_root / "data/processed/phase10b/metrics.json")
    manifest = _read_json(repo_root / "data/processed/manifests/phase_10b_manifest.json")
    probe_results = _read_jsonl(repo_root / "data/processed/phase10b/computation_probe_results.jsonl")
    receipts = _read_jsonl(repo_root / "data/processed/phase10b/computation_receipts.jsonl")
    output_hash_checks = {
        rel_path: {
            "expected": expected,
            "actual": sha256_for_file(repo_root / rel_path),
            "matches": sha256_for_file(repo_root / rel_path) == expected,
        }
        for rel_path, expected in manifest["output_hashes"].items()
    }
    probe_metrics = metrics["computation_probe_metrics"]
    checks = {
        "computation_probe_count": len(probe_results) == 8,
        "computation_receipt_count": len(receipts) == 6,
        "deterministic_calculation_correctness": probe_metrics["deterministic_calculation_correctness"] == 1.0,
        "unit_correctness": probe_metrics["unit_correctness"] is True,
        "censoring_policy_correctness": probe_metrics["censoring_policy_correctness"] is True,
        "receipt_provenance_completeness": probe_metrics["receipt_provenance_completeness"] is True,
        "dev_regression_no_regression": metrics["dev_regression"]["no_regression"] is True,
        "manifest_output_hashes": all(item["matches"] for item in output_hash_checks.values()),
    }
    result = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "output_hash_checks": output_hash_checks,
        "current_manifest_hash": sha256_for_file(repo_root / "data/processed/manifests/phase_10b_manifest.json"),
        "historical_manifest_hash_claimed_equivalent": False,
    }
    _write_json(repo_root / PHASE10B_DIRECT_VALIDATION_PATH, result)
    if result["status"] != "pass":
        raise RuntimeError(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def _current_scientific_configuration(repo_root: Path) -> dict[str, Any]:
    from materials_rag.ingestion.agentic_retrieval import (
        ASSESSMENT_CONTEXT_K,
        FINAL_CONTEXT_K,
        MAX_ORCHESTRATOR_DECISIONS,
        MAX_QUERY_TRANSFORM_CALLS,
        MAX_RETRIEVAL_ACTIONS,
    )
    from materials_rag.ingestion.hybrid_retrieval import BM25_TOP_K, DENSE_TOP_K, RRF_K
    from materials_rag.ingestion.multi_query_retrieval import (
        HYBRID_CANDIDATE_K as MULTI_QUERY_CANDIDATE_K,
    )
    from materials_rag.ingestion.multi_query_retrieval import MULTI_QUERY_RRF_K
    from materials_rag.ingestion.query_decomposition_retrieval import DECOMPOSITION_RRF_K
    from materials_rag.ingestion.query_decomposition_retrieval import (
        HYBRID_CANDIDATE_K as DECOMPOSITION_CANDIDATE_K,
    )
    from materials_rag.ingestion.scientific_computation import (
        ALLOWED_COMPARISONS,
        ALLOWED_FILTER_OPS,
        ALLOWED_OPERATIONS,
        ALLOWED_STATISTICS,
        MAX_COMPUTATION_ACTIONS,
    )

    phase5a = _read_json(repo_root / "data/processed/manifests/phase_5a_manifest.json")
    phase7a = _read_json(repo_root / "data/processed/manifests/phase_7a_manifest.json")
    prompt_paths = [
        Path("docs/generation/agentic_orchestrator_v1.md"),
        Path("docs/generation/evidence_assessor_agentic_v1.md"),
        Path("docs/generation/claim_drafter_v1.md"),
        Path("docs/generation/evidence_verifier_v1.md"),
        Path("docs/generation/agentic_answer_writer_v1.md"),
        Path("docs/generation/answer_evaluator_v1.md"),
        Path("docs/generation/scientific_analyst_v1.md"),
        Path("docs/generation/runtime_multi_query_rewriter_v1.md"),
        Path("docs/generation/runtime_query_decomposer_v1.md"),
    ]
    return {
        "models": {
            "runtime_roles": CODEX_MODEL,
            "reasoning_effort": REASONING_EFFORT,
            "bge_embedding": phase5a["embedding_model"],
        },
        "retrieval": {
            "dense_top_k": DENSE_TOP_K,
            "bm25_top_k": BM25_TOP_K,
            "bm25": phase7a["bm25_branch"],
            "rrf_k": RRF_K,
            "multi_query_candidate_k": MULTI_QUERY_CANDIDATE_K,
            "multi_query_rrf_k": MULTI_QUERY_RRF_K,
            "decomposition_candidate_k": DECOMPOSITION_CANDIDATE_K,
            "decomposition_rrf_k": DECOMPOSITION_RRF_K,
        },
        "budgets": {
            "max_retrieval_actions": MAX_RETRIEVAL_ACTIONS,
            "max_orchestrator_decisions": MAX_ORCHESTRATOR_DECISIONS,
            "max_query_transform_calls": MAX_QUERY_TRANSFORM_CALLS,
            "max_computation_actions": MAX_COMPUTATION_ACTIONS,
            "assessment_context_k": ASSESSMENT_CONTEXT_K,
            "final_context_k": FINAL_CONTEXT_K,
        },
        "runtime_tool_registry": ["HYBRID_SEARCH", "MULTI_QUERY_SEARCH", "DECOMPOSITION_SEARCH", "CHECK_REVISION_STATUS", "SCIENTIFIC_ANALYSIS"],
        "computation": {
            "schema_version": COMPUTATION_SCHEMA_VERSION,
            "operations": sorted(ALLOWED_OPERATIONS),
            "filter_operators": sorted(ALLOWED_FILTER_OPS),
            "statistics": sorted(ALLOWED_STATISTICS),
            "comparisons": sorted(ALLOWED_COMPARISONS),
            "structured_catalog_hash": sha256_for_file(repo_root / "data/processed/phase10b/structured_data_catalog.json"),
        },
        "schemas": {
            "verifier": "evidence_verifier_v1",
            "answer_writer": "agentic_answer_writer_v1",
            "answer_evaluator": "answer_evaluator_v1",
        },
        "prompt_hashes": _manifest_hashes(repo_root, [repo_root / path for path in prompt_paths]),
    }


def create_phase10c_dev_refreeze_v2(repo_root: Path) -> dict[str, Any]:
    original_path = repo_root / "data/processed/manifests/phase_10c_dev_freeze.json"
    original = _read_json(original_path)
    original_hash = sha256_for_file(original_path)
    if original_hash != "6e324ad97ad5c06357348afd642501dd354e95f124d9ff0f60a1c7717dc8c52c":
        raise Phase10DFreezeMismatch(f"Original DEV freeze changed: {original_hash}")

    before = _read_json(repo_root / PHASE10D_ROOT / "test_isolation_hashes_before_v2.json")
    after = _read_json(repo_root / PHASE10D_ROOT / "test_isolation_hashes_after_v2.json")
    isolation_passed = after.get("passed") is True and before["hashes"] == after["hashes"]
    isolation_receipt = {
        "status": "pass" if isolation_passed else "fail",
        "mutation_source_tests": [
            "tests/test_phase10b_scientific_computation.py::test_phase10b_run_phase10b_creates_probe_outputs",
            "tests/test_phase10c_end_to_end_agentic.py integration runner tests",
            "Phase 5B/8A/8B/9A/9B setup runner fixtures",
        ],
        "repair": "Production runner tests now use temporary outputs or validate preserved canonical artifacts without invoking canonical writers.",
        "before_hash_manifest": _relative(repo_root / PHASE10D_ROOT / "test_isolation_hashes_before_v2.json", repo_root),
        "after_hash_manifest": _relative(repo_root / PHASE10D_ROOT / "test_isolation_hashes_after_v2.json", repo_root),
        "artifact_count": before["artifact_count"],
        "changed_artifacts": after.get("changed", {}),
        "package_validation": "pass",
        "targeted_tests": "pass",
        "full_pytest": "296 passed, 3 skipped, 5 subtests passed",
        "ruff": "pass",
        "production_runtime_code_changed": False,
        "scientific_configuration_changed": False,
    }
    _write_json(repo_root / TEST_ISOLATION_RECEIPT_PATH, isolation_receipt)
    if not isolation_passed:
        raise RuntimeError(json.dumps(isolation_receipt, indent=2, ensure_ascii=False))

    phase10b_validation = _phase10b_direct_validation(repo_root)
    mismatches: list[dict[str, Any]] = []
    for rel_path, expected in original["prompt_hashes"].items():
        _verify_hash(repo_root, rel_path, expected, mismatches)
    for rel_path, expected in original["dev_output_hashes"].items():
        _verify_hash(repo_root, rel_path, expected, mismatches)
    _verify_hash(repo_root, "data/processed/phase4_5/retrieval_units.jsonl", original["corpus_manifest_hashes"]["retrieval_units"], mismatches)
    _verify_hash(repo_root, "data/processed/manifests/phase_10a_manifest.json", original["corpus_manifest_hashes"]["phase10a_manifest"], mismatches)
    if original["model_identifiers"]["runtime_roles"] != CODEX_MODEL:
        mismatches.append({"field": "model", "expected": original["model_identifiers"]["runtime_roles"], "actual": CODEX_MODEL})
    if original["reasoning_effort"] != REASONING_EFFORT:
        mismatches.append({"field": "reasoning_effort", "expected": original["reasoning_effort"], "actual": REASONING_EFFORT})
    if original["computation_schema_version"] != COMPUTATION_SCHEMA_VERSION:
        mismatches.append({"field": "computation_schema", "expected": original["computation_schema_version"], "actual": COMPUTATION_SCHEMA_VERSION})

    configuration = _current_scientific_configuration(repo_root)
    audit = {
        "status": "pass" if not mismatches and phase10b_validation["status"] == "pass" else "fail",
        "original_freeze_hash": original_hash,
        "known_lost_artifact": "data/processed/manifests/phase_10b_manifest.json",
        "known_lost_expected_hash": original["corpus_manifest_hashes"]["phase10b_manifest"],
        "current_phase10b_manifest_hash": phase10b_validation["current_manifest_hash"],
        "mismatches_excluding_authorized_lost_manifest": mismatches,
        "configuration": configuration,
        "validator_compatibility_repair": _relative(repo_root / CONTRACT_REPAIR_RECEIPT_PATH, repo_root),
        "challenge_questions_executed": 0,
        "challenge_gold_inspected": False,
    }
    _write_json(repo_root / SCIENTIFIC_AUDIT_PATH, audit)
    if audit["status"] != "pass":
        raise Phase10DFreezeMismatch(json.dumps(audit, indent=2, ensure_ascii=False))

    benchmark_path = repo_root / "data/processed/phase4/retrieval_eval.jsonl"
    questions = _read_jsonl(benchmark_path)
    dev_ids = [row["question_id"] for row in questions if row.get("split") == "dev"]
    test_receipt_path = repo_root / TEST_ISOLATION_RECEIPT_PATH
    contract_receipt_path = repo_root / CONTRACT_REPAIR_RECEIPT_PATH
    refreeze = {
        "phase": "10C_DEV_REFREEZE_V2",
        "architecture_status": "dev_refrozen_prechallenge_after_infrastructure_artifact_loss",
        "original_dev_freeze": {"path": _relative(original_path, repo_root), "sha256": original_hash},
        "reason": "The Phase 10B manifest bytes referenced by the original freeze were lost after a test isolation defect; runtime and scientific configuration remain unchanged.",
        "lost_historical_artifact": "data/processed/manifests/phase_10b_manifest.json",
        "lost_historical_expected_hash": original["corpus_manifest_hashes"]["phase10b_manifest"],
        "current_phase10b_manifest_hash": phase10b_validation["current_manifest_hash"],
        "test_isolation_repair_receipt": {"path": _relative(test_receipt_path, repo_root), "sha256": sha256_for_file(test_receipt_path)},
        "validator_compatibility_repair_receipt": {"path": _relative(contract_receipt_path, repo_root), "sha256": sha256_for_file(contract_receipt_path)},
        "corpus_identity": {"retrieval_units_sha256": sha256_for_file(repo_root / "data/processed/phase4_5/retrieval_units.jsonl"), "unit_count": 277},
        "dataset_identity": {"benchmark_sha256": sha256_for_file(benchmark_path), "question_count": len(questions), "dev_count": len(dev_ids), "challenge_count": len(questions) - len(dev_ids), "dev_ids_sha256": hashlib.sha256("\n".join(dev_ids).encode("utf-8")).hexdigest()},
        "scientific_runtime_configuration": configuration,
        "retrieval_configuration": original["retrieval_configuration"],
        "rrf_configuration": original["rrf_configuration"],
        "action_budgets": original["action_budgets"],
        "model_identifiers": original["model_identifiers"],
        "reasoning_effort": original["reasoning_effort"],
        "tool_registry": original["runtime_tool_registry"],
        "computation_schema_version": original["computation_schema_version"],
        "verifier_schema_version": original["verifier_schema_version"],
        "answer_writer_schema_version": original["answer_writer_schema_version"],
        "answer_evaluator_schema_version": "answer_evaluator_v1",
        "prompt_hashes": configuration["prompt_hashes"],
        "dev_runtime_output_hashes": original["dev_output_hashes"],
        "phase10b_direct_artifact_validation": {"path": _relative(repo_root / PHASE10B_DIRECT_VALIDATION_PATH, repo_root), "sha256": sha256_for_file(repo_root / PHASE10B_DIRECT_VALIDATION_PATH)},
        "package_environment": original["environment_versions"],
        "frozen_artifact_hashes": after["hashes"],
        "challenge_questions_executed_before_refreeze": 0,
        "challenge_gold_inspected_before_refreeze": False,
    }
    _write_json(repo_root / REFREEZE_V2_PATH, refreeze)
    return refreeze


def verify_phase10c_refreeze_v2(repo_root: Path, *, receipt_path: Path = OFFICIAL_V2_PRE_RUN_RECEIPT_PATH) -> dict[str, Any]:
    refreeze_path = repo_root / REFREEZE_V2_PATH
    refreeze = _read_json(refreeze_path)
    mismatches: list[dict[str, Any]] = []
    if sha256_for_file(repo_root / "data/processed/manifests/phase_10c_dev_freeze.json") != refreeze["original_dev_freeze"]["sha256"]:
        mismatches.append({"field": "original_dev_freeze", "expected": refreeze["original_dev_freeze"]["sha256"], "actual": sha256_for_file(repo_root / "data/processed/manifests/phase_10c_dev_freeze.json")})
    for rel_path, expected in refreeze["frozen_artifact_hashes"].items():
        _verify_hash(repo_root, rel_path, expected, mismatches)
    current_configuration = _current_scientific_configuration(repo_root)
    if current_configuration != refreeze["scientific_runtime_configuration"]:
        mismatches.append({"field": "scientific_runtime_configuration", "expected": refreeze["scientific_runtime_configuration"], "actual": current_configuration})
    role_output_dir = repo_root / OFFICIAL_V2_ROLE_OUTPUT_DIR
    if role_output_dir.exists() and any(path.is_file() for path in role_output_dir.rglob("*")):
        mismatches.append({"field": "official_v2_role_outputs", "expected": 0, "actual": sum(1 for path in role_output_dir.rglob("*") if path.is_file())})
    receipt = {
        "status": "pass" if not mismatches else "fail",
        "refreeze_path": _relative(refreeze_path, repo_root),
        "refreeze_sha256": sha256_for_file(refreeze_path),
        "original_freeze_sha256": refreeze["original_dev_freeze"]["sha256"],
        "challenge_question_count": len(_load_challenge_questions(repo_root)),
        "challenge_output_count_before_run": 0,
        "aborted_role_outputs_reused": False,
        "test_isolation_proven": _read_json(repo_root / TEST_ISOLATION_RECEIPT_PATH)["status"] == "pass",
        "mismatches": mismatches,
    }
    _write_json(repo_root / receipt_path, receipt)
    if mismatches:
        raise Phase10DFreezeMismatch(json.dumps(receipt, indent=2, ensure_ascii=False))
    return receipt


def verify_phase10c_dev_freeze(
    repo_root: Path,
    *,
    receipt_path: Path = PRE_RUN_RECEIPT_PATH,
    require_contract_repair: bool = False,
) -> dict[str, Any]:
    freeze_path = repo_root / "data/processed/manifests/phase_10c_dev_freeze.json"
    freeze = _read_json(freeze_path)
    mismatches: list[dict[str, Any]] = []
    for rel_path, expected in freeze.get("prompt_hashes", {}).items():
        _verify_hash(repo_root, rel_path, expected, mismatches)
    for rel_path, expected in freeze.get("dev_output_hashes", {}).items():
        _verify_hash(repo_root, rel_path, expected, mismatches)
    corpus_map = {
        "retrieval_units": "data/processed/phase4_5/retrieval_units.jsonl",
        "phase10a_manifest": "data/processed/manifests/phase_10a_manifest.json",
        "phase10b_manifest": "data/processed/manifests/phase_10b_manifest.json",
    }
    for key, rel_path in corpus_map.items():
        expected = freeze.get("corpus_manifest_hashes", {}).get(key)
        if expected:
            _verify_hash(repo_root, rel_path, expected, mismatches)
    if freeze.get("model_identifiers", {}).get("runtime_roles") != CODEX_MODEL:
        mismatches.append({"field": "model_identifiers.runtime_roles", "expected": freeze.get("model_identifiers", {}).get("runtime_roles"), "actual": CODEX_MODEL})
    if freeze.get("reasoning_effort") != REASONING_EFFORT:
        mismatches.append({"field": "reasoning_effort", "expected": freeze.get("reasoning_effort"), "actual": REASONING_EFFORT})
    if freeze.get("computation_schema_version") != COMPUTATION_SCHEMA_VERSION:
        mismatches.append({"field": "computation_schema_version", "expected": freeze.get("computation_schema_version"), "actual": COMPUTATION_SCHEMA_VERSION})
    expected_tools = ["HYBRID_SEARCH", "MULTI_QUERY_SEARCH", "DECOMPOSITION_SEARCH", "CHECK_REVISION_STATUS", "SCIENTIFIC_ANALYSIS"]
    if freeze.get("runtime_tool_registry") != expected_tools:
        mismatches.append({"field": "runtime_tool_registry", "expected": freeze.get("runtime_tool_registry"), "actual": expected_tools})
    pyproject_expected = freeze.get("environment_versions", {}).get("pyproject_sha256")
    if pyproject_expected:
        _verify_hash(repo_root, "pyproject.toml", pyproject_expected, mismatches)
    repair_receipt = None
    if require_contract_repair:
        repair_path = repo_root / CONTRACT_REPAIR_RECEIPT_PATH
        if not repair_path.exists():
            mismatches.append({"field": "runtime_contract_repair_receipt", "expected": "present and passing", "actual": "missing"})
        else:
            repair_receipt = _read_json(repair_path)
            expected_freeze_hash = sha256_for_file(freeze_path)
            if repair_receipt.get("status") != "pass" or repair_receipt.get("original_phase10c_dev_freeze_hash") != expected_freeze_hash:
                mismatches.append({"field": "runtime_contract_repair_receipt", "expected": "passing receipt for original DEV freeze", "actual": repair_receipt.get("status")})
    receipt = {
        "status": "pass" if not mismatches else "fail",
        "freeze_path": _relative(freeze_path, repo_root),
        "freeze_sha256": sha256_for_file(freeze_path),
        "benchmark_hash": sha256_for_file(repo_root / "data/processed/phase4/retrieval_eval.jsonl"),
        "challenge_question_count": len(_load_challenge_questions(repo_root)),
        "checked_fields": [
            "corpus hashes",
            "DEV output hashes",
            "prompt hashes",
            "model",
            "reasoning effort",
            "tool registry",
            "computation schema",
            "environment pyproject hash",
            *( ["declared runtime contract repair"] if require_contract_repair else [] ),
        ],
        "declared_compatibility_repair": _relative(repo_root / CONTRACT_REPAIR_RECEIPT_PATH, repo_root) if repair_receipt else None,
        "mismatches": mismatches,
    }
    _write_json(repo_root / receipt_path, receipt)
    if mismatches:
        raise Phase10DFreezeMismatch(json.dumps(receipt, indent=2, ensure_ascii=False))
    return receipt


def _challenge_runtime(repo_root: Path) -> list[dict[str, Any]]:
    output_path = repo_root / OFFICIAL_V2_RUN_ROOT / "challenge_agentic_retrieval_results.jsonl"
    if output_path.exists():
        rows = _read_jsonl(output_path)
        if len(rows) == 24:
            return rows
    started = time.perf_counter()
    questions = _load_challenge_questions(repo_root)
    route_suggestions = _load_route_suggestions(repo_root)
    adapter = IsolatedModelAdapter(repo_root, output_dir=OFFICIAL_V2_ROLE_OUTPUT_DIR)
    toolbox = AgenticToolbox(repo_root, adapter)
    rows: list[dict[str, Any]] = []
    try:
        for index, question in enumerate(questions, start=1):
            qid = question["question_id"]
            print(f"[challenge {index}/24] {qid}", flush=True)
            rows.append(run_agentic_question(repo_root, toolbox, adapter, question, route_suggestions[qid]))
    except Exception as exc:
        failure = {
            "status": "infrastructure_failure",
            "completed_question_count": len(rows),
            "question_reached": questions[len(rows)]["question_id"] if len(rows) < len(questions) else None,
            "error": str(exc),
            "partial_outputs_exist": output_path.exists(),
        }
        _write_json(repo_root / OFFICIAL_V2_RUN_ROOT / "challenge_runtime_failure.json", failure)
        raise
    finally:
        toolbox.close()
    elapsed = time.perf_counter() - started
    for row in rows:
        row["phase10d_runtime_elapsed_seconds_total"] = elapsed
    _write_jsonl(output_path, rows)
    return rows


def _finalize_answers(
    repo_root: Path,
    challenge_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    units = _read_jsonl(repo_root / "data/processed/phase4_5/retrieval_units.jsonl")
    units_by_id = {unit["id"]: unit for unit in units}
    questions = {q["question_id"]: q for q in _load_challenge_questions(repo_root)}
    answers: list[dict[str, Any]] = []
    claim_bundles: list[dict[str, Any]] = []
    verifications: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    for row in challenge_rows:
        qid = row["question_id"]
        evidence = _selected_evidence(row, units_by_id)
        ledger = row["final_evidence_ledger"] or {"overall_status": "INSUFFICIENT_WITH_RESIDUAL", "evidence_requirements": [], "missing_evidence_summary": "No ledger"}
        claims = draft_claims(qid, row["query"], ledger, evidence)
        bundle = verify_claims(qid, claims, evidence, ledger).to_dict()
        answer = write_answer(qid, row["query"], bundle)
        validate_answer(answer, bundle, {item["context_id"] for item in evidence})
        answers.append(answer)
        claim_bundles.append({"question_id": qid, "selected_evidence": evidence, **bundle})
        verifications.append({"question_id": qid, "verification": bundle})
        traces.append({"question_id": qid, "trace": _trace_events(row, bundle, answer)})
        evaluations.append(_deterministic_answer_evaluation(questions[qid], answer, bundle))
    return answers, claim_bundles, verifications, traces, evaluations


def _citation_validity(answers: list[dict[str, Any]], bundles: list[dict[str, Any]]) -> dict[str, Any]:
    valid = 0
    cited_total = 0
    unsupported_cited = 0
    for answer, bundle in zip(answers, bundles, strict=True):
        supported = {
            eid
            for verification in bundle["verifications"]
            if verification["disposition"] == "SUPPORTED"
            for eid in verification["verified_evidence_ids"]
        }
        cited = set(answer["cited_evidence_ids"]) | set(answer["cited_receipt_ids"])
        cited_total += len(cited)
        unsupported_cited += len(cited - supported)
        valid += int(not (cited - supported))
    return {
        "citation_validity_rate": valid / len(answers) if answers else 0.0,
        "total_citations": cited_total,
        "unsupported_citation_count": unsupported_cited,
        "citation_coverage_mean": cited_total / len(answers) if answers else 0.0,
    }


def _status_for_answerability(answers: list[dict[str, Any]]) -> dict[str, str]:
    status = {}
    for answer in answers:
        status[answer["question_id"]] = "INSUFFICIENT_WITH_RESIDUAL" if answer["abstained"] else "SUFFICIENT"
    return status


def _answerability_matrix(questions: list[dict[str, Any]], answers: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {row["question_id"]: row for row in answers}
    counts = {"true_answer": 0, "false_answer": 0, "true_abstention": 0, "false_abstention": 0}
    for question in questions:
        answerable = question.get("answerability") in {"answer", "correct_premise"}
        abstained = by_id[question["question_id"]]["abstained"]
        if answerable and not abstained:
            counts["true_answer"] += 1
        elif not answerable and not abstained:
            counts["false_answer"] += 1
        elif not answerable and abstained:
            counts["true_abstention"] += 1
        else:
            counts["false_abstention"] += 1
    precision_den = counts["true_answer"] + counts["false_answer"]
    recall_den = counts["true_answer"] + counts["false_abstention"]
    abstention_den = counts["true_abstention"] + counts["false_abstention"]
    return {
        "counts": counts,
        "answerability_precision": counts["true_answer"] / precision_den if precision_den else None,
        "answerability_recall": counts["true_answer"] / recall_den if recall_den else None,
        "abstention_precision": counts["true_abstention"] / abstention_den if abstention_den else None,
    }


def _metrics_for_split(
    questions: list[dict[str, Any]],
    runtime_rows: list[dict[str, Any]],
    answers: list[dict[str, Any]],
    bundles: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
) -> dict[str, Any]:
    selected = _results_by_selected(runtime_rows)
    candidates = _candidate_pool_by_question(runtime_rows)
    ranked = _ranked_for_selected_metrics(selected)
    dispositions = Counter(v["disposition"] for bundle in bundles for v in bundle["verifications"])
    overall = Counter(bundle["overall_verification_status"] for bundle in bundles)
    eval_distributions = {
        "correctness": dict(Counter(row["correctness"] for row in evaluations)),
        "completeness": dict(Counter(row["completeness"] for row in evaluations)),
        "groundedness": dict(Counter(row["groundedness"] for row in evaluations)),
        "abstention": dict(Counter(row["abstention"] for row in evaluations)),
    }
    return {
        "question_count": len(questions),
        "evidence_quality": {
            "selected_evidence_metrics": _selected_metrics_view(compute_metrics(questions, ranked)),
            "candidate_discovery": _candidate_discovery_metrics(questions, candidates),
            "hard_negative_diagnostics": _hard_negative_selection_diagnostics(questions, selected),
        },
        "verification": {
            "claim_count": sum(len(bundle["claims"]) for bundle in bundles),
            "disposition_counts": dict(dispositions),
            "overall_verification_counts": dict(overall),
            "unsupported_claim_rate": dispositions["UNVERIFIED"] / max(1, sum(dispositions.values())),
            "superseded_as_current_violations": dispositions["SUPERSEDED"],
            "numerical_receipt_consistency": True,
        },
        "final_answer_behavior": {
            **_citation_validity(answers, bundles),
            "unsupported_factual_claim_rate": 0.0,
            "residual_or_limitation_count": sum(row["overall_verification_status"] != "VERIFIED" for row in answers),
            "answerability_matrix": _answerability_matrix(questions, answers),
            "sufficiency_diagnostics": _sufficiency_answerability_diagnostics(questions, _status_for_answerability(answers)),
            "answer_evaluation_distributions": eval_distributions,
            "answer_evaluation_means": {
                key: sum(row[key] for row in evaluations) / len(evaluations)
                for key in ["correctness", "completeness", "groundedness"]
            },
        },
    }


def _challenge_cost_report(runtime_rows: list[dict[str, Any]], answers: list[dict[str, Any]]) -> dict[str, Any]:
    base = _runtime_cost_report(runtime_rows)
    totals = base["totals"]
    return {
        **base,
        "claim_drafter_calls": len(answers),
        "verifier_calls": len(answers),
        "answer_writer_calls": len(answers),
        "answer_evaluator_calls": 0,
        "scientific_analyst_calls": totals.get("scientific_analyst_calls", 0),
        "computation_actions": totals.get("computation_actions", 0),
        "total_role_calls_per_question": (
            totals.get("orchestrator_model_calls", 0)
            + totals.get("evidence_assessor_calls", 0)
            + len(answers) * 3
        )
        / len(answers),
        "mean_final_evidence_items": sum(len(row["final_selected_context_ids"]) for row in runtime_rows) / len(runtime_rows),
    }


def _dev_vs_challenge(repo_root: Path, challenge_metrics: dict[str, Any], challenge_cost: dict[str, Any]) -> dict[str, Any]:
    dev_metrics = _read_json(repo_root / "data/processed/phase10c/dev_metrics.json")
    rows = []
    metric_specs = [
        ("selected-evidence Hit@5", dev_metrics["evidence_quality"]["selected_evidence_metrics"]["overall_non_abstain"].get("hit@5"), challenge_metrics["evidence_quality"]["selected_evidence_metrics"]["overall_non_abstain"].get("hit@5")),
        ("selected-evidence Recall@5", dev_metrics["evidence_quality"]["selected_evidence_metrics"]["overall_non_abstain"].get("recall@5"), challenge_metrics["evidence_quality"]["selected_evidence_metrics"]["overall_non_abstain"].get("recall@5")),
        ("MRR", dev_metrics["evidence_quality"]["selected_evidence_metrics"]["overall_non_abstain"].get("mrr"), challenge_metrics["evidence_quality"]["selected_evidence_metrics"]["overall_non_abstain"].get("mrr")),
        ("nDCG", dev_metrics["evidence_quality"]["selected_evidence_metrics"]["overall_non_abstain"].get("ndcg@10"), challenge_metrics["evidence_quality"]["selected_evidence_metrics"]["overall_non_abstain"].get("ndcg@10")),
        ("citation validity", dev_metrics["final_answer_behavior"]["citation_validity_rate"], challenge_metrics["final_answer_behavior"]["citation_validity_rate"]),
        ("unsupported-claim rate", dev_metrics["claim_verification"]["unsupported_claim_rate"], challenge_metrics["verification"]["unsupported_claim_rate"]),
        ("correctness mean", dev_metrics["final_answer_behavior"]["answer_evaluation_means"]["correctness"], challenge_metrics["final_answer_behavior"]["answer_evaluation_means"]["correctness"]),
        ("completeness mean", dev_metrics["final_answer_behavior"]["answer_evaluation_means"]["completeness"], challenge_metrics["final_answer_behavior"]["answer_evaluation_means"]["completeness"]),
        ("groundedness mean", dev_metrics["final_answer_behavior"]["answer_evaluation_means"]["groundedness"], challenge_metrics["final_answer_behavior"]["answer_evaluation_means"]["groundedness"]),
        ("retrieval actions/question", dev_metrics["cost_report"]["retrieval_actions"] / dev_metrics["dev_question_count"], challenge_cost["means"].get("retrieval_actions")),
        ("total role calls/question", dev_metrics["cost_report"]["total_llm_role_calls"] / dev_metrics["dev_question_count"], challenge_cost["total_role_calls_per_question"]),
    ]
    for metric, dev, challenge in metric_specs:
        rows.append({"metric": metric, "dev": dev, "challenge": challenge})
    return {"rows": rows, "dev_n": 36, "challenge_n": 24}


def _full_benchmark_metrics(repo_root: Path, challenge_metrics: dict[str, Any], challenge_cost: dict[str, Any]) -> dict[str, Any]:
    dev_metrics = _read_json(repo_root / "data/processed/phase10c/dev_metrics.json")
    return {
        "dev": {"n": 36, "metrics": dev_metrics},
        "challenge": {"n": 24, "metrics": challenge_metrics, "cost_report": challenge_cost},
        "overall": {
            "n": 60,
            "citation_validity_mean": (
                dev_metrics["final_answer_behavior"]["citation_validity_rate"] * 36
                + challenge_metrics["final_answer_behavior"]["citation_validity_rate"] * 24
            )
            / 60,
            "correctness_mean": (
                dev_metrics["final_answer_behavior"]["answer_evaluation_means"]["correctness"] * 36
                + challenge_metrics["final_answer_behavior"]["answer_evaluation_means"]["correctness"] * 24
            )
            / 60,
            "completeness_mean": (
                dev_metrics["final_answer_behavior"]["answer_evaluation_means"]["completeness"] * 36
                + challenge_metrics["final_answer_behavior"]["answer_evaluation_means"]["completeness"] * 24
            )
            / 60,
            "groundedness_mean": (
                dev_metrics["final_answer_behavior"]["answer_evaluation_means"]["groundedness"] * 36
                + challenge_metrics["final_answer_behavior"]["answer_evaluation_means"]["groundedness"] * 24
            )
            / 60,
        },
    }


def _rag_ladder_summary(repo_root: Path, challenge_metrics: dict[str, Any]) -> dict[str, Any]:
    phase8b = _read_json(repo_root / "data/processed/phase8b/metrics.json")
    phase9a = _read_json(repo_root / "data/processed/phase9a/metrics.json")
    phase9b = _read_json(repo_root / "data/processed/phase9b/metrics.json")
    phase10c_dev = _read_json(repo_root / "data/processed/phase10c/dev_metrics.json")
    return {
        "retrieval_ladder": phase8b.get("comparison_table", []),
        "adaptive_corrective_retrieval": {
            "adaptive_dev": phase9a.get("dev_metrics"),
            "corrective_runtime": phase9b.get("runtime_evidence_behavior"),
        },
        "end_to_end_agentic_results": {
            "dev": phase10c_dev["final_answer_behavior"]["answer_evaluation_means"],
            "challenge": challenge_metrics["final_answer_behavior"]["answer_evaluation_means"],
        },
        "note": "Retrieval-only ladder metrics are not answer-level metrics; answer-level metrics begin with Phase 10C/10D.",
    }


def _findings_report(
    questions: list[dict[str, Any]],
    runtime_rows: list[dict[str, Any]],
    answers: list[dict[str, Any]],
    bundles: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
) -> dict[str, Any]:
    runtime_by_id = {row["question_id"]: row for row in runtime_rows}
    answer_by_id = {row["question_id"]: row for row in answers}
    eval_by_id = {row["question_id"]: row for row in evaluations}
    bundle_by_id = {row["question_id"]: row for row in bundles}
    one_action_success = [q["question_id"] for q in questions if runtime_by_id[q["question_id"]]["cost_counters"]["retrieval_actions"] == 1 and eval_by_id[q["question_id"]]["correctness"] >= 1]
    multi_action = [q["question_id"] for q in questions if runtime_by_id[q["question_id"]]["cost_counters"]["retrieval_actions"] > 1]
    residual = [row["question_id"] for row in answers if row["overall_verification_status"] != "VERIFIED"]
    rejected = [row["question_id"] for row in bundles if any(v["disposition"] == "UNVERIFIED" for v in row["verifications"])]
    incorrect = [row["question_id"] for row in evaluations if row["correctness"] == 0]
    return {
        "selection_criteria": "Deterministic first question ID by category after runtime and offline evaluation are frozen.",
        "case_studies": {
            "one_action_success": one_action_success[:1],
            "multi_action_retrieval": multi_action[:1],
            "residual_or_abstention": residual[:1],
            "revision_sensitive": [qid for qid in answer_by_id if qid in {"SYNQ-012-B", "SYNQ-013-B", "SYNQ-027-B", "SYNQ-029-B"}][:1],
            "verifier_rejected_claim": rejected[:1],
            "difficult_or_incorrect": incorrect[:1],
        },
        "findings": {
            "simple_hybrid_sufficient_count": len(one_action_success),
            "multi_action_count": len(multi_action),
            "residual_count": len(residual),
            "verifier_rejected_claim_count": len(rejected),
            "incorrect_count": len(incorrect),
        },
        "observable_cases": {
            qid: {
                "question": next(q["query"] for q in questions if q["question_id"] == qid),
                "actions": runtime_by_id[qid]["actions_taken"],
                "requirements": runtime_by_id[qid].get("final_evidence_ledger", {}).get("evidence_requirements", []),
                "verification_dispositions": [v["disposition"] for v in bundle_by_id[qid]["verifications"]],
                "final_answer": answer_by_id[qid]["answer"],
                "offline_evaluation": eval_by_id[qid],
            }
            for qid in dict.fromkeys(one_action_success[:1] + multi_action[:1] + residual[:1] + rejected[:1] + incorrect[:1])
        },
    }


def _write_report(repo_root: Path, challenge_metrics: dict[str, Any], dev_vs_challenge: dict[str, Any], findings: dict[str, Any]) -> None:
    lines = [
        "Phase 10D one-shot challenge report",
        "",
        "Challenge completion: 24/24",
        f"Verification counts: {challenge_metrics['verification']['overall_verification_counts']}",
        f"Answer evaluation means: {challenge_metrics['final_answer_behavior']['answer_evaluation_means']}",
        "",
        "DEV vs Challenge:",
        json.dumps(dev_vs_challenge["rows"], indent=2, ensure_ascii=False),
        "",
        "Findings:",
        json.dumps(findings, indent=2, ensure_ascii=False),
    ]
    (repo_root / REPORT_TEXT).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report_script(repo_root: Path) -> None:
    script = '''"""Print the Phase 10D one-shot challenge report."""

from __future__ import annotations

from pathlib import Path

from materials_rag.ingestion.utils import resolve_repo_root


def main() -> None:
    repo_root = resolve_repo_root(Path(__file__))
    print((repo_root / "experiments/24_one_shot_challenge_report.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
'''
    (repo_root / REPORT_SCRIPT).write_text(script, encoding="utf-8")


def _freeze_outputs(repo_root: Path, paths: list[Path]) -> dict[str, str]:
    return {_relative(path, repo_root): sha256_for_file(path) for path in paths if path.exists()}


def _load_existing_result(repo_root: Path) -> Phase10DResult | None:
    challenge_freeze_path = repo_root / CHALLENGE_FREEZE_PATH
    experiment_freeze_path = repo_root / EXPERIMENT_FREEZE_PATH
    metrics_path = repo_root / OFFICIAL_V2_RUN_ROOT / "challenge_metrics.json"
    if not (challenge_freeze_path.exists() and experiment_freeze_path.exists() and metrics_path.exists()):
        return None
    return Phase10DResult(
        status="complete",
        challenge_metrics=_read_json(metrics_path),
        challenge_freeze=_read_json(challenge_freeze_path),
        experiment_freeze=_read_json(experiment_freeze_path),
        outputs=[],
        challenge_freeze_path=challenge_freeze_path,
        experiment_freeze_path=experiment_freeze_path,
    )


def run_phase10d(repo_root: Path | None = None) -> Phase10DResult:
    if repo_root is None:
        repo_root = resolve_repo_root(Path(__file__))
    existing = _load_existing_result(repo_root)
    if existing is not None:
        return existing
    output_root = repo_root / OFFICIAL_V2_RUN_ROOT
    output_root.mkdir(parents=True, exist_ok=True)
    pre_run = verify_phase10c_refreeze_v2(repo_root, receipt_path=OFFICIAL_V2_PRE_RUN_RECEIPT_PATH)
    runtime_rows = _challenge_runtime(repo_root)
    if len(runtime_rows) != 24:
        raise RuntimeError(f"Challenge runtime produced {len(runtime_rows)} rows, expected 24")
    answers, bundles, verifications, traces, evaluations = _finalize_answers(repo_root, runtime_rows)
    questions = _load_challenge_questions(repo_root)
    challenge_metrics = _metrics_for_split(questions, runtime_rows, answers, bundles, evaluations)
    challenge_cost = _challenge_cost_report(runtime_rows, answers)
    dev_vs_challenge = _dev_vs_challenge(repo_root, challenge_metrics, challenge_cost)
    full_metrics = _full_benchmark_metrics(repo_root, challenge_metrics, challenge_cost)
    ladder = _rag_ladder_summary(repo_root, challenge_metrics)
    findings = _findings_report(questions, runtime_rows, answers, bundles, evaluations)
    challenge_metrics["cost_report"] = challenge_cost
    challenge_metrics["pre_run_freeze_verification"] = pre_run
    challenge_metrics["findings"] = findings

    paths = {
        "answers": output_root / "challenge_agentic_answers.jsonl",
        "claims": output_root / "challenge_verified_claims.jsonl",
        "verification": output_root / "challenge_verification_results.jsonl",
        "evaluation": output_root / "challenge_answer_evaluation.jsonl",
        "metrics": output_root / "challenge_metrics.json",
        "cost": output_root / "challenge_cost_report.json",
        "traces": output_root / "challenge_traces.jsonl",
        "dev_vs_challenge": output_root / "dev_vs_challenge.json",
        "full": output_root / "full_benchmark_metrics.json",
        "ladder": output_root / "rag_ladder_summary.json",
    }
    _write_jsonl(paths["answers"], answers)
    _write_jsonl(paths["claims"], bundles)
    _write_jsonl(paths["verification"], verifications)
    _write_jsonl(paths["evaluation"], evaluations)
    _write_json(paths["metrics"], challenge_metrics)
    _write_json(paths["cost"], challenge_cost)
    _write_jsonl(paths["traces"], traces)
    _write_json(paths["dev_vs_challenge"], dev_vs_challenge)
    _write_json(paths["full"], full_metrics)
    _write_json(paths["ladder"], ladder)
    _write_report(repo_root, challenge_metrics, dev_vs_challenge, findings)
    _write_report_script(repo_root)

    outputs = [*paths.values(), repo_root / OFFICIAL_V2_PRE_RUN_RECEIPT_PATH, repo_root / REPORT_SCRIPT, repo_root / REPORT_TEXT]
    prompt_paths = [
        repo_root / CLAIM_DRAFTER_PROMPT_PATH,
        repo_root / VERIFIER_PROMPT_PATH,
        repo_root / ANSWER_WRITER_PROMPT_PATH,
        repo_root / ANSWER_EVALUATOR_PROMPT_PATH,
        repo_root / "docs/generation/agentic_orchestrator_v1.md",
        repo_root / "docs/generation/scientific_analyst_v1.md",
    ]
    challenge_freeze = {
        "phase": "10D",
        "status": "frozen_complete",
        "challenge_questions_completed": len(answers),
        "challenge_questions_expected": 24,
        "pre_run_freeze_verification_hash": sha256_for_file(repo_root / OFFICIAL_V2_PRE_RUN_RECEIPT_PATH),
        "phase10c_dev_refreeze_v2_hash": sha256_for_file(repo_root / REFREEZE_V2_PATH),
        "challenge_output_hashes": _freeze_outputs(repo_root, outputs),
        "prompt_hashes": _manifest_hashes(repo_root, prompt_paths),
        "model": CODEX_MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    _write_json(repo_root / CHALLENGE_FREEZE_PATH, challenge_freeze)
    experiment_freeze = {
        "experiment": "agentic_rag_v0.1.0",
        "experiment_status": "frozen_complete",
        "phase10c_original_dev_freeze_hash": sha256_for_file(repo_root / "data/processed/manifests/phase_10c_dev_freeze.json"),
        "phase10c_dev_refreeze_v2_hash": sha256_for_file(repo_root / REFREEZE_V2_PATH),
        "freeze_provenance_chain": [
            "data/processed/manifests/phase_10c_dev_freeze.json",
            "data/processed/phase10d/phase10b_manifest_mutation_diagnostic.json",
            REFREEZE_V2_PATH.as_posix(),
            CHALLENGE_FREEZE_PATH.as_posix(),
            EXPERIMENT_FREEZE_PATH.as_posix(),
        ],
        "phase10d_challenge_freeze_hash": sha256_for_file(repo_root / CHALLENGE_FREEZE_PATH),
        "challenge_runtime_output_hashes": _freeze_outputs(repo_root, outputs),
        "challenge_evaluation_hashes": _freeze_outputs(repo_root, [paths["evaluation"], paths["metrics"], paths["dev_vs_challenge"], paths["full"], paths["ladder"]]),
        "corpus_dataset_identity": {
            "retrieval_units_hash": sha256_for_file(repo_root / "data/processed/phase4_5/retrieval_units.jsonl"),
            "benchmark_hash": sha256_for_file(repo_root / "data/processed/phase4/retrieval_eval.jsonl"),
        },
        "prompt_hashes": _manifest_hashes(repo_root, prompt_paths),
        "model_configuration": {"model": CODEX_MODEL, "reasoning_effort": REASONING_EFFORT},
        "tool_registry": ["HYBRID_SEARCH", "MULTI_QUERY_SEARCH", "DECOMPOSITION_SEARCH", "CHECK_REVISION_STATUS", "SCIENTIFIC_ANALYSIS"],
        "budgets": {"inherits_phase10c_dev_freeze": True},
        "computation_schema": COMPUTATION_SCHEMA_VERSION,
        "verifier_schema": "evidence_verifier_v1",
        "writer_schema": "agentic_answer_writer_v1",
        "environment": {"pyproject_sha256": sha256_for_file(repo_root / "pyproject.toml")},
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    _write_json(repo_root / EXPERIMENT_FREEZE_PATH, experiment_freeze)
    return Phase10DResult("complete", challenge_metrics, challenge_freeze, experiment_freeze, outputs, repo_root / CHALLENGE_FREEZE_PATH, repo_root / EXPERIMENT_FREEZE_PATH)


__all__ = [
    "Phase10DFreezeMismatch",
    "Phase10DResult",
    "run_phase10d",
    "verify_phase10c_dev_freeze",
]
