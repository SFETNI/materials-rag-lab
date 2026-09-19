from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from materials_rag.api import MaterialsRAG
from materials_rag.cli import app
from materials_rag.dataset import dataset_status, install_dataset
from materials_rag.ingestion.agentic_retrieval import (
    AgenticQuestionState,
    _execute_scientific_action,
)
from materials_rag.ingestion.end_to_end_agentic import _selected_evidence

runner = CliRunner()


def _unit(index: int) -> dict:
    return {
        "id": f"unit|{index:03d}",
        "kind": "chunk",
        "retrieval_text": f"IN718 fatigue evidence record {index}",
        "source": {"dataset": "test", "source_path": f"record-{index}.txt"},
        "metadata": {"evidence_id": f"E{index:03d}"},
    }


def test_public_cli_exposes_capability_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in [
        "doctor",
        "data",
        "models",
        "retrieve",
        "ask",
        "demo",
        "benchmark",
        "results",
        "reproduce",
    ]:
        assert command in result.stdout


def test_documented_cli_command_shapes_exist() -> None:
    for args in [
        ["doctor", "--help"],
        ["data", "status", "--help"],
        ["data", "install", "--help"],
        ["data", "reconstruct", "--help"],
        ["models", "status", "--help"],
        ["models", "prepare", "--help"],
        ["retrieve", "--help"],
        ["ask", "--help"],
        ["demo", "--help"],
        ["benchmark", "--help"],
        ["results", "verify", "--help"],
        ["reproduce", "--help"],
    ]:
        result = runner.invoke(app, args)
        assert result.exit_code == 0, (args, result.stdout)


def test_dataset_status_distinguishes_252_and_277(tmp_path: Path) -> None:
    target = tmp_path / "data/runtime/retrieval_units.jsonl"
    target.parent.mkdir(parents=True)
    target.write_text("".join(json.dumps(_unit(i)) + "\n" for i in range(252)), encoding="utf-8")
    assert dataset_status(tmp_path).mode == "redistributable-252"
    target.write_text("".join(json.dumps(_unit(i)) + "\n" for i in range(277)), encoding="utf-8")
    assert dataset_status(tmp_path).mode == "full-reconstructed-277"


def test_dataset_installer_verifies_and_is_idempotent(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    canonical = staging / "canonical"
    canonical.mkdir(parents=True)
    units = canonical / "retrieval_units.jsonl"
    units.write_text("".join(json.dumps(_unit(i)) + "\n" for i in range(252)), encoding="utf-8")
    digest = hashlib.sha256(units.read_bytes()).hexdigest()
    (staging / "checksums.sha256").write_text(f"{digest}  canonical/retrieval_units.jsonl\n", encoding="utf-8")
    archive = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.write(units, "canonical/retrieval_units.jsonl")
        bundle.write(staging / "checksums.sha256", "checksums.sha256")
    project = tmp_path / "project"
    first = install_dataset(archive, project)
    second = install_dataset(archive, project)
    assert first["destination_sha256"] == second["destination_sha256"] == digest
    assert dataset_status(project).retrieval_unit_count == 252


def test_public_hybrid_facade_reuses_fusion(tmp_path: Path, monkeypatch) -> None:
    units = tmp_path / "retrieval_units.jsonl"
    units.write_text("".join(json.dumps(_unit(i)) + "\n" for i in range(252)), encoding="utf-8")
    rag = MaterialsRAG(tmp_path, units)
    monkeypatch.setattr(
        rag,
        "_dense_raw",
        lambda _query, top_k: [
            {
                "rank": i + 1,
                "retrieval_unit_id": f"unit|{i:03d}",
                "kind": "chunk",
                "score": 1.0 - i / 1000,
                "source": _unit(i)["source"],
                "evidence_id": f"E{i:03d}",
                "document_id": None,
            }
            for i in range(min(top_k, 50))
        ],
    )
    results = rag.retrieve("fatigue evidence", method="hybrid", top_k=3)
    assert len(results) == 3
    assert all(set(row) == {"rank", "retrieval_unit_id", "kind", "evidence_id", "document_id", "retrieval_text", "source"} for row in results)


def test_doctor_is_capability_aware(monkeypatch) -> None:
    monkeypatch.setattr(
        "materials_rag.cli.dataset_status",
        lambda _root: SimpleNamespace(
            to_dict=lambda: {
                "outcome": "PASS",
                "mode": "redistributable-252",
                "retrieval_unit_count": 252,
            }
        ),
    )
    monkeypatch.setattr(
        "materials_rag.cli.models_status",
        lambda: SimpleNamespace(
            to_dict=lambda: {
                "outcome": "WARN",
                "embedding_available": False,
                "reranker_available": False,
                "codex_available": False,
            }
        ),
    )
    result = runner.invoke(app, ["doctor", "--capability", "hybrid", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["outcome"] == "WARN"


def test_doctor_reports_missing_checkout_without_traceback(monkeypatch) -> None:
    monkeypatch.setattr(
        "materials_rag.cli._root",
        lambda: (_ for _ in ()).throw(FileNotFoundError("no checkout")),
    )
    monkeypatch.setattr(
        "materials_rag.cli.models_status",
        lambda: SimpleNamespace(
            to_dict=lambda: {
                "outcome": "WARN",
                "embedding_available": False,
                "reranker_available": False,
                "codex_available": False,
            }
        ),
    )
    result = runner.invoke(app, ["doctor", "--capability", "core", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["project_checkout"] is False
    assert payload["outcome"] == "FAIL"


def test_scientific_action_executes_typed_registered_computation(monkeypatch, tmp_path: Path) -> None:
    request = {
        "request_id": "runtime-R1",
        "operation": "SUMMARIZE",
        "dataset_id": "nist_experiment_records",
        "filters": [],
        "group_by": [],
        "numeric_field": "cycles_to_failure",
        "status_filter": "failure_only",
        "requested_statistics": ["median"],
        "comparison_definition": None,
        "censoring_policy": "exclude_runouts_for_cycles_to_failure",
    }
    monkeypatch.setattr(
        "materials_rag.ingestion.scientific_computation.call_scientific_analyst",
        lambda *_args, **_kwargs: (
            {
                "outcome": "REQUEST_COMPUTATION",
                "reason": "A registered summary is required.",
                "computation_request": request,
            },
            [],
        ),
    )
    receipt = {
        "receipt_id": "computation_receipt|demo",
        "kind": "computation_receipt",
        "operation": "SUMMARIZE",
        "dataset_id": "nist_experiment_records",
        "source_record_ids": ["record-1"],
        "applied_filters": [],
        "group_definitions": {},
        "numeric_field": "cycles_to_failure",
        "units": "cycles",
        "censoring_status_policy": "exclude_runouts_for_cycles_to_failure",
        "sample_counts": {"selected": 1},
        "result_values": {"median": 100.0},
        "deterministic_warnings": [],
        "provenance": {"source_record_ids": ["record-1"]},
        "implementation": {"version": "test"},
        "retrieval_text": "Median cycles to failure: 100 cycles.",
    }
    monkeypatch.setattr(
        "materials_rag.ingestion.scientific_computation.execute_computation_request",
        lambda *_args, **_kwargs: SimpleNamespace(to_dict=lambda: receipt),
    )
    toolbox = SimpleNamespace(repo_root=tmp_path, adapter=object())
    state = AgenticQuestionState("Q1", "What is the median failure life?")
    observation, role_receipts, added = _execute_scientific_action(
        toolbox,
        state,
        {
            "action": "SCIENTIFIC_ANALYSIS",
            "query": "",
            "objective": "Compute median failure life.",
            "document_id": "",
            "reason": "A numerical result is required.",
        },
    )
    assert role_receipts == []
    assert added is True
    assert observation["receipt_id"] == receipt["receipt_id"]
    assert state.evidence_pool[0]["context_id"] == receipt["receipt_id"]
    assert state.cost_counters["computation_actions"] == 1
    selected = _selected_evidence(
        {
            "final_selected_context_ids": [receipt["receipt_id"]],
            "computation_receipts": [receipt],
        },
        {},
    )
    assert selected[0]["receipt"]["result_values"]["median"] == 100.0
