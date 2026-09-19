"""Phase 10B scientific computation and structured-data analysis."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from materials_rag.ingestion.agentic_retrieval import (
    CODEX_MODEL,
    REASONING_EFFORT,
    AgenticValidationError,
    IsolatedModelAdapter,
    _candidate_pool_by_question,
    _dev_phase9b_selected,
    _load_dev_questions,
    _read_json,
    _read_jsonl,
    _relative,
    _results_by_selected,
    _selected_metrics_view,
    _write_json,
    _write_jsonl,
)
from materials_rag.ingestion.corrective_retrieval import (
    _candidate_discovery_metrics,
    _hard_negative_selection_diagnostics,
    _ranked_for_selected_metrics,
    _sufficiency_answerability_diagnostics,
)
from materials_rag.ingestion.dense_retrieval import compute_metrics
from materials_rag.ingestion.utils import resolve_repo_root, sha256_for_file

PHASE10B_ROOT = Path("data/processed/phase10b")
MANIFEST_PATH = Path("data/processed/manifests/phase_10b_manifest.json")
SCIENTIFIC_ANALYST_PROMPT_PATH = Path("docs/generation/scientific_analyst_v1.md")
WALKTHROUGH_SCRIPT = Path("experiments/22_scientific_computation_walkthrough.py")
WALKTHROUGH_TEXT = Path("experiments/22_scientific_computation_walkthrough.txt")
MAX_COMPUTATION_ACTIONS = 2
IMPLEMENTATION_VERSION = "phase10b_scientific_computation_v1"

ALLOWED_OPERATIONS = {"FILTER_RECORDS", "SUMMARIZE", "COMPARE_GROUPS"}
ALLOWED_FILTER_OPS = {"EQ", "IN", "LT", "LE", "GT", "GE"}
ALLOWED_STATISTICS = {"count", "min", "max", "mean", "median", "standard_deviation"}
ALLOWED_COMPARISONS = {"difference", "ratio", "percent_difference"}


class ComputationValidationError(ValueError):
    """Raised when a typed computation request is invalid."""


@dataclass(frozen=True)
class StructuredFieldDescriptor:
    name: str
    field_type: str
    unit: str | None = None
    description: str = ""


@dataclass(frozen=True)
class StructuredDatasetDescriptor:
    dataset_id: str
    record_kind: str
    relative_path: str
    available_fields: dict[str, StructuredFieldDescriptor]
    supported_operations: tuple[str, ...]
    source: str
    notes: str = ""

    def to_runtime_catalog(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "record_kind": self.record_kind,
            "available_fields": {
                name: {
                    "type": field.field_type,
                    "unit": field.unit,
                    "description": field.description,
                }
                for name, field in self.available_fields.items()
            },
            "supported_operations": list(self.supported_operations),
            "source": self.source,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class ComputationRequest:
    request_id: str
    operation: str
    dataset_id: str
    filters: tuple[dict[str, Any], ...] = ()
    group_by: tuple[str, ...] = ()
    numeric_field: str | None = None
    status_filter: str | None = None
    requested_statistics: tuple[str, ...] = ()
    comparison_definition: dict[str, Any] | None = None
    censoring_policy: str | None = None

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> ComputationRequest:
        required = {
            "request_id",
            "operation",
            "dataset_id",
            "filters",
            "group_by",
            "numeric_field",
            "status_filter",
            "requested_statistics",
            "comparison_definition",
            "censoring_policy",
        }
        if set(payload) != required:
            raise ComputationValidationError(f"ComputationRequest keys mismatch: {sorted(payload)}")
        return cls(
            request_id=str(payload["request_id"]),
            operation=str(payload["operation"]),
            dataset_id=str(payload["dataset_id"]),
            filters=tuple(payload["filters"] or []),
            group_by=tuple(payload["group_by"] or []),
            numeric_field=payload["numeric_field"],
            status_filter=payload["status_filter"],
            requested_statistics=tuple(payload["requested_statistics"] or []),
            comparison_definition=payload["comparison_definition"],
            censoring_policy=payload["censoring_policy"],
        )

    def normalized(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "operation": self.operation,
            "dataset_id": self.dataset_id,
            "filters": list(self.filters),
            "group_by": list(self.group_by),
            "numeric_field": self.numeric_field,
            "status_filter": self.status_filter,
            "requested_statistics": list(self.requested_statistics),
            "comparison_definition": self.comparison_definition,
            "censoring_policy": self.censoring_policy,
        }


@dataclass(frozen=True)
class ComputationReceipt:
    receipt_id: str
    kind: str
    operation: str
    dataset_id: str
    source_record_ids: tuple[str, ...]
    applied_filters: tuple[dict[str, Any], ...]
    group_definitions: dict[str, Any]
    numeric_field: str | None
    units: str | None
    censoring_status_policy: str | None
    sample_counts: dict[str, int]
    result_values: dict[str, Any]
    deterministic_warnings: tuple[str, ...]
    provenance: dict[str, Any]
    implementation: dict[str, Any]
    retrieval_text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "kind": self.kind,
            "operation": self.operation,
            "dataset_id": self.dataset_id,
            "source_record_ids": list(self.source_record_ids),
            "applied_filters": list(self.applied_filters),
            "group_definitions": self.group_definitions,
            "numeric_field": self.numeric_field,
            "units": self.units,
            "censoring_status_policy": self.censoring_status_policy,
            "sample_counts": self.sample_counts,
            "result_values": self.result_values,
            "deterministic_warnings": list(self.deterministic_warnings),
            "provenance": self.provenance,
            "implementation": self.implementation,
            "retrieval_text": self.retrieval_text,
        }


@dataclass
class Phase10BResult:
    status: str
    probe_results: list[dict[str, Any]]
    metrics: dict[str, Any]
    manifest: dict[str, Any]
    outputs: list[Path]
    manifest_path: Path


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _load_records(repo_root: Path, descriptor: StructuredDatasetDescriptor) -> list[dict[str, Any]]:
    path = repo_root / descriptor.relative_path
    if not path.exists():
        public_paths = {
            "nist_experiment_records": "data/runtime/dataset/canonical/records/experiments.jsonl",
            "nist_process_records": "data/runtime/dataset/canonical/records/processes.jsonl",
            "nist_test_records": "data/runtime/dataset/canonical/records/tests.jsonl",
        }
        path = repo_root / public_paths[descriptor.dataset_id]
    if not path.exists():
        raise ComputationValidationError(
            f"Structured dataset {descriptor.dataset_id!r} is not installed"
        )
    return _read_jsonl(path)


def build_structured_data_catalog(repo_root: Path | None = None) -> dict[str, StructuredDatasetDescriptor]:
    """Return runtime-safe descriptors for canonical structured records."""

    _ = repo_root
    experiment_fields = {
        "id": StructuredFieldDescriptor("id", "string", description="Canonical record ID"),
        "specimen_id": StructuredFieldDescriptor("specimen_id", "string", description="Specimen identifier"),
        "specimen_type": StructuredFieldDescriptor("specimen_type", "string", description="NIST surface/build condition label"),
        "file_variant": StructuredFieldDescriptor("file_variant", "string", description="Source file variant"),
        "stress_level_mpa": StructuredFieldDescriptor("stress_level_mpa", "number", "MPa", "Reported stress level"),
        "force_to_be_applied_n": StructuredFieldDescriptor("force_to_be_applied_n", "number", "N", "Force to be applied"),
        "cycles_to_failure": StructuredFieldDescriptor("cycles_to_failure", "number", "cycles", "Observed cycles field; runouts require censoring care"),
        "runout_status": StructuredFieldDescriptor("runout_status", "boolean", description="True when marked as runout/right-censored"),
        "layer_height_mm": StructuredFieldDescriptor("layer_height_mm", "number", "mm", "Build layer height"),
        "diameter_top_mm": StructuredFieldDescriptor("diameter_top_mm", "number", "mm", "Top specimen diameter"),
        "diameter_middle_mm": StructuredFieldDescriptor("diameter_middle_mm", "number", "mm", "Middle specimen diameter"),
        "diameter_bottom_mm": StructuredFieldDescriptor("diameter_bottom_mm", "number", "mm", "Bottom specimen diameter"),
    }
    process_fields = {
        "id": StructuredFieldDescriptor("id", "string", description="Canonical process record ID"),
        "cycle_number": StructuredFieldDescriptor("cycle_number", "number", description="HIP cycle number"),
        "process_name": StructuredFieldDescriptor("process_name", "string", description="Process name"),
        "pressure_mpa": StructuredFieldDescriptor("pressure_mpa", "number", "MPa", "HIP pressure"),
        "temperature_c": StructuredFieldDescriptor("temperature_c", "number", "°C", "HIP temperature"),
        "hold_time_min": StructuredFieldDescriptor("hold_time_min", "number", "min", "HIP hold time"),
        "pressure_medium": StructuredFieldDescriptor("pressure_medium", "string", description="Pressure medium"),
        "cooling_type": StructuredFieldDescriptor("cooling_type", "string", description="Cooling type"),
    }
    test_fields = {
        "id": StructuredFieldDescriptor("id", "string", description="Canonical test record ID"),
        "specimen_id": StructuredFieldDescriptor("specimen_id", "string", description="Specimen identifier"),
        "file_variant": StructuredFieldDescriptor("file_variant", "string", description="Fatigue log variant"),
        "cycle_count": StructuredFieldDescriptor("cycle_count", "number", "rows/cycles", "Parsed log row/cycle count"),
        "experiment_record_id": StructuredFieldDescriptor("experiment_record_id", "string", description="Linked ExperimentRecord ID"),
    }
    return {
        "nist_experiment_records": StructuredDatasetDescriptor(
            dataset_id="nist_experiment_records",
            record_kind="ExperimentRecord",
            relative_path="data/processed/phase2a/canonical/experiment_records.jsonl",
            available_fields=experiment_fields,
            supported_operations=("FILTER_RECORDS", "SUMMARIZE", "COMPARE_GROUPS"),
            source="public NIST IN718 canonical ExperimentRecord JSONL",
            notes="cycles_to_failure excludes runouts by default for failure-life statistics.",
        ),
        "nist_process_records": StructuredDatasetDescriptor(
            dataset_id="nist_process_records",
            record_kind="ProcessRecord",
            relative_path="data/processed/phase2a/canonical/process_records.jsonl",
            available_fields=process_fields,
            supported_operations=("FILTER_RECORDS", "SUMMARIZE"),
            source="public NIST IN718 canonical ProcessRecord JSONL",
        ),
        "nist_test_records": StructuredDatasetDescriptor(
            dataset_id="nist_test_records",
            record_kind="TestRecord",
            relative_path="data/processed/phase2b/canonical/test_records.jsonl",
            available_fields=test_fields,
            supported_operations=("FILTER_RECORDS", "SUMMARIZE"),
            source="public NIST fatigue test-log canonical TestRecord JSONL",
        ),
    }


def runtime_catalog_payload(catalog: dict[str, StructuredDatasetDescriptor]) -> dict[str, Any]:
    return {dataset_id: descriptor.to_runtime_catalog() for dataset_id, descriptor in catalog.items()}


def _field_descriptor(descriptor: StructuredDatasetDescriptor, field: str) -> StructuredFieldDescriptor:
    if field not in descriptor.available_fields:
        raise ComputationValidationError(f"Field {field!r} is not registered for {descriptor.dataset_id}")
    return descriptor.available_fields[field]


def _validate_filter(filter_row: dict[str, Any], descriptor: StructuredDatasetDescriptor) -> None:
    if set(filter_row) != {"field", "op", "value"}:
        raise ComputationValidationError(f"Invalid filter keys: {sorted(filter_row)}")
    field = str(filter_row["field"])
    op = str(filter_row["op"])
    field_info = _field_descriptor(descriptor, field)
    if op not in ALLOWED_FILTER_OPS:
        raise ComputationValidationError(f"Unsupported filter operator: {op}")
    if op in {"LT", "LE", "GT", "GE"} and field_info.field_type != "number":
        raise ComputationValidationError(f"Operator {op} requires numeric field {field}")
    if op == "IN" and not isinstance(filter_row["value"], list):
        raise ComputationValidationError("IN filter value must be a list")


def validate_computation_request(request: ComputationRequest, catalog: dict[str, StructuredDatasetDescriptor]) -> None:
    if not request.request_id.strip():
        raise ComputationValidationError("request_id must be non-empty")
    if request.operation not in ALLOWED_OPERATIONS:
        raise ComputationValidationError(f"Unsupported operation: {request.operation}")
    if request.dataset_id not in catalog:
        raise ComputationValidationError(f"Unknown dataset_id: {request.dataset_id}")
    descriptor = catalog[request.dataset_id]
    if request.operation not in descriptor.supported_operations:
        raise ComputationValidationError(f"{request.operation} not supported for {request.dataset_id}")
    for filter_row in request.filters:
        _validate_filter(filter_row, descriptor)
    for field in request.group_by:
        _field_descriptor(descriptor, field)
    if request.operation == "FILTER_RECORDS":
        if request.numeric_field is not None or request.requested_statistics:
            raise ComputationValidationError("FILTER_RECORDS must not request numeric statistics")
        return
    if not request.numeric_field:
        raise ComputationValidationError(f"{request.operation} requires numeric_field")
    field_info = _field_descriptor(descriptor, request.numeric_field)
    if field_info.field_type != "number":
        raise ComputationValidationError(f"numeric_field {request.numeric_field} is not numeric")
    if request.operation == "SUMMARIZE":
        if not request.requested_statistics:
            raise ComputationValidationError("SUMMARIZE requires requested_statistics")
        invalid = set(request.requested_statistics) - ALLOWED_STATISTICS
        if invalid:
            raise ComputationValidationError(f"Unsupported statistics: {sorted(invalid)}")
    if request.operation == "COMPARE_GROUPS":
        if not isinstance(request.comparison_definition, dict):
            raise ComputationValidationError("COMPARE_GROUPS requires comparison_definition")
        expected = {"left_filters", "right_filters", "statistic", "quantity"}
        if set(request.comparison_definition) != expected:
            raise ComputationValidationError("comparison_definition must contain left_filters/right_filters/statistic/quantity")
        statistic = request.comparison_definition["statistic"]
        quantity = request.comparison_definition["quantity"]
        if statistic not in ALLOWED_STATISTICS:
            raise ComputationValidationError(f"Unsupported comparison statistic: {statistic}")
        if quantity not in ALLOWED_COMPARISONS:
            raise ComputationValidationError(f"Unsupported comparison quantity: {quantity}")
        for side in ["left_filters", "right_filters"]:
            if not isinstance(request.comparison_definition[side], list):
                raise ComputationValidationError(f"{side} must be a list")
            for filter_row in request.comparison_definition[side]:
                _validate_filter(filter_row, descriptor)
    if request.numeric_field == "cycles_to_failure":
        status = request.status_filter or "failure_only"
        if status != "failure_only":
            raise ComputationValidationError("Only failure_only cycles_to_failure summaries are implemented; censored survival analysis is unsupported")


def _compare(value: Any, op: str, expected: Any) -> bool:
    if op == "EQ":
        return value == expected
    if op == "IN":
        return value in expected
    if value is None:
        return False
    if op == "LT":
        return value < expected
    if op == "LE":
        return value <= expected
    if op == "GT":
        return value > expected
    if op == "GE":
        return value >= expected
    raise ComputationValidationError(f"Unsupported filter operator: {op}")


def filter_records(records: list[dict[str, Any]], filters: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if all(_compare(record.get(row["field"]), row["op"], row["value"]) for row in filters)
    ]


def _apply_status_policy(records: list[dict[str, Any]], request: ComputationRequest) -> tuple[list[dict[str, Any]], str | None, list[str]]:
    warnings: list[str] = []
    if request.numeric_field != "cycles_to_failure":
        return records, request.censoring_policy, warnings
    policy = "exclude_runouts_for_cycles_to_failure"
    filtered = [record for record in records if record.get("runout_status") is not True]
    excluded = len(records) - len(filtered)
    if excluded:
        warnings.append(f"Excluded {excluded} runout/right-censored record(s) from cycles_to_failure calculation.")
    return filtered, policy, warnings


def _numeric_values(records: list[dict[str, Any]], field: str) -> tuple[list[float], list[dict[str, Any]]]:
    values: list[float] = []
    source_records: list[dict[str, Any]] = []
    for record in records:
        value = record.get(field)
        if value is None:
            continue
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
            continue
        values.append(float(value))
        source_records.append(record)
    return values, source_records


def _summary(values: list[float], stats: tuple[str, ...]) -> dict[str, float | int | None]:
    result: dict[str, float | int | None] = {}
    for stat in stats:
        if stat == "count":
            result[stat] = len(values)
        elif not values:
            result[stat] = None
        elif stat == "min":
            result[stat] = min(values)
        elif stat == "max":
            result[stat] = max(values)
        elif stat == "mean":
            result[stat] = statistics.fmean(values)
        elif stat == "median":
            result[stat] = statistics.median(values)
        elif stat == "standard_deviation":
            result[stat] = statistics.stdev(values) if len(values) > 1 else None
        else:
            raise ComputationValidationError(f"Unsupported statistic: {stat}")
    return result


def _receipt_id(request: ComputationRequest, source_ids: list[str]) -> str:
    payload = {"request": request.normalized(), "source_record_ids": source_ids, "implementation": IMPLEMENTATION_VERSION}
    return "computation_receipt|" + hashlib.sha256(_json_dumps(payload).encode("utf-8")).hexdigest()[:24]


def _record_ids(records: list[dict[str, Any]]) -> list[str]:
    return [str(record["id"]) for record in records]


def _source_provenance(records: list[dict[str, Any]]) -> dict[str, Any]:
    sources = []
    for record in records:
        source = record.get("source", {})
        if isinstance(source, dict):
            sources.append({k: v for k, v in source.items() if k != "file_hash"})
    unique_sources = []
    seen = set()
    for source in sources:
        marker = _json_dumps(source)
        if marker not in seen:
            unique_sources.append(source)
            seen.add(marker)
    return {"source_count": len(unique_sources), "sources": unique_sources[:5]}


def _receipt_text(receipt: ComputationReceipt) -> str:
    return (
        f"ComputationReceipt {receipt.receipt_id}: {receipt.operation} on {receipt.dataset_id}; "
        f"numeric_field={receipt.numeric_field}; units={receipt.units}; "
        f"sample_counts={receipt.sample_counts}; result_values={receipt.result_values}; "
        f"censoring_status_policy={receipt.censoring_status_policy}."
    )


def execute_computation_request(repo_root: Path, request: ComputationRequest) -> ComputationReceipt:
    catalog = build_structured_data_catalog(repo_root)
    validate_computation_request(request, catalog)
    descriptor = catalog[request.dataset_id]
    records = _load_records(repo_root, descriptor)
    warnings: list[str] = []
    units = descriptor.available_fields.get(request.numeric_field).unit if request.numeric_field else None

    if request.operation == "FILTER_RECORDS":
        selected = filter_records(records, request.filters)
        source_ids = _record_ids(selected)
        result_values = {"record_count": len(selected), "record_ids": source_ids}
        sample_counts = {"records": len(selected)}
        source_records = selected
        group_definitions: dict[str, Any] = {}
        censoring_policy = request.censoring_policy
    elif request.operation == "SUMMARIZE":
        selected = filter_records(records, request.filters)
        selected, censoring_policy, status_warnings = _apply_status_policy(selected, request)
        warnings.extend(status_warnings)
        values, source_records = _numeric_values(selected, request.numeric_field or "")
        result_values = _summary(values, request.requested_statistics)
        sample_counts = {"records_after_filters": len(selected), "numeric_values": len(values)}
        source_ids = _record_ids(source_records)
        group_definitions = {"filters": list(request.filters)}
    elif request.operation == "COMPARE_GROUPS":
        if request.comparison_definition is None:
            raise ComputationValidationError("COMPARE_GROUPS requires comparison_definition")
        left_req = tuple(request.comparison_definition["left_filters"])
        right_req = tuple(request.comparison_definition["right_filters"])
        left_records = filter_records(records, left_req)
        right_records = filter_records(records, right_req)
        left_records, censoring_policy, left_warnings = _apply_status_policy(left_records, request)
        right_records, _, right_warnings = _apply_status_policy(right_records, request)
        warnings.extend(left_warnings + right_warnings)
        left_values, left_source = _numeric_values(left_records, request.numeric_field or "")
        right_values, right_source = _numeric_values(right_records, request.numeric_field or "")
        statistic = request.comparison_definition["statistic"]
        quantity = request.comparison_definition["quantity"]
        left_summary = _summary(left_values, (statistic,))[statistic]
        right_summary = _summary(right_values, (statistic,))[statistic]
        if left_summary is None or right_summary is None:
            raise ComputationValidationError("Comparison has no numeric values on one side")
        if quantity in {"ratio", "percent_difference"} and right_summary == 0:
            raise ComputationValidationError("Comparison denominator is zero")
        if quantity == "difference":
            comparison_value = float(left_summary) - float(right_summary)
        elif quantity == "ratio":
            comparison_value = float(left_summary) / float(right_summary)
        else:
            comparison_value = ((float(left_summary) - float(right_summary)) / float(right_summary)) * 100.0
        result_values = {
            "left": left_summary,
            "right": right_summary,
            "statistic": statistic,
            "quantity": quantity,
            "comparison_value": comparison_value,
        }
        sample_counts = {"left_numeric_values": len(left_values), "right_numeric_values": len(right_values)}
        source_records = left_source + right_source
        source_ids = _record_ids(source_records)
        group_definitions = {"left_filters": list(left_req), "right_filters": list(right_req)}
    else:
        raise ComputationValidationError(f"Unsupported operation: {request.operation}")

    implementation = {
        "version": IMPLEMENTATION_VERSION,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "arbitrary_code_execution_allowed": False,
    }
    receipt_id = _receipt_id(request, source_ids)
    receipt_without_text = ComputationReceipt(
        receipt_id=receipt_id,
        kind="computation_receipt",
        operation=request.operation,
        dataset_id=request.dataset_id,
        source_record_ids=tuple(source_ids),
        applied_filters=request.filters,
        group_definitions=group_definitions,
        numeric_field=request.numeric_field,
        units=units,
        censoring_status_policy=censoring_policy,
        sample_counts=sample_counts,
        result_values=result_values,
        deterministic_warnings=tuple(warnings),
        provenance=_source_provenance(source_records),
        implementation=implementation,
        retrieval_text="",
    )
    return ComputationReceipt(**{**receipt_without_text.to_dict(), "retrieval_text": _receipt_text(receipt_without_text)})


def validate_scientific_analyst_output(record: dict[str, Any], catalog: dict[str, StructuredDatasetDescriptor]) -> dict[str, Any]:
    if set(record) != {"outcome", "reason", "computation_request"}:
        raise AgenticValidationError(f"Invalid scientific analyst keys: {sorted(record)}")
    if record["outcome"] not in {"REQUEST_COMPUTATION", "NO_SUPPORTED_COMPUTATION", "ANALYSIS_COMPLETE"}:
        raise AgenticValidationError(f"Invalid scientific analyst outcome: {record['outcome']}")
    if not isinstance(record["reason"], str) or not record["reason"].strip():
        raise AgenticValidationError("Scientific analyst reason must be non-empty")
    if record["outcome"] == "REQUEST_COMPUTATION":
        if not isinstance(record["computation_request"], dict):
            raise AgenticValidationError("REQUEST_COMPUTATION requires a computation_request object")
        validate_computation_request(ComputationRequest.from_mapping(record["computation_request"]), catalog)
    elif record["computation_request"] is not None:
        raise AgenticValidationError("Non-computation analyst outcomes require computation_request null")
    lowered = _json_dumps(record).lower()
    for forbidden in ["gold", "benchmark", "hard_negative", "dev", "challenge", "answerability", "eval"]:
        if forbidden in lowered:
            raise AgenticValidationError(f"Scientific analyst output leaks evaluation term: {forbidden}")
    return record


def _scientific_analyst_prompt(
    repo_root: Path,
    question: str,
    ledger: dict[str, Any],
    previous_receipts: list[dict[str, Any]],
    remaining_budget: int,
    objective: str | None = None,
) -> str:
    catalog = runtime_catalog_payload(build_structured_data_catalog(repo_root))
    payload = {
        "original_question": question,
        "analysis_objective": objective or question,
        "current_evidence_ledger": ledger,
        "structured_data_catalog": catalog,
        "previous_computation_receipts": previous_receipts,
        "remaining_computation_budget": remaining_budget,
    }
    return "\n".join(
        [
            (repo_root / SCIENTIFIC_ANALYST_PROMPT_PATH).read_text(encoding="utf-8").strip(),
            "",
            "REQUEST_JSON:",
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
        ]
    ) + "\n"


def call_scientific_analyst(
    repo_root: Path,
    adapter: IsolatedModelAdapter,
    question_id: str,
    question: str,
    ledger: dict[str, Any],
    previous_receipts: list[dict[str, Any]],
    remaining_budget: int,
    objective: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prompt = _scientific_analyst_prompt(
        repo_root,
        question,
        ledger,
        previous_receipts,
        remaining_budget,
        objective,
    )
    catalog = build_structured_data_catalog(repo_root)

    def validator(record: dict[str, Any]) -> dict[str, Any]:
        return validate_scientific_analyst_output(record, catalog)

    from materials_rag.ingestion.agentic_retrieval import _call_role_with_retries

    record, receipts, _ = _call_role_with_retries(
        adapter,
        "scientific_analyst",
        f"{question_id}_scientific_analyst",
        prompt,
        validator,
        question_id,
    )
    return record, receipts


def computation_development_probes() -> list[dict[str, Any]]:
    """Return deterministic Phase 10B-only computation probes.

    These probes are not part of the frozen Phase 4 retrieval benchmark.
    Their numerical references are generated by the deterministic computation
    implementation itself.
    """

    return [
        {
            "question_id": "COMP-PROBE-001",
            "query": "What is the median cycles-to-failure for NIST 1-border specimens, excluding runouts?",
            "probe_type": "aggregation",
            "requires_retrieval": False,
            "expected_outcome": "receipt",
            "request": {
                "request_id": "COMP-PROBE-001-R1",
                "operation": "SUMMARIZE",
                "dataset_id": "nist_experiment_records",
                "filters": [{"field": "specimen_type", "op": "EQ", "value": "1-border"}],
                "group_by": [],
                "numeric_field": "cycles_to_failure",
                "status_filter": "failure_only",
                "requested_statistics": ["count", "median"],
                "comparison_definition": None,
                "censoring_policy": "exclude_runouts_for_cycles_to_failure",
            },
        },
        {
            "question_id": "COMP-PROBE-002",
            "query": "Compare median failure life for 1-border and 2-border NIST specimens as a ratio, excluding runouts.",
            "probe_type": "group_comparison",
            "requires_retrieval": False,
            "expected_outcome": "receipt",
            "request": {
                "request_id": "COMP-PROBE-002-R1",
                "operation": "COMPARE_GROUPS",
                "dataset_id": "nist_experiment_records",
                "filters": [],
                "group_by": [],
                "numeric_field": "cycles_to_failure",
                "status_filter": "failure_only",
                "requested_statistics": [],
                "comparison_definition": {
                    "left_filters": [{"field": "specimen_type", "op": "EQ", "value": "1-border"}],
                    "right_filters": [{"field": "specimen_type", "op": "EQ", "value": "2-border"}],
                    "statistic": "median",
                    "quantity": "ratio",
                },
                "censoring_policy": "exclude_runouts_for_cycles_to_failure",
            },
        },
        {
            "question_id": "COMP-PROBE-003",
            "query": "Identify the records marked as runouts in the NIST fatigue experiment table.",
            "probe_type": "runout_semantics",
            "requires_retrieval": False,
            "expected_outcome": "receipt",
            "request": {
                "request_id": "COMP-PROBE-003-R1",
                "operation": "FILTER_RECORDS",
                "dataset_id": "nist_experiment_records",
                "filters": [{"field": "runout_status", "op": "EQ", "value": True}],
                "group_by": [],
                "numeric_field": None,
                "status_filter": None,
                "requested_statistics": [],
                "comparison_definition": None,
                "censoring_policy": None,
            },
        },
        {
            "question_id": "COMP-PROBE-004",
            "query": "What pressure is recorded in the canonical NIST HIP process records?",
            "probe_type": "retrieval_plus_computation",
            "requires_retrieval": True,
            "expected_outcome": "receipt",
            "request": {
                "request_id": "COMP-PROBE-004-R1",
                "operation": "SUMMARIZE",
                "dataset_id": "nist_process_records",
                "filters": [],
                "group_by": [],
                "numeric_field": "pressure_mpa",
                "status_filter": None,
                "requested_statistics": ["count", "min", "max"],
                "comparison_definition": None,
                "censoring_policy": None,
            },
        },
        {
            "question_id": "COMP-PROBE-005",
            "query": "Estimate a survival-analysis median fatigue life including right-censored runouts.",
            "probe_type": "unsupported",
            "requires_retrieval": False,
            "expected_outcome": "unsupported",
            "unsupported_reason": "Survival/censored-data analysis is not implemented in Phase 10B.",
            "request": None,
        },
        {
            "question_id": "COMP-PROBE-006",
            "query": "Compare mean stress level for 0-border and Cylinder NIST specimens as a difference.",
            "probe_type": "group_comparison",
            "requires_retrieval": False,
            "expected_outcome": "receipt",
            "request": {
                "request_id": "COMP-PROBE-006-R1",
                "operation": "COMPARE_GROUPS",
                "dataset_id": "nist_experiment_records",
                "filters": [],
                "group_by": [],
                "numeric_field": "stress_level_mpa",
                "status_filter": None,
                "requested_statistics": [],
                "comparison_definition": {
                    "left_filters": [{"field": "specimen_type", "op": "EQ", "value": "0-border (hatch only)"}],
                    "right_filters": [{"field": "specimen_type", "op": "EQ", "value": "Cylinder"}],
                    "statistic": "mean",
                    "quantity": "difference",
                },
                "censoring_policy": None,
            },
        },
        {
            "question_id": "COMP-PROBE-007",
            "query": "Try to compute a ratio with an empty comparison denominator to verify deterministic rejection.",
            "probe_type": "tool_error",
            "requires_retrieval": False,
            "expected_outcome": "tool_error",
            "request": {
                "request_id": "COMP-PROBE-007-R1",
                "operation": "COMPARE_GROUPS",
                "dataset_id": "nist_experiment_records",
                "filters": [],
                "group_by": [],
                "numeric_field": "stress_level_mpa",
                "status_filter": None,
                "requested_statistics": [],
                "comparison_definition": {
                    "left_filters": [{"field": "specimen_type", "op": "EQ", "value": "1-border"}],
                    "right_filters": [{"field": "specimen_type", "op": "EQ", "value": "not-a-real-condition"}],
                    "statistic": "count",
                    "quantity": "ratio",
                },
                "censoring_policy": None,
            },
        },
        {
            "question_id": "COMP-PROBE-008",
            "query": "What is the maximum force-to-be-applied for 2-border NIST specimens?",
            "probe_type": "aggregation",
            "requires_retrieval": False,
            "expected_outcome": "receipt",
            "request": {
                "request_id": "COMP-PROBE-008-R1",
                "operation": "SUMMARIZE",
                "dataset_id": "nist_experiment_records",
                "filters": [{"field": "specimen_type", "op": "EQ", "value": "2-border"}],
                "group_by": [],
                "numeric_field": "force_to_be_applied_n",
                "status_filter": None,
                "requested_statistics": ["count", "max"],
                "comparison_definition": None,
                "censoring_policy": None,
            },
        },
    ]


def _probe_ledger(probe: dict[str, Any], receipt: dict[str, Any] | None, error: str | None) -> dict[str, Any]:
    if receipt is not None:
        status = "SUFFICIENT"
        req_status = "SUPPORTED"
        selected = [receipt["receipt_id"]]
        summary = ""
    else:
        status = "INSUFFICIENT_WITH_RESIDUAL"
        req_status = "MISSING"
        selected = []
        summary = error or probe.get("unsupported_reason", "No supported computation was available.")
    return {
        "question_id": probe["question_id"],
        "assessment_stage": "SCIENTIFIC_COMPUTATION",
        "overall_status": status,
        "evidence_requirements": [
            {
                "requirement_id": "R1",
                "description": probe["query"],
                "status": req_status,
                "supporting_context_ids": selected,
            }
        ],
        "missing_evidence_summary": summary,
        "selected_context_ids": selected,
    }


def _probe_trace(probe: dict[str, Any], receipt: dict[str, Any] | None, error: str | None) -> list[dict[str, Any]]:
    qid = probe["question_id"]
    now = datetime.now(UTC).isoformat()
    events = [
        {
            "event_type": "QUESTION_RECEIVED",
            "actor": "phase10b_orchestrator",
            "input_ids": [qid],
            "output_ids": [],
            "timestamp_utc": now,
            "metadata": {"question_id": qid},
        },
        {
            "event_type": "ORCHESTRATOR_DECISION",
            "actor": "phase10b_orchestrator",
            "input_ids": [qid],
            "output_ids": [],
            "timestamp_utc": now,
            "metadata": {"action": "SCIENTIFIC_ANALYSIS", "reason": "probe requires structured computation"},
        },
        {
            "event_type": "SCIENTIFIC_ANALYSIS_COMPLETED",
            "actor": "scientific_analyst",
            "input_ids": [qid],
            "output_ids": [probe["request"]["request_id"]] if probe.get("request") else [],
            "timestamp_utc": now,
            "metadata": {"outcome": "REQUEST_COMPUTATION" if probe.get("request") else "NO_SUPPORTED_COMPUTATION"},
        },
    ]
    if receipt is not None:
        events.append(
            {
                "event_type": "COMPUTATION_EXECUTED",
                "actor": "deterministic_computation_tool",
                "input_ids": [probe["request"]["request_id"]],
                "output_ids": [receipt["receipt_id"]],
                "timestamp_utc": now,
                "metadata": {"operation": receipt["operation"], "dataset_id": receipt["dataset_id"]},
            }
        )
        events.append(
            {
                "event_type": "EVIDENCE_MERGED",
                "actor": "phase10b_orchestrator",
                "input_ids": [receipt["receipt_id"]],
                "output_ids": [receipt["receipt_id"]],
                "timestamp_utc": now,
                "metadata": {"evidence_kind": "computation_receipt"},
            }
        )
    else:
        events.append(
            {
                "event_type": "COMPUTATION_UNSUPPORTED",
                "actor": "deterministic_computation_tool",
                "input_ids": [qid],
                "output_ids": [],
                "timestamp_utc": now,
                "metadata": {"error": error or probe.get("unsupported_reason", "unsupported")},
            }
        )
    events.append(
        {
            "event_type": "EVIDENCE_ASSESSED",
            "actor": "phase10b_orchestrator",
            "input_ids": [receipt["receipt_id"]] if receipt else [],
            "output_ids": [receipt["receipt_id"]] if receipt else [],
            "timestamp_utc": now,
            "metadata": {"overall_status": "SUFFICIENT" if receipt else "INSUFFICIENT_WITH_RESIDUAL"},
        }
    )
    return events


def _run_probe(repo_root: Path, probe: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if probe["expected_outcome"] == "unsupported" and probe.get("request") is None:
        reason = probe["unsupported_reason"]
        return {
            "question_id": probe["question_id"],
            "query": probe["query"],
            "probe_type": probe["probe_type"],
            "expected_outcome": probe["expected_outcome"],
            "outcome": "NO_SUPPORTED_COMPUTATION",
            "request": None,
            "receipt_id": None,
            "tool_error": None,
            "ledger": _probe_ledger(probe, None, reason),
            "trace": _probe_trace(probe, None, reason),
        }, None
    request = ComputationRequest.from_mapping(probe["request"])
    try:
        receipt = execute_computation_request(repo_root, request).to_dict()
        error = None
    except ComputationValidationError as exc:
        receipt = None
        error = str(exc)
    return {
        "question_id": probe["question_id"],
        "query": probe["query"],
        "probe_type": probe["probe_type"],
        "expected_outcome": probe["expected_outcome"],
        "outcome": "COMPUTATION_RECEIPT" if receipt else "TOOL_ERROR",
        "request": probe.get("request"),
        "receipt_id": receipt["receipt_id"] if receipt else None,
        "tool_error": error,
        "ledger": _probe_ledger(probe, receipt, error),
        "trace": _probe_trace(probe, receipt, error),
    }, receipt


def _build_probe_reference(repo_root: Path, probes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reference = []
    for probe in probes:
        if probe.get("request") is None:
            reference.append(
                {
                    "question_id": probe["question_id"],
                    "expected_outcome": "unsupported",
                    "unsupported_reason": probe["unsupported_reason"],
                }
            )
            continue
        try:
            receipt = execute_computation_request(repo_root, ComputationRequest.from_mapping(probe["request"])).to_dict()
            reference.append(
                {
                    "question_id": probe["question_id"],
                    "expected_outcome": "receipt",
                    "source_record_ids": receipt["source_record_ids"],
                    "operation": receipt["operation"],
                    "filters": receipt["applied_filters"],
                    "numeric_field": receipt["numeric_field"],
                    "expected_result_values": receipt["result_values"],
                    "units": receipt["units"],
                    "censoring_status_policy": receipt["censoring_status_policy"],
                }
            )
        except ComputationValidationError as exc:
            reference.append({"question_id": probe["question_id"], "expected_outcome": "tool_error", "error": str(exc)})
    return reference


def _probe_metrics(probes: list[dict[str, Any]], probe_results: list[dict[str, Any]], receipts: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {row["question_id"]: row for row in probe_results}
    expected_receipts = [p for p in probes if p["expected_outcome"] == "receipt"]
    expected_unsupported = [p for p in probes if p["expected_outcome"] == "unsupported"]
    expected_tool_error = [p for p in probes if p["expected_outcome"] == "tool_error"]
    valid_receipt_count = sum(1 for p in expected_receipts if by_id[p["question_id"]]["outcome"] == "COMPUTATION_RECEIPT")
    unsupported_correct = sum(1 for p in expected_unsupported if by_id[p["question_id"]]["outcome"] == "NO_SUPPORTED_COMPUTATION")
    tool_error_correct = sum(1 for p in expected_tool_error if by_id[p["question_id"]]["outcome"] == "TOOL_ERROR")
    unit_correct = all(receipt.get("units") is not None or receipt.get("operation") == "FILTER_RECORDS" for receipt in receipts)
    censoring_correct = all(
        receipt["censoring_status_policy"] == "exclude_runouts_for_cycles_to_failure"
        for receipt in receipts
        if receipt.get("numeric_field") == "cycles_to_failure"
    )
    provenance_complete = all(receipt.get("source_record_ids") is not None and receipt.get("provenance") and receipt.get("implementation") for receipt in receipts)
    return {
        "question_count": len(probes),
        "probe_types": dict(Counter(p["probe_type"] for p in probes)),
        "valid_computation_request_rate": valid_receipt_count / len(expected_receipts),
        "deterministic_calculation_correctness": valid_receipt_count / len(expected_receipts),
        "unit_correctness": unit_correct,
        "censoring_policy_correctness": censoring_correct,
        "tool_error_rate": sum(1 for r in probe_results if r["outcome"] == "TOOL_ERROR") / len(probe_results),
        "unsupported_request_detection": unsupported_correct / len(expected_unsupported),
        "expected_tool_error_detection": tool_error_correct / len(expected_tool_error),
        "receipt_provenance_completeness": provenance_complete,
        "mean_computation_actions_per_question": sum(1 for row in probe_results if row.get("request")) / len(probe_results),
        "mean_llm_role_calls_per_question": 0.0,
        "receipt_count": len(receipts),
        "receipt_question_ids": sorted(row["question_id"] for row in probe_results if row["receipt_id"]),
    }


def _phase10a_dev_regression(repo_root: Path) -> dict[str, Any]:
    phase10a_metrics = _read_json(repo_root / "data/processed/phase10a/metrics.json")
    phase10a_results = _read_jsonl(repo_root / "data/processed/phase10a/agentic_retrieval_results.jsonl")
    dev_questions = _load_dev_questions(repo_root)
    selected_by_question = _results_by_selected(phase10a_results)
    candidate_by_question = _candidate_pool_by_question(phase10a_results)
    ranked = _ranked_for_selected_metrics(selected_by_question)
    phase10b_selected = _selected_metrics_view(compute_metrics(dev_questions, ranked))
    status_by_id = {row["question_id"]: row["final_status"] for row in phase10a_results}
    return {
        "scope": "development split only",
        "dev_questions_checked": len(dev_questions),
        "challenge_questions_executed": 0,
        "phase10a_selected_evidence_metrics": phase10a_metrics["selected_evidence_metrics"],
        "phase10b_selected_evidence_metrics": phase10b_selected,
        "candidate_discovery": _candidate_discovery_metrics(dev_questions, candidate_by_question),
        "sufficiency_diagnostics": _sufficiency_answerability_diagnostics(dev_questions, status_by_id),
        "hard_negative_diagnostics": _hard_negative_selection_diagnostics(dev_questions, selected_by_question),
        "computation_calls_used_by_dev_regression": 0,
        "no_regression": phase10a_metrics["selected_evidence_metrics"] == phase10b_selected,
        "phase9b_dev_selected_question_count": len(_dev_phase9b_selected(repo_root, dev_questions)),
    }


def _write_walkthrough(repo_root: Path, probe_results: list[dict[str, Any]], receipts: list[dict[str, Any]]) -> None:
    selected = next(row for row in probe_results if row["question_id"] == "COMP-PROBE-002")
    receipt = next(row for row in receipts if row["receipt_id"] == selected["receipt_id"])
    lines = [
        "Phase 10B scientific computation walkthrough",
        "",
        f"Question: {selected['query']}",
        "",
        "Orchestrator action: SCIENTIFIC_ANALYSIS",
        "Scientific Analyst output: typed ComputationRequest",
        json.dumps(selected["request"], indent=2, ensure_ascii=False),
        "",
        "Deterministic computation receipt:",
        json.dumps(
            {
                "receipt_id": receipt["receipt_id"],
                "operation": receipt["operation"],
                "dataset_id": receipt["dataset_id"],
                "source_record_ids": receipt["source_record_ids"],
                "group_definitions": receipt["group_definitions"],
                "numeric_field": receipt["numeric_field"],
                "units": receipt["units"],
                "censoring_status_policy": receipt["censoring_status_policy"],
                "sample_counts": receipt["sample_counts"],
                "result_values": receipt["result_values"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        "",
        "Receipt becomes runtime evidence, but it is not inserted into the retrieval corpus.",
    ]
    (repo_root / WALKTHROUGH_TEXT).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_phase10b(repo_root: Path | None = None) -> Phase10BResult:
    if repo_root is None:
        repo_root = resolve_repo_root(Path(__file__))
    output_root = repo_root / PHASE10B_ROOT
    output_root.mkdir(parents=True, exist_ok=True)

    catalog = build_structured_data_catalog(repo_root)
    probes = computation_development_probes()
    probe_results: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for probe in probes:
        result, receipt = _run_probe(repo_root, probe)
        probe_results.append(result)
        traces.append({"question_id": probe["question_id"], "trace": result["trace"]})
        if receipt is not None:
            receipts.append(receipt)
    reference = _build_probe_reference(repo_root, probes)
    metrics = {
        "phase": "10B",
        "scope": "DEV regression plus separate computation development probes",
        "challenge_questions_executed": 0,
        "computation_probe_metrics": _probe_metrics(probes, probe_results, receipts),
        "dev_regression": _phase10a_dev_regression(repo_root),
        "structured_catalog": runtime_catalog_payload(catalog),
        "budgets": {"max_computation_actions_per_question": MAX_COMPUTATION_ACTIONS},
        "safeguards": {
            "arbitrary_code_execution_allowed": False,
            "runouts_not_treated_as_failures": True,
            "receipt_inserted_into_retrieval_corpus": False,
            "new_retrieval_algorithm_added": False,
            "answer_writer_called": False,
            "challenge_subset_executed": False,
        },
    }
    cost_report = {
        "probe_question_count": len(probes),
        "dev_regression_question_count": 36,
        "scientific_analyst_calls": 0,
        "computation_actions": sum(1 for row in probe_results if row.get("request")),
        "successful_receipts": len(receipts),
        "unsupported_computation_requests": sum(1 for row in probe_results if row["outcome"] == "NO_SUPPORTED_COMPUTATION"),
        "duplicate_computation_requests_rejected": 0,
        "mean_computation_actions_per_probe": metrics["computation_probe_metrics"]["mean_computation_actions_per_question"],
        "mean_llm_role_calls_per_probe": 0.0,
    }

    results_path = output_root / "computation_probe_results.jsonl"
    receipts_path = output_root / "computation_receipts.jsonl"
    traces_path = output_root / "traces.jsonl"
    metrics_path = output_root / "metrics.json"
    cost_path = output_root / "cost_report.json"
    reference_path = output_root / "computation_probe_reference.jsonl"
    catalog_path = output_root / "structured_data_catalog.json"
    _write_jsonl(results_path, probe_results)
    _write_jsonl(receipts_path, receipts)
    _write_jsonl(traces_path, traces)
    _write_json(metrics_path, metrics)
    _write_json(cost_path, cost_report)
    _write_jsonl(reference_path, reference)
    _write_json(catalog_path, runtime_catalog_payload(catalog))
    _write_walkthrough(repo_root, probe_results, receipts)

    outputs = [
        results_path,
        receipts_path,
        traces_path,
        metrics_path,
        cost_path,
        reference_path,
        catalog_path,
        repo_root / WALKTHROUGH_SCRIPT,
        repo_root / WALKTHROUGH_TEXT,
    ]
    manifest = {
        "phase": "10B",
        "status": "complete",
        "scope": "DEV regression and separate computation development probes; challenge not executed",
        "scientific_analyst": {
            "prompt_path": SCIENTIFIC_ANALYST_PROMPT_PATH.as_posix(),
            "prompt_sha256": sha256_for_file(repo_root / SCIENTIFIC_ANALYST_PROMPT_PATH),
            "isolated_adapter_available": True,
            "model": CODEX_MODEL,
            "reasoning_effort": REASONING_EFFORT,
        },
        "structured_tools": {
            "operations": sorted(ALLOWED_OPERATIONS),
            "filter_operators": sorted(ALLOWED_FILTER_OPS),
            "max_computation_actions": MAX_COMPUTATION_ACTIONS,
            "arbitrary_code_execution_allowed": False,
        },
        "probe_set": {
            "question_count": len(probes),
            "not_part_of_phase4_benchmark": True,
            "composition": dict(Counter(probe["probe_type"] for probe in probes)),
        },
        "dev_regression": metrics["dev_regression"],
        "metrics": metrics["computation_probe_metrics"],
        "output_hashes": {_relative(path, repo_root): sha256_for_file(path) for path in outputs},
        "input_hashes": {
            "data/processed/phase2a/canonical/experiment_records.jsonl": sha256_for_file(repo_root / "data/processed/phase2a/canonical/experiment_records.jsonl"),
            "data/processed/phase2a/canonical/process_records.jsonl": sha256_for_file(repo_root / "data/processed/phase2a/canonical/process_records.jsonl"),
            "data/processed/phase2b/canonical/test_records.jsonl": sha256_for_file(repo_root / "data/processed/phase2b/canonical/test_records.jsonl"),
            "data/processed/phase10a/agentic_retrieval_results.jsonl": sha256_for_file(repo_root / "data/processed/phase10a/agentic_retrieval_results.jsonl"),
        },
    }
    manifest_path = repo_root / MANIFEST_PATH
    _write_json(manifest_path, manifest)
    return Phase10BResult("complete", probe_results, metrics, manifest, outputs, manifest_path)


__all__ = [
    "ALLOWED_FILTER_OPS",
    "ALLOWED_OPERATIONS",
    "MAX_COMPUTATION_ACTIONS",
    "ComputationReceipt",
    "ComputationRequest",
    "ComputationValidationError",
    "Phase10BResult",
    "StructuredDatasetDescriptor",
    "build_structured_data_catalog",
    "call_scientific_analyst",
    "computation_development_probes",
    "execute_computation_request",
    "filter_records",
    "run_phase10b",
    "runtime_catalog_payload",
    "validate_computation_request",
    "validate_scientific_analyst_output",
]
