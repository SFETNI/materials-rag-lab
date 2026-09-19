"""Phase 10A bounded agentic retrieval orchestration."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sentence_transformers import SentenceTransformer

from materials_rag.ingestion.adaptive_routing import run_phase9a
from materials_rag.ingestion.corrective_retrieval import (
    EVIDENCE_STATUSES,
    _candidate_discovery_metrics,
    _hard_negative_selection_diagnostics,
    _ranked_for_selected_metrics,
    _selected_metrics_view,
    _sufficiency_answerability_diagnostics,
    validate_runtime_trace,
)
from materials_rag.ingestion.dense_retrieval import MODEL_NAME as BI_ENCODER_MODEL_NAME
from materials_rag.ingestion.dense_retrieval import QUERY_PREFIX, _encode_texts, compute_metrics
from materials_rag.ingestion.hybrid_retrieval import (
    BM25_TOP_K,
    DENSE_TOP_K,
    RRF_K,
    bm25_search,
    build_bm25_index,
    fuse_dense_bm25,
)
from materials_rag.ingestion.multi_query_retrieval import (
    HYBRID_CANDIDATE_K as MQ_HYBRID_CANDIDATE_K,
)
from materials_rag.ingestion.multi_query_retrieval import (
    _fuse_multi_query,
)
from materials_rag.ingestion.qdrant_retrieval import COLLECTION_NAME, open_qdrant, qdrant_search
from materials_rag.ingestion.query_decomposition_retrieval import (
    HYBRID_CANDIDATE_K as DECOMP_HYBRID_CANDIDATE_K,
)
from materials_rag.ingestion.query_decomposition_retrieval import (
    _fuse_decomposition,
)
from materials_rag.ingestion.utils import resolve_repo_root, sha256_for_file

PHASE10A_ROOT = Path("data/processed/phase10a")
MANIFEST_PATH = Path("data/processed/manifests/phase_10a_manifest.json")
ORCHESTRATOR_PROMPT_PATH = Path("docs/generation/agentic_orchestrator_v1.md")
ASSESSOR_PROMPT_PATH = Path("docs/generation/evidence_assessor_agentic_v1.md")
REWRITER_PROMPT_PATH = Path("docs/generation/runtime_multi_query_rewriter_v1.md")
DECOMPOSER_PROMPT_PATH = Path("docs/generation/runtime_query_decomposer_v1.md")
ROLE_OUTPUT_DIR = Path("experiments/phase10a_role_outputs")
WALKTHROUGH_SCRIPT = Path("experiments/21_agentic_retrieval_walkthrough.py")
WALKTHROUGH_TEXT = Path("experiments/21_agentic_retrieval_walkthrough.txt")
MAX_RETRIEVAL_ACTIONS = 4
MAX_ORCHESTRATOR_DECISIONS = 6
MAX_QUERY_TRANSFORM_CALLS = 2
MAX_COMPUTATION_ACTIONS = 2
ASSESSMENT_CONTEXT_K = 10
FINAL_CONTEXT_K = 5
CODEX_MODEL = "gpt-5.5"
REASONING_EFFORT = "low"
MAX_ATTEMPTS = 2
ACTIONS = {
    "HYBRID_SEARCH",
    "MULTI_QUERY_SEARCH",
    "DECOMPOSITION_SEARCH",
    "CHECK_REVISION_STATUS",
    "SCIENTIFIC_ANALYSIS",
    "FINISH_SUFFICIENT",
    "FINISH_WITH_RESIDUAL",
}
RETRIEVAL_ACTIONS = {"HYBRID_SEARCH", "MULTI_QUERY_SEARCH", "DECOMPOSITION_SEARCH"}
QUERY_TRANSFORM_ACTIONS = {"MULTI_QUERY_SEARCH", "DECOMPOSITION_SEARCH"}
LEAKAGE_TERMS = ["gold", "hard_negative", "answerability", "benchmark", "split", "oracle"]


class AgenticValidationError(ValueError):
    """Raised when an isolated role output violates a Phase 10A contract."""


class IsolatedModelError(RuntimeError):
    """Raised when an isolated Codex subprocess fails."""


@dataclass(frozen=True)
class AgentTool:
    name: str
    description: str


@dataclass(frozen=True)
class ModelReceipt:
    role: str
    call_id: str
    model: str
    reasoning_effort: str
    fresh_process: bool
    ephemeral: bool
    sandbox: str
    ignore_rules: bool
    ignore_user_config: bool
    working_directory_empty: bool
    tool_calls_detected: int
    event_count: int
    exit_code: int
    elapsed_seconds: float
    prompt_sha256: str
    output_sha256: str
    timestamp_utc: str


@dataclass
class CodexRunResult:
    output_text: str
    receipt: ModelReceipt
    stderr: str


@dataclass
class AgenticQuestionState:
    question_id: str
    original_question: str
    current_step: int = 0
    initial_route_suggestion: str | None = None
    actions_taken: list[dict[str, Any]] = field(default_factory=list)
    evidence_pool: list[dict[str, Any]] = field(default_factory=list)
    evidence_ledger: dict[str, Any] | None = None
    missing_evidence_summary: str = ""
    final_status: str | None = None
    trace: list[dict[str, Any]] = field(default_factory=list)
    cost_counters: dict[str, int] = field(default_factory=lambda: {
        "orchestrator_model_calls": 0,
        "evidence_assessor_calls": 0,
        "query_rewriter_calls": 0,
        "decomposer_calls": 0,
        "retrieval_actions": 0,
        "total_retrieved_candidates": 0,
        "duplicate_actions_rejected": 0,
        "revision_checks": 0,
        "scientific_analyst_calls": 0,
        "computation_actions": 0,
        "successful_computation_receipts": 0,
        "unsupported_computation_requests": 0,
        "duplicate_computation_requests_rejected": 0,
    })
    action_signatures: set[str] = field(default_factory=set)
    computation_signatures: set[str] = field(default_factory=set)
    computation_receipts: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Phase10AResult:
    status: str
    results: list[dict[str, Any]]
    metrics: dict[str, Any]
    manifest: dict[str, Any]
    outputs: list[Path]
    manifest_path: Path



def make_trace_event(
    event_type: str,
    actor: str,
    input_ids: list[str] | None = None,
    output_ids: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    allowed = {
        "QUESTION_RECEIVED",
        "INITIAL_ROUTE_SUGGESTED",
        "ORCHESTRATOR_DECISION",
        "TOOL_STARTED",
        "TOOL_COMPLETED",
        "EVIDENCE_MERGED",
        "EVIDENCE_ASSESSED",
        "REVISION_STATUS_CHECKED",
        "STOP_DECISION",
    }
    if event_type not in allowed:
        raise ValueError(f"Invalid Phase 10A trace event type: {event_type}")
    return {
        "event_type": event_type,
        "actor": actor,
        "input_ids": input_ids or [],
        "output_ids": output_ids or [],
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "metadata": metadata or {},
    }

def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records), encoding="utf-8")


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _relative(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False).lower()


def _single_json_object(text: str, question_id: str) -> dict[str, Any]:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if len(lines) != 1:
        raise AgenticValidationError(f"{question_id}: expected exactly one JSON line")
    try:
        record = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise AgenticValidationError(f"{question_id}: invalid JSON: {exc}") from exc
    if not isinstance(record, dict):
        raise AgenticValidationError(f"{question_id}: output is not an object")
    return record


class IsolatedModelAdapter:
    """Run one fresh local Codex process per role decision."""

    def __init__(self, repo_root: Path, output_dir: Path = ROLE_OUTPUT_DIR) -> None:
        self.repo_root = repo_root
        self.output_dir = repo_root / output_dir
        self.raw_dir = self.output_dir / "raw_outputs"
        self.receipt_dir = self.output_dir / "receipts"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.receipt_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def codex_command(temp_dir: Path) -> list[str]:
        return [
            "codex",
            "exec",
            "--json",
            "--ephemeral",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "--ignore-rules",
            "--ignore-user-config",
            "--model",
            CODEX_MODEL,
            "--config",
            f'model_reasoning_effort="{REASONING_EFFORT}"',
            "--cd",
            str(temp_dir),
            "-",
        ]

    @staticmethod
    def _text_from_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if isinstance(text, str):
                        parts.append(text)
            return "".join(parts)
        return ""

    @classmethod
    def _agent_message_from_event(cls, event: dict[str, Any]) -> str | None:
        event_type = str(event.get("type", ""))
        if event_type == "agent_message":
            for key in ["message", "text", "content"]:
                text = cls._text_from_content(event.get(key))
                if text:
                    return text
        item = event.get("item")
        if not isinstance(item, dict):
            return None
        item_type = str(item.get("type", ""))
        if item_type == "agent_message":
            return cls._text_from_content(item.get("text"))
        if event_type.endswith("completed") and item_type in {"message", "agent_message"} and item.get("role") == "assistant":
            return cls._text_from_content(item.get("content"))
        return None

    @staticmethod
    def _tool_call_marker(event: dict[str, Any]) -> str | None:
        event_type = str(event.get("type", "")).lower()
        if any(marker in event_type for marker in ["tool", "mcp", "web_search", "browser", "exec", "command"]):
            return event_type
        item = event.get("item")
        if isinstance(item, dict):
            item_type = str(item.get("type", "")).lower()
            if any(marker in item_type for marker in ["tool", "function_call", "mcp", "web_search", "browser", "exec", "command", "shell"]):
                return item_type
            for key in ["tool_name", "server", "command", "cmd", "function"]:
                if key in item:
                    return f"item.{key}"
        for key in ["tool_call", "tool_calls", "mcp", "command", "cmd", "function_call", "web_search"]:
            if key in event:
                return key
        return None

    @classmethod
    def parse_codex_json_events(cls, stdout: str) -> tuple[str, int, int]:
        messages: list[str] = []
        tool_calls = 0
        event_count = 0
        for line_no, line in enumerate(stdout.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise IsolatedModelError(f"Codex JSON event line {line_no} is invalid JSON: {exc}") from exc
            if not isinstance(event, dict):
                raise IsolatedModelError(f"Codex JSON event line {line_no} is not an object")
            event_count += 1
            if cls._tool_call_marker(event) is not None:
                tool_calls += 1
            message = cls._agent_message_from_event(event)
            if message is not None:
                messages.append(message)
        if not messages:
            raise IsolatedModelError("No final assistant message found in Codex JSON event stream")
        return messages[-1], tool_calls, event_count

    def run(self, role: str, call_id: str, prompt: str, timeout_seconds: int = 240) -> CodexRunResult:
        prompt_hash = _hash_text(prompt)
        with tempfile.TemporaryDirectory(prefix="phase10a_isolated_") as temp_name:
            temp_dir = Path(temp_name)
            started = time.perf_counter()
            process = subprocess.run(
                self.codex_command(temp_dir),
                input=prompt,
                text=True,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
                shell=False,
                check=False,
            )
            elapsed = time.perf_counter() - started
            output_text, tool_calls, event_count = self.parse_codex_json_events(process.stdout)
            if process.returncode != 0:
                raise IsolatedModelError(f"codex exec failed for {call_id}: {process.stderr}")
            if tool_calls:
                raise IsolatedModelError(f"isolated role {call_id} used {tool_calls} external tool events")
        output_hash = _hash_text(output_text)
        receipt = ModelReceipt(
            role=role,
            call_id=call_id,
            model=CODEX_MODEL,
            reasoning_effort=REASONING_EFFORT,
            fresh_process=True,
            ephemeral=True,
            sandbox="read-only",
            ignore_rules=True,
            ignore_user_config=True,
            working_directory_empty=True,
            tool_calls_detected=tool_calls,
            event_count=event_count,
            exit_code=process.returncode,
            elapsed_seconds=elapsed,
            prompt_sha256=prompt_hash,
            output_sha256=output_hash,
            timestamp_utc=datetime.now(UTC).isoformat(),
        )
        (self.raw_dir / f"{call_id}.jsonl").write_text(output_text, encoding="utf-8")
        _write_json(self.receipt_dir / f"{call_id}_receipt.json", {**receipt.__dict__, "stderr_present": bool(process.stderr.strip())})
        return CodexRunResult(output_text=output_text, receipt=receipt, stderr=process.stderr)


def _validate_orchestrator_action(record: dict[str, Any], question_id: str) -> dict[str, Any]:
    legacy_keys = {"action", "query", "document_id", "reason"}
    frozen_keys = {"action", "query", "objective", "document_id", "reason"}
    record_keys = set(record)
    if record_keys != legacy_keys and record_keys != frozen_keys:
        raise AgenticValidationError(f"{question_id}: invalid action keys {sorted(record)}")
    action = record["action"]
    if action not in ACTIONS:
        raise AgenticValidationError(f"{question_id}: unsupported tool action {action}")
    if action == "SCIENTIFIC_ANALYSIS" and record_keys != frozen_keys:
        raise AgenticValidationError(f"{question_id}: SCIENTIFIC_ANALYSIS requires the frozen objective field")
    for key in record_keys - {"action"}:
        if not isinstance(record[key], str):
            raise AgenticValidationError(f"{question_id}: {key} must be a string")
    objective = record.get("objective", "")
    if action in RETRIEVAL_ACTIONS and not record["query"].strip():
        raise AgenticValidationError(f"{question_id}: {action} requires non-empty query")
    if action in RETRIEVAL_ACTIONS and (objective.strip() or record["document_id"].strip()):
        raise AgenticValidationError(f"{question_id}: {action} forbids objective and document_id")
    if action == "CHECK_REVISION_STATUS" and not record["document_id"].strip():
        raise AgenticValidationError(f"{question_id}: CHECK_REVISION_STATUS requires document_id")
    if action == "CHECK_REVISION_STATUS" and (record["query"].strip() or objective.strip()):
        raise AgenticValidationError(f"{question_id}: CHECK_REVISION_STATUS forbids query and objective")
    if action == "SCIENTIFIC_ANALYSIS" and not objective.strip():
        raise AgenticValidationError(f"{question_id}: SCIENTIFIC_ANALYSIS requires non-empty objective")
    if action == "SCIENTIFIC_ANALYSIS" and (record["query"].strip() or record["document_id"].strip()):
        raise AgenticValidationError(f"{question_id}: SCIENTIFIC_ANALYSIS forbids query and document_id")
    if action in {"FINISH_SUFFICIENT", "FINISH_WITH_RESIDUAL"} and (
        record["query"].strip() or objective.strip() or record["document_id"].strip()
    ):
        raise AgenticValidationError(f"{question_id}: {action} forbids query, objective and document_id")
    if not record["reason"].strip():
        raise AgenticValidationError(f"{question_id}: reason must be non-empty")
    return record


def _validate_rewrite_record(record: dict[str, Any], question_id: str, question: str) -> None:
    if set(record) != {"question_id", "original_query", "rewrites"}:
        raise AgenticValidationError(f"{question_id}: invalid rewrite keys")
    if record["question_id"] != question_id or record["original_query"] != question:
        raise AgenticValidationError(f"{question_id}: rewrite question mismatch")
    rewrites = record["rewrites"]
    if not isinstance(rewrites, list) or len(rewrites) != 3:
        raise AgenticValidationError(f"{question_id}: expected exactly three rewrites")
    normalized = [str(item).strip() for item in rewrites]
    if any(not item for item in normalized) or len(set(normalized)) != 3 or question in normalized:
        raise AgenticValidationError(f"{question_id}: invalid rewrite content")
    record["rewrites"] = normalized


def _validate_decomposition_record(record: dict[str, Any], question_id: str, question: str) -> None:
    if set(record) != {"question_id", "original_question", "decomposable", "subqueries"}:
        raise AgenticValidationError(f"{question_id}: invalid decomposition keys")
    if record["question_id"] != question_id or record["original_question"] != question:
        raise AgenticValidationError(f"{question_id}: decomposition question mismatch")
    if not isinstance(record["decomposable"], bool):
        raise AgenticValidationError(f"{question_id}: decomposable must be boolean")
    subqueries = record["subqueries"]
    if not isinstance(subqueries, list):
        raise AgenticValidationError(f"{question_id}: subqueries must be list")
    normalized = [str(item).strip() for item in subqueries]
    if record["decomposable"]:
        if not 2 <= len(normalized) <= 4 or any(not item for item in normalized) or len(set(normalized)) != len(normalized) or question in normalized:
            raise AgenticValidationError(f"{question_id}: invalid decomposable subqueries")
    elif normalized:
        raise AgenticValidationError(f"{question_id}: atomic question must have no subqueries")
    record["subqueries"] = normalized


def _validate_agentic_assessment(record: dict[str, Any], question_id: str, allowed_context_ids: set[str]) -> dict[str, Any]:
    expected = {"question_id", "assessment_stage", "evidence_requirements", "overall_status", "missing_evidence_summary", "selected_context_ids"}
    if set(record) != expected:
        raise AgenticValidationError(f"{question_id}: invalid assessment keys {sorted(record)}")
    if record["question_id"] != question_id or record["assessment_stage"] != "AGENTIC":
        raise AgenticValidationError(f"{question_id}: invalid assessment identity/stage")
    if record["overall_status"] not in {"SUFFICIENT", "NEEDS_MORE_EVIDENCE"}:
        raise AgenticValidationError(f"{question_id}: invalid overall_status")
    requirements = record["evidence_requirements"]
    if not isinstance(requirements, list) or not requirements:
        raise AgenticValidationError(f"{question_id}: evidence_requirements must be non-empty")
    req_ids = []
    for req in requirements:
        if set(req) != {"requirement_id", "description", "status", "supporting_context_ids"}:
            raise AgenticValidationError(f"{question_id}: invalid requirement keys")
        req_ids.append(req["requirement_id"])
        if req["status"] not in EVIDENCE_STATUSES:
            raise AgenticValidationError(f"{question_id}: invalid requirement status")
        supporting = req["supporting_context_ids"]
        if not isinstance(supporting, list) or set(supporting) - allowed_context_ids:
            raise AgenticValidationError(f"{question_id}: invalid supporting IDs")
    if len(req_ids) != len(set(req_ids)):
        raise AgenticValidationError(f"{question_id}: duplicate requirement IDs")
    selected = record["selected_context_ids"]
    if not isinstance(selected, list) or len(selected) > FINAL_CONTEXT_K or len(selected) != len(set(selected)):
        raise AgenticValidationError(f"{question_id}: invalid selected_context_ids")
    if set(selected) - allowed_context_ids:
        raise AgenticValidationError(f"{question_id}: selected IDs outside allowed context")
    if not isinstance(record["missing_evidence_summary"], str):
        raise AgenticValidationError(f"{question_id}: missing_evidence_summary must be string")
    return record


def _call_role_with_retries(adapter: IsolatedModelAdapter, role: str, call_id: str, prompt: str, validator, question_id: str) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    receipts = []
    errors = []
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            result = adapter.run(role, f"{call_id}_attempt{attempt}", prompt)
            receipts.append(dict(result.receipt.__dict__))
            record = _single_json_object(result.output_text, question_id)
            return validator(record), receipts, attempt - 1
        except (AgenticValidationError, IsolatedModelError, subprocess.TimeoutExpired) as exc:
            errors.append(str(exc))
    raise AgenticValidationError(f"{question_id}: {role} failed after {MAX_ATTEMPTS} attempts: {errors}")


def _context_from_unit(unit: dict[str, Any]) -> dict[str, Any]:
    metadata = unit.get("metadata", {})
    return {
        "context_id": unit["id"],
        "kind": unit["kind"],
        "evidence_id": metadata.get("evidence_id"),
        "document_id": metadata.get("document_source_id") or metadata.get("document_id"),
        "source": {k: v for k, v in unit.get("source", {}).items() if k not in {"file_hash", "source_path"}},
        "retrieval_text": unit["retrieval_text"],
    }


def _tool_safe_result(row: dict[str, Any], unit: dict[str, Any], rank_key: str) -> dict[str, Any]:
    metadata = unit.get("metadata", {})
    return {
        "retrieval_unit_id": row["retrieval_unit_id"],
        "rank": int(row[rank_key]),
        "kind": unit["kind"],
        "evidence_id": metadata.get("evidence_id"),
        "document_id": metadata.get("document_source_id") or metadata.get("document_id"),
        "source": {k: v for k, v in unit.get("source", {}).items() if k not in {"file_hash", "source_path"}},
        "retrieval_text": unit["retrieval_text"],
        "internal_retrieval_metadata": {k: row.get(k) for k in ["rrf_score", "dense_rank", "dense_score", "bm25_rank", "bm25_score", "multi_query_rrf_score", "decomposition_rrf_score", "original_hybrid_rank", "rewrite_1_hybrid_rank", "rewrite_2_hybrid_rank", "rewrite_3_hybrid_rank", "subquery_1_hybrid_rank", "subquery_2_hybrid_rank", "subquery_3_hybrid_rank", "subquery_4_hybrid_rank"] if k in row},
    }


class AgenticToolbox:
    """Runtime tools composed from existing lower-level retrieval components."""

    def __init__(self, repo_root: Path, adapter: IsolatedModelAdapter | None = None) -> None:
        self.repo_root = repo_root
        self.adapter = adapter
        self.units = _read_jsonl(repo_root / "data/processed/phase4_5/retrieval_units.jsonl")
        self.units_by_id = {unit["id"]: unit for unit in self.units}
        if len(self.units) != 277:
            raise ValueError(f"Phase 10A expected 277 retrieval units, got {len(self.units)}")
        self.bm25_index = build_bm25_index(self.units)
        self.encoder = SentenceTransformer(BI_ENCODER_MODEL_NAME, local_files_only=True)
        self.encoder.eval()
        self.qdrant_client = open_qdrant(repo_root / "data/processed/phase5b/qdrant_storage")

    def close(self) -> None:
        self.qdrant_client.close()

    def _dense_search(self, query: str, top_k: int = DENSE_TOP_K) -> list[dict[str, Any]]:
        query_vector = _encode_texts(self.encoder, [QUERY_PREFIX + query])[0]
        return qdrant_search(self.qdrant_client, COLLECTION_NAME, query_vector, top_k)

    def hybrid_search(self, query: str, k: int = 10, question_id: str = "runtime") -> dict[str, Any]:
        if not query.strip():
            raise AgenticValidationError("HYBRID_SEARCH query must be non-empty")
        dense = self._dense_search(query, DENSE_TOP_K)
        bm25 = bm25_search(self.bm25_index, query, BM25_TOP_K)
        fused = fuse_dense_bm25({"question_id": question_id, "query": query}, dense, bm25, self.units_by_id, RRF_K)
        fused["results"] = fused["results"][:k]
        return fused

    def _rewrite_query(self, question_id: str, question: str) -> dict[str, Any]:
        if self.adapter is None:
            raise AgenticValidationError("runtime rewriter requires an isolated model adapter")
        prompt = "\n\n".join([
            (self.repo_root / REWRITER_PROMPT_PATH).read_text(encoding="utf-8").strip(),
            "REQUEST_JSON:",
            json.dumps({"question_id": question_id, "original_query": question}, ensure_ascii=False, sort_keys=True),
        ])
        result = self.adapter.run("query_rewriter", f"{question_id}_rewrite", prompt)
        record = _single_json_object(result.output_text, question_id)
        _validate_rewrite_record(record, question_id, question)
        return record

    def multi_query_search(self, question_id: str, question: str, k: int = 10) -> tuple[dict[str, Any], dict[str, Any]]:
        rewrites = self._rewrite_query(question_id, question)
        hybrid_by_variant = {}
        for variant, query in [("original", question), ("rewrite_1", rewrites["rewrites"][0]), ("rewrite_2", rewrites["rewrites"][1]), ("rewrite_3", rewrites["rewrites"][2])]:
            hybrid_by_variant[f"{question_id}::{variant}"] = self.hybrid_search(query, MQ_HYBRID_CANDIDATE_K, f"{question_id}::{variant}")
        fused = _fuse_multi_query({"question_id": question_id, "original_query": question, "rewrites": rewrites["rewrites"]}, hybrid_by_variant)
        fused["results"] = fused["results"][:k]
        return fused, rewrites

    def _decompose_query(self, question_id: str, question: str) -> dict[str, Any]:
        if self.adapter is None:
            raise AgenticValidationError("runtime decomposer requires an isolated model adapter")
        prompt = "\n\n".join([
            (self.repo_root / DECOMPOSER_PROMPT_PATH).read_text(encoding="utf-8").strip(),
            "REQUEST_JSON:",
            json.dumps({"question_id": question_id, "original_question": question}, ensure_ascii=False, sort_keys=True),
        ])
        result = self.adapter.run("query_decomposer", f"{question_id}_decompose", prompt)
        record = _single_json_object(result.output_text, question_id)
        _validate_decomposition_record(record, question_id, question)
        return record

    def decomposition_search(self, question_id: str, question: str, k: int = 10) -> tuple[dict[str, Any], dict[str, Any]]:
        decomposition = self._decompose_query(question_id, question)
        variants = [{"variant": "original", "query": question}]
        if decomposition["decomposable"]:
            variants.extend({"variant": f"subquery_{i}", "query": q} for i, q in enumerate(decomposition["subqueries"], start=1))
        hybrid_by_variant = {f"{question_id}::{row['variant']}": self.hybrid_search(row["query"], DECOMP_HYBRID_CANDIDATE_K, f"{question_id}::{row['variant']}") for row in variants}
        fused = _fuse_decomposition(decomposition, hybrid_by_variant)
        fused["results"] = fused["results"][:k]
        return fused, decomposition

    def check_revision_status(self, document_id_or_family: str) -> dict[str, Any]:
        query = document_id_or_family.strip()
        if not query:
            raise AgenticValidationError("CHECK_REVISION_STATUS requires a document id or family")
        matches = []
        seen = set()
        for unit in self.units:
            metadata = unit.get("metadata", {})
            doc_id = metadata.get("document_source_id") or metadata.get("document_id")
            family = metadata.get("revision_family_id")
            if query not in {doc_id, family}:
                continue
            if doc_id in seen:
                continue
            seen.add(doc_id)
            matches.append({
                "document_id": doc_id,
                "revision_family_id": family,
                "revision": metadata.get("revision"),
                "is_current": metadata.get("is_current"),
                "supersedes": metadata.get("supersedes"),
                "superseded_by": metadata.get("superseded_by"),
                "issued_on": metadata.get("issued_on"),
                "effective_from": metadata.get("effective_from"),
                "effective_to": metadata.get("effective_to"),
                "authority": metadata.get("authority") or unit.get("source", {}).get("authority"),
            })
        matches.sort(key=lambda row: (str(row.get("revision_family_id")), row.get("revision") or 0, str(row.get("document_id"))))
        return {"query": query, "matches": matches}


def _merge_evidence(pool: list[dict[str, Any]], tool_results: list[dict[str, Any]], units_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    seen = {item["context_id"] for item in pool}
    merged = list(pool)
    for result in sorted(tool_results, key=lambda row: row["rank"]):
        unit_id = result["retrieval_unit_id"]
        if unit_id in seen:
            continue
        merged.append(_context_from_unit(units_by_id[unit_id]))
        seen.add(unit_id)
    return merged


def _action_signature(action: dict[str, Any]) -> str:
    if action["action"] == "CHECK_REVISION_STATUS":
        value = action["document_id"]
    elif action["action"] == "SCIENTIFIC_ANALYSIS":
        value = action.get("objective", "")
    else:
        value = action["query"]
    return f"{action['action']}::{value.strip().lower()}"


def _initial_action_from_route(route: str, question: str) -> dict[str, Any]:
    return {
        "action": {"HYBRID": "HYBRID_SEARCH", "MULTI_QUERY": "MULTI_QUERY_SEARCH", "DECOMPOSITION": "DECOMPOSITION_SEARCH"}.get(route, "HYBRID_SEARCH"),
        "query": question,
        "document_id": "",
        "reason": "Initial action follows the Phase 9A question-only route suggestion.",
    }


def _assessment_prompt(repo_root: Path, question_id: str, question: str, evidence_pool: list[dict[str, Any]]) -> str:
    allowed = [item["context_id"] for item in evidence_pool]
    packet = {"question_id": question_id, "original_question": question, "evidence_pool": evidence_pool}
    lines = [
        (repo_root / ASSESSOR_PROMPT_PATH).read_text(encoding="utf-8").strip(),
        "",
        "ALLOWED_CONTEXT_IDS:",
        *[f"- {context_id}" for context_id in allowed],
        "",
        "supporting_context_ids and selected_context_ids may contain ONLY IDs copied verbatim from ALLOWED_CONTEXT_IDS.",
        "Return ONLY one JSON object on one line.",
        "",
        "REQUEST_JSON:",
        json.dumps(packet, ensure_ascii=False, sort_keys=True),
    ]
    return "\n".join(lines) + "\n"


def _orchestrator_prompt(repo_root: Path, state: AgenticQuestionState) -> str:
    remaining = {
        "retrieval_actions": MAX_RETRIEVAL_ACTIONS - state.cost_counters["retrieval_actions"],
        "orchestrator_decisions": MAX_ORCHESTRATOR_DECISIONS - state.cost_counters["orchestrator_model_calls"],
        "query_transform_calls": MAX_QUERY_TRANSFORM_CALLS - state.cost_counters["query_rewriter_calls"] - state.cost_counters["decomposer_calls"],
        "computation_actions": MAX_COMPUTATION_ACTIONS - state.cost_counters["computation_actions"],
    }
    safe_state = {
        "question_id": state.question_id,
        "original_question": state.original_question,
        "initial_route_suggestion": state.initial_route_suggestion,
        "current_evidence_ledger": state.evidence_ledger,
        "missing_evidence_summary": state.missing_evidence_summary,
        "previous_actions": state.actions_taken,
        "remaining_budget": remaining,
    }
    return "\n".join([
        (repo_root / ORCHESTRATOR_PROMPT_PATH).read_text(encoding="utf-8").strip(),
        "",
        "RUNTIME_STATE_JSON:",
        json.dumps(safe_state, ensure_ascii=False, sort_keys=True),
    ]) + "\n"


def assess_evidence(repo_root: Path, adapter: IsolatedModelAdapter, state: AgenticQuestionState) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    prompt = _assessment_prompt(repo_root, state.question_id, state.original_question, state.evidence_pool)
    allowed = {item["context_id"] for item in state.evidence_pool}

    def validator(record: dict[str, Any]) -> dict[str, Any]:
        return _validate_agentic_assessment(record, state.question_id, allowed)

    record, receipts, retries = _call_role_with_retries(adapter, "evidence_assessor", f"{state.question_id}_assess_{state.current_step}", prompt, validator, state.question_id)
    state.cost_counters["evidence_assessor_calls"] += 1 + retries
    return record, receipts, retries


def decide_next_action(repo_root: Path, adapter: IsolatedModelAdapter, state: AgenticQuestionState) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    prompt = _orchestrator_prompt(repo_root, state)

    def validator(record: dict[str, Any]) -> dict[str, Any]:
        return _validate_orchestrator_action(record, state.question_id)

    record, receipts, retries = _call_role_with_retries(adapter, "orchestrator", f"{state.question_id}_orchestrator_{state.current_step}", prompt, validator, state.question_id)
    state.cost_counters["orchestrator_model_calls"] += 1 + retries
    return record, receipts, retries


def _execute_tool(toolbox: AgenticToolbox, state: AgenticQuestionState, action: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    name = action["action"]
    if name == "HYBRID_SEARCH":
        row = toolbox.hybrid_search(action["query"], ASSESSMENT_CONTEXT_K, state.question_id)
        safe = [_tool_safe_result(result, toolbox.units_by_id[result["retrieval_unit_id"]], "hybrid_rank") for result in row["results"]]
        return safe, {"tool": name, "query": action["query"], "result_count": len(safe)}
    if name == "MULTI_QUERY_SEARCH":
        row, rewrites = toolbox.multi_query_search(state.question_id, action["query"], ASSESSMENT_CONTEXT_K)
        state.cost_counters["query_rewriter_calls"] += 1
        safe = [_tool_safe_result(result, toolbox.units_by_id[result["retrieval_unit_id"]], "multi_query_rank") for result in row["results"]]
        return safe, {"tool": name, "query": action["query"], "result_count": len(safe), "rewrite_count": len(rewrites["rewrites"])}
    if name == "DECOMPOSITION_SEARCH":
        row, decomp = toolbox.decomposition_search(state.question_id, action["query"], ASSESSMENT_CONTEXT_K)
        state.cost_counters["decomposer_calls"] += 1
        safe = [_tool_safe_result(result, toolbox.units_by_id[result["retrieval_unit_id"]], "decomposition_rank") for result in row["results"]]
        return safe, {"tool": name, "query": action["query"], "result_count": len(safe), "decomposable": decomp["decomposable"], "subquery_count": len(decomp["subqueries"])}
    if name == "CHECK_REVISION_STATUS":
        status = toolbox.check_revision_status(action["document_id"])
        state.cost_counters["revision_checks"] += 1
        return [], {"tool": name, "document_id": action["document_id"], "revision_status": status}
    raise AgenticValidationError(f"Unsupported executable action: {name}")


def _computation_context(receipt: dict[str, Any]) -> dict[str, Any]:
    return {
        "context_id": receipt["receipt_id"],
        "kind": "computation_receipt",
        "evidence_id": None,
        "document_id": None,
        "source": {
            "dataset_id": receipt["dataset_id"],
            "type": "deterministic_computation",
        },
        "retrieval_text": receipt["retrieval_text"],
        "receipt": receipt,
    }


def _execute_scientific_action(
    toolbox: AgenticToolbox,
    state: AgenticQuestionState,
    action: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], bool]:
    from materials_rag.ingestion.scientific_computation import (
        ComputationRequest,
        call_scientific_analyst,
        execute_computation_request,
    )

    if toolbox.adapter is None:
        raise AgenticValidationError("SCIENTIFIC_ANALYSIS requires an isolated model adapter")
    ledger = state.evidence_ledger or {
        "overall_status": "NEEDS_MORE_EVIDENCE",
        "evidence_requirements": [],
        "missing_evidence_summary": state.missing_evidence_summary,
        "selected_context_ids": [],
    }
    analyst, role_receipts = call_scientific_analyst(
        toolbox.repo_root,
        toolbox.adapter,
        state.question_id,
        state.original_question,
        ledger,
        state.computation_receipts,
        MAX_COMPUTATION_ACTIONS - state.cost_counters["computation_actions"],
        action["objective"],
    )
    state.cost_counters["scientific_analyst_calls"] += len(role_receipts)
    if analyst["outcome"] != "REQUEST_COMPUTATION":
        state.cost_counters["unsupported_computation_requests"] += 1
        return {
            "tool": "SCIENTIFIC_ANALYSIS",
            "objective": action["objective"],
            "outcome": analyst["outcome"],
            "reason": analyst["reason"],
        }, role_receipts, False

    request = ComputationRequest.from_mapping(analyst["computation_request"])
    signature = json.dumps(
        request.normalized(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    if signature in state.computation_signatures:
        state.cost_counters["duplicate_computation_requests_rejected"] += 1
        return {
            "tool": "SCIENTIFIC_ANALYSIS",
            "objective": action["objective"],
            "outcome": "DUPLICATE_COMPUTATION_REJECTED",
        }, role_receipts, False
    state.computation_signatures.add(signature)
    receipt = execute_computation_request(toolbox.repo_root, request).to_dict()
    state.cost_counters["computation_actions"] += 1
    state.cost_counters["successful_computation_receipts"] += 1
    state.computation_receipts.append(receipt)
    context = _computation_context(receipt)
    if context["context_id"] not in {item["context_id"] for item in state.evidence_pool}:
        state.evidence_pool.append(context)
    return {
        "tool": "SCIENTIFIC_ANALYSIS",
        "objective": action["objective"],
        "outcome": "COMPUTATION_RECEIPT",
        "receipt_id": receipt["receipt_id"],
    }, role_receipts, True


def run_agentic_question(repo_root: Path, toolbox: AgenticToolbox, adapter: IsolatedModelAdapter, question: dict[str, Any], route_suggestion: str) -> dict[str, Any]:
    qid = question["question_id"]
    query = question["query"]
    state = AgenticQuestionState(question_id=qid, original_question=query, initial_route_suggestion=route_suggestion)
    state.trace.append(make_trace_event("QUESTION_RECEIVED", "phase10a_orchestrator", input_ids=[qid], metadata={"question_id": qid}))
    state.trace.append(make_trace_event("INITIAL_ROUTE_SUGGESTED", "phase9a_router", input_ids=[qid], metadata={"strategy": route_suggestion}))
    receipts = []
    duplicate_observations = []
    while state.current_step < MAX_ORCHESTRATOR_DECISIONS:
        state.current_step += 1
        if state.evidence_ledger is not None and state.evidence_ledger["overall_status"] == "SUFFICIENT":
            state.final_status = "SUFFICIENT"
            state.trace.append(make_trace_event("STOP_DECISION", "phase10a_orchestrator", metadata={"final_status": state.final_status, "reason": "ledger sufficient"}))
            break
        action, action_receipts, _ = decide_next_action(repo_root, adapter, state)
        receipts.extend(action_receipts)
        state.trace.append(make_trace_event("ORCHESTRATOR_DECISION", "isolated_orchestrator", metadata={"action": action["action"], "reason": action["reason"]}))
        if action["action"] == "FINISH_SUFFICIENT":
            state.final_status = "SUFFICIENT"
            state.trace.append(make_trace_event("STOP_DECISION", "phase10a_orchestrator", metadata={"final_status": state.final_status, "reason": action["reason"]}))
            break
        if action["action"] == "FINISH_WITH_RESIDUAL":
            state.final_status = "INSUFFICIENT_WITH_RESIDUAL"
            state.trace.append(make_trace_event("STOP_DECISION", "phase10a_orchestrator", metadata={"final_status": state.final_status, "reason": action["reason"]}))
            break
        if action["action"] in RETRIEVAL_ACTIONS and state.cost_counters["retrieval_actions"] >= MAX_RETRIEVAL_ACTIONS:
            state.final_status = "INSUFFICIENT_WITH_RESIDUAL"
            state.trace.append(make_trace_event("STOP_DECISION", "phase10a_orchestrator", metadata={"final_status": state.final_status, "reason": "retrieval budget exhausted"}))
            break
        if action["action"] in QUERY_TRANSFORM_ACTIONS and state.cost_counters["query_rewriter_calls"] + state.cost_counters["decomposer_calls"] >= MAX_QUERY_TRANSFORM_CALLS:
            state.actions_taken.append({**action, "observation": "query transform budget exhausted"})
            action = {"action": "HYBRID_SEARCH", "query": action["query"], "document_id": "", "reason": "Fallback to Hybrid because query-transform budget is exhausted."}
        if action["action"] == "SCIENTIFIC_ANALYSIS" and state.cost_counters["computation_actions"] >= MAX_COMPUTATION_ACTIONS:
            state.final_status = "INSUFFICIENT_WITH_RESIDUAL"
            state.trace.append(make_trace_event("STOP_DECISION", "phase10a_orchestrator", metadata={"final_status": state.final_status, "reason": "computation budget exhausted"}))
            break
        signature = _action_signature(action)
        if signature in state.action_signatures:
            state.cost_counters["duplicate_actions_rejected"] += 1
            duplicate_observations.append({"step": state.current_step, "action_signature": signature})
            state.actions_taken.append({**action, "observation": "duplicate action rejected"})
            continue
        state.action_signatures.add(signature)
        state.trace.append(make_trace_event("TOOL_STARTED", "agent_tool", metadata={"tool": action["action"]}))
        if action["action"] == "SCIENTIFIC_ANALYSIS":
            observation, scientific_receipts, evidence_added = _execute_scientific_action(
                toolbox, state, action
            )
            receipts.extend(scientific_receipts)
            results = []
        else:
            results, observation = _execute_tool(toolbox, state, action)
            evidence_added = False
        state.actions_taken.append({**action, "observation": observation})
        if action["action"] in RETRIEVAL_ACTIONS:
            state.cost_counters["retrieval_actions"] += 1
            state.cost_counters["total_retrieved_candidates"] += len(results)
            state.evidence_pool = _merge_evidence(state.evidence_pool, results, toolbox.units_by_id)
            state.trace.append(make_trace_event("TOOL_COMPLETED", "agent_tool", output_ids=[r["retrieval_unit_id"] for r in results], metadata={"tool": action["action"], "result_count": len(results)}))
            state.trace.append(make_trace_event("EVIDENCE_MERGED", "phase10a_orchestrator", output_ids=[item["context_id"] for item in state.evidence_pool], metadata={"evidence_pool_size": len(state.evidence_pool)}))
            assessment, assessment_receipts, _ = assess_evidence(repo_root, adapter, state)
            receipts.extend(assessment_receipts)
            state.evidence_ledger = assessment
            state.missing_evidence_summary = assessment["missing_evidence_summary"]
            state.trace.append(make_trace_event("EVIDENCE_ASSESSED", "isolated_evidence_assessor_agentic_v1", input_ids=[item["context_id"] for item in state.evidence_pool], output_ids=assessment["selected_context_ids"], metadata={"overall_status": assessment["overall_status"], "requirement_statuses": [req["status"] for req in assessment["evidence_requirements"]]}))
        elif action["action"] == "SCIENTIFIC_ANALYSIS":
            state.trace.append(
                make_trace_event(
                    "TOOL_COMPLETED",
                    "deterministic_computation_tool",
                    output_ids=[observation["receipt_id"]]
                    if observation.get("receipt_id")
                    else [],
                    metadata=observation,
                )
            )
            if evidence_added:
                state.trace.append(
                    make_trace_event(
                        "EVIDENCE_MERGED",
                        "phase10b_orchestrator",
                        output_ids=[observation["receipt_id"]],
                        metadata={"evidence_pool_size": len(state.evidence_pool)},
                    )
                )
                assessment, assessment_receipts, _ = assess_evidence(
                    repo_root, adapter, state
                )
                receipts.extend(assessment_receipts)
                state.evidence_ledger = assessment
                state.missing_evidence_summary = assessment["missing_evidence_summary"]
                state.trace.append(
                    make_trace_event(
                        "EVIDENCE_ASSESSED",
                        "isolated_evidence_assessor_agentic_v1",
                        input_ids=[item["context_id"] for item in state.evidence_pool],
                        output_ids=assessment["selected_context_ids"],
                        metadata={
                            "overall_status": assessment["overall_status"],
                            "requirement_statuses": [
                                req["status"]
                                for req in assessment["evidence_requirements"]
                            ],
                        },
                    )
                )
        else:
            state.trace.append(make_trace_event("REVISION_STATUS_CHECKED", "revision_status_tool", metadata=observation))
    if state.final_status is None:
        state.final_status = "SUFFICIENT" if state.evidence_ledger and state.evidence_ledger["overall_status"] == "SUFFICIENT" else "INSUFFICIENT_WITH_RESIDUAL"
        state.trace.append(make_trace_event("STOP_DECISION", "phase10a_orchestrator", metadata={"final_status": state.final_status, "reason": "decision budget exhausted"}))
    selected = list(state.evidence_ledger["selected_context_ids"][:FINAL_CONTEXT_K]) if state.evidence_ledger else []
    return {
        "question_id": qid,
        "query": query,
        "initial_route_suggestion": route_suggestion,
        "actions_taken": state.actions_taken,
        "evidence_pool_ids": [item["context_id"] for item in state.evidence_pool],
        "evidence_pool_size": len(state.evidence_pool),
        "final_evidence_ledger": state.evidence_ledger,
        "missing_evidence_summary": state.missing_evidence_summary,
        "final_status": state.final_status,
        "final_selected_context_ids": selected,
        "trace": state.trace,
        "cost_counters": state.cost_counters,
        "role_receipts": receipts,
        "computation_receipts": state.computation_receipts,
        "duplicate_action_observations": duplicate_observations,
    }


def _load_dev_questions(repo_root: Path) -> list[dict[str, Any]]:
    return [q for q in _read_jsonl(repo_root / "data/processed/phase4/retrieval_eval.jsonl") if q.get("split") == "dev"]


def _load_route_suggestions(repo_root: Path) -> dict[str, str]:
    return {row["question_id"]: row["selected_strategy"] for row in _read_jsonl(repo_root / "data/processed/phase9a/adaptive_results.jsonl")}


def _results_by_selected(results: list[dict[str, Any]]) -> dict[str, list[str]]:
    return {row["question_id"]: row["final_selected_context_ids"] for row in results}


def _candidate_pool_by_question(results: list[dict[str, Any]]) -> dict[str, list[str]]:
    return {row["question_id"]: row["evidence_pool_ids"] for row in results}


def _dev_phase9b_selected(repo_root: Path, dev_questions: list[dict[str, Any]]) -> dict[str, list[str]]:
    dev_ids = {q["question_id"] for q in dev_questions}
    return {row["question_id"]: row["final_selected_context_ids"] for row in _read_jsonl(repo_root / "data/processed/phase9b/corrective_results.jsonl") if row["question_id"] in dev_ids}


def _runtime_cost_report(results: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(results)
    totals = Counter()
    for row in results:
        totals.update(row["cost_counters"])
    return {
        "question_count": count,
        "totals": dict(sorted(totals.items())),
        "means": {k: v / count if count else 0.0 for k, v in sorted(totals.items())},
        "percentage_solved_after_first_retrieval": sum(row["final_status"] == "SUFFICIENT" and row["cost_counters"]["retrieval_actions"] <= 1 for row in results) / count if count else 0.0,
        "percentage_requiring_more_than_one_retrieval_action": sum(row["cost_counters"]["retrieval_actions"] > 1 for row in results) / count if count else 0.0,
        "percentage_stopping_with_residual": sum(row["final_status"] == "INSUFFICIENT_WITH_RESIDUAL" for row in results) / count if count else 0.0,
        "duplicate_action_rate": totals["duplicate_actions_rejected"] / max(1, totals["orchestrator_model_calls"]),
    }


def _metrics_for_dev(repo_root: Path, dev_questions: list[dict[str, Any]], results: list[dict[str, Any]]) -> dict[str, Any]:
    selected_by_id = _results_by_selected(results)
    candidate_by_id = _candidate_pool_by_question(results)
    selected_metrics = _selected_metrics_view(compute_metrics(dev_questions, _ranked_for_selected_metrics(selected_by_id)))
    candidate_discovery = _candidate_discovery_metrics(dev_questions, candidate_by_id)
    status_by_id = {row["question_id"]: row["final_status"] for row in results}
    sufficiency = _sufficiency_answerability_diagnostics(dev_questions, status_by_id)
    hard_negative = _hard_negative_selection_diagnostics(dev_questions, selected_by_id)
    phase8b_metrics = _read_json(repo_root / "data/processed/phase8b/metrics.json")
    phase9a_metrics = _read_json(repo_root / "data/processed/phase9a/metrics.json")
    phase9b_metrics = _read_json(repo_root / "data/processed/phase9b/metrics.json")
    phase9b_selected = compute_metrics(dev_questions, _ranked_for_selected_metrics(_dev_phase9b_selected(repo_root, dev_questions)))
    comparison = []
    for metric in ["hit@1", "hit@3", "hit@5", "recall@1", "recall@3", "recall@5", "mrr", "ndcg@10"]:
        comparison.append({
            "metric": metric,
            "hybrid": phase8b_metrics["hybrid_rrf"]["by_split"]["dev"].get(metric),
            "multi_query": phase8b_metrics["multi_query_hybrid"]["by_split"]["dev"].get(metric),
            "decomposition": phase8b_metrics["decomposition_hybrid"]["by_split"]["dev"].get(metric),
            "phase9a_adaptive": phase9a_metrics["dev_metrics"].get(metric),
            "phase9b_corrective_selected": phase9b_selected["overall_non_abstain"].get(metric),
            "phase10a_agentic_selected": selected_metrics["overall_non_abstain"].get(metric),
        })
    return {
        "phase": "10A",
        "scope": "development split only",
        "question_count": len(dev_questions),
        "selected_evidence_metrics": selected_metrics,
        "candidate_discovery": candidate_discovery,
        "sufficiency_diagnostics": sufficiency,
        "hard_negative_diagnostics": hard_negative,
        "comparison_table_dev": comparison,
        "phase9b_reference_runtime": phase9b_metrics["runtime_evidence_behavior"],
    }


def _phase9b_vs_phase10a_corrections(repo_root: Path, results: list[dict[str, Any]]) -> dict[str, Any]:
    phase9b = _read_jsonl(repo_root / "data/processed/phase9b/corrective_results.jsonl")
    by_id = {row["question_id"]: row for row in results}
    rows = []
    for row in phase9b:
        qid = row["question_id"]
        if qid not in by_id or not row["correction_performed"]:
            continue
        agent = by_id[qid]
        rows.append({
            "question_id": qid,
            "phase9b_correction_strategy": row["correction_strategy"],
            "phase9b_final_status": row["final_overall_status"],
            "phase10a_actions": [a["action"] for a in agent["actions_taken"]],
            "phase10a_final_status": agent["final_status"],
            "new_context_ids_vs_phase9b_pool": sorted(set(agent["evidence_pool_ids"]) - set(row["combined_evidence_pool_ids"])),
        })
    return {
        "phase9b_corrections_on_dev": len(rows),
        "phase9b_resolved_after_correction_on_dev": sum(r["phase9b_final_status"] == "SUFFICIENT" for r in rows),
        "phase10a_sufficient_on_same_questions": sum(r["phase10a_final_status"] == "SUFFICIENT" for r in rows),
        "per_question": rows,
    }


def _case_studies(dev_questions: list[dict[str, Any]], results: list[dict[str, Any]]) -> dict[str, Any]:
    questions_by_id = {q["question_id"]: q for q in dev_questions}
    by_id = {row["question_id"]: row for row in results}
    selected_ids = [qid for qid in ["SYNQ-001-A", "SYNQ-002-A", "SYNQ-009-B", "SYNQ-012-A", "SYNQ-014-B"] if qid in by_id]
    extras = [
        next((r["question_id"] for r in results if r["actions_taken"] and r["actions_taken"][0]["action"] == "HYBRID_SEARCH" and r["cost_counters"]["retrieval_actions"] == 1), None),
        next((r["question_id"] for r in results if r["cost_counters"]["retrieval_actions"] > 1), None),
        next((r["question_id"] for r in results if r["final_status"] == "INSUFFICIENT_WITH_RESIDUAL"), None),
        next((r["question_id"] for r in results if "revision" in r["query"].lower() or "current" in r["query"].lower()), None),
    ]
    for qid in extras:
        if qid and qid not in selected_ids:
            selected_ids.append(qid)
    cases = {}
    for qid in selected_ids:
        row = by_id[qid]
        q = questions_by_id[qid]
        gold = set(q.get("gold_chunk_ids", []))
        hard = set(q.get("hard_negative_chunk_ids", []))
        cases[qid] = {
            "runtime": {
                "initial_route_suggestion": row["initial_route_suggestion"],
                "actions_taken": row["actions_taken"],
                "final_status": row["final_status"],
                "final_selected_context_ids": row["final_selected_context_ids"],
                "cost_counters": row["cost_counters"],
            },
            "offline_evaluation": {
                "selected_gold_ids": [u for u in row["final_selected_context_ids"] if u in gold],
                "selected_hard_negative_ids": [u for u in row["final_selected_context_ids"] if u in hard],
                "gold_count": len(gold),
            },
        }
    return cases


def _validate_phase10a_runtime(results: list[dict[str, Any]], dev_questions: list[dict[str, Any]], units_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    violations = []
    if {r["question_id"] for r in results} != {q["question_id"] for q in dev_questions}:
        violations.append("results do not match dev question IDs")
    for row in results:
        text = _json_text({"trace": row["trace"], "actions": row["actions_taken"], "ledger": row["final_evidence_ledger"]})
        hits = [term for term in LEAKAGE_TERMS if term in text]
        if hits:
            violations.append(f"{row['question_id']}: leakage terms {hits}")
        for unit_id in row["evidence_pool_ids"] + row["final_selected_context_ids"]:
            if unit_id not in units_by_id:
                violations.append(f"{row['question_id']}: unknown evidence id {unit_id}")
        if row["cost_counters"]["retrieval_actions"] > MAX_RETRIEVAL_ACTIONS:
            violations.append(f"{row['question_id']}: retrieval budget exceeded")
        if row["cost_counters"]["orchestrator_model_calls"] > MAX_ORCHESTRATOR_DECISIONS:
            violations.append(f"{row['question_id']}: decision budget exceeded")
        if row["cost_counters"]["query_rewriter_calls"] + row["cost_counters"]["decomposer_calls"] > MAX_QUERY_TRANSFORM_CALLS:
            violations.append(f"{row['question_id']}: query transform budget exceeded")
        if any(receipt["tool_calls_detected"] != 0 for receipt in row["role_receipts"]):
            violations.append(f"{row['question_id']}: tool call detected in receipt")
    trace_validation = validate_runtime_trace(results)
    if not trace_validation["passed"]:
        violations.append(f"trace validation failed: {trace_validation['violations']}")
    return {"passed": not violations, "violations": violations}


def _write_walkthrough(repo_root: Path, results: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    script = '''"""Print a compact Phase 10A agentic retrieval walkthrough."""\n\nfrom __future__ import annotations\n\nimport argparse\nimport json\nfrom pathlib import Path\n\n\ndef read_jsonl(path: Path) -> list[dict]:\n    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]\n\n\ndef main() -> int:\n    parser = argparse.ArgumentParser()\n    parser.add_argument("--question-id", default="SYNQ-001-A")\n    args = parser.parse_args()\n    rows = {\n        row["question_id"]: row\n        for row in read_jsonl(Path("data/processed/phase10a/agentic_retrieval_results.jsonl"))\n    }\n    row = rows[args.question_id]\n    print("Question:", row["query"])\n    print("Initial route:", row["initial_route_suggestion"])\n    print("Actions:")\n    for action in row["actions_taken"]:\n        print(\n            "-",\n            action["action"],\n            action.get("query") or action.get("document_id"),\n            "=>",\n            action.get("observation", {}),\n        )\n    print("Final status:", row["final_status"])\n    print("Selected evidence IDs:")\n    for unit_id in row["final_selected_context_ids"]:\n        print("-", unit_id)\n    return 0\n\n\nif __name__ == "__main__":\n    raise SystemExit(main())\n'''
    (repo_root / WALKTHROUGH_SCRIPT).write_text(script, encoding="utf-8")
    qid = "SYNQ-001-A" if any(r["question_id"] == "SYNQ-001-A" for r in results) else results[0]["question_id"]
    row = next(r for r in results if r["question_id"] == qid)
    lines = ["Phase 10A agentic retrieval walkthrough", "", f"Question ID: {qid}", f"Question: {row['query']}", f"Initial route suggestion: {row['initial_route_suggestion']}", "", "Runtime actions:"]
    lines.extend(f"- {a['action']}: {a.get('query') or a.get('document_id')} | {a['reason']}" for a in row["actions_taken"])
    lines.extend(["", f"Final status: {row['final_status']}", "Selected evidence:", *[f"- {u}" for u in row["final_selected_context_ids"]], "", "Evaluation-only summary:", json.dumps(metrics["case_studies"].get(qid, {}), indent=2, ensure_ascii=False)])
    (repo_root / WALKTHROUGH_TEXT).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_phase10a(repo_root: Path | None = None, *, resume: bool = True) -> Phase10AResult:
    if repo_root is None:
        repo_root = resolve_repo_root(Path(__file__))
    run_phase9a(repo_root)
    output_root = repo_root / PHASE10A_ROOT
    output_root.mkdir(parents=True, exist_ok=True)
    role_dir = repo_root / ROLE_OUTPUT_DIR
    if not resume and role_dir.exists():
        shutil.rmtree(role_dir)
    role_dir.mkdir(parents=True, exist_ok=True)
    dev_questions = _load_dev_questions(repo_root)
    if len(dev_questions) != 36:
        raise ValueError(f"Phase 10A must run 36 DEV questions, got {len(dev_questions)}")
    route_suggestions = _load_route_suggestions(repo_root)
    adapter = IsolatedModelAdapter(repo_root)
    toolbox = AgenticToolbox(repo_root, adapter)
    results: list[dict[str, Any]] = []
    try:
        for index, question in enumerate(dev_questions, start=1):
            qid = question["question_id"]
            print(f"[{index}/36] {qid} agentic run", flush=True)
            results.append(run_agentic_question(repo_root, toolbox, adapter, question, route_suggestions[qid]))
    finally:
        toolbox.close()
    validation = _validate_phase10a_runtime(results, dev_questions, toolbox.units_by_id)
    if not validation["passed"]:
        raise ValueError(f"Phase 10A runtime validation failed: {validation['violations'][:10]}")
    metrics = _metrics_for_dev(repo_root, dev_questions, results)
    cost_report = _runtime_cost_report(results)
    metrics["runtime_costs"] = cost_report
    metrics["phase9b_vs_phase10a_correction_comparison"] = _phase9b_vs_phase10a_corrections(repo_root, results)
    metrics["case_studies"] = _case_studies(dev_questions, results)
    metrics["runtime_validation"] = validation
    ledgers = [
        {
            "question_id": row["question_id"],
            "assessment_stage": "AGENTIC_FINAL",
            "overall_status": row["final_status"],
            "evidence_requirements": row["final_evidence_ledger"].get("evidence_requirements", []) if row["final_evidence_ledger"] else [],
            "missing_evidence_summary": row["missing_evidence_summary"],
            "selected_context_ids": row["final_selected_context_ids"],
        }
        for row in results
    ]
    traces = [{"question_id": row["question_id"], "trace": row["trace"]} for row in results]
    results_path = output_root / "agentic_retrieval_results.jsonl"
    ledgers_path = output_root / "evidence_ledgers.jsonl"
    traces_path = output_root / "traces.jsonl"
    metrics_path = output_root / "metrics.json"
    cost_path = output_root / "cost_report.json"
    _write_jsonl(results_path, results)
    _write_jsonl(ledgers_path, ledgers)
    _write_jsonl(traces_path, traces)
    _write_json(metrics_path, metrics)
    _write_json(cost_path, cost_report)
    _write_walkthrough(repo_root, results, metrics)
    outputs = [results_path, ledgers_path, traces_path, metrics_path, cost_path, repo_root / WALKTHROUGH_SCRIPT, repo_root / WALKTHROUGH_TEXT]
    manifest = {
        "phase": "10A",
        "status": "complete",
        "scope": "development split only",
        "dev_questions_executed": len(results),
        "challenge_questions_executed": 0,
        "max_retrieval_actions": MAX_RETRIEVAL_ACTIONS,
        "max_orchestrator_decisions": MAX_ORCHESTRATOR_DECISIONS,
        "max_query_transform_calls": MAX_QUERY_TRANSFORM_CALLS,
        "agent_framework_added": False,
        "new_retrieval_algorithm_added": False,
        "answer_writer_called": False,
        "scientific_computation_implemented": False,
        "model": CODEX_MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "prompts": {
            "orchestrator": {"path": ORCHESTRATOR_PROMPT_PATH.as_posix(), "sha256": sha256_for_file(repo_root / ORCHESTRATOR_PROMPT_PATH)},
            "assessor": {"path": ASSESSOR_PROMPT_PATH.as_posix(), "sha256": sha256_for_file(repo_root / ASSESSOR_PROMPT_PATH)},
            "rewriter": {"path": REWRITER_PROMPT_PATH.as_posix(), "sha256": sha256_for_file(repo_root / REWRITER_PROMPT_PATH)},
            "decomposer": {"path": DECOMPOSER_PROMPT_PATH.as_posix(), "sha256": sha256_for_file(repo_root / DECOMPOSER_PROMPT_PATH)},
        },
        "runtime_costs": cost_report,
        "final_status_counts": dict(Counter(row["final_status"] for row in results)),
        "output_hashes": {_relative(path, repo_root): sha256_for_file(path) for path in outputs},
    }
    manifest_path = repo_root / MANIFEST_PATH
    _write_json(manifest_path, manifest)
    return Phase10AResult("complete", results, metrics, manifest, outputs, manifest_path)


__all__ = [
    "ACTIONS",
    "MAX_ORCHESTRATOR_DECISIONS",
    "MAX_QUERY_TRANSFORM_CALLS",
    "MAX_RETRIEVAL_ACTIONS",
    "AgentTool",
    "AgenticQuestionState",
    "AgenticToolbox",
    "AgenticValidationError",
    "IsolatedModelAdapter",
    "Phase10AResult",
    "_validate_orchestrator_action",
    "run_phase10a",
]
