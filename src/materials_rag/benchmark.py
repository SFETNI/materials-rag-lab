"""Public, non-destructive benchmark execution."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from materials_rag.api import MaterialsRAG
from materials_rag.dataset import dataset_status
from materials_rag.ingestion.dense_retrieval import compute_metrics
from materials_rag.models import EMBEDDING_MODEL, RERANKER_MODEL, RERANKER_REVISION

RETRIEVAL_LADDER_METHODS = (
    "dense",
    "dense-reranked",
    "hybrid",
    "hybrid-reranked",
    "multi-query",
    "decomposition",
)
PUBLIC_RETRIEVAL_METHODS = ("dense", "hybrid")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def run_benchmark(
    project_root: Path,
    output_root: Path,
    *,
    suite: str = "retrieval-ladder",
    methods: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Execute retrieval methods over the installed benchmark into a new output root."""

    project_root = project_root.resolve()
    output_root = output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError("Benchmark output directory must not already exist")
    status = dataset_status(project_root)
    if status.outcome != "PASS":
        raise ValueError(status.detail)
    if suite == "retrieval-ladder":
        if status.retrieval_unit_count != 277:
            raise ValueError("retrieval-ladder reproduction requires the reconstructed 277-unit corpus")
        default_methods = RETRIEVAL_LADDER_METHODS
    elif suite == "public-retrieval":
        default_methods = PUBLIC_RETRIEVAL_METHODS
    else:
        raise ValueError("suite must be retrieval-ladder or public-retrieval")
    selected = methods or default_methods
    if not selected or set(selected) - set(RETRIEVAL_LADDER_METHODS):
        raise ValueError(f"Unsupported benchmark methods: {selected}")

    benchmark_path = project_root / "data/runtime/dataset/benchmark/synthetic/retrieval_eval.jsonl"
    if not benchmark_path.exists():
        raise FileNotFoundError("Installed companion dataset is missing the synthetic benchmark")
    questions = _read_jsonl(benchmark_path)
    if len(questions) != 60:
        raise ValueError(f"Expected 60 benchmark questions, found {len(questions)}")

    output_root.mkdir(parents=True)
    rag = MaterialsRAG.from_dataset(
        project_root,
        require_full=suite == "retrieval-ladder",
        runtime_output_root=output_root / "role_outputs",
    )
    metrics_by_method: dict[str, Any] = {}
    output_hashes: dict[str, str] = {}
    for method in selected:
        rows: list[dict[str, Any]] = []
        metric_rows: dict[str, list[dict[str, Any]]] = {}
        for index, question in enumerate(questions, start=1):
            question_id = question["question_id"]
            query = question.get("query") or question.get("question") or question["original_question"]
            retrieved = rag.retrieve(query, method=method, top_k=50)
            ids = [row["retrieval_unit_id"] for row in retrieved]
            rows.append(
                {
                    "question_id": question_id,
                    "query": query,
                    "method": method,
                    "retrieval_unit_ids": ids,
                }
            )
            metric_rows[question_id] = [
                {"retrieval_unit_id": unit_id, "score": 0.0} for unit_id in ids
            ]
            print(f"[{index:02d}/60] {method} {question_id}", flush=True)
        result_path = output_root / f"{method}_results.jsonl"
        _write_jsonl(result_path, rows)
        metrics_by_method[method] = compute_metrics(questions, metric_rows)
        output_hashes[result_path.name] = _sha256(result_path)

    metrics_path = output_root / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics_by_method, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_hashes[metrics_path.name] = _sha256(metrics_path)
    manifest = {
        "status": "completed",
        "suite": suite,
        "question_count": len(questions),
        "methods": list(selected),
        "dataset_mode": status.mode,
        "retrieval_units_sha256": _sha256(Path(status.retrieval_units_path)),
        "benchmark_sha256": _sha256(benchmark_path),
        "embedding_model": EMBEDDING_MODEL,
        "reranker_model": RERANKER_MODEL,
        "reranker_revision": RERANKER_REVISION,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "output_hashes": output_hashes,
        "published_results_overwritten": False,
        "note": "Model-backed query transformations are fresh replication calls and may vary.",
    }
    manifest_path = output_root / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


__all__ = ["PUBLIC_RETRIEVAL_METHODS", "RETRIEVAL_LADDER_METHODS", "run_benchmark"]
