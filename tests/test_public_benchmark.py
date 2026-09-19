from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from materials_rag.benchmark import run_benchmark


class _FakeRAG:
    def retrieve(self, query: str, *, method: str, top_k: int) -> list[dict[str, object]]:
        return [
            {
                "rank": 1,
                "retrieval_unit_id": "unit|gold",
                "kind": "chunk",
                "evidence_id": None,
                "document_id": None,
                "retrieval_text": query,
                "source": {},
            }
        ][:top_k]


def test_public_benchmark_executes_and_writes_new_metrics(monkeypatch, tmp_path: Path) -> None:
    project = tmp_path / "project"
    benchmark = project / "data/runtime/dataset/benchmark/synthetic/retrieval_eval.jsonl"
    benchmark.parent.mkdir(parents=True)
    questions = [
        {
            "question_id": f"Q{index:02d}",
            "query": f"Question {index}",
            "gold_chunk_ids": ["unit|gold"],
            "hard_negative_chunk_ids": [],
            "answerability": "answer",
            "split": "dev" if index < 36 else "challenge",
            "task_family": "test",
        }
        for index in range(60)
    ]
    benchmark.write_text(
        "".join(json.dumps(row) + "\n" for row in questions), encoding="utf-8"
    )
    units = project / "data/runtime/retrieval_units.jsonl"
    units.parent.mkdir(parents=True, exist_ok=True)
    units.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        "materials_rag.benchmark.dataset_status",
        lambda _root: SimpleNamespace(
            outcome="PASS",
            detail="",
            retrieval_unit_count=252,
            retrieval_units_path=str(units),
            mode="redistributable-252",
        ),
    )
    monkeypatch.setattr(
        "materials_rag.benchmark.MaterialsRAG.from_dataset",
        lambda *_args, **_kwargs: _FakeRAG(),
    )
    output = tmp_path / "new-output"
    report = run_benchmark(
        project, output, suite="public-retrieval", methods=("dense", "hybrid")
    )
    assert report["status"] == "completed"
    assert report["question_count"] == 60
    assert (output / "dense_results.jsonl").exists()
    assert (output / "hybrid_results.jsonl").exists()
    assert (output / "metrics.json").exists()
    assert (output / "run_manifest.json").exists()
