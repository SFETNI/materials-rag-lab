"""Shared support for capability-named public examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from materials_rag.api import MaterialsRAG
from materials_rag.dataset import dataset_status

ROOT = Path(__file__).resolve().parents[1]

def retrieval_example(method: str, label: str) -> None:
    parser = argparse.ArgumentParser(description=label)
    parser.add_argument("--query", default="IN718 fatigue runout stress definition")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    status = dataset_status(ROOT)
    if status.retrieval_unit_count == 0:
        print(f"{label}: dataset not installed")
        print("Run: materials-rag-lab data install <archive-or-directory>")
        return
    rows = MaterialsRAG.from_dataset(ROOT).retrieve(args.query, method=method, top_k=args.top_k)
    print(json.dumps(rows, indent=2, ensure_ascii=False))

def result_example(label: str, keys: tuple[str, ...]) -> None:
    payload = json.loads((ROOT / "data/results/retrieval_ladder_metrics.json").read_text(encoding="utf-8"))
    rows = [{"metric": row["metric"], **{key: row.get(key) for key in keys}} for row in payload["systems"]]
    print(label)
    print(json.dumps(rows, indent=2))

def agentic_example(label: str) -> None:
    parser = argparse.ArgumentParser(description=label)
    parser.add_argument("--query")
    args = parser.parse_args()
    if not args.query:
        print(f"{label}: pass --query to run a new bounded Agentic invocation.")
        print("Run materials-rag-lab doctor first; Agentic use requires the 277-unit corpus.")
        return
    result = MaterialsRAG.from_dataset(ROOT, require_full=True).ask(args.query)
    print(json.dumps(result, indent=2, ensure_ascii=False))
