"""Capability-oriented public API over the frozen Materials RAG components."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from materials_rag.dataset import dataset_status
from materials_rag.project import find_project_root


def _project_root() -> Path:
    return find_project_root()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


class MaterialsRAG:
    """Public facade for retrieval and bounded Agentic RAG.

    The facade delegates ranking, fusion, model isolation, evidence assessment,
    and verification to the versioned implementation modules. It never loads
    benchmark gold or reference-answer files at runtime.
    """

    def __init__(
        self,
        project_root: Path,
        retrieval_units_path: Path,
        runtime_output_root: Path | None = None,
    ) -> None:
        self.project_root = project_root.resolve()
        self.retrieval_units_path = retrieval_units_path.resolve()
        self.units = _read_jsonl(self.retrieval_units_path)
        if len(self.units) not in {252, 277}:
            raise ValueError(f"Expected 252 or 277 retrieval units, found {len(self.units)}")
        self.units_by_id = {unit["id"]: unit for unit in self.units}
        if len(self.units_by_id) != len(self.units):
            raise ValueError("Retrieval unit IDs must be unique")
        from materials_rag.ingestion.hybrid_retrieval import build_bm25_index

        self._bm25_index = build_bm25_index(self.units)
        self._encoder: Any | None = None
        self._document_matrix: np.ndarray | None = None
        self._reranker: Any | None = None
        self.runtime_output_root = (
            runtime_output_root.resolve()
            if runtime_output_root is not None
            else self.project_root / "data/runtime/runs"
        )

    @classmethod
    def from_dataset(
        cls,
        project_root: str | Path | None = None,
        *,
        require_full: bool = False,
        runtime_output_root: str | Path | None = None,
    ) -> MaterialsRAG:
        root = Path(project_root).resolve() if project_root else _project_root()
        status = dataset_status(root)
        if status.outcome == "FAIL" or status.retrieval_unit_count == 0:
            raise FileNotFoundError(status.detail)
        if require_full and status.retrieval_unit_count != 277:
            raise ValueError("Full frozen replication requires the reconstructed 277-unit corpus")
        role_root = Path(runtime_output_root) if runtime_output_root is not None else None
        return cls(root, Path(status.retrieval_units_path), role_root)

    def _ensure_dense(self) -> None:
        if self._document_matrix is not None:
            return
        from sentence_transformers import SentenceTransformer

        from materials_rag.ingestion.dense_retrieval import MODEL_NAME, _encode_texts

        self._encoder = SentenceTransformer(MODEL_NAME, local_files_only=True)
        self._encoder.eval()
        self._document_matrix = _encode_texts(
            self._encoder,
            [str(unit["retrieval_text"]) for unit in self.units],
        )

    def _dense_raw(self, query: str, top_k: int) -> list[dict[str, Any]]:
        if not query.strip():
            raise ValueError("query must be non-empty")
        self._ensure_dense()
        from materials_rag.ingestion.dense_retrieval import QUERY_PREFIX, _encode_texts, _rank_query

        assert self._encoder is not None and self._document_matrix is not None
        vector = _encode_texts(self._encoder, [QUERY_PREFIX + query])[0]
        return _rank_query(vector, self._document_matrix, self.units, min(top_k, len(self.units)))

    def _hybrid_raw(self, query: str, top_k: int, question_id: str) -> dict[str, Any]:
        from materials_rag.ingestion.hybrid_retrieval import (
            BM25_TOP_K,
            DENSE_TOP_K,
            RRF_K,
            bm25_search,
            fuse_dense_bm25,
        )

        dense = self._dense_raw(query, DENSE_TOP_K)
        lexical = bm25_search(self._bm25_index, query, BM25_TOP_K)
        row = fuse_dense_bm25(
            {"question_id": question_id, "query": query}, dense, lexical, self.units_by_id, RRF_K
        )
        row["results"] = row["results"][:top_k]
        return row

    def _rerank_raw(self, query: str, rows: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        from materials_rag.ingestion.cross_encoder_reranking import (
            DENSE_CANDIDATE_K,
            _score_pairs,
            load_reranker,
        )

        candidates = rows[:DENSE_CANDIDATE_K]
        if self._reranker is None:
            self._reranker = load_reranker()
        scores = _score_pairs(
            self._reranker,
            [(query, self.units_by_id[row["retrieval_unit_id"]]["retrieval_text"]) for row in candidates],
        )
        reranked = [
            dict(row, rerank_score=score, candidate_rank=index)
            for index, (row, score) in enumerate(zip(candidates, scores, strict=True), start=1)
        ]
        reranked.sort(key=lambda row: (-row["rerank_score"], row["candidate_rank"]))
        for rank, row in enumerate(reranked, start=1):
            row["rank"] = rank
        return reranked[:top_k]

    def _public_result(self, row: dict[str, Any], rank: int) -> dict[str, Any]:
        unit = self.units_by_id[row["retrieval_unit_id"]]
        metadata = unit.get("metadata", {})
        return {
            "rank": rank,
            "retrieval_unit_id": unit["id"],
            "kind": unit["kind"],
            "evidence_id": metadata.get("evidence_id"),
            "document_id": metadata.get("document_source_id") or metadata.get("document_id"),
            "retrieval_text": unit["retrieval_text"],
            "source": unit.get("source", {}),
        }

    def retrieve(self, query: str, *, method: str = "hybrid", top_k: int = 10) -> list[dict[str, Any]]:
        """Retrieve evidence with a named public capability."""

        if top_k < 1:
            raise ValueError("top_k must be positive")
        method = method.lower().replace("_", "-")
        question_id = "runtime-" + hashlib.sha256(query.encode("utf-8")).hexdigest()[:12]
        if method == "dense":
            rows = self._dense_raw(query, top_k)
        elif method == "dense-reranked":
            rows = self._rerank_raw(query, self._dense_raw(query, 20), top_k)
        elif method == "hybrid":
            rows = self._hybrid_raw(query, top_k, question_id)["results"]
        elif method == "hybrid-reranked":
            candidates = self._hybrid_raw(query, 20, question_id)["results"]
            rows = self._rerank_raw(query, candidates, top_k)
        elif method in {"multi-query", "decomposition"}:
            toolbox, _adapter = self._toolbox(question_id)
            if method == "multi-query":
                fused, _ = toolbox.multi_query_search(question_id, query, top_k)
            else:
                fused, _ = toolbox.decomposition_search(question_id, query, top_k)
            rows = fused["results"]
        else:
            raise ValueError(
                "method must be dense, dense-reranked, hybrid, hybrid-reranked, "
                "multi-query, or decomposition"
            )
        return [self._public_result(row, index) for index, row in enumerate(rows, start=1)]

    def _toolbox(self, run_id: str):
        from materials_rag.ingestion.agentic_retrieval import AgenticToolbox, IsolatedModelAdapter
        from materials_rag.ingestion.hybrid_retrieval import build_bm25_index

        owner = self

        class PublicToolbox(AgenticToolbox):
            def __init__(self) -> None:
                self.repo_root = owner.project_root
                self.adapter = IsolatedModelAdapter(
                    owner.project_root, owner.runtime_output_root / run_id / "role_outputs"
                )
                self.units = owner.units
                self.units_by_id = owner.units_by_id
                self.bm25_index = build_bm25_index(self.units)

            def close(self) -> None:
                return None

            def _dense_search(self, query: str, top_k: int = 50) -> list[dict[str, Any]]:
                return owner._dense_raw(query, top_k)

        toolbox = PublicToolbox()
        return toolbox, toolbox.adapter

    def _initial_route(self, question_id: str, query: str, adapter: Any) -> dict[str, Any]:
        from materials_rag.ingestion.agentic_retrieval import _single_json_object

        prompt = "\n\n".join(
            [
                (self.project_root / "docs/generation/adaptive_router_v1.md").read_text(encoding="utf-8").strip(),
                "Return ONLY one JSON object on one line.",
                "REQUEST_JSON:",
                json.dumps({"question_id": question_id, "original_question": query}, ensure_ascii=False, sort_keys=True),
            ]
        )
        result = adapter.run("initial_router", f"{question_id}_route", prompt)
        record = _single_json_object(result.output_text, question_id)
        if set(record) != {"question_id", "original_question", "strategy", "reason"}:
            raise ValueError("initial router returned an invalid schema")
        if record["question_id"] != question_id or record["original_question"] != query:
            raise ValueError("initial router changed the question identity")
        if record["strategy"] not in {"HYBRID", "MULTI_QUERY", "DECOMPOSITION"}:
            raise ValueError("initial router returned an unsupported strategy")
        return record

    def ask(self, query: str) -> dict[str, Any]:
        """Run bounded retrieval, evidence assessment, verification, and answer assembly."""

        from materials_rag.ingestion.agentic_retrieval import run_agentic_question
        from materials_rag.ingestion.end_to_end_agentic import (
            _selected_evidence,
            draft_claims,
            validate_answer,
            verify_claims,
            write_answer,
        )

        if len(self.units) != 277:
            raise ValueError("Agentic ask requires the reconstructed 277-unit corpus")
        question_id = "user-" + hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]
        toolbox, adapter = self._toolbox(question_id)
        route = self._initial_route(question_id, query, adapter)
        runtime = run_agentic_question(
            self.project_root,
            toolbox,
            adapter,
            {"question_id": question_id, "query": query},
            route["strategy"],
        )
        evidence = _selected_evidence(runtime, self.units_by_id)
        ledger = runtime["final_evidence_ledger"] or {
            "overall_status": "INSUFFICIENT_WITH_RESIDUAL",
            "evidence_requirements": [],
            "missing_evidence_summary": "No evidence assessment was produced.",
        }
        claims = draft_claims(question_id, query, ledger, evidence)
        verified = verify_claims(question_id, claims, evidence, ledger).to_dict()
        answer = write_answer(question_id, query, verified)
        validate_answer(answer, verified, {item["context_id"] for item in evidence})
        return {
            "question_id": question_id,
            "answer": answer,
            "initial_route": route,
            "actions": runtime["actions_taken"],
            "evidence_ledger": ledger,
            "selected_evidence": evidence,
            "verified_claims": verified,
            "trace": runtime["trace"],
            "cost_counters": runtime["cost_counters"],
            "computation_receipts": runtime.get("computation_receipts", []),
        }


__all__ = ["MaterialsRAG"]
