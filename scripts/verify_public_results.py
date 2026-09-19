"""Verify compact public metric summaries and README headline values."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / "data/results" / name).read_text(encoding="utf-8"))


def _metric_rows(payload: dict) -> dict[str, dict]:
    return {row["metric"]: row for row in payload["systems"]}


def main() -> None:
    retrieval = _load("retrieval_ladder_metrics.json")
    agentic = _load("agentic_dev_challenge_metrics.json")
    computation = _load("computation_probe_metrics.json")
    rows = _metric_rows(retrieval)
    challenge = agentic["challenge"]["metrics"]
    selected = challenge["evidence_quality"]["selected_evidence_metrics"]["overall_non_abstain"]
    answer = challenge["final_answer_behavior"]
    expected = {
        "Dense Hit@5": (rows["hit@5"]["dense"], 0.7884615384615384),
        "Hybrid Hit@5": (rows["hit@5"]["hybrid_rrf"], 0.9230769230769231),
        "Challenge Hit@5": (selected["hit@5"], 1.0),
        "Challenge Recall@5": (selected["recall@5"], 0.7916666666666666),
        "Challenge MRR": (selected["mrr"], 0.9375),
        "Challenge nDCG": (selected["ndcg@10"], 0.8016091390426794),
        "Citation validity": (answer["citation_validity_rate"], 1.0),
        "Probe correctness": (
            computation["probe_metrics"]["deterministic_calculation_correctness"],
            1.0,
        ),
    }
    mismatches = [name for name, (actual, wanted) in expected.items() if abs(actual - wanted) > 1e-12]
    if mismatches:
        raise SystemExit(f"Compact frozen result mismatch: {mismatches}")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    evaluation = (ROOT / "docs/evaluation.md").read_text(encoding="utf-8")
    required = ["0.7885", "0.9231", "1.0000", "0.7917", "0.9375", "0.8016", "84 claims"]
    missing = [token for token in required if token not in readme]
    if missing:
        raise SystemExit(f"README is missing frozen headline values: {missing}")
    missing = [token for token in required if token not in evaluation]
    if missing:
        raise SystemExit(f"Evaluation documentation is missing frozen headline values: {missing}")
    print("Public result summaries: PASS")


if __name__ == "__main__":
    main()
