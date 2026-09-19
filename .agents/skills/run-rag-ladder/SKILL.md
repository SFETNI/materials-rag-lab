---
name: run-rag-ladder
description: Run one named Materials RAG capability and explain its output without requiring internal experiment phase knowledge.
---

# Run the RAG ladder

Ask which capability the user wants: Vanilla Dense Retrieval, Reranking, Hybrid Retrieval, Multi-query, Decomposition, Adaptive Routing, or Corrective RAG. Do not silently escalate.

- To execute one query, use `uv run materials-rag-lab retrieve --method <dense|dense-reranked|hybrid|hybrid-reranked|multi-query|decomposition> --query "..."`.
- For a guided example, use `uv run materials-rag-lab demo <capability>`.
- To execute a new benchmark, use `uv run materials-rag-lab benchmark public-retrieval --output <new-directory>` or the documented full reconstruction suite.
- To inspect published findings without rerunning models, use `uv run materials-rag-lab results verify` and the compact files under `data/results/`.
- Explain retrieved IDs, evidence text, and method-specific behavior. Keep benchmark gold outside runtime.
- If data or models are unavailable, use `materials-rag-lab doctor`, `data status`, and `models status` and provide the public setup step.

Never describe `demo` or `results verify` as a benchmark rerun. Describe Hit@K, Recall@K, MRR, and nDCG only when the user is running an evaluation suite. Use capability names rather than development phase names.
