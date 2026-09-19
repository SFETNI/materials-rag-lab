---
name: run-agentic-rag
description: Operate and inspect the bounded Materials RAG Agentic workflow through the public CLI, including evidence, receipts, verification, and residuals.
---

# Run Agentic RAG

1. Run `uv run materials-rag-lab doctor --capability agentic`; Agentic use requires the reconstructed 277-unit corpus and compatible isolated role runtime.
2. Run `uv run materials-rag-lab ask "<question>" --output <new-file>.json`.
3. Inspect `initial_route`, `actions`, `evidence_ledger`, `selected_evidence`, `verified_claims`, `trace`, and `cost_counters`.
4. Explain observable decisions from the policy. For example, Decomposition indicates separable evidence needs; a targeted Hybrid action indicates a requirement remained `MISSING` or `PARTIAL`.
5. When the registered Scientific Analysis action occurs, surface its typed request, deterministic computation receipt ID, units, censoring policy, and provenance. Surface revision status, verifier dispositions, and residual uncertainty. Never claim access to hidden reasoning.

Do not expose benchmark gold, override action budgets, or use a different model while describing the run as frozen v0.1.0.
