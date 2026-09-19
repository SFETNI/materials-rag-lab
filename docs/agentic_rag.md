# Agentic RAG

## How the system decides what to do next

The system first uses a question-only router to suggest Hybrid, Multi-query, or Decomposition. After evidence is retrieved, the bounded orchestrator sees the Evidence Ledger, previous observable actions, and remaining budgets. It may perform a targeted Hybrid search, broaden one need with Multi-query, split several needs with Decomposition, check revision metadata, request deterministic scientific analysis, or stop. The public `ask` facade can execute each action in this typed set.

The orchestrator is model-guided within a typed action set. Retrieval is capped at 4 actions, orchestrator decisions at 6, query transformations at 2, and computations at 2. Duplicate actions are rejected. It stops when evidence is sufficient, when no productive tool remains, or when a budget is exhausted. Missing support is returned as residual uncertainty for human review.

See the complete [orchestration policy](orchestration_policy.md).

## Evidence Ledger

Each question becomes explicit requirements with `SUPPORTED`, `PARTIAL`, `MISSING`, `CONFLICTING`, or `SUPERSEDED` states. `MISSING` means unsupported by the supplied evidence pool, not proven absent from the corpus. The ledger stores context IDs and concise observable reasons; it does not store chain-of-thought.

## Runtime roles and tools

| Component | Responsibility |
|---|---|
| Orchestrator | Chooses a typed next action and controls stopping. |
| Hybrid / Multi-query / Decomposition | Retrieval tools built from the same lower-level Dense, BM25, and RRF implementation. |
| Revision status | Deterministic lookup over canonical metadata. |
| Scientific Analyst | Translates a numerical need into a typed computation request. |
| Computation executor | Deterministically validates fields, units, filters, censoring, and operations, then emits a receipt. |
| Evidence Assessor | Updates requirement support from supplied evidence only. |
| Verifier / Critic | Checks claim support, revision role, numerical receipts, and overreach. |
| Answer assembly | Uses supported claims and residuals only. |
| Human | Owns consequential engineering decisions. |

The model-backed roles exercised in frozen v0.1.0 were the router, orchestrator, Evidence Assessor, query rewriter, and decomposer. A model-backed Scientific Analyst interface was implemented, but the frozen probes and held-out challenge recorded zero Scientific Analyst calls. Claim drafting, verification, answer assembly, lexical answer evaluation, and all reported numerical calculations were deterministic structured functions. Prompt files remain versioned contracts and design provenance; see [generation contracts](generation/README.md).

## Scientific computation

Models do not author authoritative numerical results. The Scientific Analyst can request only `FILTER_RECORDS`, `SUMMARIZE`, or `COMPARE_GROUPS` over registered fields. Deterministic code preserves units, excludes runouts from cycles-to-failure statistics unless explicitly supported, rejects zero-denominator ratios, and emits a provenance-bearing `ComputationReceipt`.

The public release runtime connects this registered computation path to `ask`. This post-freeze packaging integration does not alter or retroactively extend the frozen experiment: the frozen computation probes exercised the deterministic operations separately, and the held-out challenge invoked no scientific-computation action.

Receipts are runtime evidence; they are not inserted into the semantic retrieval corpus.

## Verification and residuals

Claims are classified as factual, numerical, comparative, causal, revision-status, or limitation claims. Verification dispositions are `SUPPORTED`, `CONTRADICTED`, `UNVERIFIED`, and `SUPERSEDED`. Superseded evidence can support historical chronology but cannot establish a current approved conclusion.

Unsupported critical claims are removed. The result is `VERIFIED`, `VERIFIED_WITH_RESIDUAL`, or `NOT_ANSWERABLE_FROM_EVIDENCE`. Verification does not reopen retrieval in frozen v0.1.0.

## Inspecting a run

```bash
materials-rag-lab ask "What was recorded for B017-F03?" --output run.json
```

Inspect `initial_route`, `actions`, `evidence_ledger`, `selected_evidence`, `verified_claims`, `trace`, and `cost_counters`. These fields explain observable policy decisions and evidence state without exposing private reasoning.
