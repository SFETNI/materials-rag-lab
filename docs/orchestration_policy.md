# Orchestration policy

The frozen controller uses two distinct decisions.

```text
INITIAL ROUTING
question only -> choose the first retrieval capability

AGENTIC CONTROL
question + observed evidence + previous actions + remaining budget
-> choose the next useful action or stop
```

Initial routing is model-guided under a predefined rubric. It is not a hard-coded classifier and it does not see retrieval scores, benchmark labels, gold evidence, or prior benchmark performance.

## Initial routing rubric

| Capability | Use when |
|---|---|
| `HYBRID` | The request is a direct factual lookup, exact identifiers or technical terms matter, one main evidence need is expected, and there is no clear reason for a more expensive transformation. |
| `MULTI_QUERY` | The information need is one thing, but terminology, synonyms, wording, or formulation may vary. Three reformulations can expose evidence missed by the original wording. |
| `DECOMPOSITION` | The question has multiple factual requirements, requests several attributes, makes a comparison, or contains separable evidence needs best retrieved independently. |

The conservative default is Hybrid. No strategy is assumed to be universally best.

## Evidence feedback loop

```text
Question
  -> initial route
  -> retrieve or analyze
  -> Evidence Ledger
  -> inspect unsupported requirements
       sufficient                         -> FINISH_SUFFICIENT
       specific missing fact              -> HYBRID_SEARCH with a targeted query
       one need with terminology variance -> MULTI_QUERY_SEARCH
       several independent needs          -> DECOMPOSITION_SEARCH
       current/superseded ambiguity        -> CHECK_REVISION_STATUS
       authoritative numerical need        -> SCIENTIFIC_ANALYSIS
       no productive action or no budget   -> FINISH_WITH_RESIDUAL
```

`MISSING` means unsupported by the supplied evidence pool. It does not establish that a fact is absent from the full corpus or the wider literature. The ledger stores observable requirement states and cited context IDs, not hidden reasoning.

## Next-action policy

| Observed state | Preferred action | Purpose |
|---|---|---|
| Direct evidence appears sufficient | `FINISH_SUFFICIENT` | Avoid unnecessary calls. |
| A specific factual requirement is `MISSING` or `PARTIAL` | `HYBRID_SEARCH` | Search explicitly for the missing entity, identifier, date, or attribute. |
| The same missing need may use varying terminology | `MULTI_QUERY_SEARCH` | Broaden lexical and semantic formulation. |
| Several independent requirements remain | `DECOMPOSITION_SEARCH` | Search each need separately. |
| Revision authority matters | `CHECK_REVISION_STATUS` | Read current/superseded relationships from canonical metadata. |
| A supported numerical result is required | `SCIENTIFIC_ANALYSIS` | Request a typed deterministic computation over registered data. |
| Tools cannot close the residual or a budget is exhausted | `FINISH_WITH_RESIDUAL` | Preserve uncertainty rather than invent support. |

## Policy and model discretion

The orchestrator is LLM-guided but constrained by typed actions, explicit evidence state, deterministic tool interfaces, bounded budgets, duplicate-action protection, and explicit stopping conditions.

Frozen v0.1.0 limits:

- at most **4 retrieval actions**;
- at most **6 orchestrator decisions**;
- at most **2 query-transform calls**;
- at most **2 computation actions**;
- exact duplicate normalized actions are rejected;
- stop on sufficient evidence, exhausted budget, or an explicit residual decision.

Retrieval, revision checks, schema validation, scientific computation, citation validation, and final v0.1.0 claim/answer assembly are deterministic executors. Initial routing, orchestration, evidence assessment, rewriting, decomposition, and typed computation-request translation are model-guided roles using the declared isolated runtime. The public release runtime wires the unchanged Scientific Analyst contract into `ask`; the frozen probes and held-out challenge recorded zero calls to that role, so the frozen metrics remain unchanged.

## Why the controller behaves this way

The design follows observations in this benchmark:

- Dense retrieval supplied the semantic baseline.
- Cross-encoder reranking improved some Dense ordering but degraded the broader Hybrid candidate pool.
- Hybrid Dense + BM25 + RRF was a strong simple baseline.
- Multi-query improved candidate discovery, although fusion could demote strong rewrite-specific evidence.
- Decomposition exposed distinct evidence for compound questions but was not universally superior.
- Question-only adaptive routing did not dominate fixed Hybrid.
- The bounded fixed corrective policy flagged incomplete evidence but resolved none of its 11 flagged cases.
- Targeted Agentic retrieval resolved cases that fixed correction did not.
- Revision-aware verification prevented superseded evidence from supporting a current conclusion.
- Strict context-ID validation caught cross-question citation contamination.
- Parsing, normalization, canonicalization, and chunking materially affected retrieval quality.

The operating principle is: **start simple, inspect evidence, and escalate only for a specific unresolved need**.

## Role boundaries

| Role or capability | Implementation boundary |
|---|---|
| Orchestrator | Model-guided next-action selection over typed state; it does not retrieve or fabricate evidence. |
| Retrieval capabilities | Hybrid, Multi-query, Decomposition, and revision lookup are tools, not persistent agents. |
| Scientific Analyst | Model-guided translation from a numerical need to a typed request. Deterministic code computes authoritative values. |
| Evidence Assessor / Ledger | Model-guided assessment constrained to supplied context IDs and fixed statuses. |
| Verifier / Critic | Frozen v0.1.0 deterministic claim checks for support, receipts, and revision appropriateness; it cannot search. |
| Answer assembly | Frozen v0.1.0 deterministic assembly from verified claims and residuals; it cannot retrieve or calculate. |
| Human | Reviews evidence, residual uncertainty, and consequential R&D decisions. |

## Data preparation is part of the system

```text
raw evidence -> parsing -> normalization -> chunking
-> canonical objects -> Knowledge Units -> retrieval corpus
-> retrieval and orchestration
```

The canonical families include `Document`, `Chunk`, `ExperimentRecord`, `ProcessRecord`, `AnalysisRecord`, and `TestRecord`.

```text
SOURCE TRUTH
!= RETRIEVAL REPRESENTATION
!= EMBEDDING VECTOR
!= DERIVED COMPUTATION RECEIPT
```

This separation keeps provenance, retrieval text, model representations, and derived numerical evidence auditable.
