# Methodology

For the concise record of the pre-challenge integrity checks and one-shot run, see [Experiment integrity](experiment_integrity.md).

## Experimental principle

The project adds one retrieval or orchestration capability at a time, evaluates it against frozen labels, and preserves negative results. Later capabilities import earlier implementations instead of redefining them.

| Public capability | Main implementation | Frozen artifact family |
|---|---|---|
| Canonical knowledge preparation | `parsers.py`, `normalization.py`, `chunker.py`, `retrieval_corpus.py` | canonical records and 277 retrieval units |
| Vanilla dense retrieval | `dense_retrieval.py`, `qdrant_retrieval.py` | dense and Qdrant rankings |
| Cross-encoder reranking | `cross_encoder_reranking.py` | dense Top-20 reranking |
| Hybrid retrieval | `hybrid_retrieval.py` | BM25 and RRF rankings |
| Multi-query | `multi_query_retrieval.py` | four-query second-stage RRF |
| Decomposition | `query_decomposition_retrieval.py` | evidence-need second-stage RRF |
| Adaptive routing | `adaptive_routing.py` | one-strategy question routing |
| Corrective evidence loop | `corrective_retrieval.py` | Evidence Ledger and one correction |
| Agentic retrieval | `agentic_retrieval.py` | bounded targeted actions and traces |
| Scientific computation | `scientific_computation.py` | typed requests and receipts |
| Verified answers | `end_to_end_agentic.py` | claims, verification, and answers |
| Held-out evaluation | `one_shot_challenge.py` | frozen one-shot challenge outputs |

## Corpus construction

Public NASA and NIST source material is parsed into canonical documents, chunks, experiments, processes, analyses, and tests. A controlled fictional internal corpus supplies revision-sensitive manufacturing and fatigue cases. Each unit retains stable IDs and source metadata.

Structured data are normalized into Parquet and typed records. They are not flattened into prose for authoritative numerical calculation. Retrieval text remains an evidence-discovery representation.

## Benchmark

The frozen synthetic benchmark contains 60 questions representing 30 paired intents. The development split has 36 questions and the one-shot challenge has 24. Twenty-six hard-negative annotations test plausible but incorrect, stale, or scope-mismatched evidence.

The benchmark labels are `authored_not_expert_validated`. They support controlled software experiments and do not claim independent scientific validation.

## Isolation discipline

Question rewrites, decompositions, routing decisions, evidence assessments, and runtime roles use isolated processes that receive only their declared runtime-safe inputs. Gold evidence, reference answers, hard-negative labels, split labels, previous performance, and oracle analyses remain offline.

Strict per-question context-ID allowlists prevent a model from constructing or carrying citations from another question. This validation caught cross-question contamination during assessor protocol development.

## Evaluation discipline

Retrieval metrics retain the same gold labels and denominators across fixed retrieval ablations. Public sanity cases and computation probes are reported separately. Development configuration was frozen before the challenge set was executed.

The held-out challenge was run once after a preflight verified all frozen inputs. Infrastructure incidents before that run are documented in [Experiment integrity](experiment_integrity.md).

## Interpretation

The system reports candidate discovery separately from final evidence selection. It also separates deterministic checks, such as citation validity and receipt equality, from lexical answer-regression checks. No single score is treated as a universal measure of scientific answer quality.
