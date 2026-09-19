# Evaluation

## Scope and denominators

The retrieval benchmark has 60 synthetic questions. Ordinary retrieval aggregates use the frozen non-abstain denominator where specified by the phase artifacts. DEV and challenge results are always shown separately for the end-to-end system.

The public NASA/NIST sanity set and eight computation development probes are separate from the synthetic benchmark.

## Retrieval ladder

The first seven rows are retrieval rankings. The Corrective row evaluates the final evidence selected by the Evidence Ledger workflow, so it includes a selection stage and should not be read as a pure ranker ablation.

| System | Hit@1 | Hit@3 | Hit@5 | Recall@5 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| Dense | 0.5192 | 0.7308 | 0.7885 | 0.5625 | 0.6402 | 0.5665 |
| Dense + Cross-Encoder | 0.5000 | 0.8462 | 0.8846 | 0.6154 | 0.6837 | 0.6030 |
| Hybrid RRF | 0.5385 | 0.8077 | **0.9231** | **0.6635** | 0.6894 | 0.6543 |
| Hybrid RRF + Cross-Encoder | 0.4615 | 0.8269 | 0.8654 | 0.5962 | 0.6539 | 0.5789 |
| Multi-query Hybrid | 0.5577 | 0.8269 | 0.8654 | 0.6442 | **0.7121** | **0.6711** |
| Decomposition Hybrid | 0.5769 | 0.7500 | 0.8846 | 0.6234 | 0.7008 | 0.6447 |
| Adaptive Router | 0.5962 | 0.7500 | 0.8846 | 0.6298 | 0.7104 | 0.6547 |
| Corrective selected evidence | **0.8462** | **0.9615** | **0.9615** | **0.7516** | **0.8910** | **0.7607** |

The broad Hybrid candidate set was degraded by the cross-encoder on this benchmark. Multi-query improved candidate coverage. Decomposition sometimes retrieved missing evidence in one branch but demoted it during fusion. The question-only router did not dominate fixed Hybrid.

The Corrective workflow flagged 11 questions and resolved none after its single fixed alternate strategy. This motivated targeted Agentic retrieval rather than an unbounded loop.

## End-to-end Agentic results

### Selected evidence

| Metric | DEV, n=36 | Challenge, n=24 |
|---|---:|---:|
| Hit@5 | 0.9444 | **1.0000** |
| Recall@5 | 0.7569 | **0.7917** |
| MRR | 0.8773 | **0.9375** |
| nDCG@10 | 0.7631 | **0.8016** |
| Citation validity | 1.0000 | **1.0000** |

Challenge candidate discovery gold recall was 0.8125 at @10, @20, and @50. No hard negative was selected into final challenge evidence.

### Verification

The challenge verifier evaluated 84 claims:

| Disposition | Count |
|---|---:|
| Supported | 68 |
| Unverified | 15 |
| Superseded | 1 |
| Unsupported factual claims entering final answers | 0 |

Overall verification status was 10 `VERIFIED`, 11 `VERIFIED_WITH_RESIDUAL`, and 3 `NOT_ANSWERABLE_FROM_EVIDENCE`. Citation validity was 1.0, with 41 citations and no unsupported citation IDs.

### Answerability

| Offline outcome | Count |
|---|---:|
| True answer | 13 |
| False answer | 1 |
| True abstention | 7 |
| False abstention | 3 |

Answerability precision was 0.9286, answerability recall 0.8125, and abstention precision 0.7000.

### Correctness and completeness limitation

The frozen correctness and completeness results use a deterministic lexical reference check. The challenge distributions were evenly split across scores 0, 1, and 2, with means of 1.0; DEV means were 0.4722. These values are regression indicators and are not equivalent to expert semantic assessment. Groundedness was checked deterministically and had a mean of 2.0 for both splits.

## Efficiency

| Measure | Challenge |
|---|---:|
| Mean retrieval actions/question | 2.00 |
| Mean total role calls/question | 7.67 |
| Mean final evidence items | 2.79 |
| Solved after first retrieval | 62.5% |
| Required more than one retrieval | 37.5% |
| Ended with residual evidence | 37.5% |
| Duplicate-action rate | 0.0% |

No scientific computation action occurred during the held-out challenge.

## Computation probes

The separate computation development set contains eight probes and six receipts. It includes one unsupported computation and one intentional tool rejection. Deterministic calculation correctness was 1.0; unit correctness, censoring semantics, and provenance completeness passed.

## Frozen sources

The exact runtime and evaluation artifact hashes are recorded in the frozen manifests under `data/manifests/`. Large generated outputs remain outside this compact software export.

The canonical experiment freeze SHA-256 is `859418aa8f80d0c9e2ac28275d418ecf1c7fe2b46c71401f83c9cc33d2124b22`.
