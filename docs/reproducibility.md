# Reproducibility

## Three levels

| Level | Inputs | What it establishes |
|---|---|---|
| A — Quick demonstration | Repository only | Canonical formats, sample preparation, public commands, frozen-result verification |
| B — Redistributable replication | Repository + Zenodo 252-unit dataset + declared models | Public retrieval over redistributable evidence; not the full experiment |
| C — Full experiment replication | Level B + lawful reconstruction + frozen runtime | A new 277-unit retrieval-ladder execution in a new output root |

## Freeze chain

The canonical final freeze is `data/manifests/agentic_rag_v0.1.0_experiment_freeze.json`, SHA-256 `859418aa8f80d0c9e2ac28275d418ecf1c7fe2b46c71401f83c9cc33d2124b22`.

```text
original DEV freeze
-> prechallenge infrastructure incidents with 0 challenge questions executed
-> test isolation repair
-> DEV refreeze v2
-> one-shot 24-question challenge
-> final experiment freeze
```

Challenge gold was not inspected by runtime roles and no challenge-informed tuning occurred. See [Release provenance](release_provenance.md).

## Verify before running

```bash
uv sync --locked --dev
uv run materials-rag-lab doctor
uv run materials-rag-lab data status
uv run materials-rag-lab models status
uv run materials-rag-lab results verify
uv run python scripts/validate_public_candidate.py
uv run pytest
uv run ruff check .
```

## Verification is not reproduction

```bash
uv run materials-rag-lab results verify
```

This checks that compact public result artifacts and documentation agree with the frozen manifests. It does not execute retrieval models.

## Fresh, non-destructive execution

Never overwrite frozen files under `data/manifests/` or `data/results/`.

With the 252-unit projection, rerun the public Dense and Hybrid benchmark:

```bash
uv run materials-rag-lab benchmark public-retrieval --output /new/empty/output/path
```

For the six-method retrieval ladder, first reconstruct the 277-unit corpus and then run:

```bash
uv run materials-rag-lab reproduce --suite retrieval-ladder --output /new/empty/output/path
```

This executes Dense, Dense + reranking, Hybrid, Hybrid + reranking, Multi-query, and Decomposition over all 60 questions. It writes fresh rankings, metrics, role outputs, and a run manifest. Model-backed query transformations are fresh calls and may vary. The command refuses an existing directory and never writes to `data/manifests/` or `data/results/`.

Adaptive, Corrective, and end-to-end Agentic headline values remain frozen-result inspection in v0.1.0; the public command does not mislabel their compact summaries as a fresh rerun.

## Environment identity

The root `pyproject.toml` and `uv.lock` define the supported public environment. The exact historical project specification is retained at `data/reproducibility/pyproject.frozen.toml`; it is intentionally distinct from release packaging. Recovered model IDs, revision limitations, retrieval parameters, and budgets are in `data/reproducibility/frozen_environment.json`.

The exact Dense embedding commit was not recoverable. The reranker commit was recovered. A different model, revision, role model, reasoning effort, or provider is a new replication configuration.

## Evaluation limits

- Labels are `authored_not_expert_validated`.
- Correctness/completeness uses deterministic lexical regression, not expert semantic review.
- Computation probes are separate from the 60-question benchmark.
- Model-backed calls are stochastic even when configuration is held fixed.
