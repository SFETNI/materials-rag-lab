# Quickstarts

Install and inspect the public package:

```bash
uv sync --locked --dev
uv run materials-rag-lab doctor
```

Install the companion dataset:

```bash
uv run materials-rag-lab data install /path/to/materials-rag-data-v0.1.0.zip --check
uv run materials-rag-lab data install /path/to/materials-rag-data-v0.1.0.zip
uv run materials-rag-lab data status
```

For authorized full-corpus reconstruction:

```bash
uv run materials-rag-lab data reconstruct --source-root /path/to/authorized-sources --check
uv run materials-rag-lab data reconstruct --source-root /path/to/authorized-sources --output data/runtime
```

## Data and knowledge preparation

```bash
uv run python experiments/00_data_and_knowledge_preparation.py
```

This safe sample shows raw input, identifier normalization, a canonical `Document`, a `Chunk`, retrieval text, and provenance. It does not rebuild the whole dataset.

## Vanilla Dense Retrieval

```bash
uv run materials-rag-lab retrieve --method dense --query "IN718 fatigue runout"
```

## Hybrid Retrieval

```bash
uv run materials-rag-lab retrieve --method hybrid --query "B017-F03 stress definition fatigue outcome"
```

Hybrid uses the same Dense and BM25 branches and RRF `k=60` as the frozen implementation.

## Query transformations

Use Multi-query for one need whose wording may vary:

```bash
uv run materials-rag-lab retrieve --method multi-query --query "fracture origin in B021 fatigue specimens"
```

Use Decomposition for independent needs:

```bash
uv run materials-rag-lab retrieve --method decomposition --query "Compare loading definitions and fatigue outcomes for B017 and B021"
```

These commands require the isolated model runtime described in [Model and runtime setup](model_runtime.md).

## Agentic RAG

Agentic RAG requires the reconstructed 277-unit corpus and a compatible authenticated Codex CLI. The registered `SCIENTIFIC_ANALYSIS` action can request only typed operations; deterministic code produces any numerical receipt.

```bash
uv run materials-rag-lab ask "What fatigue result was recorded for B017-F03, including stress definition and outcome?" --output run.json
```

Inspect `initial_route`, `actions`, `evidence_ledger`, `selected_evidence`, `verified_claims`, `trace`, and `cost_counters`. The trace shows observable policy and evidence transitions, not hidden reasoning.

## Python API

```python
from materials_rag.api import MaterialsRAG

rag = MaterialsRAG.from_dataset()
results = rag.retrieve("IN718 fatigue runout", method="hybrid", top_k=5)
print(results[0]["retrieval_unit_id"])
```

## Query, demo, benchmark, and result commands

- `retrieve` executes one query.
- `demo` runs a small capability example or displays its published summary.
- `benchmark` executes a new evaluation into a new directory.
- `results verify` verifies published compact artifacts without rerunning models.

```bash
uv run materials-rag-lab demo data-preparation
uv run materials-rag-lab demo hybrid --query "IN718 HIP hold time"
uv run materials-rag-lab benchmark public-retrieval --output public-benchmark/
uv run materials-rag-lab results verify
```

Full retrieval-ladder reproduction is described separately in [Reproducibility](reproducibility.md).
