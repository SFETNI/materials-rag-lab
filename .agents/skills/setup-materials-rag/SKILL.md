---
name: setup-materials-rag
description: Set up and diagnose a public Materials RAG Lab installation, companion dataset, retrieval models, and optional Agentic runtime.
---

# Set up Materials RAG Lab

1. Run `uv sync --locked --dev` and `uv run materials-rag-lab doctor --capability core`.
2. If data are missing, direct the user to DOI `10.5281/zenodo.22837229`, then run `uv run materials-rag-lab data install <archive-or-directory> --check` before installation.
3. Run `uv run materials-rag-lab data status`. Explain whether the installation is the 252-unit redistributable projection or reconstructed 277-unit corpus. For Level C, verify authorized sources with `uv run materials-rag-lab data reconstruct --source-root <authorized-sources> --check`, then reconstruct into `data/runtime`.
4. Run `uv run materials-rag-lab models status`. Download only when requested with `uv run materials-rag-lab models prepare`.
5. Run `uv run materials-rag-lab doctor --capability <dense|hybrid|reranking|agentic|scientific>` for the requested workflow. Require a compatible authenticated local Codex runtime only for Multi-query, Decomposition, Agentic, or model-guided Scientific Analysis.

Use only public paths and commands. Do not depend on development artifacts, private caches, benchmark gold, or release staging files.
