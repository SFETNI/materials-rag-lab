# Contributing

Thank you for helping improve Materials RAG Lab.

## Development setup

1. Install Python 3.12 or newer.
2. Run `uv sync --locked --dev`.
3. Download the companion dataset described in `data/README.md` when a test or example needs the full corpus.
4. Run `python -m pytest` and `python -m ruff check src tests experiments scripts`.

## Project boundaries

- Keep retrieval, query transformation, evidence assessment, deterministic computation, verification, and answer writing in separate modules.
- Preserve stable IDs and source provenance.
- Keep structured numerical records separate from semantic retrieval text.
- Never treat benchmark gold, reference answers, or hard-negative annotations as runtime evidence.
- Use temporary output directories in tests. Tests must not rewrite frozen manifests or canonical artifacts.
- Do not redistribute reference-only source files. Follow `THIRD_PARTY_NOTICES.md` and `docs/data_and_provenance.md`.

## Pull requests

Explain the behavior changed, why it changed, and how it was tested. Changes to frozen scientific configuration or evaluation methodology should be proposed as a new experiment version rather than silently replacing the published v0.1.0 result.

## Optional Codex skills

The repository-local skills under `.agents/skills/` map setup, ladder, Agentic, and reproduction tasks to the same public CLI and Python API documented for every user. They are conveniences only; no core workflow depends on an agent product.
