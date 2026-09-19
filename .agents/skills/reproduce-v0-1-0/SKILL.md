---
name: reproduce-v0-1-0
description: Verify frozen v0.1.0 artifacts and execute a fresh six-method retrieval-ladder replication into a new output root.
---

# Reproduce v0.1.0

1. Read `docs/reproducibility.md` and choose Level A, B, or C.
2. Verify the dataset mode with `uv run materials-rag-lab data status`: 252 units support redistributable replication. For 277 units, run `uv run materials-rag-lab data reconstruct --source-root <authorized-sources> --check` and then reconstruct into `data/runtime`.
3. Verify models and role runtime with `uv run materials-rag-lab models status`.
4. Verify frozen summaries with `uv run materials-rag-lab results verify` and the canonical freeze hash with `uv run python scripts/validate_public_candidate.py`. State explicitly that this is verification.
5. Run a fresh execution with `uv run materials-rag-lab reproduce --suite retrieval-ladder --output <new-empty-directory>`. Confirm the new run manifest lists the six executed retrieval methods.

Stop on any frozen-input mismatch. Never overwrite `data/manifests/` or `data/results/`. State that lexical correctness/completeness checks are deterministic regression checks, not expert semantic adjudication.
