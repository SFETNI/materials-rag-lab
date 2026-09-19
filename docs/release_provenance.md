# Experiment integrity and release provenance

The public v0.1.0 result preserves both scientific outputs and the prechallenge infrastructure history.

1. The development architecture was frozen before challenge execution.
2. A prompt/validator compatibility mismatch stopped the first preflight with **0 challenge questions completed**.
3. A test then mutated a referenced scientific manifest. The original historical bytes could not be recovered, so the original DEV freeze remained preserved rather than rewritten.
4. Tests were isolated from canonical artifacts. Scientific configuration and preserved DEV outputs were audited directly.
5. A new prechallenge DEV refreeze v2 documented the lost artifact and unchanged runtime configuration.
6. The official held-out challenge ran once from question 1 under that refreeze.
7. The final experiment freeze links the original freeze, incident records, refreeze, challenge freeze, and final outputs.

Before the final preflight passed:

- challenge questions executed: **0**;
- challenge gold inspected: **no**;
- challenge-derived tuning: **none**;
- original DEV freeze preserved: **yes**.

These incidents were infrastructure and test-isolation failures, not scientific challenge results. The authoritative public provenance records are the three manifests under [`data/manifests/`](../data/manifests/).

## Public runtime hardening after the experiment freeze

Release-facing interfaces were added without regenerating scientific outputs: capability-aware diagnostics, authorized 252→277 reconstruction, non-destructive benchmark execution, and execution of the already-declared `SCIENTIFIC_ANALYSIS` action through the existing typed computation request and receipt machinery. Prompt files, benchmark records, frozen manifests, model settings, and reported metrics remain unchanged.

The scientific-analysis integration is a public runtime capability, not a retroactive challenge result. The held-out challenge recorded zero scientific-analysis actions; numerical evidence remains validated by the separate computation probes. Public reconstruction verifies the frozen ordered IDs and retrieval text, while preserving the release-normalized metadata of the redistributable projection.
