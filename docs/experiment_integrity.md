# Experiment integrity

The held-out challenge was run only after the development architecture and its scientific configuration were frozen.

Two infrastructure issues were detected before any challenge question completed:

1. A documented orchestrator output field and its strict validator disagreed. The validator was repaired to match the already-frozen prompt contract, with unrelated unknown fields still rejected.
2. A test wrote a regenerated manifest into a canonical artifact location. Test outputs were isolated, the original development freeze was preserved, and a new pre-challenge development freeze was created because the exact historical manifest bytes were no longer available.

Before the official challenge run:

- zero challenge questions had completed;
- challenge references had not been inspected;
- no challenge-derived tuning occurred;
- the original development freeze remained preserved;
- prompts, models, retrieval settings, action budgets, computation schemas, verifier behavior, and writer behavior were confirmed unchanged.

The official 24-question challenge was then executed once. The final provenance chain is recorded in the three manifests under `data/manifests/`.

