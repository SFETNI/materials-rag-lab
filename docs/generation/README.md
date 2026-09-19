# Versioned role contracts

Prompt files are public interfaces and experiment provenance. Their presence does not mean every role was model-executed in the frozen v0.1.0 end-to-end path.

All model-backed frozen roles used fresh isolated `gpt-5.5` processes with low reasoning, no tool use, and strict structured validation. Runtime modules live under `src/materials_rag/ingestion/`.

| Contract | Role and purpose | Frozen v0.1.0 status | Runtime implementation |
|---|---|---|---|
| `adaptive_router_v1.md` | Question-only initial strategy suggestion | Model-executed | `adaptive_routing.py`, public facade router |
| `agentic_orchestrator_v1.md` | Next bounded action from evidence state | Model-executed | `agentic_retrieval.py` |
| `evidence_assessor_agentic_v1.md` | Agentic Evidence Ledger update | Model-executed | `agentic_retrieval.py` |
| `runtime_multi_query_rewriter_v1.md` | Exactly three runtime query reformulations | Model-executed when Multi-query was called | `agentic_retrieval.py`, `multi_query_retrieval.py` |
| `runtime_query_decomposer_v1.md` | Runtime atomic evidence needs | Model-executed when Decomposition was called | `agentic_retrieval.py`, `query_decomposition_retrieval.py` |
| `scientific_analyst_v1.md` | Typed computation-request translation | Retained model-backed interface; the frozen probes and held-out challenge recorded zero Scientific Analyst calls. The public release runtime can invoke this unchanged contract, while deterministic code owns all values. | `scientific_computation.py`, public `ask` facade |
| `claim_drafter_v1.md` | Candidate claim contract | Retained contract; frozen final DEV/challenge path used deterministic drafting | `end_to_end_agentic.py` |
| `evidence_verifier_v1.md` | Claim verification contract | Retained contract; frozen final path used deterministic verification | `end_to_end_agentic.py` |
| `agentic_answer_writer_v1.md` | Grounded answer contract | Retained contract; frozen final path used deterministic assembly | `end_to_end_agentic.py` |
| `answer_evaluator_v1.md` | Semantic evaluation design | Retained contract; reported frozen metrics used deterministic lexical checks | `end_to_end_agentic.py` |
| `multi_query_rewriter_v1.md` | Historical isolated batch rewrite experiment | Model-executed in controlled historical experiment | `multi_query_retrieval.py` |
| `query_decomposer_v1.md` | Historical isolated batch decomposition experiment | Model-executed in controlled historical experiment | `query_decomposition_retrieval.py` |
| `evidence_assessor_initial_v2.md`, `evidence_assessor_final_v1.md` | Historical bounded corrective evidence gates | Model-executed in controlled historical experiment | `corrective_retrieval.py` |
| `vanilla_generator_v1.md` | Earlier answer-generation baseline contract | Historical contract; not the final Agentic writer | generation-packet provenance |

Prompt hashes are frozen in the release manifests under [`data/manifests/`](../../data/manifests/). Public reproduction verifies those manifests rather than silently updating a prompt hash.

## Public prompt hashes

| File | SHA-256 |
|---|---|
| `adaptive_router_v1.md` | `a42016b5eef7cb4882c56947290a2ae5c9aac82ef9824c3766c90162046d419f` |
| `agentic_answer_writer_v1.md` | `887c97c6123d856ca0ece4355dcee666a15616edeaf882c5439246777778359c` |
| `agentic_orchestrator_v1.md` | `737b681a65d77364f0440df93d63b5ce0918b3b374c0111a20f206582bcdfebd` |
| `answer_evaluator_v1.md` | `fdf10980f4d5ae9265cca117a4de5ce6f7d97e26bfc7707ddae7a026c042e518` |
| `claim_drafter_v1.md` | `d66737b88c2a71296e2f5719b2509f53b650ebd88622f98093f66502c8f61836` |
| `evidence_assessor_agentic_v1.md` | `3ff22110b2df8dfebcede2c1cdd85fc8cf95a0b73f6e0be328b56a3bea5c0e48` |
| `evidence_assessor_final_v1.md` | `cd019aef5b58028f591101e89ff2031de815624b8237fe6af542755709bff0c9` |
| `evidence_assessor_initial_v2.md` | `cf8bcb4af33821fef5ee699c2ca77cc749004e164d86fddb1ed70858607c54b0` |
| `evidence_verifier_v1.md` | `e5a5b1937aae965600a699a4eadd478a66909430c93fa952cb25ee6e6642153f` |
| `multi_query_rewriter_v1.md` | `9cda703a9a17b5654777317d0bfe6c97a03b0006257002523cfa9cf3b1251aab` |
| `query_decomposer_v1.md` | `d5691b1ab9facd6b6763365be17cff1a68d501cb07b7068e76ad29634701ac6a` |
| `runtime_multi_query_rewriter_v1.md` | `bcb39fe0bd03bc29f9339bf5542d3cb232c7ef7ba61b9fa625600285fdac578c` |
| `runtime_query_decomposer_v1.md` | `4d9c4026224e8c8453192b69b5c9f1ecb445153dbd057a95117b53a9c9ffa445` |
| `scientific_analyst_v1.md` | `359de1175e5d4c2a64bb48c1d8df74603224504722f52c7c4bd3510aac213b04` |
| `vanilla_generator_v1.md` | `05066db757a4d26d0dad41c7f71c421e806ca1065891a3e0e0536b0d6927539b` |
