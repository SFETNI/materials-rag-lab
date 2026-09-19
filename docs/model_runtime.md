# Model and runtime setup

## Frozen identities

| Purpose | Identifier | Revision |
|---|---|---|
| Dense embeddings | `BAAI/bge-base-en-v1.5` | Exact commit was not preserved in the frozen dense manifest; do not invent one. |
| Cross-encoder reranker | `BAAI/bge-reranker-base` | `2cfc18c9415c912f9d8155881c133215df768a70` |
| Isolated model roles | `gpt-5.5` | reasoning effort `low`, fresh local Codex CLI process per decision |

The retrieval implementation loads Hugging Face models with local-files-only behavior during scientific runs. Prepare the declared snapshots before retrieval:

```bash
materials-rag-lab models status
materials-rag-lab models prepare
materials-rag-lab doctor --capability hybrid
materials-rag-lab doctor --capability agentic
```

The preparation command downloads the public Hugging Face artifacts into the normal cache. The status command checks the embedding model, pinned reranker snapshot, and local `codex` executable. `doctor --capability` combines those checks with the installed dataset mode and reports `PASS`, `WARN`, or `FAIL` for the requested workflow.

Agentic runs require an authenticated Codex CLI that can execute `gpt-5.5`. Each role call uses an empty temporary directory, an ephemeral session, read-only sandbox, ignored project/user rules, low reasoning, JSON event auditing, and zero external tool calls. Non-model retrieval remains usable without Codex.

Using another embedding model, reranker revision, role model, provider, or reasoning effort is a new replication configuration. It must not be described as the frozen v0.1.0 run.

The supported public installation is defined by the repository `pyproject.toml` and `uv.lock`. The exact historical project specification is retained at [`data/reproducibility/pyproject.frozen.toml`](../data/reproducibility/pyproject.frozen.toml); see [`frozen_environment.json`](../data/reproducibility/frozen_environment.json) for the recovered configuration and its limitation.
