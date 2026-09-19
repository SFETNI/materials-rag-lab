# Data setup

**Dataset:** *Materials RAG Lab Dataset: IN718 Additive-Manufacturing Evidence and Benchmark for Progressive RAG*, v0.1.0  
**DOI:** <https://doi.org/10.5281/zenodo.22837229>

## Install the redistributable projection

```bash
materials-rag-lab data install /path/to/materials-rag-data-v0.1.0.zip --check
materials-rag-lab data install /path/to/materials-rag-data-v0.1.0.zip
materials-rag-lab data status
```

The installer accepts a ZIP or extracted root. It verifies `checksums.sha256`, rejects unsafe archive paths, requires exactly 252 retrieval units, is idempotent, and refuses to overwrite different runtime data. It installs the projection under `data/runtime/`.

## Corpus modes

- **252 units — redistributable:** normal public use and Level B replication.
- **277 units — full reconstructed:** frozen replication after lawful acquisition and ingestion of the 25 excluded units.
- **25 units — reference-only:** text from two sources is not redistributed.

The registry has a third entry for the NASA 20180005513 active-media PDF container. That is a packaging exclusion: its text-derived evidence remains in the 252-unit projection and does not add to the 25-unit text exclusion.

The installer never downloads restricted files. Full reconstruction follows the source registry and [Data and provenance](../docs/data_and_provenance.md).

After obtaining the two reference-only PDFs through authorized channels, reconstruct without redistributing them:

```bash
materials-rag-lab data reconstruct --source-root /path/to/authorized-sources --check
materials-rag-lab data reconstruct --source-root /path/to/authorized-sources --output data/runtime
```

The command verifies both PDF hashes, regenerates exactly 25 chunks, combines them with the installed 252-unit projection, requires 277 unique IDs, and verifies the frozen ordered ID and retrieval-text identity. It retains no source PDF. Release metadata paths are normalized, so the reconstructed JSONL is not claimed to be byte-identical to the private historical JSONL.

## Included repository data

- `manifests/`: frozen development, challenge, and final experiment records.
- `results/`: compact metrics derived from frozen outputs.
- `samples/`: safe format examples.
- `reconstruction/`: public source and acquisition registries.
- `reproducibility/`: recovered frozen configuration.

Raw PDFs, vectors, local databases, raw model responses, and intermediate outputs are excluded.
