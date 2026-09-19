# Data and provenance

## Companion dataset

**Materials RAG Lab Dataset: IN718 Additive-Manufacturing Evidence and Benchmark for Progressive RAG**, version 0.1.0

DOI: [10.5281/zenodo.22837229](https://doi.org/10.5281/zenodo.22837229)

Expected final archive name: `materials-rag-data-v0.1.0.zip`.

## Corpus boundary

| Projection | Retrieval units | Description |
|---|---:|---|
| Full experimental corpus | 277 | Local research snapshot used for the frozen experiments |
| Redistributable dataset | 252 | Rights-filtered projection in the companion dataset |
| Reference-only exclusion | 25 | Units derived from two sources that are indexed locally but not redistributed |

Retained units keep their original IDs and retrieval text. The 252-unit projection is produced by filtering, not by renumbering or rewriting records.

The two reference-only sources are:

- Kafka et al., 2023, *International Journal of Fatigue*, DOI `10.1016/j.ijfatigue.2023.107872`;
- NASA NTRS `20230010244`, whose recorded rights determination supports government use but was conservatively classified as reference-only for public redistribution.

The registry also contains NASA NTRS `20180005513` as a **packaging-only exclusion**. Its original PDF container includes active media and is not packaged, while its text-derived canonical evidence remains in the 252-unit redistributable corpus. It is therefore not a third source contributing to the 25 excluded retrieval units.

## Reconstructing the full research corpus

The dataset archive includes a source registry, rights registry, inclusion manifest, reference-only registry, and `acquisition/verify_sources.py`.

1. Download the companion dataset from Zenodo.
2. Obtain reference-only sources through an authorized route.
3. Place them at the paths documented in `provenance/source_registry.jsonl`.
4. Install the 252-unit dataset, then run the public reconstruction command:

   ```bash
   materials-rag-lab data reconstruct --source-root /path/to/authorized-sources --check
   materials-rag-lab data reconstruct --source-root /path/to/authorized-sources --output data/runtime
   ```

5. Confirm `materials-rag-lab data status` reports `full-reconstructed-277`.

The reconstructor uses the existing PDF parser and chunker, checks the declared source hashes and 25 expected unit IDs, and verifies the ordered ID/retrieval-text identity recorded in `data/reconstruction/full_corpus_identity.json`. It does not copy the restricted PDFs into the repository or output directory.

For the 252-unit public projection, use the deterministic installer instead of manually copying files:

```bash
materials-rag-lab data install /path/to/materials-rag-data-v0.1.0.zip
```

## Evidence kinds

The canonical model includes:

- `Document`: source container and provenance;
- `Chunk`: page or section-level text evidence;
- `ExperimentRecord`: summarized experiment outcome;
- `ProcessRecord`: manufacturing or heat-treatment process data;
- `AnalysisRecord`: derived analytical artifact;
- `TestRecord`: test execution and fatigue-status record.

Structured Parquet artifacts retain numerical series and fields outside semantic retrieval. Computation receipts refer back to source record IDs and never become retrieval corpus entries.

## Synthetic benchmark data

The fictional internal corpus and its benchmark are project-authored. The benchmark contains 60 questions, 30 paired intents, a 36/24 DEV/challenge split, and 26 hard-negative annotations. Labels are `authored_not_expert_validated`.

Synthetic values are designed for controlled software evaluation. They are not NIST or NASA observations, design allowables, qualification evidence, or production acceptance criteria.

## Rights model

- Project-authored data and documentation: CC BY 4.0.
- Project-authored software, scripts, and schemas: Apache-2.0.
- NIST and NASA material: source-specific terms and attribution remain in force.
- Publisher-controlled or conservatively restricted material: reference-only and not included in the redistributable projection.

The public source registry is in `data/reconstruction/source_registry.jsonl`. Source-level rights records and archive-facing notices are included in the published companion dataset.

## Path privacy

Release projections normalize machine-local metadata to archive-relative paths. Public documentation and the companion archive should not contain usernames, local model cache paths, temporary directories, credentials, or access tokens.
