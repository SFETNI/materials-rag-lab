"""Source-specific canonical parsers for Phase 2A."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from materials_rag.ingestion.chunker import ChunkConfig, chunk_documents, write_chunks
from materials_rag.ingestion.hip_parser import parse_hip_workbook
from materials_rag.ingestion.pdf_parser import parse_pdf_documents
from materials_rag.ingestion.rbf_parser import parse_rbf_results_csv
from materials_rag.ingestion.sn_parser import parse_sn_analysis
from materials_rag.ingestion.test_log_parser import parse_rbf_test_logs
from materials_rag.ingestion.utils import resolve_repo_root, sha256_for_file, write_jsonl
from materials_rag.schemas import (
    AnalysisRecord,
    Chunk,
    Document,
    ExperimentRecord,
    ProcessRecord,
    SourceRef,
    TestRecord,
)


@dataclass
class Phase2AResult:
    """Container returned by run_phase2a."""

    experiment_records: list[ExperimentRecord]
    process_records: list[ProcessRecord]
    analysis_records: list[AnalysisRecord]
    documents: list[Document]
    warnings: list[dict[str, str]]
    outputs: list[Path]
    manifest_path: Path


@dataclass
class Phase2BResult:
    """Container returned by run_phase2b."""

    test_records: list[TestRecord]
    warnings: list[dict[str, str]]
    outputs: list[Path]
    manifest_path: Path


@dataclass
class Phase3Result:
    """Container returned by run_phase3."""

    chunks: list[Chunk]
    warnings: list[dict[str, str]]
    outputs: list[Path]
    manifest_path: Path


@dataclass
class Phase35AResult:
    """Container returned by run_phase35a."""

    documents: list[Document]
    chunks: list[Chunk]
    warnings: list[dict[str, str]]
    outputs: list[Path]
    manifest_path: Path


def _build_sources(repo_root: Path) -> dict[str, Path]:
    return {
        "nist_root": repo_root / "data" / "raw" / "external" / "nist_in718",
        "nasa_root": repo_root / "data" / "raw" / "external" / "nasa",
        "processed_root": repo_root / "data" / "processed",
    }


def discover_publication_pdf_sources(repo_root: Path) -> list[tuple[Path, SourceRef]]:
    """Discover public PDF sources for the expanded narrative corpus."""

    raw_external = repo_root / "data" / "raw" / "external"
    nasa_root = raw_external / "nasa"
    nist_root = raw_external / "nist_in718"
    search_roots = [
        ("nasa", nasa_root, "publications"),
        ("nist_in718", nist_root / "part_drawings", "part_drawings"),
        ("nist_in718", nist_root / "publications", "publications"),
    ]

    discovered: dict[str, tuple[Path, SourceRef]] = {}
    for dataset, search_root, source_dataset in search_roots:
        if not search_root.exists():
            continue
        for path in sorted(search_root.glob("*.pdf"), key=lambda value: value.name.lower()):
            source_path = path.relative_to(raw_external).as_posix()
            discovered[source_path] = (
                path,
                SourceRef(
                    dataset=dataset,
                    source_path=source_path,
                    source_dataset=source_dataset,
                    file_hash=sha256_for_file(path),
                ),
            )

    return [discovered[key] for key in sorted(discovered)]


def _make_input_hashes(repo_root: Path, paths: list[Path]) -> dict[str, str]:
    return {
        (
            path.relative_to(repo_root)
            if path.is_absolute()
            else path
        ).as_posix(): sha256_for_file(path)
        for path in paths
    }


def _load_phase2a_experiment_records(repo_root: Path) -> list[ExperimentRecord]:
    experiment_path = (
        repo_root / "data" / "processed" / "phase2a" / "canonical" / "experiment_records.jsonl"
    )
    if not experiment_path.exists():
        return []

    records: list[ExperimentRecord] = []
    for line in experiment_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload.get("kind") == "experiment_record":
            records.append(ExperimentRecord.model_validate(payload))
    return records


def _zip_member_count(zip_path: Path) -> int:
    import zipfile

    with zipfile.ZipFile(zip_path) as zf:
        return len([name for name in zf.namelist() if not name.endswith("/")])


def _summarize_phase2b_counts(records: list[TestRecord]) -> dict[str, int]:
    counts = defaultdict(int)
    for record in records:
        counts["logical_test_record_count"] += 1
        if record.link_status == "linked":
            counts["linked"] += 1
        elif record.link_status == "ambiguous":
            counts["ambiguous"] += 1
        else:
            counts["unmatched"] += 1

        has_csv = bool(record.metadata.get("raw_csv_member"))
        has_test = bool(record.metadata.get("raw_test_member"))
        if has_csv and has_test:
            counts["paired_csv_test"] += 1
        elif has_csv:
            counts["csv_only"] += 1
        elif has_test:
            counts["test_only"] += 1
    return dict(counts)


def _load_phase2a_documents(repo_root: Path) -> list[Document]:
    documents_path = repo_root / "data" / "processed" / "phase2a" / "canonical" / "documents.jsonl"
    if not documents_path.exists():
        return []
    docs: list[Document] = []
    for line in documents_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload.get("kind") == "document":
            docs.append(Document.model_validate(payload))
    return sorted(docs, key=lambda value: value.id)


def _load_phase2a_warnings(repo_root: Path) -> list[dict[str, str]]:
    manifest_path = repo_root / "data" / "processed" / "manifests" / "phase_2a_manifest.json"
    if not manifest_path.exists():
        return []
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    warnings = payload.get("warnings", [])
    return [warning for warning in warnings if isinstance(warning, dict)]


def _median(values: list[int]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2 == 1:
        return float(sorted_values[mid])
    return float((sorted_values[mid - 1] + sorted_values[mid]) / 2)


def _page_char_counts(document: Document) -> list[int]:
    counts: list[int] = []
    for page in document.metadata.get("page_text_normalized", []):
        if isinstance(page, dict):
            counts.append(len(str(page.get("text", ""))))
    return counts


def _classify_pdf_document(document: Document) -> str:
    title = (document.title or "").lower()
    metadata_title = str(document.metadata.get("pdf_metadata_title", "")).lower()
    path = document.source.source_path.lower()
    page_text = " ".join(
        str(page.get("text", ""))
        for page in document.metadata.get("page_text_normalized", [])[:3]
        if isinstance(page, dict)
    ).lower()
    evidence = f"{title} {metadata_title} {path} {page_text}"
    if "part_drawings" in path:
        return "engineering_drawing"
    if "nasa-tm" in evidence or "technical memorandum" in evidence:
        return "technical_report"
    if "powerpoint presentation" in evidence or "presentation" in evidence:
        return "presentation"
    if (
        "journal" in evidence
        or "conference" in evidence
        or "abstract" in evidence
        or "elsevier" in evidence
    ):
        return "journal_or_conference_paper"
    return "other"


def _enrich_publication_document(document: Document) -> None:
    page_char_counts = _page_char_counts(document)
    low_text_pages = [
        page_index + 1 for page_index, char_count in enumerate(page_char_counts) if char_count < 100
    ]
    figure_heavy_pages = [
        page_index + 1 for page_index, char_count in enumerate(page_char_counts) if char_count < 150
    ]
    source_path = Path(document.metadata.get("source_path", ""))
    document.metadata.update(
        {
            "document_type": _classify_pdf_document(document),
            "page_char_counts": page_char_counts,
            "low_text_pages": low_text_pages,
            "figure_heavy_candidate_pages": figure_heavy_pages,
            "source_sha256": document.source.file_hash,
        }
    )
    if source_path.exists():
        document.metadata["file_size_bytes"] = source_path.stat().st_size
    document.document_type = str(document.metadata["document_type"])


def _warning_counts_by_document_type(
    warnings: list[dict[str, str]],
) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for warning in warnings:
        source_path = warning.get("source_path") or warning.get("document_id") or "<unknown>"
        warning_type = warning.get("type", "<unknown>")
        counts[source_path][warning_type] += 1
    return {key: dict(value) for key, value in sorted(counts.items())}


def _duplicate_hash_groups(source_hashes: dict[str, str]) -> list[list[str]]:
    by_hash: dict[str, list[str]] = defaultdict(list)
    for path, digest in source_hashes.items():
        by_hash[digest].append(path)
    return [sorted(paths) for paths in by_hash.values() if len(paths) > 1]


def _load_jsonl_models(path: Path, model: type[Document | Chunk]) -> list[Document] | list[Chunk]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(model.model_validate(json.loads(line)))
    return records


def _old_corpus_stability_report(
    repo_root: Path,
    expanded_documents: list[Document],
    expanded_chunks: list[Chunk],
) -> dict[str, object]:
    old_document_path = repo_root / "data" / "processed" / "phase2a" / "canonical" / "documents.jsonl"
    old_chunk_path = repo_root / "data" / "processed" / "phase3" / "canonical" / "chunks.jsonl"
    old_documents = _load_jsonl_models(old_document_path, Document)
    old_chunks = _load_jsonl_models(old_chunk_path, Chunk)
    old_document_ids = sorted(document.id for document in old_documents)
    expanded_document_ids = {document.id for document in expanded_documents}

    old_chunk_counts = Counter(chunk.document_id for chunk in old_chunks)
    expanded_chunk_counts = Counter(chunk.document_id for chunk in expanded_chunks)
    missing_document_ids = [doc_id for doc_id in old_document_ids if doc_id not in expanded_document_ids]
    chunk_count_changes = {
        doc_id: {
            "old": old_chunk_counts.get(doc_id, 0),
            "expanded": expanded_chunk_counts.get(doc_id, 0),
        }
        for doc_id in old_document_ids
        if old_chunk_counts.get(doc_id, 0) != expanded_chunk_counts.get(doc_id, 0)
    }

    return {
        "old_document_ids": old_document_ids,
        "old_document_ids_preserved": not missing_document_ids,
        "missing_old_document_ids": missing_document_ids,
        "old_chunk_counts_preserved": not chunk_count_changes,
        "chunk_count_changes": chunk_count_changes,
    }


def run_phase2a(
    repo_root: Path | None = None,
    output_root: Path | None = None,
) -> Phase2AResult:
    """Run deterministic Phase 2A parsing and write canonical outputs."""

    if repo_root is None:
        repo_root = resolve_repo_root(Path(__file__))
    if output_root is None:
        output_root = repo_root / "data" / "processed"

    paths = _build_sources(repo_root)
    nist_root = paths["nist_root"]
    nasa_root = paths["nasa_root"]
    processed_root = output_root
    canonical_root = processed_root / "phase2a" / "canonical"

    phase_warnings: list[dict[str, str]] = []
    generated_paths: list[Path] = []

    # 1) RBF experiment rows
    rbf_csv = nist_root / "fatigue_data" / "RBF_Results_30um.csv"
    rbf_source = SourceRef(
        dataset="nist_in718",
        source_path=rbf_csv.relative_to(repo_root / "data" / "raw" / "external").as_posix(),
        source_dataset="fatigue_data",
    )
    experiment_records, rbf_warnings, _ = parse_rbf_results_csv(rbf_csv, rbf_source)
    phase_warnings.extend(rbf_warnings)
    experiments_file = canonical_root / "experiment_records.jsonl"
    write_jsonl(experiment_records, experiments_file)
    generated_paths.append(experiments_file)

    # 2) HIP process + sensor table -> parquet
    hip_xlsx = nist_root / "build_data" / "heat_treatment" / "HIP_cycles NIST In718 11292021.xlsx"
    hip_source = SourceRef(
        dataset="nist_in718",
        source_path=hip_xlsx.relative_to(repo_root / "data" / "raw" / "external").as_posix(),
        source_dataset="build_data",
    )
    process_record, hip_warnings, hip_outputs = parse_hip_workbook(hip_xlsx, hip_source, processed_root)
    process_records = [process_record]
    phase_warnings.extend(hip_warnings)
    generated_paths.extend(hip_outputs)
    process_file = canonical_root / "process_records.jsonl"
    write_jsonl(process_records, process_file)
    generated_paths.append(process_file)

    # 3) SN and confidence
    sn_csv = nist_root / "fatigue_stats" / "SN_curve_fits.csv"
    conf_csv = nist_root / "fatigue_stats" / "confidence_bands.csv"
    analysis_source = SourceRef(
        dataset="nist_in718",
        source_path=sn_csv.relative_to(repo_root / "data" / "raw" / "external").as_posix(),
        source_dataset="fatigue_stats",
    )
    analysis_records, sn_warnings, sn_outputs = parse_sn_analysis(sn_csv, conf_csv, analysis_source, processed_root)
    phase_warnings.extend(sn_warnings)
    generated_paths.extend(sn_outputs)
    analysis_file = canonical_root / "analysis_records.jsonl"
    write_jsonl(analysis_records, analysis_file)
    generated_paths.append(analysis_file)

    # 4) PDFs (no chunking yet, no OCR)
    nasa_pdf = nasa_root / "20150016245.pdf"
    kafka_pdf_a = nist_root / "part_drawings" / "Kafka_AM_RBF_finishing_steps.pdf"
    kafka_pdf_b = nist_root / "part_drawings" / "Kafka_AM_cylinder_to_RBF.pdf"
    pdf_spec = [
        (
            nasa_pdf,
            SourceRef(
                dataset="nasa",
                source_path=nasa_pdf.relative_to(repo_root / "data" / "raw" / "external").as_posix(),
                source_dataset="pdf",
            ),
        ),
        (
            kafka_pdf_a,
            SourceRef(
                dataset="nist_in718",
                source_path=kafka_pdf_a.relative_to(repo_root / "data" / "raw" / "external").as_posix(),
                source_dataset="part_drawings",
            ),
        ),
        (
            kafka_pdf_b,
            SourceRef(
                dataset="nist_in718",
                source_path=kafka_pdf_b.relative_to(repo_root / "data" / "raw" / "external").as_posix(),
                source_dataset="part_drawings",
            ),
        ),
    ]

    document_records, pdf_warnings = parse_pdf_documents(pdf_spec)
    phase_warnings.extend(pdf_warnings)
    documents_file = canonical_root / "documents.jsonl"
    write_jsonl(document_records, documents_file)
    generated_paths.append(documents_file)

    # Manifest
    input_files = [rbf_csv, hip_xlsx, sn_csv, conf_csv, nasa_pdf, kafka_pdf_a, kafka_pdf_b]
    source_hashes = _make_input_hashes(repo_root, input_files)
    counts: dict[str, int] = {
        "ExperimentRecord": len(experiment_records),
        "ProcessRecord": len(process_records),
        "AnalysisRecord": len(analysis_records),
        "Document": len(document_records),
    }
    output_artifacts = [
        {
            "path": path.relative_to(repo_root).as_posix(),
            "type": "parquet" if path.suffix == ".parquet" else "jsonl",
        }
        for path in generated_paths
    ]
    manifest = {
        "phase": "2A",
        "version": "1.0.0",
        "repo_root": repo_root.as_posix(),
        "input_files": [path.as_posix() for path in input_files],
        "source_hashes": source_hashes,
        "output_artifacts": output_artifacts,
        "counts": counts,
        "object_counts": {
            "experiment_records": len(experiment_records),
            "process_records": len(process_records),
            "analysis_records": len(analysis_records),
            "documents": len(document_records),
        },
        "warnings": phase_warnings,
    }
    manifest_root = processed_root / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_root / "phase_2a_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return Phase2AResult(
        experiment_records=experiment_records,
        process_records=process_records,
        analysis_records=analysis_records,
        documents=document_records,
        warnings=phase_warnings,
        outputs=generated_paths + [manifest_path],
        manifest_path=manifest_path,
    )


def run_phase2b(
    repo_root: Path | None = None,
    output_root: Path | None = None,
) -> Phase2BResult:
    """Run deterministic Phase 2B parsing for raw fatigue test logs."""

    if repo_root is None:
        repo_root = resolve_repo_root(Path(__file__))
    if output_root is None:
        output_root = repo_root / "data" / "processed"

    paths = _build_sources(repo_root)
    nist_root = paths["nist_root"]
    processed_root = output_root
    canonical_root = processed_root / "phase2b" / "canonical"

    phase_warnings: list[dict[str, str]] = []
    generated_paths: list[Path] = []

    zip_path = nist_root / "fatigue_data" / "IN718_30um_RBF_test_logs.zip"
    source = SourceRef(
        dataset="nist_in718",
        source_path=zip_path.relative_to(repo_root / "data" / "raw" / "external").as_posix(),
        source_dataset="fatigue_data",
    )

    experiment_records = _load_phase2a_experiment_records(repo_root)
    test_records, parse_warnings, parquet_outputs = parse_rbf_test_logs(
        zip_path,
        source,
        processed_root,
        experiment_records=experiment_records,
    )
    phase_warnings.extend(parse_warnings)

    for record in test_records:
        if record.file_variant == "buffer_full":
            phase_warnings.append(
                {
                    "type": "variant_detected",
                    "record_id": record.id,
                    "specimen_id": record.specimen_id,
                }
            )
        if record.specimen_id == "6.2" and record.run_id == "test2":
            phase_warnings.append(
                {
                    "type": "special_run_detected",
                    "record_id": record.id,
                    "specimen_id": record.specimen_id,
                    "run_id": record.run_id,
                }
            )

    test_file = canonical_root / "test_records.jsonl"
    write_jsonl(test_records, test_file)
    generated_paths.append(test_file)
    generated_paths.extend(parquet_outputs)

    counts = _summarize_phase2b_counts(test_records)
    manifest_counts = {
        "TestRecord": counts.get("logical_test_record_count", 0),
        "linked": counts.get("linked", 0),
        "unmatched": counts.get("unmatched", 0),
        "ambiguous": counts.get("ambiguous", 0),
        "paired": counts.get("paired_csv_test", 0),
        "csv_only": counts.get("csv_only", 0),
        "test_only": counts.get("test_only", 0),
    }

    source_hashes = _make_input_hashes(repo_root, [zip_path])
    output_artifacts = [
        {
            "path": path.relative_to(repo_root).as_posix(),
            "type": "parquet" if path.suffix == ".parquet" else "jsonl",
        }
        for path in generated_paths
    ]
    manifest = {
        "phase": "2B",
        "version": "1.0.0",
        "repo_root": repo_root.as_posix(),
        "input_files": [zip_path.as_posix()],
        "source_hashes": source_hashes,
        "zip_member_count": _zip_member_count(zip_path),
        "logical_test_record_count": counts.get("logical_test_record_count", 0),
        "linked_test_record_count": counts.get("linked", 0),
        "unmatched_test_record_count": counts.get("unmatched", 0),
        "ambiguous_test_record_count": counts.get("ambiguous", 0),
        "paired_csv_test_count": counts.get("paired_csv_test", 0),
        "csv_only_count": counts.get("csv_only", 0),
        "test_only_count": counts.get("test_only", 0),
        "warnings": phase_warnings,
        "object_counts": {
            "test_records": len(test_records),
        },
        "counts": manifest_counts,
        "output_artifacts": output_artifacts,
    }
    manifest_root = processed_root / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_root / "phase_2b_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return Phase2BResult(
        test_records=test_records,
        warnings=phase_warnings,
        outputs=generated_paths + [manifest_path],
        manifest_path=manifest_path,
    )


def run_phase3(
    repo_root: Path | None = None,
    output_root: Path | None = None,
    config: ChunkConfig | None = None,
) -> Phase3Result:
    """Run deterministic Phase 3 chunking for narrative Document objects only."""

    if repo_root is None:
        repo_root = resolve_repo_root(Path(__file__))
    if output_root is None:
        output_root = repo_root / "data" / "processed"

    phase3_root = output_root / "phase3" / "canonical"
    phase3_root.mkdir(parents=True, exist_ok=True)
    canonical_documents_root = repo_root / "data" / "processed" / "phase2a" / "canonical"
    phase2a_documents = canonical_documents_root / "documents.jsonl"

    documents = _load_phase2a_documents(repo_root)
    warnings: list[dict[str, str]] = []
    if not documents:
        warnings.append({"type": "no_documents", "path": phase2a_documents.as_posix()})

    chunks, chunk_warnings = chunk_documents(documents, config=config)
    warnings.extend(chunk_warnings)
    phase2a_warnings = _load_phase2a_warnings(repo_root)
    pdf_quality_warnings = [
        warning
        for warning in phase2a_warnings
        if warning.get("type")
        in {
            "compatibility_normalization_difference",
            "suspicious_unicode_glyph",
            "suspicious_technical_unit",
        }
    ]

    output_paths = write_chunks(chunks, output_root)

    input_paths: list[Path] = []
    for doc in documents:
        source_path = Path(repo_root / "data" / "raw" / "external" / doc.source.source_path)
        if source_path.exists():
            input_paths.append(source_path)
        else:
            # Fall back to repo-relative interpretation if a path was already absolute.
            candidate = Path(doc.source.source_path)
            if candidate.exists():
                input_paths.append(candidate)

    manifest_counts = {
        "total_chunks": len(chunks),
        "documents_chunked": len(documents),
        "chunks_per_document": {
            doc.id: sum(1 for chunk in chunks if chunk.document_id == doc.id) for doc in documents
        },
        "cross_page_chunks": 0,
        "very_small_chunks": 0,
        "token_count_min": 0,
        "token_count_median": 0,
        "token_count_mean": 0,
        "token_count_max": 0,
        "overlap_token_min": 0,
        "overlap_token_median": 0,
        "overlap_token_mean": 0,
        "overlap_token_max": 0,
        "suspicious_pdf_warning_count": len(pdf_quality_warnings),
    }

    if chunks:
        manifest_counts["cross_page_chunks"] = sum(
            1
            for chunk in chunks
            if chunk.page_start is not None
            and chunk.page_end is not None
            and chunk.page_end > chunk.page_start
        )
        manifest_counts["very_small_chunks"] = len([chunk for chunk in chunks if chunk.token_count < 90])
        token_counts = sorted(chunk.token_count for chunk in chunks)
        manifest_counts["token_count_min"] = token_counts[0]
        manifest_counts["token_count_max"] = token_counts[-1]
        manifest_counts["token_count_mean"] = float(sum(token_counts) / len(token_counts))
        manifest_counts["token_count_median"] = _median(token_counts)
        overlap_counts = sorted(
            int(chunk.metadata.get("overlap_tokens_with_previous", 0)) for chunk in chunks
        )
        manifest_counts["overlap_token_min"] = overlap_counts[0]
        manifest_counts["overlap_token_max"] = overlap_counts[-1]
        manifest_counts["overlap_token_mean"] = float(sum(overlap_counts) / len(overlap_counts))
        manifest_counts["overlap_token_median"] = _median(overlap_counts)

    manifest = {
        "phase": "3",
        "version": "1.0.0",
        "input_files": [path.as_posix() for path in input_paths],
        "source_hashes": _make_input_hashes(repo_root, input_paths),
        "output_artifacts": [
            {"path": path.relative_to(repo_root).as_posix(), "type": "jsonl"} for path in output_paths
        ],
        **manifest_counts,
        "pdf_quality_warnings": pdf_quality_warnings,
        "warnings": warnings,
    }

    manifest_path = output_root / "manifests" / "phase_3_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return Phase3Result(
        chunks=chunks,
        warnings=warnings,
        outputs=output_paths + [manifest_path],
        manifest_path=manifest_path,
    )


def run_phase35a(
    repo_root: Path | None = None,
    output_root: Path | None = None,
    config: ChunkConfig | None = None,
) -> Phase35AResult:
    """Run Phase 3.5A expanded public-PDF corpus parsing and chunking."""

    if repo_root is None:
        repo_root = resolve_repo_root(Path(__file__))
    if output_root is None:
        output_root = repo_root / "data" / "processed"
    if config is None:
        config = ChunkConfig()

    canonical_root = output_root / "phase3_5a" / "canonical"
    canonical_root.mkdir(parents=True, exist_ok=True)
    pdf_sources = discover_publication_pdf_sources(repo_root)

    documents, pdf_warnings = parse_pdf_documents(pdf_sources)
    for document in documents:
        _enrich_publication_document(document)

    chunks, chunk_warnings = chunk_documents(documents, config=config)
    warnings = pdf_warnings + chunk_warnings

    documents_path = canonical_root / "documents.jsonl"
    chunks_path = canonical_root / "chunks.jsonl"
    write_jsonl(documents, documents_path)
    write_jsonl(chunks, chunks_path)

    input_paths = [path for path, _ in pdf_sources]
    source_hashes = _make_input_hashes(repo_root, input_paths)
    token_counts = sorted(chunk.token_count for chunk in chunks)
    low_text_page_counts = {
        document.source.source_path: len(document.metadata.get("low_text_pages", []))
        for document in documents
    }
    chunks_per_document = {
        document.id: sum(1 for chunk in chunks if chunk.document_id == document.id)
        for document in documents
    }
    chunks_per_pdf = {
        document.source.source_path: chunks_per_document.get(document.id, 0)
        for document in documents
    }

    output_paths = [documents_path, chunks_path]
    duplicate_groups = _duplicate_hash_groups(source_hashes)
    manifest = {
        "phase": "3.5A",
        "version": "1.0.0",
        "input_files": [path.as_posix() for path in input_paths],
        "source_hashes": source_hashes,
        "output_artifacts": [
            {"path": path.relative_to(repo_root).as_posix(), "type": "jsonl"}
            for path in output_paths
        ],
        "pdf_count": len(pdf_sources),
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "chunks_per_document": chunks_per_document,
        "chunks_per_pdf": chunks_per_pdf,
        "token_count_min": token_counts[0] if token_counts else 0,
        "token_count_median": _median(token_counts),
        "token_count_mean": float(sum(token_counts) / len(token_counts)) if token_counts else 0.0,
        "token_count_max": token_counts[-1] if token_counts else 0,
        "low_text_page_counts": low_text_page_counts,
        "low_text_pages": {
            document.source.source_path: document.metadata.get("low_text_pages", [])
            for document in documents
        },
        "cross_page_chunk_count": sum(
            1
            for chunk in chunks
            if chunk.page_start is not None
            and chunk.page_end is not None
            and chunk.page_end > chunk.page_start
        ),
        "suspicious_extraction_warning_counts": _warning_counts_by_document_type(warnings),
        "duplicate_detection": {
            "duplicate_pdf_sha256_groups": duplicate_groups,
            "has_duplicates": bool(duplicate_groups),
        },
        "chunking_config": {
            "target_tokens": config.target_tokens,
            "max_tokens": config.max_tokens,
            "overlap_tokens": config.overlap_tokens,
            "tiny_chunk_threshold": config.tiny_chunk_threshold,
        },
        "old_corpus_stability": _old_corpus_stability_report(repo_root, documents, chunks),
        "warnings": warnings,
    }

    manifest_path = output_root / "manifests" / "phase_3_5a_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    return Phase35AResult(
        documents=documents,
        chunks=chunks,
        warnings=warnings,
        outputs=output_paths + [manifest_path],
        manifest_path=manifest_path,
    )
