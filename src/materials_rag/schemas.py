"""Canonical model layer for Phase 1 materials knowledge graph objects."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

CanonicalIDKind = Literal[
    "document",
    "knowledge_unit",
    "chunk",
    "experiment_record",
    "process_record",
    "analysis_record",
    "test_record",
]

LinkStatus = Literal["linked", "no_experiment_match", "ambiguous"]


class SourceRef(BaseModel):
    """Source location metadata for every retrievable object."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    dataset: str = Field(..., description="Logical dataset name, e.g. 'nist_in718'.")
    source_path: str = Field(..., description="Relative path inside the corpus.")
    authority: str = Field(default="external_reference", description="Provenance authority.")
    source_dataset: str | None = Field(default=None, description="Optional source subcategory.")
    source_revision: str | None = Field(default=None, description="Optional dataset revision.")
    file_hash: str | None = Field(default=None, description="Optional file hash.")


def make_human_readable_id(kind: CanonicalIDKind, *parts: str) -> str:
    """Create deterministic IDs from normalized components.

    Example:
        make_human_readable_id(
            "experiment_record", "nist_in718", "RBF_Results_30um", "1.2"
        ) -> "experiment_record|nist_in718|RBF_Results_30um|1.2"
    """

    normalized = [str(part).strip().replace("|", "_") for part in parts if part]
    return "|".join([kind, *normalized])


class RetrieverObject(BaseModel):
    """Shared fields required for every object used in retrieval."""

    model_config = ConfigDict(
        str_strip_whitespace=True, extra="forbid", populate_by_name=True
    )

    id: str = Field(..., description="Deterministic object id.")
    kind: CanonicalIDKind = Field(..., description="Object kind used for retrieval filters.")
    retrieval_text: str = Field(
        ..., min_length=1, description="Chunk-style human-readable retrieval text."
    )
    source: SourceRef
    metadata: dict[str, Any] = Field(default_factory=dict)


class Document(RetrieverObject):
    kind: Literal["document"] = "document"

    title: str | None = None
    document_type: str = "pdf"
    language: str | None = None


class KnowledgeUnit(RetrieverObject):
    kind: Literal["knowledge_unit"] = "knowledge_unit"

    document_id: str = Field(..., description="Parent document identifier.")
    section: str | None = None
    order: int | None = None


class Chunk(RetrieverObject):
    kind: Literal["chunk"] = "chunk"

    document_id: str = Field(..., description="Parent document identifier.")
    chunk_index: int = Field(..., ge=0, description="Deterministic chunk position.")
    page_start: int | None = Field(default=None, ge=1, description="Starting PDF page index (1-based).")
    page_end: int | None = Field(default=None, ge=1, description="Ending PDF page index (1-based).")
    start_char: int | None = Field(default=None, ge=0)
    end_char: int | None = Field(default=None, ge=0)
    text: str = Field(..., min_length=1, description="Canonical chunk text as processed.")
    token_count: int = Field(..., ge=1, description="Approximate token count for the chunk.")
    artifact_path: str | None = Field(
        default=None,
        description="Path for persisted chunk artifacts, when chunk serialization is externalized.",
    )


class ExperimentRecord(RetrieverObject):
    """Represents one row-level experimental result from the fatigue summary table."""

    kind: Literal["experiment_record"] = "experiment_record"

    specimen_id: str = Field(..., description="Canonical specimen id, e.g. '1.2'.")
    specimen_id_raw: str = Field(..., description="Original specimen id from source row.")
    specimen_run: str | None = Field(default=None, description="Optional run variant, e.g. 'test2'.")
    file_variant: str = Field(
        default="standard",
        description="File variant label, e.g. 'buffer_full' for buffer-only files.",
    )
    layer_height_mm: float | None = None
    diameter_top_mm: float | None = None
    diameter_middle_mm: float | None = None
    diameter_bottom_mm: float | None = None
    stress_level_mpa: float | None = None
    force_to_be_applied_n: float | None = None
    cycles_to_failure: float | None = None
    specimen_notes: str | None = Field(default=None, alias="notes")
    specimen_type: str | None = None
    runout_status: bool | None = None
    initial_variability_check: str | None = None
    force_variability_from_test_log: str | None = None
    stress_variability: str | None = None


class ProcessRecord(RetrieverObject):
    """Represents process-level settings / cycle metadata such as HIP treatment."""

    kind: Literal["process_record"] = "process_record"

    specimen_id: str | None = Field(default=None, description="Optional associated specimen id.")
    specimen_id_raw: str | None = Field(default=None, description="Original specimen-like identifier.")
    cycle_number: int | None = None
    pressure_mpa: float | None = None
    temperature_c: float | None = None
    hold_time_min: float | None = None
    pressure_medium: str | None = None
    cooling_type: str | None = None
    process_name: str | None = None
    cycle_data_path: str | None = Field(
        default=None,
        description="External reference to time-series or profile curve file path.",
    )


class AnalysisRecord(RetrieverObject):
    """Represents fitted analytical summaries such as SN-curve fits and confidence data."""

    kind: Literal["analysis_record"] = "analysis_record"

    analysis_type: str = Field(..., description="e.g., 'sn_curve_fit' or 'confidence_band'.")
    method: str | None = None
    x_axis_label: str | None = None
    y_axis_label: str | None = None
    fit_parameters: dict[str, float | int | str] = Field(default_factory=dict)
    fit_metrics: dict[str, float] = Field(default_factory=dict)
    source_data_path: str = Field(..., description="Path to source dataset behind the analysis.")
    output_data_path: str | None = Field(default=None, description="Optional cached model output path.")


class TestRecord(RetrieverObject):
    """Represents raw fatigue test runs stored as external files (csv/test metadata)."""

    __test__ = False

    kind: Literal["test_record"] = "test_record"

    specimen_id: str = Field(..., description="Canonical specimen id.")
    specimen_id_raw: str = Field(..., description="Raw specimen id from filename/context.")
    run_id: str | None = None
    file_variant: str = Field(
        default="standard",
        description="Run/file variant like 'buffer_full'.",
    )
    csv_artifact_path: str = Field(..., description="Path to raw fatigue .csv signal file.")
    test_metadata_path: str | None = Field(
        default=None, description="Path to paired .test metadata artifact."
    )
    signal_columns: list[str] = Field(default_factory=list)
    cycle_count: int | None = None
    statistics: dict[str, dict[str, float]] = Field(default_factory=dict)
    experiment_record_id: str | None = Field(
        default=None, description="Linked canonical experiment record identifier."
    )
    link_status: LinkStatus = Field(
        default="no_experiment_match",
        description="How confidently this log links to an ExperimentRecord.",
    )


def infer_document_id(source: SourceRef) -> str:
    """Create a deterministic document ID from source path."""

    rel = Path(source.source_path).stem
    return make_human_readable_id("document", source.dataset, rel)


def infer_chunk_id(document_id: str, chunk_index: int) -> str:
    """Create a deterministic chunk ID."""

    return make_human_readable_id("chunk", document_id, str(chunk_index))
