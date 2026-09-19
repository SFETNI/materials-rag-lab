"""Parser for RBF_Results_30um.csv into ExperimentRecord objects."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from materials_rag.ingestion.normalization import normalize_specimen_identifier
from materials_rag.schemas import ExperimentRecord, SourceRef, make_human_readable_id


def _coerce_float(raw: object) -> tuple[float | None, str | None, bool]:
    """Return (value, raw_token, numeric_ok)."""

    if pd.isna(raw):
        return None, None, True
    token = str(raw).strip()
    if token == "":
        return None, "", False
    try:
        return float(token), None, True
    except ValueError:
        return None, token, False


def _optional_text(raw: object) -> str | None:
    if pd.isna(raw):
        return None
    text = str(raw)
    return text if text != "" else None


def _derive_runout_status(raw_note: object) -> bool | None:
    if pd.isna(raw_note):
        return None
    text = str(raw_note)
    if not text:
        return None
    lower = text.lower()
    return "runout" in lower


def parse_rbf_results_csv(
    csv_path: Path,
    source: SourceRef,
) -> tuple[list[ExperimentRecord], list[dict[str, str]], int]:
    """Parse a clean fatigue summary CSV into canonical experiment records."""

    df = pd.read_csv(csv_path)
    # Canonical representation should remove completely empty trailing columns.
    df = df.dropna(axis=1, how="all").copy()

    records: list[ExperimentRecord] = []
    warnings: list[dict[str, str]] = []

    for row_index, row in df.iterrows():
        raw_specimen = row.get("Number")
        if pd.isna(raw_specimen):
            warnings.append(
                {
                    "type": "skipped_row",
                    "reason": "missing_specimen_id",
                    "row_index": str(int(row_index)),
                }
            )
            continue

        normalized = normalize_specimen_identifier(str(raw_specimen))

        stress, raw_stress_abnormal, stress_numeric_ok = _coerce_float(row.get("Stress Level [MPa]"))
        force, raw_force_abnormal, force_numeric_ok = _coerce_float(row.get("Force to be applied [N]"))

        cycles, _, _ = _coerce_float(row.get("Cycles to failure"))

        layer_height, _, _ = _coerce_float(row.get("Layer Height"))
        diameter_top, _, _ = _coerce_float(
            row.get("Diameter [mm] (caliper) (4.04 nominal) [top]")
        )
        diameter_middle, _, _ = _coerce_float(
            row.get("Diameter [mm] (caliper) (4.04 nominal) [middle]")
        )
        diameter_bottom, _, _ = _coerce_float(
            row.get("Diameter [mm] (caliper) (4.04 nominal)  [bottom]")
        )

        notes = _optional_text(row.get("Notes"))
        runout_status = _derive_runout_status(notes)
        retrieval_text = (
            f"Specimen {normalized.canonical_specimen_id} "
            f"(raw {normalized.raw_identifier}) from {source.source_path}: "
            f"Type={row.get('Type')} stress={stress if stress_numeric_ok else 'NA'} "
            f"force={force if force_numeric_ok else 'NA'} "
            f"cycles={cycles if cycles is not None else 'NA'}."
        )

        record_id = make_human_readable_id(
            "experiment_record",
            source.dataset,
            "RBF_Results_30um",
            normalized.canonical_specimen_id,
            str(int(row_index)),
        )

        metadata = {
            "source_row_number": int(row_index) + 2,
            "source_row_hash_source": csv_path.as_posix(),
            "raw_stress_level_token": row.get("Stress Level [MPa]") if not stress_numeric_ok else None,
            "raw_force_to_be_applied_token": row.get(
                "Force to be applied [N]"
            ) if not force_numeric_ok else None,
        }

        record = ExperimentRecord(
            id=record_id,
            retrieval_text=retrieval_text,
            source=source,
            specimen_id=normalized.canonical_specimen_id,
            specimen_id_raw=normalized.raw_identifier,
            specimen_run=normalized.run_id,
            file_variant=normalized.file_variant,
            layer_height_mm=layer_height,
            diameter_top_mm=diameter_top,
            diameter_middle_mm=diameter_middle,
            diameter_bottom_mm=diameter_bottom,
            stress_level_mpa=stress,
            force_to_be_applied_n=force,
            cycles_to_failure=cycles,
            notes=notes,
            initial_variability_check=str(row.get("Initial variability check"))
            if pd.notna(row.get("Initial variability check"))
            else None,
            force_variability_from_test_log=str(row.get("Force variability from test log"))
            if pd.notna(row.get("Force variability from test log"))
            else None,
            stress_variability=str(row.get("Stress variability"))
            if pd.notna(row.get("Stress variability"))
            else None,
            specimen_type=str(row.get("Type")) if pd.notna(row.get("Type")) else None,
            runout_status=runout_status,
            metadata=metadata,
        )
        records.append(record)

        if (not stress_numeric_ok and raw_stress_abnormal is not None) or (
            not force_numeric_ok and raw_force_abnormal is not None
        ):
            warnings.append(
                {
                    "type": "non_numeric_measure",
                    "row_index": str(int(row_index)),
                    "specimen_id": normalized.canonical_specimen_id,
                    "stress_raw": str(raw_stress_abnormal) if raw_stress_abnormal else "",
                    "force_raw": str(raw_force_abnormal) if raw_force_abnormal else "",
                }
            )

    return records, warnings, len(df)
