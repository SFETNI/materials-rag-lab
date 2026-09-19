"""Parser for HIP cycle workbook into ProcessRecord + normalized sensor parquet."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas import isna

from materials_rag.schemas import ProcessRecord, SourceRef, make_human_readable_id


def _normalize_column(col: object) -> str:
    name = str(col).strip()
    name = name.strip(",").strip()
    name = name.replace("\\", "_").replace("/", "_")
    return name


def _find_row_containing(values: list[object], keyword: str) -> int | None:
    marker = keyword.lower()
    for index, row in enumerate(values):
        if any(isinstance(cell, str) and marker in cell.lower() for cell in row):
            return index
    return None


def _to_optional_float(raw: object) -> float | None:
    value = pd.to_numeric(raw, errors="coerce")
    if isna(value):
        return None
    return float(value)


def parse_hip_workbook(
    workbook_path: Path,
    source: SourceRef,
    output_root: Path,
) -> tuple[ProcessRecord, list[dict[str, str]], list[Path]]:
    """Parse the Overview sheet and normalize Data_Cycle_1 into parquet."""

    overview_df = pd.read_excel(workbook_path, sheet_name="Overview", header=None)
    data_df = pd.read_excel(workbook_path, sheet_name="Data_Cycle_1", header=None)

    warnings: list[dict[str, str]] = []
    output_paths: list[Path] = []

    # Overview extraction
    header_idx = _find_row_containing(overview_df.values.tolist(), "pressure")
    if header_idx is None:
        header_idx = _find_row_containing(overview_df.values.tolist(), "cycle #")
    if header_idx is None:
        warnings.append(
            {
                "type": "overview_header_not_detected",
                "path": workbook_path.as_posix(),
            }
        )
        header_idx = 0

    header_row = overview_df.iloc[header_idx].tolist()
    value_row = (
        overview_df.iloc[header_idx + 1].tolist()
        if header_idx + 1 < len(overview_df)
        else []
    )

    def _pick(header_key: str, fallback_index: int | None = None) -> object:
        for idx, name in enumerate(header_row):
            if (
                isinstance(name, str)
                and header_key in str(name).lower()
                and idx < len(value_row)
            ):
                return value_row[idx]
        if fallback_index is not None and fallback_index < len(value_row):
            return value_row[fallback_index]
        return None

    cycle_number = _pick("cycle")
    pressure = _pick("pressure [mpa]")
    temperature = _pick("temperature")
    hold_time = _pick("hold time")
    pressure_medium = _pick("pressure medium")
    cooling_type = _pick("cooling type")
    process_name = "HIP_cycles"

    # Normalize numeric measurements for sensor table
    header_candidates = data_df.apply(lambda row: [str(v).strip().lower() for v in row.tolist()], axis=1)
    data_header_idx = None
    for i, row in enumerate(header_candidates.tolist()):
        if any(cell in {"datetime", "time", "position"} for cell in row) and any("temp" in cell for cell in row):
            data_header_idx = i
            break
    if data_header_idx is None:
        data_header_idx = _find_row_containing(data_df.values.tolist(), "datetime")
    if data_header_idx is None:
        data_header_idx = 0
        warnings.append(
            {
                "type": "data_header_not_detected",
                "path": workbook_path.as_posix(),
            }
        )

    sensor_header = data_df.iloc[data_header_idx].tolist()
    sensor_rows = data_df.iloc[data_header_idx + 1 :].reset_index(drop=True)
    sensor_rows.columns = [_normalize_column(col) for col in sensor_header]
    sensor_rows = sensor_rows.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)

    # Keep raw temporal strings as strings when parse fails.
    if "DateTime" in sensor_rows.columns:
        sensor_rows["DateTime"] = pd.to_datetime(sensor_rows["DateTime"], errors="coerce").astype(str)

    # Numeric coercion for remaining measured columns
    for col in sensor_rows.columns:
        if col == "DateTime":
            continue
        sensor_rows[col] = pd.to_numeric(sensor_rows[col], errors="coerce")

    sensor_path = (
        output_root
        / "phase2a"
        / "artifacts"
        / "hip"
        / "process_cycle_1_sensor.parquet"
    )
    sensor_path.parent.mkdir(parents=True, exist_ok=True)
    sensor_rows.to_parquet(sensor_path, index=False)
    output_paths.append(sensor_path)

    process_record_id = make_human_readable_id(
        "process_record",
        source.dataset,
        process_name,
        "cycle_1",
    )

    metadata = {
        "source_sheet": "Overview",
        "sensor_table_path": sensor_path.as_posix(),
        "sensor_rows": len(sensor_rows),
        "sensor_columns": list(sensor_rows.columns),
        "raw_workbook": workbook_path.as_posix(),
    }

    runout_text = (
        f"HIP cycle {cycle_number} at {pressure} MPa, "
        f"{temperature} C for {hold_time} min; medium={pressure_medium}."
    )

    cycle_numeric = _to_optional_float(cycle_number)
    cycle_numeric = int(cycle_numeric) if cycle_numeric is not None else None
    pressure_val = _to_optional_float(pressure)
    temp_val = _to_optional_float(temperature)
    hold_val = _to_optional_float(hold_time)

    process_record = ProcessRecord(
        id=process_record_id,
        retrieval_text=runout_text,
        source=source,
        cycle_number=cycle_numeric,
        pressure_mpa=pressure_val,
        temperature_c=temp_val,
        hold_time_min=hold_val,
        pressure_medium=str(pressure_medium) if pd.notna(pressure_medium) else None,
        cooling_type=str(cooling_type) if pd.notna(cooling_type) else None,
        process_name=str(process_name),
        cycle_data_path=sensor_path.as_posix(),
        metadata=metadata,
    )
    return process_record, warnings, output_paths
