"""Parser for raw fatigue log ZIP archive into canonical TestRecord objects."""

from __future__ import annotations

import io
import json
import re
import zipfile
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError, ParserError

from materials_rag.ingestion.normalization import normalize_specimen_identifier
from materials_rag.schemas import (
    ExperimentRecord,
    SourceRef,
    TestRecord,
    make_human_readable_id,
)


def _safe_path_token(value: str) -> str:
    token = re.sub(r"[^0-9A-Za-z_.-]+", "_", value.strip())
    return token or "run"


def _normalize_csv_columns(columns: Iterable[object]) -> list[str]:
    return [str(col).strip() for col in columns]


def _logical_run_prefix(stem: str) -> str:
    if stem.startswith("IN718_30um_"):
        return "with_30um"
    if stem.startswith("IN718_"):
        return "base_in718"
    return "raw"


def _read_csv_member(zf: zipfile.ZipFile, member_name: str) -> pd.DataFrame | None:
    raw = zf.read(member_name).decode("utf-8", errors="replace")
    lines = raw.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    if not lines:
        return None
    text = "\n".join(lines)
    try:
        df = pd.read_csv(io.StringIO(text))
    except (ParserError, UnicodeDecodeError, EmptyDataError, ValueError):
        return None

    if df.empty:
        return None
    df.columns = _normalize_csv_columns(df.columns)
    return df.dropna(axis=1, how="all")


def _compute_signal_stats(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    if df.empty:
        return {}

    stats: dict[str, dict[str, float]] = {}
    duration = None
    time_candidates = [col for col in df.columns if "time" in col.lower()]
    if time_candidates:
        time_col = time_candidates[0]
        time_series = pd.to_numeric(df[time_col], errors="coerce")
        if time_series.notna().any():
            trimmed = time_series.dropna()
            if not trimmed.empty:
                duration = float(trimmed.iloc[-1] - trimmed.iloc[0])
                stats["time_sec"] = {
                    "min": float(trimmed.min()),
                    "max": float(trimmed.max()),
                }
                if len(trimmed) > 1:
                    stats["time_sec"]["duration"] = duration

    for col in df.columns:
        lowered = col.lower()
        if not ("load" in lowered or "stress" in lowered):
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        numeric = numeric[numeric.notna()]
        if numeric.empty:
            continue
        stats[col] = {
            "mean": float(numeric.mean()),
            "min": float(numeric.min()),
            "max": float(numeric.max()),
            "std": float(numeric.std(ddof=0)),
        }
    return stats


def _decode_test_metadata(raw: bytes) -> str:
    text = raw.decode("utf-8", errors="replace").replace("\x00", "")
    if "**END**" in text:
        text = text.split("**END**", 1)[0] + "**END**"
    text = re.sub(r"[\x00-\x08\x0b\x0c\x1f\x7f-\xff]", " ", text)
    return text


def _canonical_metadata_key(raw_key: str) -> str:
    cleaned = re.sub(r"[^0-9a-z ]", " ", raw_key.lower())
    cleaned = re.sub(r"\\s+", " ", cleaned).strip()
    return cleaned.replace(" ", "_")


def _parse_test_metadata(raw: bytes) -> tuple[dict[str, str], dict[str, str], list[str], dict[str, float]]:
    normalized_text = _decode_test_metadata(raw)
    candidates = [
        "date",
        "time",
        "specimen id",
        "direction",
        "specimen type",
        "specimen length",
        "specimen diameter",
        "specimen area",
        "status",
    ]

    key_pattern = re.compile(
        r"(" + "|".join(re.escape(key) for key in candidates) + r")\s*:?",
        re.IGNORECASE,
    )
    matches = list(key_pattern.finditer(normalized_text))
    parsed_pairs: dict[str, str] = {}
    for idx, match in enumerate(matches):
        key = match.group(1).strip()
        key_start = match.end()
        value_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(normalized_text)
        value_raw = normalized_text[key_start:value_end]
        value_raw = value_raw.removeprefix(",").strip()
        if key and value_raw:
            parsed_pairs[_canonical_metadata_key(key)] = value_raw

    numeric_values: dict[str, float] = {}
    if "specimen_length" in parsed_pairs:
        try:
            numeric_values["specimen_length_mm"] = float(parsed_pairs["specimen_length"])
        except ValueError:
            pass
    if "specimen_diameter" in parsed_pairs:
        try:
            numeric_values["specimen_diameter_mm"] = float(parsed_pairs["specimen_diameter"])
        except ValueError:
            pass
    if "specimen_area" in parsed_pairs:
        try:
            numeric_values["specimen_area_sqmm"] = float(parsed_pairs["specimen_area"])
        except ValueError:
            pass
    if "status" in parsed_pairs:
        numeric_values["status_text_available"] = 1.0

    warnings: list[str] = []
    for key in ("date", "time", "specimen_id"):
        if key not in parsed_pairs:
            warnings.append(f"missing_{key}")
    return parsed_pairs, {"raw_text": normalized_text[:3000]}, warnings, numeric_values


def _build_reconciliation(
    canonical_id: str,
    test_metadata: dict[str, str],
    experiment_records_by_id: dict[str, list[ExperimentRecord]],
) -> tuple[str | None, str, list[str]]:
    warnings: list[str] = []
    matched: str | None = None
    status = "no_experiment_match"

    candidates = experiment_records_by_id.get(canonical_id, [])
    if len(candidates) == 1:
        status = "linked"
        matched = candidates[0].id
    elif len(candidates) > 1:
        status = "ambiguous"
        warnings.append(
            f"specimen {canonical_id} has {len(candidates)} candidate experiment records"
        )

    if "specimen_id" in test_metadata:
        normalized = normalize_specimen_identifier(test_metadata["specimen_id"])
        if normalized.canonical_specimen_id != canonical_id:
            warnings.append(
                f"test metadata specimen id {normalized.canonical_specimen_id} "
                f"does not match filename-based {canonical_id}"
            )

    if status == "no_experiment_match":
        warnings.append(f"no_experiment_record_for_specimen_{canonical_id}")

    if status == "linked" and matched is not None:
        # Compare limited meaningful fields where overlap is supported.
        reference = experiment_records_by_id[canonical_id][0]
        specimen_diameter = test_metadata.get("specimen_diameter")
        if specimen_diameter:
            try:
                value = float(specimen_diameter)
                for field in ["diameter_top_mm", "diameter_middle_mm", "diameter_bottom_mm"]:
                    candidate = getattr(reference, field, None)
                    if isinstance(candidate, (int, float)):
                        if abs(candidate - value) > 0.05:
                            warnings.append(
                                f"specimen_diameter mismatch vs {field}: "
                                f"test={value} exp={candidate}"
                            )
                        break
            except ValueError:
                pass

    return matched, status, warnings


def parse_rbf_test_logs(
    zip_path: Path,
    source: SourceRef,
    output_root: Path,
    experiment_records: list[ExperimentRecord] | None = None,
) -> tuple[list[TestRecord], list[dict[str, str]], list[Path]]:
    experiment_records = experiment_records or []
    by_specimen: dict[str, list[ExperimentRecord]] = defaultdict(list)
    for record in experiment_records:
        by_specimen[record.specimen_id].append(record)

    with zipfile.ZipFile(zip_path) as zf:
        member_names = sorted(
            n for n in zf.namelist() if not n.endswith("/") and (n.lower().endswith(".csv") or n.lower().endswith(".test"))
        )

        grouped: dict[tuple[str, str | None, str, str], dict[str, str]] = defaultdict(dict)
        for name in member_names:
            base = Path(name).stem
            normalized = normalize_specimen_identifier(base)
            key = (
                normalized.canonical_specimen_id,
                normalized.run_id,
                normalized.file_variant,
                _logical_run_prefix(base),
            )
            suffix = Path(name).suffix.lower()
            grouped[key][suffix] = name

        warnings: list[dict[str, str]] = []
        output_paths: list[Path] = []
        records: list[TestRecord] = []
        for (
            canonical_specimen,
            run_id,
            file_variant,
            run_prefix,
        ), members in sorted(
            grouped.items(), key=lambda item: (item[0][0], item[0][1] or "", item[0][2], item[0][3])
        ):
            csv_member = members.get(".csv")
            test_member = members.get(".test")

            raw_member_count = len(members)
            raw_member_paths = [members[".csv"]] if ".csv" in members else []
            if ".test" in members:
                raw_member_paths.append(members[".test"])
            raw_member_paths = sorted(raw_member_paths)
            raw_members_csv = members.get(".csv")
            raw_members_test = members.get(".test")
            warnings_for_record: list[str] = []

            parsed_metadata: dict[str, str] = {}
            metadata_warnings: list[str] = []
            metadata_preview: str = ""
            metadata_numeric: dict[str, float] = {}
            parquet_path: Path | None = None
            signal_columns: list[str] = []
            row_count = None
            stats: dict[str, dict[str, float]] = {}

            if test_member:
                raw = zf.read(test_member)
                parsed_metadata, meta_fields, metadata_warnings, metadata_numeric = _parse_test_metadata(raw)
                metadata_preview = json.dumps(meta_fields, ensure_ascii=False)[:2048]
                if metadata_warnings:
                    warnings_for_record.extend(metadata_warnings)

            if csv_member:
                df = _read_csv_member(zf, csv_member)
                if df is None:
                    warnings_for_record.append(f"unable_to_parse_csv:{csv_member}")
                else:
                    signal_columns = [str(col) for col in df.columns]
                    row_count = len(df)
                    stats = _compute_signal_stats(df)
                    artifact_name = _safe_path_token(
                        "_".join(
                            filter(None, [canonical_specimen, run_id or "", file_variant])
                        )
                    )
                    parquet_path = (
                        output_root
                        / "phase2b"
                        / "tables"
                        / "test_logs"
                        / f"{artifact_name}.parquet"
                    )
                    parquet_path.parent.mkdir(parents=True, exist_ok=True)
                    df.to_parquet(parquet_path, index=False)
                    output_paths.append(parquet_path)
            else:
                warnings_for_record.append("csv_missing")

            if test_member is None:
                warnings_for_record.append("test_metadata_missing")

            linked_id, link_status, link_warnings = _build_reconciliation(
                canonical_specimen,
                parsed_metadata,
                by_specimen,
            )
            warnings_for_record.extend(link_warnings)

            retrieval_parts = [
                f"specimen={canonical_specimen}",
                f"run={run_id}" if run_id else "run=standard",
                f"variant={file_variant}",
                f"csv={'present' if csv_member else 'missing'}",
                f"test={'present' if test_member else 'missing'}",
                f"rows={row_count}" if row_count is not None else "rows=na",
            ]
            if duration := stats.get("time_sec", {}).get("duration"):
                retrieval_parts.append(f"duration={duration:g}")
            retrieval_text = " | ".join(retrieval_parts)

            record_id = make_human_readable_id(
                "test_record",
                source.dataset,
                source.source_path.replace("/", "_"),
                canonical_specimen,
                run_id or "standard",
                file_variant,
                run_prefix,
            )
            raw_identifier = (
                raw_members_csv
                if raw_members_csv is not None
                else (raw_members_test if raw_members_test is not None else canonical_specimen)
            )
            if raw_members_csv:
                raw_identifier = Path(raw_members_csv).stem
            elif raw_members_test:
                raw_identifier = Path(raw_members_test).stem

            if not raw_identifier:
                raw_identifier = canonical_specimen

            record = TestRecord(
                id=record_id,
                retrieval_text=retrieval_text,
                source=source,
                specimen_id=canonical_specimen,
                specimen_id_raw=raw_identifier,
                run_id=run_id,
                file_variant=file_variant,
                csv_artifact_path=raw_members_csv or "",
                test_metadata_path=raw_members_test,
                signal_columns=signal_columns,
                cycle_count=row_count,
                statistics=stats,
                experiment_record_id=linked_id,
                link_status=link_status,
                metadata={
                    "archive_member_count": raw_member_count,
                    "archive_members": raw_member_paths,
                    "parsed_test_metadata": parsed_metadata,
                    "test_metadata_numeric": metadata_numeric,
                    "test_metadata_preview": metadata_preview,
                    "raw_test_member": test_member,
                    "raw_csv_member": csv_member,
                    "normalized_signal_path": parquet_path.as_posix() if parquet_path else None,
                    "reconciliation_warnings": warnings_for_record,
                    "zip_archive": zip_path.as_posix(),
                },
            )
            records.append(record)
            if warnings_for_record:
                warnings.append(
                    {"type": "test_record_warning", "record_id": record.id, "messages": ",".join(warnings_for_record)}
                )

    return records, warnings, output_paths
