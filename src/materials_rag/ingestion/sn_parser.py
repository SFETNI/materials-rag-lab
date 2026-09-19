"""Parser for SN curve fit and confidence tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from materials_rag.schemas import AnalysisRecord, SourceRef, make_human_readable_id


def _normalize_name(name: str) -> str:
    return name.replace(" ", "_").replace("-", "_")


def parse_sn_analysis(
    sn_csv: Path,
    conf_csv: Path,
    source: SourceRef,
    output_root: Path,
) -> tuple[list[AnalysisRecord], list[dict[str, str]], list[Path]]:
    """Parse S-N curve fits and confidence bands into one AnalysisRecord per condition."""

    sn_df = pd.read_csv(sn_csv)
    cb_df = pd.read_csv(conf_csv)

    # Remove accidental index columns.
    sn_df = sn_df.rename(columns=lambda c: str(c).strip())
    cb_df = cb_df.rename(columns=lambda c: str(c).strip())
    for col in ("Unnamed: 0",):
        sn_df = sn_df.drop(columns=[col], errors="ignore")
        cb_df = cb_df.drop(columns=[col], errors="ignore")

    warnings: list[dict[str, str]] = []
    artifact_paths: list[Path] = []
    records: list[AnalysisRecord] = []

    sn_types = sorted(set(sn_df["Type"].dropna().unique().tolist()))
    cb_types = sorted(set(cb_df["Type"].dropna().unique().tolist()))

    if sn_types != cb_types:
        warnings.append(
            {
                "type": "condition_set_mismatch",
                "sn_types": ",".join(sn_types),
                "confidence_types": ",".join(cb_types),
            }
        )

    for condition in sorted(set(sn_types) | set(cb_types)):
        sn_part = sn_df[sn_df["Type"] == condition].sort_values("N.log10").reset_index(drop=True)
        cb_part = cb_df[cb_df["Type"] == condition].sort_values("S").reset_index(drop=True)

        cond_name = str(condition)
        cond_id = _normalize_name(cond_name)
        fit_path = (
            output_root
            / "phase2a"
            / "artifacts"
            / "sn"
            / f"sn_curve_{cond_id}.parquet"
        )
        conf_path = (
            output_root
            / "phase2a"
            / "artifacts"
            / "sn"
            / f"confidence_{cond_id}.parquet"
        )
        fit_path.parent.mkdir(parents=True, exist_ok=True)

        merged_by_condition = (
            not sn_part.empty and not cb_part.empty and len(sn_part) == len(cb_part)
        )
        if merged_by_condition and set(cb_part.columns) >= {"S", "lower", "upper"}:
            relation = "paired_by_type_and_rowcount"
        sn_part.to_parquet(fit_path, index=False)
        cb_part.to_parquet(conf_path, index=False)
        artifact_paths.extend([fit_path, conf_path])

        if not merged_by_condition:
            warnings.append(
                {
                    "type": "condition_mismatch",
                    "condition": cond_name,
                    "sn_rows": str(len(sn_part)),
                    "confidence_rows": str(len(cb_part)),
                    "message": "Keeping outputs separate due non-deterministic pairing.",
                }
            )

            source_data = fit_path.as_posix()
            output_data = conf_path.as_posix()
            record = AnalysisRecord(
                id=make_human_readable_id(
                    "analysis_record", source.dataset, "sn_analysis", cond_id, "unpaired"
                ),
                analysis_type="sn_curve_unpaired",
                retrieval_text=(
                    f"Unpaired SN and confidence tables for condition {condition}: "
                    f"{len(sn_part)} fit points and {len(cb_part)} confidence points."
                ),
                source=source,
                method="conditional_grouping",
                x_axis_label="N.log10",
                y_axis_label="S",
                source_data_path=source_data,
                output_data_path=output_data,
                fit_parameters={
                    "condition": cond_name,
                    "paired": False,
                    "fit_points": len(sn_part),
                    "band_points": len(cb_part),
                },
                fit_metrics={"point_count": float(len(sn_part) + len(cb_part))},
                metadata={"pairing": relation if merged_by_condition else "unpaired"},
            )
            records.append(record)
            continue

        record = AnalysisRecord(
            id=make_human_readable_id(
                "analysis_record", source.dataset, "sn_analysis", cond_id, "paired"
            ),
            analysis_type="sn_curve_with_confidence",
            retrieval_text=(
                f"SN fit with confidence band for condition {condition}: "
                f"{len(sn_part)} points each."
            ),
            source=source,
            method="paired_by_condition_and_length",
            x_axis_label="N.log10",
            y_axis_label="S (MPa)",
            source_data_path=fit_path.as_posix(),
            output_data_path=conf_path.as_posix(),
            fit_parameters={
                "condition": cond_name,
                "paired": True,
                "point_count": len(sn_part),
            },
            fit_metrics={"point_count": float(len(sn_part))},
            metadata={
                "paired": True,
                "sn_columns": [str(c) for c in sn_part.columns],
                "confidence_columns": [str(c) for c in cb_part.columns],
                "sn_source": sn_csv.as_posix(),
                "confidence_source": conf_csv.as_posix(),
            },
        )
        records.append(record)

    return records, warnings, artifact_paths
