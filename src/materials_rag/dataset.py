"""Deterministic installation and status checks for the companion dataset."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DATASET_DOI = "10.5281/zenodo.22837229"
RUNTIME_RELATIVE_PATH = Path("data/runtime/retrieval_units.jsonl")
EXPECTED_REDISTRIBUTABLE_UNITS = 252
EXPECTED_FULL_UNITS = 277


@dataclass(frozen=True)
class DatasetStatus:
    outcome: str
    mode: str
    retrieval_unit_count: int
    retrieval_units_path: str
    missing_reference_sources: tuple[str, ...]
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl_count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip())


def _reference_sources(project_root: Path) -> tuple[str, ...]:
    path = project_root / "data/reconstruction/reference_only_sources.jsonl"
    if not path.exists():
        return ()
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    return tuple(
        row["source_id"]
        for row in records
        if str(row.get("source_id", "")).startswith("reference_only:")
    )


def dataset_status(project_root: Path) -> DatasetStatus:
    project_root = project_root.resolve()
    candidates = [project_root / RUNTIME_RELATIVE_PATH]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        return DatasetStatus(
            "WARN",
            "not-installed",
            0,
            str(candidates[0]),
            _reference_sources(project_root),
            f"Install dataset v0.1.0 from doi:{DATASET_DOI}.",
        )
    count = _jsonl_count(path)
    if count == EXPECTED_FULL_UNITS:
        mode, outcome = "full-reconstructed-277", "PASS"
    elif count == EXPECTED_REDISTRIBUTABLE_UNITS:
        mode, outcome = "redistributable-252", "PASS"
    else:
        mode, outcome = f"unrecognized-{count}", "FAIL"
    return DatasetStatus(
        outcome,
        mode,
        count,
        str(path),
        _reference_sources(project_root) if count < EXPECTED_FULL_UNITS else (),
        "Full frozen replication requires 277 units; normal public use supports the 252-unit projection.",
    )


def _safe_extract(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as bundle:
        root = destination.resolve()
        for member in bundle.infolist():
            resolved = (destination / member.filename).resolve()
            if not resolved.is_relative_to(root):
                raise ValueError(f"Unsafe archive member: {member.filename}")
        bundle.extractall(destination)


def _dataset_root(source: Path) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    if source.is_dir():
        roots = [source, *[path for path in source.iterdir() if path.is_dir()]]
        match = next((path for path in roots if (path / "canonical/retrieval_units.jsonl").exists()), None)
        if match is None:
            raise ValueError("Dataset root must contain canonical/retrieval_units.jsonl")
        return match, None
    if source.suffix.lower() != ".zip":
        raise ValueError("Dataset source must be the v0.1.0 ZIP or an extracted dataset directory")
    holder = tempfile.TemporaryDirectory(prefix="materials_rag_dataset_")
    destination = Path(holder.name)
    _safe_extract(source, destination)
    roots = [destination, *[path for path in destination.iterdir() if path.is_dir()]]
    match = next((path for path in roots if (path / "canonical/retrieval_units.jsonl").exists()), None)
    if match is None:
        holder.cleanup()
        raise ValueError("Dataset archive does not contain canonical/retrieval_units.jsonl")
    return match, holder


def _verify_checksums(dataset_root: Path) -> int:
    checksum_path = dataset_root / "checksums.sha256"
    if not checksum_path.exists():
        raise ValueError("Dataset is missing checksums.sha256")
    checked = 0
    for line in checksum_path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line:
            continue
        expected, relative = line.split(maxsplit=1)
        relative = relative.lstrip("* ").replace("\\", "/")
        path = dataset_root / relative
        if not path.exists():
            raise ValueError(f"Dataset checksum entry is missing: {relative}")
        actual = _sha256(path)
        if actual != expected.lower():
            raise ValueError(f"Dataset checksum mismatch for {relative}: {actual}")
        checked += 1
    return checked


def install_dataset(source: Path, project_root: Path, *, check_only: bool = False) -> dict[str, Any]:
    source = source.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    dataset_root, holder = _dataset_root(source)
    try:
        checked = _verify_checksums(dataset_root)
        retrieval_source = dataset_root / "canonical/retrieval_units.jsonl"
        count = _jsonl_count(retrieval_source)
        if count != EXPECTED_REDISTRIBUTABLE_UNITS:
            raise ValueError(f"Expected 252 redistributable retrieval units, found {count}")
        runtime_root = project_root.resolve() / "data/runtime"
        dataset_destination = runtime_root / "dataset"
        destination = project_root.resolve() / RUNTIME_RELATIVE_PATH
        if check_only:
            return {
                "outcome": "PASS",
                "check_only": True,
                "source": str(source),
                "verified_files": checked,
                "retrieval_unit_count": count,
                "destination": str(dataset_destination),
            }
        destination.parent.mkdir(parents=True, exist_ok=True)
        for source_file in sorted(path for path in dataset_root.rglob("*") if path.is_file()):
            relative = source_file.relative_to(dataset_root)
            target = dataset_destination / relative
            if target.exists() and _sha256(target) != _sha256(source_file):
                raise ValueError(f"Refusing to overwrite different installed dataset file: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, target)
        if destination.exists() and _sha256(destination) != _sha256(retrieval_source):
            raise ValueError(f"Refusing to overwrite different runtime data: {destination}")
        shutil.copy2(retrieval_source, destination)
        manifest_source = dataset_root / "release_manifest.json"
        if manifest_source.exists():
            shutil.copy2(manifest_source, destination.parent / "dataset_release_manifest.json")
        return {
            "outcome": "PASS",
            "check_only": False,
            "source": str(source),
            "verified_files": checked,
            "retrieval_unit_count": count,
            "destination": str(dataset_destination),
            "retrieval_units_path": str(destination),
            "destination_sha256": _sha256(destination),
        }
    finally:
        if holder is not None:
            holder.cleanup()


__all__ = ["DATASET_DOI", "DatasetStatus", "dataset_status", "install_dataset"]
