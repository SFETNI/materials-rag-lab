import hashlib
import json
from pathlib import Path

import materials_rag

ROOT = Path(__file__).resolve().parents[1]


def test_import_resolves_inside_public_candidate() -> None:
    module_path = Path(materials_rag.__file__).resolve()
    assert module_path.is_relative_to(ROOT)


def test_canonical_freeze_manifest_hash() -> None:
    path = ROOT / "data/manifests/agentic_rag_v0.1.0_experiment_freeze.json"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == "859418aa8f80d0c9e2ac28275d418ecf1c7fe2b46c71401f83c9cc33d2124b22"


def test_sample_records_are_valid_jsonl() -> None:
    path = ROOT / "data/samples/retrieval_units_sample.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()]
    assert records
    assert all("id" in record for record in records)


def test_development_directories_are_absent() -> None:
    forbidden = [".git", ".venv", "release", "authoring", "storage", "tmp", "Redaction"]
    assert not [name for name in forbidden if (ROOT / name).exists()]
