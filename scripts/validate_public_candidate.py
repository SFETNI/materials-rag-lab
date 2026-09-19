"""Validate the allowlisted public export without running scientific experiments."""

from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_FREEZE = "859418aa8f80d0c9e2ac28275d418ecf1c7fe2b46c71401f83c9cc33d2124b22"
REQUIRED_TOP_LEVEL = {
    ".agents",
    ".github",
    ".gitignore",
    ".python-version",
    "CHANGELOG.md",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "LICENSE",
    "LICENSE-CC-BY-4.0",
    "README.md",
    "THIRD_PARTY_NOTICES.md",
    "data",
    "docs",
    "experiments",
    "pyproject.toml",
    "schemas",
    "scripts",
    "src",
    "tests",
    "uv.lock",
}
FORBIDDEN_TOP_LEVEL = {
    ".git",
    ".venv",
    "Redaction",
    "authoring",
    "release",
    "storage",
    "tmp",
}


def main() -> None:
    names = {path.name for path in ROOT.iterdir()}
    missing = sorted(REQUIRED_TOP_LEVEL - names)
    forbidden = sorted(FORBIDDEN_TOP_LEVEL & names)
    if missing or forbidden:
        raise SystemExit(f"missing={missing}; forbidden={forbidden}")

    freeze = ROOT / "data/manifests/agentic_rag_v0.1.0_experiment_freeze.json"
    actual = hashlib.sha256(freeze.read_bytes()).hexdigest()
    if actual != EXPECTED_FREEZE:
        raise SystemExit(f"experiment freeze hash mismatch: {actual}")

    if list(ROOT.rglob("*.pyc")) or list(ROOT.rglob("__pycache__")):
        raise SystemExit("generated Python cache files are present")
    if (ROOT / "data/runtime").exists():
        raise SystemExit("runtime-installed data must not be included in the public source tree")

    required_runtime_files = {
        ROOT / "src/materials_rag/api.py",
        ROOT / "src/materials_rag/benchmark.py",
        ROOT / "src/materials_rag/reconstruction.py",
        ROOT / "data/reconstruction/full_corpus_identity.json",
        ROOT / "docs/orchestration_policy.md",
    }
    missing_runtime_files = sorted(str(path) for path in required_runtime_files if not path.exists())
    if missing_runtime_files:
        raise SystemExit(f"public runtime files are missing: {missing_runtime_files}")

    source_files = list((ROOT / "src/materials_rag").rglob("*.py"))
    experiments = list((ROOT / "experiments").glob("[0-9][0-9]_*.py"))
    if not source_files or len(experiments) != 11:
        raise SystemExit("source modules or experiment ladder are incomplete")

    zero_byte = [path for path in source_files if path.stat().st_size == 0]
    if zero_byte:
        raise SystemExit(f"zero-byte source modules are present: {zero_byte}")

    required_results = {
        "retrieval_ladder_metrics.json",
        "agentic_dev_challenge_metrics.json",
        "computation_probe_metrics.json",
    }
    if {path.name for path in (ROOT / "data/results").glob("*.json")} != required_results:
        raise SystemExit("compact public result set is incomplete or unexpected")

    print(f"Public candidate: PASS ({len(source_files)} source modules, {len(experiments)} experiments)")


if __name__ == "__main__":
    main()
