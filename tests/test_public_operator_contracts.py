from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_public_skills_are_valid_and_use_public_commands() -> None:
    skills = sorted((ROOT / ".agents/skills").glob("*/SKILL.md"))
    assert [path.parent.name for path in skills] == [
        "reproduce-v0-1-0",
        "run-agentic-rag",
        "run-rag-ladder",
        "setup-materials-rag",
    ]
    forbidden = [
        "run_" + "phase",
        "Cloud" + "PC",
        "C:" + "\\Users",
        "phase" + "10a_" + "role_outputs",
        "external " + "gate",
    ]
    for path in skills:
        text = path.read_text(encoding="utf-8")
        assert text.startswith("---\nname:")
        assert "\ndescription:" in text.split("---", 2)[1]
        assert not [term for term in forbidden if term in text]


def test_public_result_verifier_passes() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/verify_public_results.py")],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_frozen_manifests_remain_exact() -> None:
    expected = {
        "agentic_rag_v0.1.0_experiment_freeze.json": "859418aa8f80d0c9e2ac28275d418ecf1c7fe2b46c71401f83c9cc33d2124b22",
        "phase_10c_dev_refreeze_v2.json": "bf5b9efc4688d9496792aded2ab749ca56373a963231ccb2896fad03d90a587e",
        "phase_10d_challenge_freeze.json": "798f03e8fb4464d56839d2ce0764a4ff59544a2daddf83bc95a1dd501eb105b4",
    }
    for name, digest in expected.items():
        path = ROOT / "data/manifests" / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_source_tree_has_no_empty_scaffolds() -> None:
    assert not [path for path in (ROOT / "src/materials_rag").rglob("*.py") if path.stat().st_size == 0]


def test_normal_public_navigation_has_no_development_transport_terms() -> None:
    public_files = [
        ROOT / "README.md",
        ROOT / "CONTRIBUTING.md",
        ROOT / "docs/quickstarts.md",
        ROOT / "docs/architecture.md",
        ROOT / "docs/agentic_rag.md",
        ROOT / "docs/orchestration_policy.md",
        ROOT / "docs/model_runtime.md",
        ROOT / "docs/data_and_provenance.md",
    ]
    forbidden = [
        "external " + "gate",
        "low-" + "effort session",
        "working " + "tree",
        "Cloud" + "PC",
        "C:" + "\\Users",
        "RC" + "1",
        "RC" + "2",
    ]
    for path in public_files:
        text = path.read_text(encoding="utf-8")
        assert not [term for term in forbidden if term in text], path
