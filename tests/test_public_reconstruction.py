from __future__ import annotations

import hashlib
import json
from pathlib import Path

from materials_rag.reconstruction import reconstruct_full_corpus


def _compact_hash(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def test_reconstruction_combines_252_and_25_without_source_redistribution(
    monkeypatch, tmp_path: Path
) -> None:
    project = tmp_path / "project"
    canonical = project / "data/runtime/dataset/canonical"
    canonical.mkdir(parents=True)
    public = [
        {
            "id": f"unit|{index:03d}",
            "kind": "chunk_internal",
            "retrieval_text": f"Public evidence {index}",
            "source": {"dataset": "public", "source_path": f"public-{index}.txt"},
            "metadata": {},
        }
        for index in range(252)
    ]
    (canonical / "retrieval_units.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in public), encoding="utf-8"
    )
    missing = [
        {
            "id": f"missing|{index:02d}",
            "kind": "chunk_public",
            "retrieval_text": f"Authorized evidence {index}",
            "source": {"dataset": "reference", "source_path": f"ref-{index:02d}.pdf"},
            "metadata": {"chunk_index": 0},
        }
        for index in range(25)
    ]
    combined = [*missing, *public]
    identity_root = project / "data/reconstruction"
    identity_root.mkdir(parents=True)
    identity = {
        "frozen_jsonl_sha256": "historical-byte-hash",
        "ordered_ids_sha256": _compact_hash([row["id"] for row in combined]),
        "ordered_id_and_retrieval_text_sha256": _compact_hash(
            [{"id": row["id"], "retrieval_text": row["retrieval_text"]} for row in combined]
        ),
    }
    (identity_root / "full_corpus_identity.json").write_text(
        json.dumps(identity), encoding="utf-8"
    )
    monkeypatch.setattr(
        "materials_rag.reconstruction._missing_units", lambda *_args: (missing, [])
    )
    report = reconstruct_full_corpus(project, tmp_path / "authorized", project / "data/runtime")
    assert report["retrieval_unit_count"] == 277
    assert report["retrieval_identity_matches_frozen"] is True
    assert report["source_files_retained"] is False
    assert len(
        (project / "data/runtime/retrieval_units.jsonl").read_text().splitlines()
    ) == 277
