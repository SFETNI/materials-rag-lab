"""Normalization utilities for specimen and run identifiers.

This module intentionally performs only canonicalization and no file parsing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

RunFileVariant = Literal["standard", "buffer_full", "other"]


@dataclass(frozen=True)
class SpecimenIdentifier:
    """Normalized specimen identifier payload."""

    raw_identifier: str
    canonical_specimen_id: str
    run_id: str | None
    file_variant: RunFileVariant


def _strip_prefixes(identifier: str) -> str:
    token = identifier.strip().lower()
    token = token.replace(" ", "")
    token = token.replace("\\", "/").split("/")[-1]
    extension = Path(token).suffix.lower()
    if extension in {".csv", ".txt", ".xlsx", ".xls", ".pdf", ".zip", ".test"}:
        token = Path(token).stem
    for prefix in ("in718_30um_", "in718_", "30um_"):
        if token.startswith(prefix):
            token = token[len(prefix) :]
            break
    return token


def _extract_run_suffix(token: str) -> tuple[str, str | None]:
    """Split a trailing run suffix such as '_test2'."""

    m = re.match(r"^(.*)_(test\d+)$", token, flags=re.IGNORECASE)
    if m:
        return m.group(1), m.group(2).lower()
    return token, None


def _apply_variant_rules(token: str) -> tuple[str, RunFileVariant]:
    if token.endswith("_buffer_full"):
        return token[: -len("_buffer_full")], "buffer_full"

    if "_buffer_full_" in token:
        token = token.replace("_buffer_full_", "_")
        return token, "buffer_full"

    return token, "standard"


def _to_decimal_notation(specimen_token: str) -> str:
    """Convert the NIST decimal coding convention to a standard float-like form."""

    return re.sub(r"(?<=\d)p(?=\d)", ".", specimen_token)


def normalize_specimen_identifier(raw_identifier: str) -> SpecimenIdentifier:
    """Normalize one specimen identifier preserving both raw and canonical forms.

    Examples
    --------
    >>> normalize_specimen_identifier("1p2").canonical_specimen_id
    '1.2'
    >>> normalize_specimen_identifier("6p2_test2").canonical_specimen_id
    '6.2'
    >>> normalize_specimen_identifier("6p2_test2").run_id
    'test2'
    """

    raw = raw_identifier.strip()
    token = _strip_prefixes(raw)
    token, file_variant = _apply_variant_rules(token)
    token, run_id = _extract_run_suffix(token)
    canonical = _to_decimal_notation(token)
    return SpecimenIdentifier(
        raw_identifier=raw,
        canonical_specimen_id=canonical,
        run_id=run_id,
        file_variant=file_variant,
    )


def normalize_specimen_id_and_run(raw_identifier: str) -> tuple[str, str | None, RunFileVariant]:
    """Convenience helper returning a tuple for quick integrations."""

    normalized = normalize_specimen_identifier(raw_identifier)
    return normalized.canonical_specimen_id, normalized.run_id, normalized.file_variant
