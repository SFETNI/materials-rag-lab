"""Conservative text-quality diagnostics for extracted technical PDFs."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

KNOWN_SUSPICIOUS_GLYPHS = {"\u01cb"}
REPLACEMENT_CHARACTER = "\ufffd"

_TECHNICAL_UNIT_PATTERNS = (
    re.compile(r"\d+(?:\.\d+)?-?\u01cbm", re.IGNORECASE),
    re.compile(r"\d+(?:\.\d+)?-?\u01cbin\.?", re.IGNORECASE),
)
_MALFORMED_MAGNIFICATION_PATTERN = re.compile(
    r"\b\d+\s*\u00ee\b(?=.{0,60}\bmagnification\b)",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class TextQualityFinding:
    """A suspicious text diagnostic; detection only, never correction."""

    category: str
    suspicious_substring: str
    snippet: str
    character: str | None = None
    codepoint: str | None = None
    unicode_category: str | None = None
    unicode_name: str | None = None

    def as_warning_fields(self) -> dict[str, str]:
        payload = {
            "category": self.category,
            "suspicious_substring": self.suspicious_substring,
            "snippet": self.snippet,
        }
        if self.character is not None:
            payload["character"] = self.character
        if self.codepoint is not None:
            payload["codepoint"] = self.codepoint
        if self.unicode_category is not None:
            payload["unicode_category"] = self.unicode_category
        if self.unicode_name is not None:
            payload["unicode_name"] = self.unicode_name
        return payload


def snippet_around(text: str, start_index: int, end_index: int, width: int = 40) -> str:
    """Return compact context around a suspicious span."""

    start = max(0, start_index - width)
    end = min(len(text), end_index + width)
    return text[start:end].replace("\n", " ")


def is_intrinsically_suspicious_char(char: str) -> bool:
    """Return whether a single Unicode character is suspicious by itself."""

    return (
        unicodedata.category(char) == "Co"
        or char == REPLACEMENT_CHARACTER
        or char in KNOWN_SUSPICIOUS_GLYPHS
    )


def _char_finding(text: str, index: int, char: str) -> TextQualityFinding:
    return TextQualityFinding(
        category="suspicious_unicode_glyph",
        suspicious_substring=char,
        snippet=snippet_around(text, index, index + 1),
        character=char,
        codepoint=f"U+{ord(char):04X}",
        unicode_category=unicodedata.category(char),
        unicode_name=unicodedata.name(char, "<unknown>"),
    )


def find_intrinsic_suspicious_glyphs(text: str) -> list[TextQualityFinding]:
    """Find suspicious Unicode glyphs without flagging normal non-ASCII typography."""

    findings: list[TextQualityFinding] = []
    seen: set[tuple[str, str]] = set()
    for index, char in enumerate(text):
        if not is_intrinsically_suspicious_char(char):
            continue
        key = (char, snippet_around(text, index, index + 1, width=12))
        if key in seen:
            continue
        seen.add(key)
        findings.append(_char_finding(text, index, char))
    return findings


def find_suspicious_technical_sequences(text: str) -> list[TextQualityFinding]:
    """Find known malformed technical sequences in this corpus."""

    findings: list[TextQualityFinding] = []
    for pattern in (*_TECHNICAL_UNIT_PATTERNS, _MALFORMED_MAGNIFICATION_PATTERN):
        for match in pattern.finditer(text):
            findings.append(
                TextQualityFinding(
                    category="suspicious_technical_unit",
                    suspicious_substring=match.group(0),
                    snippet=snippet_around(text, match.start(), match.end()),
                )
            )
    return findings


def find_suspicious_text(text: str) -> list[TextQualityFinding]:
    """Return all conservative suspicious-text diagnostics."""

    return find_intrinsic_suspicious_glyphs(text) + find_suspicious_technical_sequences(text)
