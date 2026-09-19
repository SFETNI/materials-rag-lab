"""Check local Markdown links and required public assets."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def _anchors(path: Path) -> set[str]:
    anchors: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("#"):
            continue
        heading = line.lstrip("#").strip().lower()
        slug = re.sub(r"[^a-z0-9 _-]", "", heading).replace(" ", "-")
        slug = re.sub(r"-+", "-", slug).strip("-")
        if slug:
            anchors.add(slug)
    return anchors


def main() -> None:
    markdown_files = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
    failures: list[str] = []
    checked = 0
    for document in markdown_files:
        text = document.read_text(encoding="utf-8")
        for target in MARKDOWN_LINK.findall(text):
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            clean_target, _, anchor = target.partition("#")
            if not clean_target:
                if anchor and anchor not in _anchors(document):
                    failures.append(f"{document.relative_to(ROOT)} -> {target} (missing anchor)")
                continue
            checked += 1
            resolved = (document.parent / clean_target).resolve()
            if not resolved.exists():
                failures.append(f"{document.relative_to(ROOT)} -> {target}")
            elif anchor and resolved.suffix.lower() == ".md" and anchor not in _anchors(resolved):
                failures.append(f"{document.relative_to(ROOT)} -> {target} (missing anchor)")
    if failures:
        raise SystemExit("Broken local links:\n" + "\n".join(failures))
    print(f"Documentation links: PASS ({checked} local targets checked)")


if __name__ == "__main__":
    main()
