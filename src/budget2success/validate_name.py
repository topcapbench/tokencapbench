from __future__ import annotations

import argparse
from pathlib import Path

_ALLOWED_BINARY_SUFFIXES = {".png", ".pdf", ".jpg", ".jpeg", ".gif", ".zip", ".pyc"}
_LEGACY_NAME = "".join(["resource", "green", "bench"]).lower()


def find_legacy_name_hits(root: Path) -> list[str]:
    hits: list[str] = []
    for path in root.rglob("*"):
        if ".git" in path.parts or not path.is_file() or path.suffix.lower() in _ALLOWED_BINARY_SUFFIXES:
            continue
        try:
            text = path.read_text(errors="ignore").lower()
        except Exception:
            continue
        if _LEGACY_NAME in text:
            hits.append(str(path))
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(description="Check that the legacy benchmark name is absent from text artifacts.")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    hits = find_legacy_name_hits(Path(args.root))
    if hits:
        print("\n".join(hits))
        raise SystemExit(1)
    print("OK: no legacy benchmark name found")


if __name__ == "__main__":
    main()
