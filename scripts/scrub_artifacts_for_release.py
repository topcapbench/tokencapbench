#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


TEXT_SUFFIXES = {".csv", ".json", ".jsonl", ".md", ".txt", ".yaml", ".yml"}
SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|authorization|bearer[_-]?token|client[_-]?secret|refresh[_-]?token)\s*[:=]\s*(?!<redacted>|null|none)[\"']?([A-Za-z0-9_\-./+=]{16,})"),
]


def scrub_artifacts_for_release(
    *,
    artifact_root: str | Path = "artifacts,reports/artifacts,metadata,reports/tables,configs,prompts",
    dry_run: bool = False,
    output: str | Path = "reports/release_scrub_audit.json",
    csv_output: str | Path = "reports/tables/secret_scrub_audit.csv",
) -> dict[str, Any]:
    roots = _scan_roots(artifact_root)
    findings: list[dict[str, Any]] = []
    scanned = 0
    for root in roots:
        paths = [root] if root.is_file() else sorted(root.rglob("*")) if root.exists() else []
        for path in paths:
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            scanned += 1
            text = path.read_text(encoding="utf-8", errors="replace")
            for pattern in SECRET_PATTERNS:
                for match in pattern.finditer(text):
                    findings.append(
                        {
                            "path": str(path),
                            "line": text.count("\n", 0, match.start()) + 1,
                            "kind": match.group(1),
                            "action": "would_redact" if dry_run else "manual_review_required",
                        }
                    )
    payload = {
        "artifact_root": ",".join(str(root) for root in roots),
        "dry_run": dry_run,
        "files_scanned": scanned,
        "findings": findings,
        "status": "PASS" if not findings else "REVIEW",
    }
    if not dry_run:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        _write_csv(Path(csv_output), payload)
    return payload


def _scan_roots(value: str | Path) -> list[Path]:
    if isinstance(value, Path):
        return [value]
    return [Path(part.strip()) for part in str(value).split(",") if part.strip()]


def _write_csv(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = payload["findings"] or [
        {
            "path": str(payload["artifact_root"]),
            "line": "",
            "kind": "none",
            "action": "pass",
            "patterns_checked": len(SECRET_PATTERNS),
            "files_scanned": payload["files_scanned"],
            "status": payload["status"],
        }
    ]
    normalized = []
    for row in rows:
        normalized.append(
            {
                "path": row.get("path", ""),
                "line": row.get("line", ""),
                "kind": row.get("kind", ""),
                "action": row.get("action", ""),
                "patterns_checked": row.get("patterns_checked", len(SECRET_PATTERNS)),
                "files_scanned": row.get("files_scanned", payload["files_scanned"]),
                "status": row.get("status", payload["status"]),
            }
        )
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(normalized[0].keys()))
        writer.writeheader()
        writer.writerows(normalized)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan packaged artifacts for release-sensitive secrets.")
    parser.add_argument("--artifact-root", default="artifacts,reports/artifacts,metadata,reports/tables,configs,prompts")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", default="reports/release_scrub_audit.json")
    parser.add_argument("--csv-output", default="reports/tables/secret_scrub_audit.csv")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero when findings are present.")
    args = parser.parse_args()
    payload = scrub_artifacts_for_release(
        artifact_root=args.artifact_root,
        dry_run=args.dry_run,
        output=args.output,
        csv_output=args.csv_output,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.strict and payload["findings"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
