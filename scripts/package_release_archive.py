#!/usr/bin/env python
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CORE_ROWS = 33008
ARTIFACT_PREFIX = "tokencapbench"
RELEASE_ARCHIVE = "reports/tokencapbench_release_archive.zip"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def count_jsonl(path: Path) -> int:
    opener = gzip.open if str(path).endswith(".gz") else open
    count = 0
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def gzip_copy(src: Path, dst: Path, *, preserve_if_sha: str | None = None) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if preserve_if_sha and dst.exists() and sha256_file(dst) == preserve_if_sha:
        return dst
    with src.open("rb") as fin, dst.open("wb") as raw_out:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_out, compresslevel=9, mtime=0) as fout:
            shutil.copyfileobj(fin, fout)
    return dst


def artifact_rows(external_store: Path, include_core_outcomes: bool) -> list[dict[str, Any]]:
    specs = [
        ("core", "core", "forecasts"),
        ("extension", "extensions", "forecasts"),
        ("extension", "extensions", "outcomes"),
        ("timing_diagnostic", "timing", "forecasts"),
        ("timing_diagnostic", "timing", "outcomes"),
    ]
    if include_core_outcomes:
        specs.insert(1, ("core", "core", "outcomes"))
    rows: list[dict[str, Any]] = []
    for scope, subdir, kind in specs:
        src = ROOT / "artifacts" / f"{ARTIFACT_PREFIX}_{scope}_{kind}.jsonl"
        if not src.exists():
            rows.append({"scope": scope, "kind": kind, "status": "missing", "source": str(src)})
            continue
        dst = external_store / subdir / f"{ARTIFACT_PREFIX}_{scope}_{kind}.jsonl.gz"
        gzip_copy(src, dst)
        rows.append(
            {
                "scope": scope,
                "kind": kind,
                "status": "present",
                "source": rel(src),
                "external_path": rel(dst),
                "raw_rows": count_jsonl(src),
                "raw_bytes": src.stat().st_size,
                "raw_sha256": sha256_file(src),
                "gzip_bytes": dst.stat().st_size,
                "gzip_sha256": sha256_file(dst),
            }
        )
    for scope, subdir in [("core", "core"), ("extension", "extensions"), ("timing_diagnostic", "timing")]:
        src = ROOT / "artifacts" / f"{ARTIFACT_PREFIX}_{scope}_metrics.json"
        if src.exists():
            dst = external_store / subdir / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            rows.append(
                {
                    "scope": scope,
                    "kind": "metrics",
                    "status": "present",
                    "source": rel(src),
                    "external_path": rel(dst),
                    "raw_rows": "",
                    "raw_bytes": src.stat().st_size,
                    "raw_sha256": sha256_file(src),
                    "gzip_bytes": "",
                    "gzip_sha256": "",
                }
            )
    return rows


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def update_croissant(external_store: Path, manifest: dict[str, Any]) -> None:
    path = ROOT / "metadata" / "croissant.json"
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    payload.update(
        {
            "@context": payload.get("@context") or {"@vocab": "https://schema.org/", "cr": "http://mlcommons.org/croissant/"},
            "@type": payload.get("@type") or "cr:Dataset",
            "name": "TokenCapBench",
            "description": "Forecasts, budgeted solver outcomes, metrics, and manifests for the TokenCapBench benchmark artifact snapshot.",
            "license": payload.get("license") or "MIT",
            "version": payload.get("version") or "0.3.0",
            "url": RELEASE_ARCHIVE,
            "artifactReleaseUrl": RELEASE_ARCHIVE,
            "codeUrl": ".",
            "releaseMode": "relative_path_archive",
            "releaseModeDescription": "Distribution contentUrl paths are relative to the repository or release archive until authors upload an anonymous hosted copy.",
            "hosting_status": "AUTHOR_OWNED_NEEDS_ANONYMOUS_URL",
        }
    )
    distribution = []
    for row in manifest.get("artifacts", []):
        if row.get("status") != "present":
            continue
        if row.get("external_path"):
            rel_path = Path(row["external_path"])
            try:
                rel_path = rel_path.relative_to(ROOT)
            except ValueError:
                pass
            distribution.append(
                {
                    "@type": "cr:FileObject",
                    "name": str(rel_path),
                    "contentUrl": str(rel_path),
                    "encodingFormat": "application/gzip" if str(rel_path).endswith(".gz") else "application/json",
                    "sha256": row.get("gzip_sha256") or row.get("raw_sha256"),
                    "contentSize": row.get("gzip_bytes") or row.get("raw_bytes"),
                }
            )
    distribution.append(
        {
            "@type": "cr:FileObject",
            "name": RELEASE_ARCHIVE,
            "contentUrl": RELEASE_ARCHIVE,
            "encodingFormat": "application/zip",
        }
    )
    payload["distribution"] = distribution
    payload["recordSet"] = payload.get("recordSet") or [{"@type": "cr:RecordSet", "name": "tokencapbench_artifacts"}]
    for key in ["ANONYMIZED_RELEASE_URL_TO_BE_FILLED", "ANONYMIZED_CODE_URL_TO_BE_FILLED"]:
        for field in ["artifactReleaseUrl", "codeUrl"]:
            if payload.get(field) == key:
                payload[field] = "."
    write_json(path, payload)


def zip_inputs() -> list[Path]:
    roots = [
        ROOT / "artifacts",
        ROOT / "external_artifact_store" / "20260504T212800Z",
        ROOT / "paper" / "figures",
        ROOT / "paper" / "tables",
        ROOT / "reports" / "tables",
        ROOT / "reports" / "figures",
        ROOT / "configs",
        ROOT / "prompts",
        ROOT / "metadata",
    ]
    files: list[Path] = []
    for root in roots:
        if root.exists():
            files.extend(
                path
                for path in root.rglob("*")
                if path.is_file() and path.name != "artifact_manifest.json"
                and not _is_private_release_file(path)
            )
    for path in [
        ROOT / "README.md",
        ROOT / "BENCHMARK_CARD.md",
        ROOT / "DATA_PROVENANCE.md",
        ROOT / "REPRODUCING.md",
        ROOT / "CITATION.cff",
        ROOT / "LICENSE",
        ROOT / "paper" / "neurips2026_tokencapbench.tex",
        ROOT / "paper" / "neurips2026_tokencapbench.pdf",
        ROOT / "paper" / "references.bib",
    ]:
        if path.exists():
            files.append(path)
    return sorted(set(files))


def _is_private_release_file(path: Path) -> bool:
    parts = set(path.parts)
    if "__pycache__" in parts or ".pytest_cache" in parts:
        return True
    if path.suffix == ".pyc":
        return True
    if path.suffix.lower() in {".yaml", ".yml"} and (
        path.name.endswith("_local.yaml")
        or path.name.endswith("_local.yml")
        or path.name.endswith("_mock_smoke.yaml")
        or path.name.endswith("_mock_smoke.yml")
    ):
        return True
    return False


def write_archive(output: Path) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    files = zip_inputs()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT))
    file_rows = []
    for path in files:
        rel_name = str(path.relative_to(ROOT))
        file_rows.append({"path": rel_name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {"path": rel(output), "bytes": output.stat().st_size, "sha256": sha256_file(output), "files": file_rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Package the TokenCapBench release archive and external artifact manifest.")
    parser.add_argument("--external-store", required=True)
    parser.add_argument("--include-core-outcomes", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    external_store = Path(args.external_store)
    if not external_store.is_absolute():
        external_store = ROOT / external_store
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output

    artifacts = artifact_rows(external_store, include_core_outcomes=args.include_core_outcomes)
    core_row = next((row for row in artifacts if row.get("scope") == "core" and row.get("kind") == "outcomes"), {})
    if args.include_core_outcomes:
        if core_row.get("raw_rows") != EXPECTED_CORE_ROWS:
            raise SystemExit(f"core outcome row-count mismatch: {core_row}")
    archive_info = write_archive(output)
    manifest = {
        "created_utc": utc_now(),
        "external_store": str(external_store.relative_to(ROOT) if external_store.is_relative_to(ROOT) else external_store),
        "artifacts": artifacts,
        "release_archive": str(output.relative_to(ROOT) if output.is_relative_to(ROOT) else output),
        "release_archive_bytes": archive_info["bytes"],
        "release_archive_sha256": archive_info["sha256"],
        "core_outcomes_expected_rows": EXPECTED_CORE_ROWS,
    }
    write_json(external_store / "artifact_manifest.json", manifest)
    write_json(output.with_suffix(output.suffix + ".sha256_manifest.json"), {"archive": archive_info, "files": archive_info["files"]})
    update_croissant(external_store, manifest)
    # Rebuild the archive after Croissant receives the final relative release fields.
    archive_info = write_archive(output)
    manifest["release_archive_bytes"] = archive_info["bytes"]
    manifest["release_archive_sha256"] = archive_info["sha256"]
    write_json(external_store / "artifact_manifest.json", manifest)
    write_json(output.with_suffix(output.suffix + ".sha256_manifest.json"), {"archive": archive_info, "files": archive_info["files"]})
    update_croissant(external_store, manifest)
    release_manifest = {
        "created_utc": utc_now(),
        "code_commit": git_commit(),
        "commands": [
            "python scripts/make_paper_tables.py --artifact-root reports/artifacts",
            "python scripts/make_paper_figures.py --artifact-root reports/artifacts",
            f"python scripts/package_release_archive.py --external-store external_artifact_store/20260504T212800Z --include-core-outcomes --output {RELEASE_ARCHIVE}",
        ],
        "files": {row["path"]: row["sha256"] for row in archive_info["files"][:500]},
        "live_api_calls_made": False,
        "new_api_spend_usd": 0.0,
        "math_label_mode": "frozen_paper_artifacts",
    }
    write_json(ROOT / "reports" / "release_manifest.json", release_manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


def git_commit() -> str:
    import subprocess

    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


if __name__ == "__main__":
    main()
