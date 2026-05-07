#!/usr/bin/env python
from __future__ import annotations

import argparse
import glob
import json
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from budget2success.analysis.paper import load_run_config
from budget2success.utils.manifest import (
    dataset_summary,
    prompt_hashes,
    redact_secrets,
    sha256_file,
    write_redacted_config_snapshot,
    write_run_manifest,
)


REQUIRED_RUN_FILES = ("forecasts.jsonl", "outcomes.jsonl", "metrics.json")
CONFIG_NAMES = ("config_snapshot.yaml", "config.yaml")


def package_run_artifacts(
    run_dir: str | Path,
    *,
    artifact_root: str | Path = "reports/artifacts",
    strict: bool = True,
) -> Path | None:
    run_path = Path(run_dir)
    if not run_path.exists():
        raise FileNotFoundError(run_path)
    missing = [name for name in REQUIRED_RUN_FILES if not (run_path / name).exists()]
    config_path = _find_config_path(run_path)
    if config_path is None:
        missing.append("config_snapshot.yaml or config.yaml")
    if missing:
        message = f"{run_path} is missing required run artifacts: {', '.join(missing)}"
        if strict:
            raise FileNotFoundError(message)
        print(f"SKIP: {message}", file=sys.stderr)
        return None

    run_id = run_path.name
    artifact_id = _artifact_id(run_path)
    artifact_dir = Path(artifact_root) / artifact_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    for name in REQUIRED_RUN_FILES:
        shutil.copy2(run_path / name, artifact_dir / name)

    config = load_run_config(run_path)
    write_redacted_config_snapshot(config, artifact_dir / "config_snapshot.yaml")
    if config_path and config_path.exists():
        shutil.copy2(config_path, artifact_dir / f"source_{config_path.name}")

    _copy_prompt_snapshots(config, artifact_dir)
    _write_task_file_hash(config, artifact_dir)

    run_manifest = write_run_manifest(
        artifact_dir,
        config=config,
        command_line_arguments=sys.argv[1:],
        phase="packaged",
        extra={
            "source_run_dir": str(run_path),
            "artifact_id": artifact_id,
            "dataset": dataset_summary(config.get("task_file")),
            "prompt_hashes": prompt_hashes(config),
            "packaging_note": "No environment variables or secrets are copied.",
        },
    )
    sha_manifest = _write_sha_manifest(artifact_dir)
    return artifact_dir if run_manifest.exists() and sha_manifest.exists() else Path()


def package_all_runs(*, artifact_root: str | Path = "reports/artifacts", strict: bool = True) -> list[Path]:
    packaged: list[Path] = []
    for metrics_path in sorted(Path("reports/runs").glob("**/metrics.json")):
        artifact_dir = package_run_artifacts(metrics_path.parent, artifact_root=artifact_root, strict=strict)
        if artifact_dir is not None:
            packaged.append(artifact_dir)
    return packaged


def package_release_archive(
    *,
    artifact_root: str | Path = "reports/artifacts",
    tables_dir: str | Path = "reports/tables",
    figures_dir: str | Path = "reports/figures",
    metadata_dir: str | Path = "metadata",
    output: str | Path = "reports/tokencapbench_release_archive.zip",
) -> Path:
    roots = {
        "reports/artifacts": Path(artifact_root),
        "reports/tables": Path(tables_dir),
        "reports/figures": Path(figures_dir),
        "metadata": Path(metadata_dir),
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    files = _release_files(roots)
    if not files:
        raise FileNotFoundError("No release files matched the required artifact patterns.")

    manifest: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "archive": str(output_path),
        "files": {},
    }
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source, arcname in files:
            archive.write(source, arcname)
            manifest["files"][arcname] = sha256_file(source)
        archive.writestr("SHA256_MANIFEST.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    sidecar = output_path.with_suffix(output_path.suffix + ".sha256_manifest.json")
    manifest["archive_sha256"] = sha256_file(output_path)
    sidecar.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def _release_files(roots: dict[str, Path]) -> list[tuple[Path, str]]:
    selected: dict[str, Path] = {}
    dynamic_globs = [
        f"{roots['reports/artifacts']}/*/forecasts.jsonl",
        f"{roots['reports/artifacts']}/*/outcomes.jsonl",
        f"{roots['reports/artifacts']}/*/metrics.json",
        f"{roots['reports/tables']}/*.csv",
        f"{roots['reports/figures']}/*.png",
        "configs/**/*.yaml",
        "prompts/*.md",
        str(roots["metadata"] / "croissant.json"),
    ]
    for pattern in dynamic_globs:
        for match in sorted(glob.glob(pattern, recursive=True)):
            path = Path(match)
            if not path.is_file():
                continue
            arcname = _release_arcname(path, roots)
            selected[arcname] = path
    return [(path, arcname) for arcname, path in sorted(selected.items())]


def _release_arcname(path: Path, roots: dict[str, Path]) -> str:
    path = path.resolve()
    for canonical, root in roots.items():
        try:
            rel = path.relative_to(root.resolve())
        except ValueError:
            continue
        return str(Path(canonical) / rel).replace("\\", "/")
    return str(Path(path).relative_to(Path.cwd().resolve())).replace("\\", "/")


def _artifact_id(run_path: Path) -> str:
    if run_path.parent.name != "runs" and run_path.parent.parent.name == "runs":
        return f"{run_path.parent.name}__{run_path.name}"
    return run_path.name


def _find_config_path(run_path: Path) -> Path | None:
    for name in CONFIG_NAMES:
        path = run_path / name
        if path.exists():
            return path
    legacy = Path("reports/live_configs") / f"{run_path.name}.yaml"
    if legacy.exists():
        return legacy
    return None


def _copy_prompt_snapshots(config: dict[str, Any], artifact_dir: Path) -> None:
    prompt_dir = artifact_dir / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    forecast_prompt = config.get("forecast_prompt")
    if forecast_prompt and Path(forecast_prompt).exists():
        shutil.copy2(forecast_prompt, prompt_dir / f"forecast_{Path(forecast_prompt).name}")
    solver_prompts = config.get("solver_prompts") or {}
    if isinstance(solver_prompts, dict):
        for track, prompt_path in sorted(solver_prompts.items()):
            if prompt_path and Path(prompt_path).exists():
                shutil.copy2(prompt_path, prompt_dir / f"solver_{track}_{Path(prompt_path).name}")


def _write_task_file_hash(config: dict[str, Any], artifact_dir: Path) -> None:
    task_file = config.get("task_file")
    if not task_file:
        return
    path = Path(task_file)
    payload = {"task_file": str(path), "exists": path.exists()}
    if path.exists():
        payload["sha256"] = sha256_file(path)
    (artifact_dir / "task_file_hash.json").write_text(
        json.dumps(redact_secrets(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_sha_manifest(artifact_dir: Path) -> Path:
    rows = {}
    for path in sorted(artifact_dir.rglob("*")):
        if path.is_file() and path.name != "sha256_manifest.json":
            rows[str(path.relative_to(artifact_dir))] = sha256_file(path)
    out_path = artifact_dir / "sha256_manifest.json"
    out_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Package raw TokenCapBench run artifacts for reproducibility.")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--run-dir", help="Run directory, e.g. reports/runs/<run_id>")
    group.add_argument("--all-runs", action="store_true", help="Package all run directories with metrics.json.")
    parser.add_argument("--artifact-root", default="reports/artifacts")
    parser.add_argument("--tables-dir", default="reports/tables")
    parser.add_argument("--figures-dir", default="reports/figures")
    parser.add_argument("--metadata-dir", default="metadata")
    parser.add_argument("--output", default="reports/tokencapbench_release_archive.zip")
    parser.add_argument("--skip-incomplete", action="store_true", help="Skip incomplete runs instead of failing.")
    args = parser.parse_args()

    if args.all_runs:
        packaged = package_all_runs(artifact_root=args.artifact_root, strict=not args.skip_incomplete)
        print(json.dumps({"packaged": [str(path) for path in packaged]}, indent=2))
    elif args.run_dir:
        artifact_dir = package_run_artifacts(args.run_dir, artifact_root=args.artifact_root, strict=not args.skip_incomplete)
        packaged = [artifact_dir] if artifact_dir is not None else []
        print(json.dumps({"packaged": [str(path) for path in packaged]}, indent=2))
    else:
        archive = package_release_archive(
            artifact_root=args.artifact_root,
            tables_dir=args.tables_dir,
            figures_dir=args.figures_dir,
            metadata_dir=args.metadata_dir,
            output=args.output,
        )
        print(json.dumps({"archive": str(archive), "sha256": sha256_file(archive)}, indent=2))


if __name__ == "__main__":
    main()
