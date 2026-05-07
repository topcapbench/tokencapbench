#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from budget2success.analysis.paper import load_paper_runs
from budget2success.utils.manifest import sha256_file


def make_croissant_metadata(
    *,
    artifact_root: str | Path = "reports/artifacts",
    output: str | Path = "metadata/croissant.json",
    dataset_url: str = "ANONYMIZED_CODE_URL_TO_BE_FILLED",
    release_url: str = "ANONYMIZED_RELEASE_URL_TO_BE_FILLED",
    creator_name: str = "TokenCapBench authors",
    version: str = "2026-04-29",
    date_published: str = "2026-04-29",
    license_name: str = "MIT",
) -> Path:
    artifact_root = Path(artifact_root)
    runs = load_paper_runs(
        run_root=Path(artifact_root) / "__no_reports_runs__",
        artifact_root=artifact_root,
        include_artifacts=True,
    )
    distributions: list[dict[str, Any]] = []
    for path in sorted(artifact_root.glob("*")):
        if not path.is_dir():
            continue
        for name in (
            "forecasts.jsonl",
            "outcomes.jsonl",
            "metrics.json",
            "config_snapshot.yaml",
            "source_config_snapshot.yaml",
            "run_manifest.json",
            "sha256_manifest.json",
            "task_file_hash.json",
        ):
            file_path = path / name
            if file_path.exists():
                distributions.append(
                    {
                        "@type": "cr:FileObject",
                        "name": f"{path.name}/{name}",
                        "contentUrl": str(file_path),
                        "encodingFormat": _encoding_format(name),
                        "sha256": sha256_file(file_path),
                    }
                )
    for root_name in ("reports/artifacts_corrected", "reports/splits", "reports/tables", "reports/figures"):
        root = Path(root_name)
        for file_path in sorted(root.rglob("*")) if root.exists() else []:
            if file_path.is_file() and file_path.suffix.lower() in {".csv", ".json", ".jsonl", ".yaml", ".yml", ".png", ".svg", ".md"}:
                distributions.append(
                    {
                        "@type": "cr:FileObject",
                        "name": str(file_path),
                        "contentUrl": str(file_path),
                        "encodingFormat": _encoding_format(file_path.name),
                        "sha256": sha256_file(file_path),
                    }
                )
    release_manifest = Path("reports/release_manifest.json")
    if release_manifest.exists():
        distributions.append(
            {
                "@type": "cr:FileObject",
                "name": str(release_manifest),
                "contentUrl": str(release_manifest),
                "encodingFormat": _encoding_format(release_manifest.name),
                "sha256": sha256_file(release_manifest),
            }
        )
    metadata = {
        "@context": {
            "@vocab": "https://schema.org/",
            "cr": "http://mlcommons.org/croissant/",
        },
        "@type": "cr:Dataset",
        "name": "TokenCapBench frozen paper artifacts",
        "description": (
            "Forecasts, budgeted solver outcomes, metrics, and manifests for the "
            "TokenCapBench paper artifact snapshot. "
            "The benchmark measures calibrated forecasts of verified success under generated-token budgets."
        ),
        "license": license_name,
        "version": version,
        "datePublished": date_published,
        "creator": [{"@type": "Organization", "name": creator_name}],
        "citation": "See CITATION.cff",
        "url": dataset_url,
        "codeUrl": dataset_url,
        "artifactReleaseUrl": release_url,
        "releaseMode": "relative_path_archive",
        "releaseModeDescription": (
            "Distribution contentUrl values are repository-relative paths for archival release. "
            "Replace them with hosted HTTPS URLs for strict final submission packaging."
        ),
        "keywords": ["calibration", "token budgets", "verified success", "LLM evaluation"],
        "intendedUse": (
            "Evaluate whether language models can forecast their probability of verified task success "
            "under generated-token budgets before solver tokens are spent."
        ),
        "outOfScopeUse": (
            "Do not use these artifacts as a raw output-length benchmark or as evidence for full SWE/agentic "
            "coverage without official harness runs."
        ),
        "dataSources": [
            "GSM8K/MATH core",
            "EvalPlus HumanEval+/MBPP+ core",
            "BigCodeBench-Hard extension with official-package labels",
            "CanItEdit provided-test editing bridge",
            "LiveCodeBench-300 appendix freshness",
            "token-proxy analysis splits",
            "strict fixed-budget scheduling summaries",
        ],
        "annotationAndVerifierProcess": (
            "Forecasts are produced in separate contexts from solver runs. Solver outputs are generated under "
            "hard token caps and scored with deterministic task verifiers or official benchmark harness bridges."
        ),
        "knownLimitations": (
            "Budgets are generated-token caps, not full hidden reasoning compute. CanItEdit uses provided tests, "
            "not hidden tests. The package makes no official SWE/Docker/agent-runtime claim. Public benchmark "
            "contamination is possible for standard substrates."
        ),
        "sensitiveContentStatement": (
            "The benchmark uses public math and programming tasks and does not intentionally collect personal data. "
            "Generated model outputs may still contain unexpected text and should be scrubbed before release."
        ),
        "maintenancePlan": (
            "Refresh task splits, provider pricing, official harness pins, and SHA-256 manifests when rerunning "
            "paper experiments or adding new benchmark sources."
        ),
        "rai": {
            "intended_use": "Token-budget self-forecasting research and evaluation.",
            "out_of_scope_use": "Raw output-length prediction or unsupported SWE/agentic claims.",
            "data_sources": "Public benchmark tasks plus generated model forecasts and solver outcomes.",
            "annotation_verifier_process": "Deterministic external verification of hard-capped solver outputs.",
            "known_limitations": "Generated-token caps do not include hidden reasoning compute; CanItEdit uses provided tests; no official SWE/Docker/agent claim is made.",
            "sensitive_content": "No intentional personal-data collection; release scrub is required.",
            "maintenance_plan": "Pin harness versions and regenerate manifests on every release refresh.",
        },
        "recordSet": [
            {
                "@type": "cr:RecordSet",
                "name": "paper_runs",
                "description": "One record per packaged run directory.",
                "field": [
                    {"@type": "cr:Field", "name": "suite", "dataType": "sc:Text"},
                    {"@type": "cr:Field", "name": "run_id", "dataType": "sc:Text"},
                    {"@type": "cr:Field", "name": "model", "dataType": "sc:Text"},
                    {"@type": "cr:Field", "name": "forecasts", "dataType": "sc:Integer"},
                    {"@type": "cr:Field", "name": "outcomes", "dataType": "sc:Integer"},
                ],
            }
        ],
        "runSummary": [
            {
                "suite": run.suite or "",
                "run_id": run.run_id,
                "model": run.model,
                "forecasts": len(run.forecasts),
                "outcomes": len(run.outcomes),
                "artifact_source": run.artifact_source,
            }
            for run in runs
        ],
        "distribution": distributions,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return output


def _encoding_format(name: str) -> str:
    if name.endswith(".jsonl"):
        return "application/jsonlines"
    if name.endswith(".json"):
        return "application/json"
    if name.endswith((".yaml", ".yml")):
        return "application/x-yaml"
    if name.endswith(".csv"):
        return "text/csv"
    if name.endswith(".png"):
        return "image/png"
    if name.endswith(".svg"):
        return "image/svg+xml"
    if name.endswith(".md"):
        return "text/markdown"
    return "text/plain"


def main() -> None:
    parser = argparse.ArgumentParser(description="Write Croissant-style metadata for packaged paper artifacts.")
    parser.add_argument("--artifact-root", default="reports/artifacts")
    parser.add_argument("--output", default="metadata/croissant.json")
    parser.add_argument("--dataset-url", default="ANONYMIZED_CODE_URL_TO_BE_FILLED")
    parser.add_argument("--release-url", default="ANONYMIZED_RELEASE_URL_TO_BE_FILLED")
    parser.add_argument("--creator-name", default="TokenCapBench authors")
    parser.add_argument("--version", default="2026-04-29")
    parser.add_argument("--date-published", default="2026-04-29")
    parser.add_argument("--license-name", default="MIT")
    args = parser.parse_args()
    path = make_croissant_metadata(
        artifact_root=args.artifact_root,
        output=args.output,
        dataset_url=args.dataset_url,
        release_url=args.release_url,
        creator_name=args.creator_name,
        version=args.version,
        date_published=args.date_published,
        license_name=args.license_name,
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
