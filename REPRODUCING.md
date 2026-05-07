# Reproducing TokenCapBench

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```

Optional extras are declared in `pyproject.toml` for dataset loading, provider clients, math verification, coding harnesses, and report generation.

## Reproduce Tables and Figures from Frozen Artifacts

```bash
python scripts/make_paper_tables.py --artifact-root reports/artifacts
python scripts/make_paper_figures.py --artifact-root reports/artifacts
python scripts/validate_submission_package.py \
  --artifact-root reports/artifacts \
  --tables-dir reports/tables \
  --figures-dir reports/figures \
  --metadata-dir metadata
```

These commands use the frozen artifacts already present in the repository. They do not make live API calls.

## Rebuild the Release Archive

```bash
python scripts/package_release_archive.py \
  --external-store external_artifact_store/20260504T212800Z \
  --include-core-outcomes \
  --output reports/tokencapbench_release_archive.zip
```

The archive includes public benchmark documentation, frozen artifacts, paper tables and figures, prompts, configs, metadata, and release manifests.

## Live Reruns

Live reruns require provider credentials and may incur provider costs. Configure an OpenAI-compatible endpoint through environment variables or the relevant experiment config:

```bash
export OPENAI_COMPATIBLE_BASE_URL="https://provider.example/v1"
export OPENAI_COMPATIBLE_API_KEY="<api-key>"
```

Provider reruns should write fresh run directories with `forecasts.jsonl`, `outcomes.jsonl`, `metrics.json`, `run_manifest.json`, and `sha256_manifest.json`. Do not overwrite the frozen paper artifacts unless intentionally preparing a new release.

## Validation

Before publishing a new artifact bundle, run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
python scripts/validate_submission_package.py
```

Then inspect `reports/tables/submission_package_validation.csv` and `reports/tables/secret_scrub_audit.csv`.
