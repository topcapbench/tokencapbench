import json

from scripts.validate_submission_package import REQUIRED_DOCS, REQUIRED_FIGURE_PREFIXES, REQUIRED_TABLES, validate_submission_package


def test_validate_submission_package_minimal(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for path in [*REQUIRED_DOCS, "CITATION.cff", "LICENSE"]:
        file_path = tmp_path / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("x\n", encoding="utf-8")
    metadata = tmp_path / "metadata" / "croissant.json"
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(
        json.dumps(
            {
                "@context": {},
                "@type": "cr:Dataset",
                "name": "TokenCapBench",
                "description": "x",
                "license": "MIT",
                "version": "test",
                "url": "https://github.com/example/tokencapbench",
                "releaseMode": "relative_path_archive",
                "releaseModeDescription": "Relative paths are resolved inside the release archive.",
                "distribution": [{"name": "x"}],
                "recordSet": [{"name": "records"}],
            }
        ),
        encoding="utf-8",
    )
    tables = tmp_path / "reports" / "tables"
    tables.mkdir(parents=True)
    for table in REQUIRED_TABLES:
        (tables / table).write_text("a\n1\n", encoding="utf-8")
    (tables / "paper_table_clean_evidence_scope.csv").write_text(
        "track,source,tasks,models,verifier,role\n"
        "math,GSM8K + MATH,1000,4,numeric/symbolic,core evidence\n"
        "coding,HumanEval+ + MBPP+,542,4,EvalPlus,core evidence\n"
        "hard coding,BigCodeBench-Hard,148,5,official BigCodeBench package,main extension\n"
        "code editing,CanItEdit,105,4,provided tests,editing bridge\n"
        "fresh coding,LiveCodeBench-300,300,2,official LiveCodeBench labels,appendix freshness\n"
        "stability,prompt variants,150,2+,frozen verifier outcomes,appendix stability\n",
        encoding="utf-8",
    )
    (tables / "secret_scrub_audit.csv").write_text(
        "path,line,kind,action,patterns_checked,files_scanned,status\nreports/artifacts,,none,pass,1,1,PASS\n",
        encoding="utf-8",
    )
    (tables / "forecast_leakage_audit.csv").write_text("check,status\nno_forecast_leakage,PASS\n", encoding="utf-8")
    figures = tmp_path / "reports" / "figures"
    figures.mkdir(parents=True)
    for prefix in REQUIRED_FIGURE_PREFIXES:
        for suffix in [".png", ".svg"]:
            (figures / f"{prefix}{suffix}").write_bytes(b"x")
    artifact = tmp_path / "reports" / "artifacts" / "run"
    artifact.mkdir(parents=True)
    for name in ["forecasts.jsonl", "outcomes.jsonl", "sha256_manifest.json"]:
        (artifact / name).write_text("{}\n", encoding="utf-8")
    release_manifest = tmp_path / "reports" / "release_manifest.json"
    release_manifest.write_text(
        json.dumps(
            {
                "created_utc": "2026-04-29T00:00:00+00:00",
                "code_commit": "abc",
                "commands": ["pytest"],
                "files": {"README.md": "hash"},
                "live_api_calls_made": False,
                "new_api_spend_usd": 0.0,
                "math_label_mode": "corrected",
            }
        ),
        encoding="utf-8",
    )
    archive = tmp_path / "reports" / "tokencapbench_release_archive.zip"
    import zipfile

    with zipfile.ZipFile(archive, "w") as zf:
        for name in [
            *REQUIRED_DOCS,
            "reports/artifacts/run/forecasts.jsonl",
            "reports/artifacts/run/outcomes.jsonl",
            "reports/artifacts/run/metrics.json",
            "reports/tables/table.csv",
            "reports/figures/figure.png",
            "configs/config.yaml",
            "prompts/forecast.md",
            "metadata/croissant.json",
        ]:
            zf.writestr(name, "x\n")
    archive.with_suffix(archive.suffix + ".sha256_manifest.json").write_text('{"files": {}}\n', encoding="utf-8")

    path, ok = validate_submission_package(run_pytest=False)

    assert ok
    assert path.exists()


def test_validate_submission_package_fails_on_secret_findings(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    test_validate_submission_package_minimal(tmp_path, monkeypatch)
    (tmp_path / "reports" / "tables" / "secret_scrub_audit.csv").write_text(
        "path,line,kind,action,patterns_checked,files_scanned,status\nx,1,api_key,manual_review_required,1,1,REVIEW\n",
        encoding="utf-8",
    )

    _path, ok = validate_submission_package(run_pytest=False)

    assert not ok


def test_validate_submission_package_fails_if_required_table_deleted(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    test_validate_submission_package_minimal(tmp_path, monkeypatch)
    (tmp_path / "reports" / "tables" / REQUIRED_TABLES[0]).unlink()

    _path, ok = validate_submission_package(run_pytest=False)

    assert not ok


def test_validate_submission_package_fails_on_strict_fixed_budget_overshoot(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    test_validate_submission_package_minimal(tmp_path, monkeypatch)
    table = tmp_path / "reports" / "tables" / "paper_table15_fixed_budget_scheduling.csv"
    table.write_text(
        "suite,model,method,policy,budget_fraction,target_total_budget,selected_total_budget,"
        "budget_used,budget_slack_tokens,strict_budget_feasible\n"
        "s,m,self_forecast_raw,policy_b,0.50,100,100,120,-20,0\n",
        encoding="utf-8",
    )

    _path, ok = validate_submission_package(run_pytest=False)

    assert not ok


def test_strict_final_submission_requires_hosted_distribution_urls(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    test_validate_submission_package_minimal(tmp_path, monkeypatch)

    _path, ok = validate_submission_package(run_pytest=False, strict_final_submission=True)

    assert not ok


def test_validate_submission_package_rejects_incompatible_main_rows(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    test_validate_submission_package_minimal(tmp_path, monkeypatch)
    task_dir = tmp_path / "data" / "processed"
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "paper_bad.jsonl").write_text(
        json.dumps(
            {
                "task_id": "swebench_bad",
                "track": "swe",
                "prompt": "p",
                "verifier": "swebench",
                "source": "swebench",
                "metadata": {"paper_role": "paper", "chat_completion_compatible": False},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    _path, ok = validate_submission_package(run_pytest=False)

    assert not ok
