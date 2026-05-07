import json

from scripts.make_croissant_metadata import make_croissant_metadata


def test_make_croissant_metadata_includes_required_and_rai_fields(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "reports" / "artifacts" / "paper_math_core__m"
    run_dir.mkdir(parents=True)
    for name in ["forecasts.jsonl", "outcomes.jsonl", "metrics.json", "config_snapshot.yaml", "sha256_manifest.json"]:
        (run_dir / name).write_text("{}\n", encoding="utf-8")
    (tmp_path / "reports" / "tables").mkdir(parents=True)
    (tmp_path / "reports" / "tables" / "paper_table1_related_work.csv").write_text("a\n1\n", encoding="utf-8")
    output = make_croissant_metadata(
        artifact_root=tmp_path / "reports" / "artifacts",
        output=tmp_path / "metadata" / "croissant.json",
        dataset_url="https://github.com/example/budget2success",
        creator_name="Example",
        version="test",
        date_published="2026-04-29",
        license_name="MIT",
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    for field in ["@context", "@type", "name", "description", "license", "version", "distribution", "recordSet"]:
        assert field in payload
    assert payload["releaseMode"] == "relative_path_archive"
    assert payload["releaseModeDescription"]
    for field in ["intendedUse", "outOfScopeUse", "dataSources", "annotationAndVerifierProcess", "knownLimitations", "sensitiveContentStatement", "maintenancePlan", "rai"]:
        assert field in payload
    distribution_names = {item["name"] for item in payload["distribution"]}
    assert "paper_math_core__m/forecasts.jsonl" in distribution_names
    assert "reports/tables/paper_table1_related_work.csv" in distribution_names
    assert not any(name.startswith("docs/") for name in distribution_names)
