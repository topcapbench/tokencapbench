import subprocess
import sys


def test_make_paper_results_report_contains_expected_structure(tmp_path):
    output = tmp_path / "paper_results_report.md"
    subprocess.run(
        [
            sys.executable,
            "scripts/make_paper_results_report.py",
            "--output",
            str(output),
            "--corrected-artifact-root",
            "reports/artifacts_corrected",
            "--math-label-mode",
            "corrected",
        ],
        check=True,
    )

    text = output.read_text(encoding="utf-8")

    assert "TokenCapBench asks" in text
    assert "Figure 2" in text
    assert "Table 3a" in text
    assert "Are we ready for a first draft?" in text
    assert "official LiveCodeBench" in text
    assert "repeatability" in text
    assert "math plus standalone coding" in text or "math plus EvalPlus coding" in text
    assert "configured_not_run" not in text
