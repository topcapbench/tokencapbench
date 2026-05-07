import pytest

from scripts import make_paper_tables


def test_manual_math_audit_final_mode_rejects_zero_annotations(tmp_path, monkeypatch):
    monkeypatch.setattr(make_paper_tables, "TABLE_DIR", tmp_path)
    (tmp_path / "manual_math_verifier_audit_sample.csv").write_text(
        "task_id,source,model,budget,old_success,new_success,solution_excerpt,gold_answer,human_audit_label,audit_note\n"
        "gsm8k_test_0,gsm8k,model,128,False,False,x,1,,\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="at least one annotated"):
        make_paper_tables._manual_math_audit_summary_rows(final_paper_mode=True)
