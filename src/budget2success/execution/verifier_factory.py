from __future__ import annotations

from budget2success.execution.verifier import Verifier
from budget2success.execution.verifier_registry import get_verifier


def build_verifier(name: str) -> Verifier:
    """Backward-compatible factory wrapper around the active verifier registry."""
    aliases = {
        "math_answer": "exact_match",
        "exact_match_math": "exact_match",
        "python_unit_tests": "python_unit_test",
        "swebench_official": "swebench",
        "external_command": "external_command",
    }
    return get_verifier(aliases.get(name, name))
