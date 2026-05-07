from __future__ import annotations


def observed_budget2success(outcomes: dict[int, bool]) -> int | None:
    """Return the smallest budget with success, or None if all failed."""
    for budget in sorted(outcomes):
        if outcomes[budget]:
            return budget
    return None
