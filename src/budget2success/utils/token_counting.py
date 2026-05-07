from __future__ import annotations


def approximate_token_count(text: str) -> int:
    """Cheap fallback token estimate for smoke tests.

    Paper results should use provider-reported or tokenizer-counted tokens.
    This fallback deliberately over-documents its weakness to avoid accidental
    use as a scientific measurement.
    """
    if not text:
        return 0
    # A common rough heuristic: one token is around four English characters.
    return max(1, round(len(text) / 4))
