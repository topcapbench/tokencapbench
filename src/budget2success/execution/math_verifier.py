from __future__ import annotations

from enum import Enum
from fractions import Fraction
import re

from budget2success.execution.verifier import Verifier
from budget2success.schemas.records import TaskRecord, VerificationResult

_BOX_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
_FRAC_RE = re.compile(r"\\frac\{(-?\d+)\}\{(-?\d+)\}")
_NUM_RE = re.compile(r"-?(?:\d+\s*/\s*\d+|\d*\.\d+|\d+)")
_PLAIN_NUMBER_RE = re.compile(r"^-?(?:\d+(?:\.\d*)?|\.\d+)$")
_PLAIN_FRACTION_RE = re.compile(r"^-?\d+\s*/\s*-?\d+$")
_LATEX_FRACTION_RE = re.compile(r"^\\frac\{[^{}]+\}\{[^{}]+\}$")


class MathVerifierMode(str, Enum):
    STRICT = "strict"
    LENIENT = "lenient"
    MATH_VERIFY_OPTIONAL = "math_verify_optional"


def extract_boxed_answer(text: str) -> str | None:
    matches = _BOX_RE.findall(text or "")
    if matches:
        return matches[-1].strip()
    return None


def normalize_numeric(text: str) -> str:
    text = _latex_fractions_to_plain(str(text))
    return text.strip().replace(",", "").replace("$", "").replace(" ", "")


def extract_final_numeric_answer(text: str) -> str | None:
    """Extract the answer the model presents as final, not any number in the reasoning."""
    text = _latex_fractions_to_plain(str(text or ""))
    boxed = extract_boxed_answer(text)
    if boxed is not None:
        numbers = _NUM_RE.findall(boxed)
        return normalize_numeric(numbers[-1] if numbers else boxed)
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    finalish = [
        line
        for line in lines
        if re.search(r"(final\s+answer|answer\s+is|^answer\s*:|therefore)", line, flags=re.IGNORECASE)
    ]
    search_space = "\n".join(finalish[-2:] or lines[-2:])
    numbers = _NUM_RE.findall(search_space)
    return normalize_numeric(numbers[-1]) if numbers else None


def classify_math_answer(answer: str | None) -> str:
    """Coarse answer-shape tag for verifier audit tables.

    This is intentionally syntactic metadata. It does not try to decide whether
    two math expressions are equivalent.
    """
    if answer is None:
        return "missing"
    text = str(answer).strip()
    if not text:
        return "missing"
    boxed = extract_boxed_answer(text)
    if boxed is not None:
        text = boxed.strip()
    compact = re.sub(r"\s+", "", text.strip().strip("$"))
    lower = compact.lower()
    if not compact:
        return "missing"
    if _looks_like_interval_or_set(lower):
        return "interval_or_set"
    if _looks_like_tuple_or_coordinate(lower):
        return "tuple_or_coordinate"
    if _PLAIN_FRACTION_RE.fullmatch(compact) or _LATEX_FRACTION_RE.fullmatch(compact):
        return "fraction"
    if _PLAIN_NUMBER_RE.fullmatch(compact.replace(",", "")):
        return "numeric"
    if _looks_like_expression(lower):
        return "expression"
    return "text_or_other"


def _latex_fractions_to_plain(text: str) -> str:
    return _FRAC_RE.sub(lambda match: f"{match.group(1)}/{match.group(2)}", text)


def _looks_like_tuple_or_coordinate(text: str) -> bool:
    if "," not in text:
        return False
    has_parens = (
        text.startswith("(")
        or text.startswith("\\left(")
        or text.startswith("\\big(")
        or text.startswith("\\bigl(")
    )
    has_closing = text.endswith(")") or text.endswith("\\right)") or text.endswith("\\big)") or text.endswith("\\bigr)")
    return has_parens and has_closing


def _looks_like_interval_or_set(text: str) -> bool:
    if any(marker in text for marker in ["\\infty", "\\cup", "\\cap", "≤", "≥"]):
        return True
    if re.search(r"\\(?:leq?|geq?|in)(?![A-Za-z])", text):
        return True
    if text.startswith(("[", "\\left[")) and "," in text:
        return True
    if text.startswith(("{", "\\{", "\\left\\{")) and text.endswith(("}", "\\}", "\\right\\}")):
        return True
    if re.search(r"[\[\]]", text) and "," in text:
        return True
    return False


def _looks_like_expression(text: str) -> bool:
    if any(marker in text for marker in ["\\sqrt", "\\pi", "\\sin", "\\cos", "\\tan", "\\log", "\\ln", "^", "="]):
        return True
    if re.search(r"[a-zA-Z]", text):
        return True
    if any(op in text for op in ["+", "*"]):
        return True
    # A minus sign outside a leading numeric sign usually indicates an
    # expression after numeric and fraction cases have already been removed.
    return "-" in text[1:]


def _numeric_equivalent(left: str | None, right: str | None) -> bool:
    if left is None or right is None:
        return False
    left_norm = normalize_numeric(left)
    right_norm = normalize_numeric(right)
    if left_norm == right_norm:
        return True
    try:
        return Fraction(left_norm) == Fraction(right_norm)
    except Exception:
        try:
            return abs(float(left_norm) - float(right_norm)) <= 1e-9
        except Exception:
            return False


def _math_metadata(
    *,
    mode: MathVerifierMode | str,
    extracted_prediction: str | None,
    extracted_gold: str | None,
    math_verify_available: bool,
    **extra: object,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "verifier_mode": str(mode.value if isinstance(mode, MathVerifierMode) else mode),
        "extracted_prediction": extracted_prediction,
        "extracted_gold": extracted_gold,
        "math_verify_available": bool(math_verify_available),
    }
    metadata.update(extra)
    return metadata


class ExactMatchMathVerifier(Verifier):
    def verify(self, task: TaskRecord, solution: str) -> VerificationResult:
        if task.answer is None:
            metadata = _math_metadata(
                mode=MathVerifierMode.STRICT,
                extracted_prediction=None,
                extracted_gold=None,
                math_verify_available=False,
            )
            return VerificationResult.error(error="missing_answer", metadata=metadata)
        candidate = extract_boxed_answer(solution)
        if candidate is None:
            lines = [line.strip() for line in solution.strip().splitlines() if line.strip()]
            candidate = lines[-1] if lines else solution.strip()
        candidate_norm = normalize_numeric(candidate)
        answer_norm = normalize_numeric(task.answer)
        metadata = _math_metadata(
            mode=MathVerifierMode.STRICT,
            extracted_prediction=candidate_norm,
            extracted_gold=answer_norm,
            math_verify_available=False,
        )
        return VerificationResult(
            status="success" if candidate_norm == answer_norm else "failure",
            success=candidate_norm == answer_norm,
            details={"candidate": candidate, "candidate_norm": candidate_norm, "answer_norm": answer_norm},
            metadata=metadata,
        )


class NumericExactVerifier(Verifier):
    def __init__(self, mode: str = "strict") -> None:
        if mode not in {MathVerifierMode.STRICT.value, MathVerifierMode.LENIENT.value}:
            raise ValueError("mode must be 'strict' or 'lenient'")
        self.mode = MathVerifierMode(mode)

    def verify(self, task: TaskRecord, solution: str) -> VerificationResult:
        if task.answer is None:
            metadata = _math_metadata(
                mode=self.mode,
                extracted_prediction=None,
                extracted_gold=None,
                math_verify_available=False,
            )
            return VerificationResult.error(error="missing_answer", metadata=metadata)
        answer = normalize_numeric(task.answer)
        extracted = extract_final_numeric_answer(solution)
        candidates = [normalize_numeric(c) for c in _NUM_RE.findall(solution)]
        if self.mode == MathVerifierMode.STRICT:
            success = _numeric_equivalent(extracted, answer)
        else:
            success = any(_numeric_equivalent(candidate, answer) for candidate in candidates[-3:])
        metadata = _math_metadata(
            mode=self.mode,
            extracted_prediction=extracted,
            extracted_gold=answer,
            math_verify_available=False,
        )
        return VerificationResult(
            status="success" if success else "failure",
            success=success,
            details={
                "mode": self.mode.value,
                "extracted": extracted,
                "gold": answer,
                "candidates": candidates[-5:],
            },
            metadata=metadata,
        )


class MathVerifyOptionalVerifier(Verifier):
    """Use Hugging Face math-verify if installed; otherwise fall back."""

    def __init__(self) -> None:
        self._fallback = ExactMatchMathVerifier()
        try:
            from math_verify import parse, verify  # type: ignore

            self._parse = parse
            self._verify = verify
        except Exception:
            self._parse = None
            self._verify = None

    def verify(self, task: TaskRecord, solution: str) -> VerificationResult:
        if task.answer is None:
            metadata = _math_metadata(
                mode=MathVerifierMode.MATH_VERIFY_OPTIONAL,
                extracted_prediction=None,
                extracted_gold=None,
                math_verify_available=self._parse is not None and self._verify is not None,
            )
            return VerificationResult.error(error="missing_answer", metadata=metadata)
        if self._parse is None or self._verify is None:
            result = self._fallback.verify(task, solution)
            result.details["math_verify_available"] = False
            result.metadata.update(
                _math_metadata(
                    mode=MathVerifierMode.MATH_VERIFY_OPTIONAL,
                    extracted_prediction=result.metadata.get("extracted_prediction"),
                    extracted_gold=result.metadata.get("extracted_gold"),
                    math_verify_available=False,
                )
            )
            return result
        try:
            gold = self._parse(task.answer)
            pred = self._parse(solution)
            success = bool(self._verify(gold, pred))
            extracted = extract_final_numeric_answer(solution)
            answer = normalize_numeric(task.answer)
            metadata = _math_metadata(
                mode=MathVerifierMode.MATH_VERIFY_OPTIONAL,
                extracted_prediction=extracted,
                extracted_gold=answer,
                math_verify_available=True,
            )
            return VerificationResult(
                status="success" if success else "failure",
                success=success,
                details={"math_verify_available": True, "extracted": extracted, "gold": answer},
                metadata=metadata,
            )
        except Exception as exc:
            result = self._fallback.verify(task, solution)
            result.details.update({"math_verify_available": True, "math_verify_error": str(exc)})
            result.metadata.update(
                _math_metadata(
                    mode=MathVerifierMode.MATH_VERIFY_OPTIONAL,
                    extracted_prediction=result.metadata.get("extracted_prediction"),
                    extracted_gold=result.metadata.get("extracted_gold"),
                    math_verify_available=True,
                    math_verify_error=str(exc),
                )
            )
            return result


def _math_verify_available(verifier: MathVerifyOptionalVerifier) -> bool:
    return verifier._parse is not None and verifier._verify is not None


def _is_symbolic_math_source(source: str) -> bool:
    source = (source or "").lower()
    if source == "gsm8k":
        return False
    return source in {"hendrycks_math", "math", "math500"} or ("math" in source and "gsm8k" not in source)


class TaskAwareMathVerifier(Verifier):
    """Route GSM8K to strict numeric verification and MATH-style tasks to math-verify."""

    def __init__(self, require_math_verify_for_symbolic: bool = False) -> None:
        self.require_math_verify_for_symbolic = bool(require_math_verify_for_symbolic)
        self._numeric_strict = NumericExactVerifier(mode="strict")
        self._math_verify = MathVerifyOptionalVerifier()

    def verify(self, task: TaskRecord, solution: str) -> VerificationResult:
        source = (task.source or "").lower()
        if _is_symbolic_math_source(source):
            if self.require_math_verify_for_symbolic and not _math_verify_available(self._math_verify):
                metadata = self._metadata(
                    source=source,
                    mode="task_aware_strict",
                    policy="math_verify_required_for_symbolic",
                    extracted_prediction=extract_final_numeric_answer(solution),
                    extracted_gold=str(task.answer) if task.answer is not None else None,
                    math_verify_available=False,
                )
                return VerificationResult.error(error="math_verify_required", metadata=metadata)
            result = self._math_verify.verify(task, solution)
            result.metadata.update(
                self._metadata(
                    source=source,
                    mode="task_aware_strict" if self.require_math_verify_for_symbolic else "task_aware_math",
                    policy="math_verify_for_symbolic",
                    extracted_prediction=result.metadata.get("extracted_prediction"),
                    extracted_gold=result.metadata.get("extracted_gold") or (str(task.answer) if task.answer is not None else None),
                    math_verify_available=bool(result.metadata.get("math_verify_available", False)),
                )
            )
            result.details["task_aware_policy"] = "math_verify_for_symbolic"
            return result

        result = self._numeric_strict.verify(task, solution)
        result.metadata.update(
            self._metadata(
                source=source or "unknown",
                mode="task_aware_strict" if self.require_math_verify_for_symbolic else "task_aware_math",
                policy="numeric_strict_for_gsm8k_or_non_symbolic",
                extracted_prediction=result.metadata.get("extracted_prediction"),
                extracted_gold=result.metadata.get("extracted_gold"),
                math_verify_available=_math_verify_available(self._math_verify),
            )
        )
        result.details["task_aware_policy"] = "numeric_strict_for_gsm8k_or_non_symbolic"
        return result

    def _metadata(
        self,
        *,
        source: str,
        mode: str,
        policy: str,
        extracted_prediction: object,
        extracted_gold: object,
        math_verify_available: bool,
    ) -> dict[str, object]:
        return {
            "verifier_mode": mode,
            "task_aware_policy": policy,
            "source": source or "unknown",
            "math_verify_available": bool(math_verify_available),
            "extracted_prediction": extracted_prediction,
            "extracted_gold": extracted_gold,
        }
