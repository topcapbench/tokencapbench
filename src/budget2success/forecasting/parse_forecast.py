from __future__ import annotations

import json
import re
from typing import Any

from budget2success.schemas.records import ForecastRecord

_FENCED_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first JSON object from a model response."""
    stripped = text.strip()

    errors: list[str] = []
    for block in _FENCED_BLOCK_RE.findall(stripped):
        try:
            return _raw_decode_first_object(block)
        except json.JSONDecodeError as exc:
            errors.append(str(exc))

    try:
        return _raw_decode_first_object(stripped)
    except json.JSONDecodeError as exc:
        errors.append(str(exc))

    detail = f": {'; '.join(errors[:3])}" if errors else ""
    raise ValueError(f"No JSON object found in forecast response{detail}")


def _raw_decode_first_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            obj, _end = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise json.JSONDecodeError("No JSON object found", text, 0)


def validate_forecast_budget_grid(record: ForecastRecord, budget_grid: list[int]) -> None:
    """Require one forecast probability for every requested budget."""
    expected = {int(b) for b in budget_grid}
    observed = {int(b) for b in (record.p_success_by_budget or {})}
    if expected != observed:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"Forecast budget keys do not match requested grid; missing={missing}, extra={extra}")


def _coerce_probability(value: Any) -> float:
    if isinstance(value, str):
        text = value.strip()
        if text.endswith("%"):
            return float(text[:-1].strip()) / 100.0
        return float(text)
    return float(value)


def _clip_probability_maps(data: dict[str, Any]) -> dict[str, Any]:
    """Clip model-produced probabilities at parse time and record repairs.

    Direct schema construction still rejects out-of-range probabilities. The
    parser is intentionally more forgiving because malformed model JSON should
    be repairable and auditable rather than silently discarded.
    """
    repairs: list[dict[str, Any]] = []
    for field in ("p_success_by_budget", "forecast"):
        raw_map = data.get(field)
        if not isinstance(raw_map, dict):
            continue
        repaired: dict[str, Any] = {}
        for budget, raw_probability in raw_map.items():
            probability = _coerce_probability(raw_probability)
            clipped = min(1.0, max(0.0, probability))
            if clipped != probability:
                repairs.append(
                    {
                        "field": field,
                        "budget": str(budget),
                        "raw_probability": probability,
                        "clipped_probability": clipped,
                    }
                )
            repaired[str(budget)] = clipped
        data[field] = repaired
    if repairs:
        existing = data.get("forecast_extras")
        merged = dict(existing) if isinstance(existing, dict) else {}
        merged.setdefault("probability_repairs", []).extend(repairs)
        data["forecast_extras"] = merged
    return data


def parse_forecast_json(text: str) -> ForecastRecord:
    data = _clip_probability_maps(extract_json_object(text))
    if "median_success_budget" in data and "median_budget2success" not in data:
        data["median_budget2success"] = data["median_success_budget"]
    fields = getattr(ForecastRecord, "model_fields", None)
    if fields is None:
        fields = getattr(ForecastRecord, "__fields__", {})
    known_fields = set(fields)
    extras = {key: value for key, value in data.items() if key not in known_fields}
    if extras:
        existing = data.get("forecast_extras")
        merged = dict(existing) if isinstance(existing, dict) else {}
        merged.update(extras)
        data = {key: value for key, value in data.items() if key in known_fields}
        data["forecast_extras"] = merged
    record = ForecastRecord.model_validate(data)
    record.raw_text = text
    return record
