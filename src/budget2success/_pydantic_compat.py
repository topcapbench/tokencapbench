"""Small compatibility layer for Pydantic v1/v2.

The project targets Pydantic v2, but some lightweight CI images still carry
Pydantic v1. These shims keep the repository smoke tests usable in those
environments without changing the public record schemas.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any
import inspect

try:  # Pydantic v2 path.
    from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator  # type: ignore
except ImportError:  # pragma: no cover - exercised only in Pydantic v1 environments.
    from pydantic import BaseModel as _BaseModel  # type: ignore
    from pydantic import Field, root_validator, validator  # type: ignore

    class BaseModel(_BaseModel):  # type: ignore
        @classmethod
        def model_validate(cls, data: Any):
            return cls.parse_obj(data)

        def model_dump(self, mode: str | None = None, **kwargs: Any) -> dict[str, Any]:
            return self.dict(**{k: v for k, v in kwargs.items() if k in {"include", "exclude", "by_alias", "exclude_unset", "exclude_defaults", "exclude_none"}})

    def ConfigDict(**kwargs: Any) -> dict[str, Any]:
        return dict(kwargs)

    def _unwrap(func: Any):
        return func.__func__ if isinstance(func, classmethod) else func

    def field_validator(*fields: str, mode: str | None = None):
        pre = mode == "before"

        def decorate(func: Any):
            raw = _unwrap(func)
            params = list(inspect.signature(raw).parameters)
            wants_info = len(params) >= 3

            def _wrapped(cls, value, values=None, field=None, config=None):
                if wants_info:
                    info = SimpleNamespace(field_name=getattr(field, "name", fields[0] if fields else ""))
                    return raw(cls, value, info)
                return raw(cls, value)

            _wrapped.__name__ = getattr(raw, "__name__", "field_validator_wrapper")
            _wrapped.__qualname__ = getattr(raw, "__qualname__", _wrapped.__name__)
            return validator(*fields, pre=pre, allow_reuse=True)(_wrapped)

        return decorate

    def model_validator(mode: str = "after"):
        if mode != "after":
            raise NotImplementedError("Only after model validators are supported by the v1 compatibility shim.")

        def decorate(func: Any):
            raw = _unwrap(func)

            def _wrapped(cls, values):
                obj = cls.construct(**values)
                result = raw(obj)
                if result is None:
                    result = obj
                if hasattr(result, "dict"):
                    return result.dict()
                return dict(values)

            _wrapped.__name__ = getattr(raw, "__name__", "model_validator_wrapper")
            _wrapped.__qualname__ = getattr(raw, "__qualname__", _wrapped.__name__)
            return root_validator(allow_reuse=True, skip_on_failure=True)(_wrapped)

        return decorate
