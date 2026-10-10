"""YAML -> PipelineConfig loader with strict validation."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import yaml

from .schema import PipelineConfig

_NONE_TYPE = type(None)


class ConfigError(Exception):
    """Raised when a config file does not match the expected schema."""


def _unwrap_optional(tp: Any) -> tuple[Any, bool]:
    """Return (inner_type, is_optional) for an `X | None` annotation."""
    if get_origin(tp) is not None and _NONE_TYPE in get_args(tp):
        args = [a for a in get_args(tp) if a is not _NONE_TYPE]
        if len(args) == 1:
            return args[0], True
    return tp, False


def _coerce_scalar(value: Any, tp: Any, path: str) -> Any:
    if tp is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if tp is int and isinstance(value, bool):
        raise ConfigError(f"{path}: expected int, got bool")
    if tp in (int, float, str, bool) and not isinstance(value, tp):
        raise ConfigError(f"{path}: expected {tp.__name__}, got {type(value).__name__}")
    return value


def _build(cls: type, data: Any, path: str) -> Any:
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping, got {type(data).__name__}")

    hints = get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls)}

    unknown = set(data) - set(fields)
    if unknown:
        bad = sorted(unknown)[0]
        bad_path = f"{path}.{bad}" if path else bad
        raise ConfigError(f"{bad_path}: unknown key")

    kwargs: dict[str, Any] = {}
    for name, field in fields.items():
        field_path = f"{path}.{name}" if path else name
        tp = hints[name]
        inner_tp, is_optional = _unwrap_optional(tp)
        has_default = field.default is not dataclasses.MISSING or field.default_factory is not dataclasses.MISSING

        if name not in data:
            if is_optional or has_default:
                continue
            raise ConfigError(f"{field_path}: missing required key")

        raw = data[name]

        if raw is None:
            if is_optional:
                kwargs[name] = None
                continue
            raise ConfigError(f"{field_path}: expected {inner_tp}, got None")

        if dataclasses.is_dataclass(inner_tp):
            kwargs[name] = _build(inner_tp, raw, field_path)
        elif get_origin(inner_tp) is tuple:
            if not isinstance(raw, (list, tuple)):
                raise ConfigError(f"{field_path}: expected a list, got {type(raw).__name__}")
            elem_types = get_args(inner_tp)
            elem_tp = elem_types[0] if elem_types else Any
            kwargs[name] = tuple(
                _coerce_scalar(v, elem_tp, f"{field_path}[{i}]") if elem_tp is not Any else v for i, v in enumerate(raw)
            )
        else:
            kwargs[name] = _coerce_scalar(raw, inner_tp, field_path)

    try:
        return cls(**kwargs)
    except ValueError as exc:
        raise ConfigError(f"{path}: {exc}") from exc


def load_config(path: str | Path) -> PipelineConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raw = {}
    return _build(PipelineConfig, raw, "")
