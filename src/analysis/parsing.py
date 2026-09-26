"""Parse CSV fields written by the current linear-model runners."""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping
from typing import Any


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _parse_scalar(value: str) -> Any:
    value = re.sub(r"np\.(?:float|int)\d*\((.*?)\)", r"\1", value).strip()
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value.strip("'\"")


def _extract_method_option(value: str, name: str) -> Any | None:
    match = re.search(rf"\b{re.escape(name)}=([^,)]*)", value)
    return None if match is None else _parse_scalar(match.group(1))


def parse_config_string(value: Any) -> dict[str, Any]:
    """Parse one current runner ``args`` value without evaluating callables."""
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str) or _is_missing(value):
        return {}

    parsed = {
        key: _parse_scalar(item)
        for key, item in re.findall(r"'([^']+)':\s*([^,}]+)", value)
    }
    method_match = re.search(
        r"'method':\s*(.*?)(?=,\s*'[^']+':|$|})", value, re.DOTALL
    )
    if method_match:
        method_value = method_match.group(1)
        class_match = re.search(r"<class '([^']+)'>", method_value)
        if class_match:
            parsed["method"] = class_match.group(1).split(".")[-1]
        for name in (
            "approximation",
            "permutation_type",
            "test_method",
            "asymptotic_null",
            "gamma",
        ):
            option = _extract_method_option(method_value, name)
            if option is not None:
                parsed[name] = option

    solver_match = re.search(
        r"'solver':\s*(?:functools\.partial\()?<function ([^ ]+)", value
    )
    if solver_match:
        parsed["solver"] = solver_match.group(1)
    return parsed


def parse_result_string(value: Any) -> dict[str, Any]:
    """Parse one current runner metric dictionary."""
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str) or _is_missing(value):
        return {}
    try:
        parsed = ast.literal_eval(
            re.sub(r"np\.(?:float|int)\d*\((.*?)\)", r"\1", value)
        )
    except (ValueError, SyntaxError):
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}
