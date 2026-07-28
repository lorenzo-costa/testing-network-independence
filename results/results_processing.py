"""Reusable processing utilities for simulation result dataframes.

The public entry point is :func:`process_results`.  Processing is split into
small steps so that fields or transformations can be added without changing
the rest of the pipeline.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd


NA_VALUE = "NA"
_NOT_FOUND = object()


METHOD_LABELS = {
    "DistanceCorrelationTest": "DC",
    "RVTest": "RV",
    "ObservedCVM": "CVM",
    "CanonicalCorrelationTest": "CCA",
    "MultivariateACTest_0.5": "AC_sqrt",
    "MultivariateACTest_third": "AC_third",
    "MultivariateACTest_sqrt_avg": "AC_sqrt_avg",
    "MultivariateACTest_sqrt_max": "AC_sqrt_max",
    "MultivariateACTest_third_avg": "AC_third_avg",
    "MultivariateACTest_third_max": "AC_third_max",
    "MultivariateACTest_quarter": "AC_quarter",
    "MultivariateACTest_quarter_max": "AC_quarter_max",
    "MultivariateACTest_1.0": "AC_1",
    "MultivariateACTest_True": "AC_adaptive",
}

LATENT_SIM_LABELS = {
    "multiplicative_noise": "mult_noise",
    "uncorrelated_bernoulli": "bernoulli",
    "joint_normal": "normal",
    "logarithmic": "log",
}

MARGINAL_LABELS = {
    "chi 5": "chi df=5",
    "t 5": "t df=5",
    "uniform -1 1": "unif(-1, 1)",
}


def _is_missing_scalar(value: Any) -> bool:
    """Return whether *value* is a scalar missing value."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip() or value.strip() == NA_VALUE

    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False

    return bool(missing) if np.isscalar(missing) else False


def parse_matrix(value: Any) -> np.ndarray | None:
    """Convert a dataframe cell into a two-dimensional NumPy array."""
    if _is_missing_scalar(value):
        return None

    if isinstance(value, str):
        try:
            value = ast.literal_eval(value.strip())
        except (ValueError, SyntaxError):
            return None

    try:
        matrix = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None

    if matrix.ndim != 2 or 0 in matrix.shape:
        return None

    return matrix


def classify_covariance(
    value: Any,
    atol: float = 1e-8,
    rtol: float = 1e-5,
) -> str:
    """Classify a covariance matrix by its structure."""
    matrix = parse_matrix(value)
    if matrix is None:
        return "invalid"

    n_rows, n_cols = matrix.shape
    if n_rows != n_cols or not np.all(np.isfinite(matrix)):
        return "invalid"

    if not np.allclose(matrix, matrix.T, atol=atol, rtol=rtol):
        return "other"

    if np.allclose(matrix, np.eye(n_rows), atol=atol, rtol=rtol):
        return "identity"

    diagonal = np.diag(matrix)
    off_diagonal = matrix[~np.eye(n_rows, dtype=bool)]
    equal_diagonal = np.allclose(
        diagonal,
        diagonal[0],
        atol=atol,
        rtol=rtol,
    )
    equal_off_diagonal = n_rows <= 1 or np.allclose(
        off_diagonal,
        off_diagonal[0],
        atol=atol,
        rtol=rtol,
    )

    if equal_diagonal and equal_off_diagonal:
        return "equicorrelated"

    if np.allclose(
        matrix - np.diag(diagonal),
        0,
        atol=atol,
        rtol=rtol,
    ):
        return "diagonal"

    return "other"


def covariance_condition_number(value: Any) -> float:
    """Return a square matrix's condition number, or ``np.nan`` if invalid."""
    matrix = parse_matrix(value)
    if matrix is None:
        return np.nan

    n_rows, n_cols = matrix.shape
    if n_rows != n_cols or not np.all(np.isfinite(matrix)):
        return np.nan

    try:
        return float(np.linalg.cond(matrix))
    except np.linalg.LinAlgError:
        return np.nan


def add_covariance_summary(
    df: pd.DataFrame,
    covariance_column: str,
    classification_column: str | None = None,
    condition_number_column: str | None = None,
) -> pd.DataFrame:
    """Add covariance classification and condition-number columns.

    If ``covariance_column`` does not exist, both output columns are filled
    with ``"NA"``.
    """
    result = df.copy()
    classification_column = (
        classification_column or f"{covariance_column}_type"
    )
    condition_number_column = (
        condition_number_column
        or f"{covariance_column}_condition_number"
    )

    if covariance_column not in result:
        result[classification_column] = NA_VALUE
        result[condition_number_column] = NA_VALUE
        return result

    available = ~result[covariance_column].map(_is_missing_scalar)
    result[classification_column] = NA_VALUE
    result[condition_number_column] = NA_VALUE
    result.loc[available, classification_column] = (
        result.loc[available, covariance_column].map(classify_covariance)
    )
    result.loc[available, condition_number_column] = (
        result.loc[available, covariance_column].map(
            covariance_condition_number
        )
    )
    return result


def parse_value(value: Any) -> Any:
    """Parse a Python literal, returning a cleaned string if parsing fails."""
    if not isinstance(value, str):
        return value

    value = value.strip()
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value.strip("'\"")


def extract_argument(text: Any, name: str) -> Any:
    """Extract a named ``key=value`` or ``'key': value`` argument.

    Nested lists, dictionaries, tuples, and quoted strings are respected.
    ``_NOT_FOUND`` is returned when the argument is absent.
    """
    if not isinstance(text, str):
        return _NOT_FOUND

    match = re.search(
        rf"(?:['\"]?{re.escape(name)}['\"]?)\s*(?:=|:)\s*",
        text,
    )
    if not match:
        return _NOT_FOUND

    start = match.end()
    index = start
    bracket_pairs = {"[": "]", "{": "}", "(": ")"}
    closing_brackets = set(bracket_pairs.values())
    stack: list[str] = []
    quote: str | None = None
    escaped = False

    while index < len(text):
        character = text[index]

        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
        elif character in {"'", '"'}:
            quote = character
        elif character in bracket_pairs:
            stack.append(bracket_pairs[character])
        elif character in closing_brackets:
            if stack and character == stack[-1]:
                stack.pop()
            elif not stack:
                break
        elif character == "," and not stack:
            break

        index += 1

    return parse_value(text[start:index])


def parse_config_string(value: Any) -> dict[str, Any]:
    """Parse a serialized simulation configuration into a dictionary."""
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str) or _is_missing_scalar(value):
        return {}

    pairs: dict[str, Any] = dict(
        re.findall(r"'([^']+)':\s*([^,}]+)", value)
    )
    method_match = re.search(
        r"'method':\s*(.*?)(?=,\s*'[^']+':|$|})",
        value,
        re.DOTALL,
    )
    method_class = None
    test_function = None

    if method_match:
        method_value = method_match.group(1)
        class_match = re.search(r"<class '([^']+)'>", method_value)
        if class_match:
            method_class = class_match.group(1).split(".")[-1]

        function_match = re.search(
            r"test_function=<function ([^ ]+)",
            method_value,
        )
        if function_match:
            test_function = function_match.group(1)

        method_arguments = {
            "NN_number": "M",
            "aggregate": "aggregate_coeff",
            "approximation": "approximation",
            "permutation_type": "permutation_type",
            "adaptive_m": "adaptive_m",
        }
        for output_name, argument_name in method_arguments.items():
            extracted = extract_argument(method_value, argument_name)
            if extracted is not _NOT_FOUND:
                pairs[output_name] = extracted

    if method_class == "PermutationTest" and test_function:
        pairs["method"] = f"{method_class}_{test_function}"
    elif method_class is not None:
        pairs["method"] = method_class

    optional_arguments = {
        "conditional_copula": "conditional_copula",
        "post_nonlinear_noise": "post_nonlinear_noise",
        "n_strata": "C",
        "stratum_covariance": "stratum_covariance",
        "copula_model": "copula_model",
        "latent_sim": "latent_sim",
    }
    for output_name, argument_name in optional_arguments.items():
        extracted = extract_argument(value, argument_name)
        if extracted is not _NOT_FOUND:
            pairs[output_name] = extracted

    column_covariance = extract_argument(value, "column_covariance")
    column_covariance_z = extract_argument(value, "column_covariance_z")
    if column_covariance is not _NOT_FOUND:
        pairs["column_covariance"] = column_covariance
        pairs["column_covariance_source"] = "column_covariance"
    elif column_covariance_z is not _NOT_FOUND:
        pairs["column_covariance"] = column_covariance_z
        pairs["column_covariance_source"] = "column_covariance_z"

    solver_match = re.search(r"'solver':\s*<function ([^ ]+)", value)
    if solver_match:
        pairs["solver"] = solver_match.group(1)

    for key, item in list(pairs.items()):
        wrapper_match = re.fullmatch(
            r"np\.\w+\(([^)]+)\)",
            str(item).strip(),
        )
        if wrapper_match:
            pairs[key] = parse_value(wrapper_match.group(1))

    degree_match = re.search(r"\bdegree\s*=\s*(\d+)", value)
    if degree_match:
        pairs["degree"] = int(degree_match.group(1))

    marginals_match = re.search(
        r"'marginals':\s*\{(.*?)\}",
        value,
        re.DOTALL,
    )
    if marginals_match:
        marginals = dict(
            re.findall(
                r"'([^']+)'\s*:\s*'([^']+)'",
                marginals_match.group(1),
            )
        )
        pairs["marginals"] = marginals
        for variable, distribution in marginals.items():
            pairs[f"marginal_{variable}"] = distribution

    return pairs


def parse_result_string(value: Any) -> dict[str, Any]:
    """Parse a serialized result dictionary, including NumPy scalars."""
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str) or _is_missing_scalar(value):
        return {}

    cleaned = re.sub(
        r"np\.(?:float|int)\d*\((.*?)\)",
        r"\1",
        value,
    )
    try:
        parsed = ast.literal_eval(cleaned)
    except (ValueError, SyntaxError):
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _clean_string(value: Any) -> Any:
    if _is_missing_scalar(value):
        return NA_VALUE
    return value.strip("'\"") if isinstance(value, str) else value


def _to_int_or_na(value: Any) -> int | str:
    if _is_missing_scalar(value):
        return NA_VALUE
    cleaned = re.sub(r"np\.int\d*\((.*?)\)", r"\1", str(value))
    try:
        return int(cleaned)
    except (TypeError, ValueError):
        return NA_VALUE


def _to_float_or_na(value: Any) -> float | str:
    if _is_missing_scalar(value):
        return NA_VALUE
    cleaned = re.sub(r"np\.float\d*\((.*?)\)", r"\1", str(value))
    try:
        return float(cleaned)
    except (TypeError, ValueError):
        return NA_VALUE


def _mapping_value(
    value: Any,
    key: str,
    converter: Callable[[Any], Any] | None = None,
) -> Any:
    if not isinstance(value, Mapping):
        return NA_VALUE
    item = value.get(key, NA_VALUE)
    return converter(item) if converter is not None else item


def _parse_source_columns(results: pd.DataFrame) -> pd.DataFrame:
    """Parse serialized input columns without requiring either one."""
    result = results.copy()
    if "args" in result:
        result["args"] = result["args"].map(parse_config_string)
    else:
        result["args"] = NA_VALUE

    if "ComputeAll" in result:
        result["ComputeAll"] = result["ComputeAll"].map(parse_result_string)
    else:
        result["ComputeAll"] = NA_VALUE

    return result


def _parse_density(value: Any) -> list[Any]:
    if isinstance(value, str):
        extracted = re.findall(r"np\.float64\((.*?)\)", value)
        return extracted
    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)
    return []


def _add_density_columns(results: pd.DataFrame) -> pd.DataFrame:
    result = results.copy()
    if "density" not in result:
        result["density"] = NA_VALUE
        result["density_A"] = NA_VALUE
        result["density_B"] = NA_VALUE
        return result

    result["density"] = result["density"].map(_parse_density)
    result["density_A"] = result["density"].map(
        lambda values: values[0] if len(values) > 0 else NA_VALUE
    )
    result["density_B"] = result["density"].map(
        lambda values: values[1] if len(values) > 1 else NA_VALUE
    )
    return result


def _add_config_columns(results: pd.DataFrame) -> pd.DataFrame:
    """Expand selected configuration fields from the parsed ``args`` column."""
    result = results.copy()
    field_specs: dict[str, tuple[str, Callable[[Any], Any] | None]] = {
        "edge_var": ("edge_var", None),
        "n": ("n", _to_int_or_na),
        "k": ("k", _to_int_or_na),
        "method": ("method", _clean_string),
        "solver": ("solver", _clean_string),
        "make_sparse": ("make_sparse", None),
        "use_true_x": ("use_true_latent_x", None),
        "use_true_z": ("use_true_latent_z", None),
        "latent_sim": ("latent_sim", _clean_string),
        "aggregate": ("aggregate", None),
        "copula": ("copula_model", None),
        "dgp": ("dgp_name", None),
        "marginal_y": ("marginal_y", _clean_string),
        "marginal_z": ("marginal_z", _clean_string),
        "rho": ("rho", _to_float_or_na),
        "degree": ("degree", None),
        "NN_number": ("NN_number", None),
        "functional_form": ("functional_form", _clean_string),
        "sbm_covariate_sampling": (
            "sbm_covariate_sampling",
            _clean_string,
        ),
        "assortativity": ("assortativity", None),
        "x_distribution": ("x_distribution", _clean_string),
        "approximation": ("approximation", _clean_string),
        "adaptive_m": ("adaptive_m", None),
        "column_covariance": ("column_covariance", None),
    }

    for output_column, (key, converter) in field_specs.items():
        result[output_column] = result["args"].map(
            lambda value, key=key, converter=converter: _mapping_value(
                value,
                key,
                converter,
            )
        )

    result["dgp_name"] = result["dgp"].map(
        lambda value: (
            _clean_string(value).split("_")[0]
            if not _is_missing_scalar(value)
            else NA_VALUE
        )
    )
    return result


def _add_result_columns(results: pd.DataFrame) -> pd.DataFrame:
    """Expand metrics from the parsed ``ComputeAll`` column."""
    result = results.copy()
    metric_fields = {
        "RelativeFrobeniusNorm_x": "RelativeFrobeniusNorm_x",
        "RelativeFrobeniusNorm_z": "RelativeFrobeniusNorm_z",
        "ProcrustesDistance_x": "ProcrustesDistance_x",
        "ProcrustesDistance_z": "ProcrustesDistance_z",
        "FalseRejection": "FalseRejection",
        "Power": "TrueRejection",
        "Rejection": "Rejection",
    }
    for output_column, key in metric_fields.items():
        result[output_column] = result["ComputeAll"].map(
            lambda value, key=key: _mapping_value(value, key)
        )
    return result


def _add_group_averages(results: pd.DataFrame) -> pd.DataFrame:
    result = results.copy()
    group_columns = ["n", "method", "marginals", "copula", "dgp_name"]
    average_specs = {
        "avg_rel_frob_z": "RelativeFrobeniusNorm_z",
        "avg_proc_dist_z": "ProcrustesDistance_z",
    }
    missing_group_columns = [
        column for column in group_columns if column not in result
    ]
    if missing_group_columns:
        for column in missing_group_columns:
            result[column] = NA_VALUE
        for output_column in average_specs:
            result[output_column] = NA_VALUE
        return result

    for output_column, value_column in average_specs.items():
        numeric_values = pd.to_numeric(result[value_column], errors="coerce")
        if numeric_values.notna().any():
            result[output_column] = numeric_values.groupby(
                [result[column] for column in group_columns],
                dropna=False,
            ).transform("mean")
        else:
            result[output_column] = NA_VALUE

    return result


def _add_permutation_type(
    results: pd.DataFrame,
    compute_all_was_present: bool,
) -> pd.DataFrame:
    result = results.copy()
    if not compute_all_was_present:
        result["permutation_type"] = NA_VALUE
        return result

    numeric_values = pd.to_numeric(
        result["RelativeFrobeniusNorm_z"],
        errors="coerce",
    )
    result["permutation_type"] = np.where(
        numeric_values.isna(),
        "observed",
        "latent",
    )
    return result


def _normalize_labels(results: pd.DataFrame) -> pd.DataFrame:
    result = results.copy()
    for column in ("marginal_y", "marginal_z"):
        result[column] = result[column].replace(MARGINAL_LABELS)
    result["copula"] = result["copula"].replace(
        {"mixture_uniform": "mixture"}
    )
    return result


def _append_method_variant(
    results: pd.DataFrame,
    variant_column: str,
) -> pd.DataFrame:
    result = results.copy()
    available = (
        ~result[variant_column].map(_is_missing_scalar)
        & ~result["method"].map(_is_missing_scalar)
    )
    if available.any():
        variants = result.loc[available, variant_column].map(_clean_string)
        result.loc[available, "method"] = (
            result.loc[available, "method"].map(_clean_string).astype(str)
            + "_"
            + variants.astype(str)
        )
    return result


def _augment_and_rename_methods(results: pd.DataFrame) -> pd.DataFrame:
    result = results.copy()
    for variant_column in ("NN_number", "approximation", "adaptive_m"):
        result = _append_method_variant(result, variant_column)
    result["method"] = result["method"].replace(METHOD_LABELS)
    result["latent_sim"] = result["latent_sim"].replace(LATENT_SIM_LABELS)
    return result


def _fill_missing_values(results: pd.DataFrame) -> pd.DataFrame:
    """Replace every scalar pandas/NumPy missing value with ``"NA"``."""
    result = results.copy()
    for column in result.columns:
        missing = result[column].map(_is_missing_scalar)
        if missing.any():
            result.loc[missing, column] = NA_VALUE
    return result


def process_results(results_concat: pd.DataFrame) -> pd.DataFrame:
    """Run the complete result-processing pipeline and return a new dataframe.

    Missing input columns do not raise errors.  Their corresponding source and
    derived columns are created and filled with ``"NA"``.
    """
    if not isinstance(results_concat, pd.DataFrame):
        raise TypeError("results_concat must be a pandas DataFrame")

    compute_all_was_present = "ComputeAll" in results_concat
    results = _parse_source_columns(results_concat)
    results = _add_density_columns(results)
    results = _add_config_columns(results)
    results = _add_result_columns(results)
    results = _add_group_averages(results)
    results = _add_permutation_type(results, compute_all_was_present)
    results = _normalize_labels(results)
    results = _augment_and_rename_methods(results)
    results = add_covariance_summary(
        results,
        covariance_column="column_covariance",
    )
    return _fill_missing_values(results)


__all__ = [
    "NA_VALUE",
    "add_covariance_summary",
    "classify_covariance",
    "covariance_condition_number",
    "extract_argument",
    "parse_config_string",
    "parse_matrix",
    "parse_result_string",
    "parse_value",
    "process_results",
]
