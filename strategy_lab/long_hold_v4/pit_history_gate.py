"""Strict qualification gate for historical PIT inputs used by Long Hold V4.

This module validates data eligibility only. Passing this gate does not run a
backtest and never promotes a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .core import ContractError


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "long_hold_v4_pit_gate.json"
CHECK_COLUMNS = ["dataset_id", "sleeve", "check", "status", "detail"]
SUMMARY_COLUMNS = [
    "dataset_id",
    "sleeve",
    "priority",
    "path",
    "status",
    "rows",
    "assets",
    "coverage_start",
    "coverage_end",
    "failed_checks",
]
QUEUE_COLUMNS = ["priority", "dataset_id", "sleeve", "target_path", "status", "failed_checks", "provider_options", "manual_action"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _value_set_sha256(values: set[str]) -> str:
    normalized = sorted(str(value).strip() for value in values if str(value).strip())
    return hashlib.sha256("\n".join(normalized).encode("utf-8")).hexdigest()


def _safe_path(root: Path, relative: str) -> Path:
    root_resolved = root.resolve()
    path = (root / relative).resolve()
    if path != root_resolved and root_resolved not in path.parents:
        raise ContractError(f"PIT dataset path escapes project root: {relative}")
    return path


def load_gate_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"model", "output_directory", "datasets"}
    missing = sorted(required.difference(config))
    if missing:
        raise ContractError(f"PIT gate config missing fields: {missing}")
    if not isinstance(config["datasets"], list) or not config["datasets"]:
        raise ContractError("PIT gate datasets must be a non-empty list")
    ids = [str(item.get("dataset_id", "")) for item in config["datasets"]]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ContractError("PIT gate dataset_id values must be unique and non-empty")
    for item in config["datasets"]:
        dataset_required = {
            "dataset_id",
            "sleeve",
            "priority",
            "path",
            "required_columns",
            "primary_key",
            "date_columns",
            "available_date_column",
            "coverage_column",
            "minimum_start_date",
            "minimum_end_date",
            "minimum_rows",
            "minimum_assets",
            "provider_options",
        }
        item_missing = sorted(dataset_required.difference(item))
        if item_missing:
            raise ContractError(f"PIT dataset {item.get('dataset_id')} missing fields: {item_missing}")
        if item["sleeve"] not in {"stock", "etf", "macro", "shared"}:
            raise ContractError(f"unsupported PIT sleeve: {item['sleeve']}")
        if int(item["minimum_rows"]) < 1 or int(item["minimum_assets"]) < 0:
            raise ContractError(f"invalid minimum rows/assets: {item['dataset_id']}")
        if item.get("require_external_evidence") is True:
            evidence_missing = [
                field
                for field in ("lineage_manifest", "validation_manifest")
                if not str(item.get(field, "")).strip()
            ]
            if evidence_missing:
                raise ContractError(
                    f"PIT dataset {item['dataset_id']} requires external evidence fields: {evidence_missing}"
                )
        vintage_mode = str(item.get("lineage_source_vintage_mode", "single"))
        if vintage_mode not in {"single", "set_sha256"}:
            raise ContractError(
                f"unsupported lineage source-vintage mode for {item['dataset_id']}: {vintage_mode}"
            )
        if item.get("validation_require_dataset_input_match", True) not in {True, False}:
            raise ContractError(
                f"validation_require_dataset_input_match must be boolean: {item['dataset_id']}"
            )
        maximum_lag = item.get("maximum_availability_lag_days")
        lag_reference = str(item.get("availability_lag_reference_column", "")).strip()
        if maximum_lag is not None:
            if int(maximum_lag) < 0 or not lag_reference:
                raise ContractError(f"invalid availability lag contract: {item['dataset_id']}")
            if lag_reference not in set(item["required_columns"]):
                raise ContractError(
                    f"PIT dataset {item['dataset_id']} availability lag reference is not required: {lag_reference}"
                )
    return config


def _check(rows: list[dict[str, str]], rule: dict[str, Any], name: str, passed: bool, detail: str) -> None:
    rows.append(
        {
            "dataset_id": str(rule["dataset_id"]),
            "sleeve": str(rule["sleeve"]),
            "check": name,
            "status": "pass" if passed else "fail",
            "detail": detail,
        }
    )


def _parse_dates(frame: pd.DataFrame, columns: list[str]) -> tuple[dict[str, pd.Series], dict[str, int]]:
    parsed: dict[str, pd.Series] = {}
    bad: dict[str, int] = {}
    for column in columns:
        raw = frame[column]
        values = pd.to_datetime(raw, errors="coerce")
        nonblank = raw.notna() & raw.astype(str).str.strip().ne("")
        parsed[column] = values
        bad[column] = int((nonblank & values.isna()).sum())
    return parsed, bad


def _overlap_count(frame: pd.DataFrame, rule: dict[str, Any], parsed: dict[str, pd.Series]) -> int:
    interval = rule.get("interval")
    if not interval:
        return 0
    groups = list(interval["group_columns"])
    start_col = str(interval["start_column"])
    end_col = str(interval["end_column"])
    work = frame[groups].copy()
    work["_start"] = parsed[start_col]
    work["_end"] = parsed[end_col].fillna(pd.Timestamp.max.normalize())
    work = work.dropna(subset=["_start"]).sort_values(groups + ["_start", "_end"])
    overlaps = 0
    for _, group in work.groupby(groups, dropna=False, sort=False):
        previous_end: pd.Timestamp | None = None
        for start_value, end_value in group[["_start", "_end"]].itertuples(index=False, name=None):
            start = pd.Timestamp(start_value)
            end = pd.Timestamp(end_value)
            if end <= start:
                overlaps += 1
            if previous_end is not None and start < previous_end:
                overlaps += 1
            previous_end = max(previous_end, end) if previous_end is not None else end
    return overlaps


def _validate_external_evidence(
    checks: list[dict[str, str]],
    root: Path,
    rule: dict[str, Any],
    path: Path,
    *,
    dataset_sha256: str,
    source_vintages: set[str],
) -> list[Path]:
    """Validate lineage and independent-validation manifests without loading the dataset."""
    lineage_paths = [path]
    lineage_relative = rule.get("lineage_manifest")
    if lineage_relative:
        lineage_path = _safe_path(root, str(lineage_relative))
        lineage_exists = lineage_path.is_file()
        _check(checks, rule, "lineage_manifest_exists", lineage_exists, str(lineage_relative))
        if lineage_exists:
            lineage_paths.append(lineage_path)
            try:
                lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                _check(checks, rule, "lineage_manifest_readable", False, repr(exc))
            else:
                _check(checks, rule, "lineage_manifest_readable", True, str(lineage_relative))
                required_output_role = rule.get("lineage_required_output_role")
                if required_output_role is None:
                    output_path = str(lineage.get("output_path", "")).replace("\\", "/")
                    output_hash = str(lineage.get("output_sha256", ""))
                else:
                    output_items = lineage.get("outputs")
                    output_items = output_items if isinstance(output_items, list) else []
                    roles = [str(item.get("role", "")) for item in output_items if isinstance(item, dict)]
                    _check(
                        checks,
                        rule,
                        "lineage_output_roles_unique",
                        len(roles) == len(set(roles)),
                        f"roles={roles}",
                    )
                    role_item = next(
                        (
                            item
                            for item in output_items
                            if isinstance(item, dict) and str(item.get("role", "")) == str(required_output_role)
                        ),
                        {},
                    )
                    output_path = str(role_item.get("path", "")).replace("\\", "/")
                    output_hash = str(role_item.get("sha256", ""))
                output_ok = output_path == str(rule["path"]).replace("\\", "/") and output_hash == dataset_sha256
                _check(checks, rule, "lineage_output_hash_match", output_ok, f"output_path={output_path}")

                code_relative = str(lineage.get("code_path", ""))
                try:
                    code_path = _safe_path(root, code_relative) if code_relative else None
                except ValueError:
                    code_path = None
                code_ok = bool(
                    code_path
                    and code_path.is_file()
                    and _sha256(code_path) == str(lineage.get("code_sha256", ""))
                )
                _check(checks, rule, "lineage_code_hash_match", code_ok, code_relative)
                if code_path is not None and code_path.is_file():
                    lineage_paths.append(code_path)

                inputs = lineage.get("inputs")
                if not isinstance(inputs, list):
                    input_relative = lineage.get("input_path")
                    input_hash = lineage.get("input_sha256")
                    inputs = (
                        [{"path": input_relative, "sha256": input_hash}]
                        if input_relative and input_hash
                        else []
                    )
                input_failures: list[str] = []
                for item in inputs:
                    if not isinstance(item, dict):
                        input_failures.append("invalid_manifest_item")
                        continue
                    input_relative = str(item.get("path", ""))
                    try:
                        input_path = _safe_path(root, input_relative) if input_relative else None
                    except ValueError:
                        input_path = None
                    expected_hash = str(item.get("sha256", ""))
                    if (
                        not input_path
                        or not input_path.is_file()
                        or not expected_hash
                        or _sha256(input_path) != expected_hash
                    ):
                        input_failures.append(input_relative or "missing_path")
                        continue
                    lineage_paths.append(input_path)
                _check(
                    checks,
                    rule,
                    "lineage_inputs_hash_match",
                    bool(inputs) and not input_failures,
                    f"inputs={len(inputs)};failures={input_failures[:10]}",
                )

                vintage_mode = str(rule.get("lineage_source_vintage_mode", "single"))
                if vintage_mode == "set_sha256":
                    expected_vintage_hash = str(lineage.get("source_vintage_set_sha256", ""))
                    expected_vintage_count = lineage.get("source_vintage_count")
                    actual_vintage_hash = _value_set_sha256(source_vintages)
                    vintage_ok = bool(
                        expected_vintage_hash
                        and expected_vintage_hash == actual_vintage_hash
                        and expected_vintage_count == len(source_vintages)
                    )
                    vintage_detail = (
                        f"mode=set_sha256;count={len(source_vintages)};"
                        f"expected_count={expected_vintage_count};actual_hash={actual_vintage_hash}"
                    )
                else:
                    expected_vintage = str(lineage.get("source_vintage", ""))
                    if not expected_vintage and lineage.get("input_sha256"):
                        expected_vintage = f"macro_pit_panel_sha256:{lineage['input_sha256']}"
                    vintage_ok = bool(expected_vintage and source_vintages == {expected_vintage})
                    vintage_detail = expected_vintage
                _check(checks, rule, "lineage_source_vintage_match", vintage_ok, vintage_detail)
                _check(
                    checks,
                    rule,
                    "lineage_historical_use_approved",
                    lineage.get("historical_backtest_allowed") is True,
                    f"historical_backtest_allowed={lineage.get('historical_backtest_allowed')!r}",
                )
                for field, expected in rule.get("lineage_required_values", {}).items():
                    actual = lineage.get(field)
                    _check(
                        checks,
                        rule,
                        f"lineage_required_value:{field}",
                        actual == expected,
                        f"expected={expected!r};actual={actual!r}",
                    )

    validation_relative = rule.get("validation_manifest")
    if validation_relative:
        validation_path = _safe_path(root, str(validation_relative))
        validation_exists = validation_path.is_file()
        _check(checks, rule, "validation_manifest_exists", validation_exists, str(validation_relative))
        if validation_exists:
            lineage_paths.append(validation_path)
            try:
                validation = json.loads(validation_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                _check(checks, rule, "validation_manifest_readable", False, repr(exc))
            else:
                _check(checks, rule, "validation_manifest_readable", True, str(validation_relative))
                _check(
                    checks,
                    rule,
                    "validation_qualification_passed",
                    validation.get("qualification_status") == "PASS",
                    f"qualification_status={validation.get('qualification_status')!r}",
                )
                _check(
                    checks,
                    rule,
                    "validation_historical_use_approved",
                    validation.get("historical_backtest_allowed") is True,
                    f"historical_backtest_allowed={validation.get('historical_backtest_allowed')!r}",
                )
                _check(
                    checks,
                    rule,
                    "validation_cannot_promote_model",
                    validation.get("model_promotion_allowed") is False,
                    f"model_promotion_allowed={validation.get('model_promotion_allowed')!r}",
                )
                for field, expected in rule.get("validation_required_values", {}).items():
                    actual = validation.get(field)
                    _check(
                        checks,
                        rule,
                        f"validation_required_value:{field}",
                        actual == expected,
                        f"expected={expected!r};actual={actual!r}",
                    )
                expected_schema = rule.get("validation_schema")
                if expected_schema is not None:
                    _check(
                        checks,
                        rule,
                        "validation_schema_match",
                        validation.get("validation_schema") == expected_schema,
                        f"expected={expected_schema!r};actual={validation.get('validation_schema')!r}",
                    )

                zero_count_fields = rule.get("validation_required_zero_counts")
                if zero_count_fields is None:
                    zero_count_fields = ["continuous_trading_large_jump_count"]
                for field in zero_count_fields:
                    _check(
                        checks,
                        rule,
                        f"validation_zero_count:{field}",
                        validation.get(field) == 0,
                        f"count={validation.get(field)!r}",
                    )

                allowed_validation_values = rule.get("validation_allowed_values")
                if allowed_validation_values is None:
                    allowed_validation_values = {
                        "review_status": ["CLEAR", "LONG_SUSPENSION_REVIEW_REQUIRED"]
                    }
                for field, allowed in allowed_validation_values.items():
                    actual = validation.get(field)
                    _check(
                        checks,
                        rule,
                        f"validation_allowed_value:{field}",
                        actual in set(allowed),
                        f"value={actual!r};allowed={allowed!r}",
                    )

                code_relative = str(validation.get("code_path", ""))
                try:
                    validation_code = _safe_path(root, code_relative) if code_relative else None
                except ValueError:
                    validation_code = None
                code_ok = bool(
                    validation_code
                    and validation_code.is_file()
                    and _sha256(validation_code) == str(validation.get("code_sha256", ""))
                )
                _check(checks, rule, "validation_code_hash_match", code_ok, code_relative)
                if validation_code is not None and validation_code.is_file():
                    lineage_paths.append(validation_code)

                validation_inputs = validation.get("inputs")
                input_failures: list[str] = []
                input_paths: list[str] = []
                dataset_input_match = False
                if not isinstance(validation_inputs, list):
                    validation_inputs = []
                    input_failures.append("inputs_not_a_list")
                for item in validation_inputs:
                    if not isinstance(item, dict):
                        input_failures.append("invalid_manifest_item")
                        continue
                    input_relative = str(item.get("path", ""))
                    normalized_relative = input_relative.replace("\\", "/")
                    input_paths.append(normalized_relative)
                    try:
                        input_path = _safe_path(root, input_relative) if input_relative else None
                    except ValueError:
                        input_path = None
                    expected_hash = str(item.get("sha256", ""))
                    if (
                        not input_path
                        or not input_path.is_file()
                        or not expected_hash
                        or _sha256(input_path) != expected_hash
                    ):
                        input_failures.append(input_relative or "missing_path")
                        continue
                    if normalized_relative == str(rule["path"]).replace("\\", "/"):
                        dataset_input_match = expected_hash == dataset_sha256
                    lineage_paths.append(input_path)
                minimum_inputs = int(rule.get("validation_minimum_input_count", 1))
                duplicate_inputs = len(input_paths) - len(set(input_paths))
                _check(
                    checks,
                    rule,
                    "validation_inputs_hash_match",
                    len(validation_inputs) >= minimum_inputs
                    and not input_failures
                    and duplicate_inputs == 0,
                    (
                        f"inputs={len(validation_inputs)};min={minimum_inputs};"
                        f"duplicates={duplicate_inputs};failures={input_failures[:10]}"
                    ),
                )
                require_dataset_input_match = bool(rule.get("validation_require_dataset_input_match", True))
                _check(
                    checks,
                    rule,
                    "validation_dataset_input_match",
                    dataset_input_match or not require_dataset_input_match,
                    (
                        str(rule["path"])
                        if require_dataset_input_match
                        else "disabled_by_explicit_post_validation_promotion_contract"
                    ),
                )

                required_output_roles = rule.get("validation_required_output_roles")
                if required_output_roles is not None:
                    output_items = validation.get("outputs")
                    output_items = output_items if isinstance(output_items, list) else []
                    roles = [str(item.get("role", "")) for item in output_items if isinstance(item, dict)]
                    role_map = {
                        str(item.get("role", "")): item
                        for item in output_items
                        if isinstance(item, dict) and item.get("role")
                    }
                    _check(
                        checks,
                        rule,
                        "validation_output_roles_unique",
                        len(roles) == len(set(roles)),
                        f"roles={roles}",
                    )
                    for role in required_output_roles:
                        item = role_map.get(str(role), {})
                        output_relative = str(item.get("path", ""))
                        try:
                            validation_output = _safe_path(root, output_relative) if output_relative else None
                        except ValueError:
                            validation_output = None
                        output_ok = bool(
                            validation_output
                            and validation_output.is_file()
                            and _sha256(validation_output) == str(item.get("sha256", ""))
                        )
                        _check(
                            checks,
                            rule,
                            f"validation_output_hash_match:{role}",
                            output_ok,
                            output_relative,
                        )
                        if validation_output is not None and validation_output.is_file():
                            lineage_paths.append(validation_output)
                else:
                    for path_key, hash_key in (
                        ("output_path", "output_sha256"),
                        ("exceptions_path", "exceptions_sha256"),
                        ("long_gap_review_path", "long_gap_review_sha256"),
                        ("report_path", "report_sha256"),
                    ):
                        output_relative = str(validation.get(path_key, ""))
                        try:
                            validation_output = _safe_path(root, output_relative) if output_relative else None
                        except ValueError:
                            validation_output = None
                        output_ok = bool(
                            validation_output
                            and validation_output.is_file()
                            and _sha256(validation_output) == str(validation.get(hash_key, ""))
                        )
                        _check(
                            checks,
                            rule,
                            f"validation_output_hash_match:{path_key}",
                            output_ok,
                            output_relative,
                        )
                        if validation_output is not None and validation_output.is_file():
                            lineage_paths.append(validation_output)

                exceptions_relative = str(validation.get("exceptions_path", ""))
                if not exceptions_relative:
                    for item in validation.get("outputs", []):
                        if isinstance(item, dict) and item.get("role") == "exceptions":
                            exceptions_relative = str(item.get("path", ""))
                            break
                try:
                    exceptions_path = _safe_path(root, exceptions_relative) if exceptions_relative else None
                    exceptions = (
                        pd.read_csv(exceptions_path)
                        if exceptions_path and exceptions_path.is_file()
                        else None
                    )
                except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError, ValueError):
                    exceptions = None
                _check(
                    checks,
                    rule,
                    "validation_exception_file_empty",
                    exceptions is not None and exceptions.empty,
                    f"rows={len(exceptions) if exceptions is not None else 'unreadable'}",
                )
    return lineage_paths


def _validate_dataset_streaming(
    root: Path,
    rule: dict[str, Any],
    as_of: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any], list[Path]]:
    """Validate a large, primary-key-sorted CSV with bounded memory."""
    checks: list[dict[str, str]] = []
    path = _safe_path(root, str(rule["path"]))
    exists = path.is_file()
    _check(checks, rule, "file_exists", exists, str(rule["path"]))
    empty_summary = {
        "dataset_id": rule["dataset_id"],
        "sleeve": rule["sleeve"],
        "priority": rule["priority"],
        "path": rule["path"],
        "status": "blocked",
        "rows": 0,
        "assets": 0,
        "coverage_start": "",
        "coverage_end": "",
        "failed_checks": "file_exists",
    }
    if not exists:
        return pd.DataFrame(checks, columns=CHECK_COLUMNS), empty_summary, []
    if rule.get("interval"):
        raise ContractError(f"streaming interval validation is unsupported: {rule['dataset_id']}")

    try:
        header = pd.read_csv(path, nrows=0)
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        _check(checks, rule, "csv_readable", False, repr(exc))
        empty_summary["failed_checks"] = "csv_readable"
        return pd.DataFrame(checks, columns=CHECK_COLUMNS), empty_summary, [path]
    required_columns = set(rule["required_columns"])
    missing_columns = sorted(required_columns.difference(header.columns))
    _check(checks, rule, "required_columns", not missing_columns, ",".join(missing_columns))
    if missing_columns:
        empty_summary["failed_checks"] = "required_columns"
        return pd.DataFrame(checks, columns=CHECK_COLUMNS), empty_summary, [path]

    primary_key = list(rule["primary_key"])
    date_columns = list(
        dict.fromkeys(
            [*rule["date_columns"], rule["available_date_column"], rule["coverage_column"]]
        )
    )
    lag_reference_column = str(rule.get("availability_lag_reference_column", "")).strip()
    if lag_reference_column:
        date_columns = list(dict.fromkeys([*date_columns, lag_reference_column]))
    asset_column = str(rule.get("asset_column", "asset"))
    chunksize = max(1, int(rule.get("streaming_chunksize", 250_000)))
    rows = 0
    assets_seen: set[str] = set()
    missing_key = 0
    duplicate_rows = 0
    order_violations = 0
    previous_last_key: tuple[str, ...] | None = None
    bad_dates = {column: 0 for column in date_columns}
    future_available = 0
    floor_bad = 0
    availability_lag_bad = 0
    coverage_start = pd.NaT
    coverage_end = pd.NaT
    allowed_invalid = {column: set() for column in rule.get("allowed_values", {})}
    finite_invalid = {column: 0 for column in rule.get("finite_numeric_columns", [])}
    conditional_finite_invalid: dict[tuple[str, str], int] = {}
    conditional_null_invalid: dict[tuple[str, str], int] = {}
    relation_invalid: dict[str, int] = {}
    ratio_counts = [0 for _ in rule.get("value_ratio_limits", [])]
    numeric_bound_invalid = {column: 0 for column in rule.get("numeric_bounds", {})}
    required_population = [0 for _ in rule.get("require_any_values", [])]
    source_bad_values = 0
    source_vintages: set[str] = set()

    try:
        reader = pd.read_csv(
            path,
            dtype={asset_column: str},
            low_memory=False,
            chunksize=chunksize,
        )
        for frame in reader:
            rows += len(frame)
            if asset_column in frame.columns:
                assets_seen.update(frame[asset_column].dropna().astype(str))

            key_missing = frame[primary_key].isna().any(axis=1)
            missing_key += int(key_missing.sum())
            valid_keys = frame.loc[~key_missing, primary_key].astype(str)
            if not valid_keys.empty:
                key_index = pd.MultiIndex.from_frame(valid_keys)
                duplicate_rows += int(key_index.duplicated(keep=False).sum())
                if not key_index.is_monotonic_increasing:
                    order_violations += 1
                first_key = tuple(valid_keys.iloc[0].tolist())
                last_key = tuple(valid_keys.iloc[-1].tolist())
                if previous_last_key is not None:
                    if first_key == previous_last_key:
                        duplicate_rows += 2
                    elif first_key < previous_last_key:
                        order_violations += 1
                previous_last_key = last_key

            parsed: dict[str, pd.Series] = {}
            for column in date_columns:
                raw = frame[column]
                values = pd.to_datetime(raw, errors="coerce")
                nonblank = raw.notna() & raw.astype(str).str.strip().ne("")
                parsed[column] = values
                bad_dates[column] += int((nonblank & values.isna()).sum())
            available = parsed[str(rule["available_date_column"])]
            future_available += int((available > as_of).sum())
            floor_columns = list(rule.get("availability_floor_columns", []))
            if floor_columns:
                floor = pd.concat([parsed[column] for column in floor_columns], axis=1).max(axis=1)
                floor_bad += int((available.notna() & floor.notna() & (available < floor)).sum())
            maximum_lag = rule.get("maximum_availability_lag_days")
            if maximum_lag is not None:
                lag_reference = parsed[lag_reference_column]
                lag_days = (available - lag_reference).dt.days
                availability_lag_bad += int(
                    (available.notna() & lag_reference.notna() & (lag_days > int(maximum_lag))).sum()
                )
            coverage = parsed[str(rule["coverage_column"])].dropna()
            if not coverage.empty:
                chunk_start = coverage.min()
                chunk_end = coverage.max()
                coverage_start = chunk_start if pd.isna(coverage_start) else min(coverage_start, chunk_start)
                coverage_end = chunk_end if pd.isna(coverage_end) else max(coverage_end, chunk_end)

            for column, accepted in rule.get("allowed_values", {}).items():
                actual = set(frame[column].dropna().astype(str).str.lower().str.strip())
                allowed = {str(value).lower().strip() for value in accepted}
                allowed_invalid[column].update(actual.difference(allowed))

            for column in rule.get("finite_numeric_columns", []):
                values = pd.to_numeric(frame[column], errors="coerce")
                finite_invalid[column] += int((values.isna() | ~values.map(math.isfinite)).sum())

            for requirement in rule.get("conditional_finite_numeric_columns", []):
                condition_column = str(requirement["when_column"])
                accepted = {str(value).lower().strip() for value in requirement["when_values"]}
                condition = frame[condition_column].astype(str).str.lower().str.strip().isin(accepted)
                for column in requirement["columns"]:
                    values = pd.to_numeric(frame[column], errors="coerce")
                    key = (condition_column, str(column))
                    conditional_finite_invalid[key] = conditional_finite_invalid.get(key, 0) + int(
                        (condition & (values.isna() | ~values.map(math.isfinite))).sum()
                    )

            for requirement in rule.get("conditional_null_columns", []):
                condition_column = str(requirement["when_column"])
                accepted = {str(value).lower().strip() for value in requirement["when_values"]}
                condition = frame[condition_column].astype(str).str.lower().str.strip().isin(accepted)
                for column in requirement["columns"]:
                    key = (condition_column, str(column))
                    conditional_null_invalid[key] = conditional_null_invalid.get(key, 0) + int(
                        (condition & frame[column].notna()).sum()
                    )

            for requirement in rule.get("conditional_numeric_relations", []):
                condition_column = str(requirement["when_column"])
                accepted = {str(value).lower().strip() for value in requirement["when_values"]}
                condition = frame[condition_column].astype(str).str.lower().str.strip().isin(accepted)
                left = pd.to_numeric(frame[str(requirement["left_column"])], errors="coerce")
                right = pd.to_numeric(frame[str(requirement["right_column"])], errors="coerce")
                operator = str(requirement["operator"])
                if operator == ">":
                    valid = left > right
                elif operator == ">=":
                    valid = left >= right
                elif operator == "<":
                    valid = left < right
                elif operator == "<=":
                    valid = left <= right
                else:
                    raise ValueError(f"unsupported numeric relation operator: {operator}")
                label = f"{requirement['left_column']}{operator}{requirement['right_column']}"
                relation_invalid[label] = relation_invalid.get(label, 0) + int((condition & ~valid).sum())

            for index, requirement in enumerate(rule.get("value_ratio_limits", [])):
                accepted = {str(value).lower().strip() for value in requirement["values"]}
                values = frame[str(requirement["column"])].astype(str).str.lower().str.strip()
                ratio_counts[index] += int(values.isin(accepted).sum())

            for column, bounds in rule.get("numeric_bounds", {}).items():
                values = pd.to_numeric(frame[column], errors="coerce")
                valid = values.notna() & values.map(math.isfinite)
                if "minimum" in bounds:
                    valid &= values >= float(bounds["minimum"])
                if "exclusive_minimum" in bounds:
                    valid &= values > float(bounds["exclusive_minimum"])
                if "maximum" in bounds:
                    valid &= values <= float(bounds["maximum"])
                if "exclusive_maximum" in bounds:
                    valid &= values < float(bounds["exclusive_maximum"])
                numeric_bound_invalid[column] += int((~valid).sum())

            for index, requirement in enumerate(rule.get("require_any_values", [])):
                accepted = {str(value).lower().strip() for value in requirement["values"]}
                actual = frame[str(requirement["column"])].dropna().astype(str).str.lower().str.strip()
                required_population[index] += int(actual.isin(accepted).sum())

            source_columns = [column for column in ("data_source", "source_vintage") if column in frame.columns]
            source_bad_values += sum(
                int(frame[column].isna().sum() + frame[column].astype(str).str.strip().eq("").sum())
                for column in source_columns
            )
            if "source_vintage" in frame.columns:
                source_vintages.update(
                    frame["source_vintage"].dropna().astype(str).str.strip().loc[lambda item: item.ne("")].unique()
                )
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as exc:
        _check(checks, rule, "csv_readable", False, repr(exc))
        empty_summary.update({"rows": rows, "assets": len(assets_seen), "failed_checks": "csv_readable"})
        return pd.DataFrame(checks, columns=CHECK_COLUMNS), empty_summary, [path]

    _check(checks, rule, "csv_readable", True, f"rows={rows};chunksize={chunksize}")
    _check(checks, rule, "minimum_rows", rows >= int(rule["minimum_rows"]), f"rows={rows};min={rule['minimum_rows']}")
    _check(checks, rule, "primary_key_complete", missing_key == 0, f"bad_rows={missing_key}")
    unique_proven = duplicate_rows == 0 and order_violations == 0
    _check(
        checks,
        rule,
        "primary_key_unique",
        unique_proven,
        f"duplicate_rows={duplicate_rows};order_violations={order_violations}",
    )
    _check(
        checks,
        rule,
        "streaming_primary_key_order",
        order_violations == 0,
        f"order_violations={order_violations};order={primary_key}",
    )
    _check(checks, rule, "dates_parseable", sum(bad_dates.values()) == 0, json.dumps(bad_dates, ensure_ascii=False, sort_keys=True))
    _check(checks, rule, "available_date_not_future", future_available == 0, f"bad_rows={future_available}")
    _check(checks, rule, "available_date_respects_events", floor_bad == 0, f"bad_rows={floor_bad}")
    if rule.get("maximum_availability_lag_days") is not None:
        _check(
            checks,
            rule,
            "availability_lag_within_limit",
            availability_lag_bad == 0,
            f"bad_rows={availability_lag_bad};max_days={int(rule['maximum_availability_lag_days'])}",
        )
    required_start = pd.Timestamp(rule["minimum_start_date"])
    required_end = pd.Timestamp(rule["minimum_end_date"])
    start_ok = pd.notna(coverage_start) and pd.Timestamp(coverage_start) <= required_start
    end_ok = pd.notna(coverage_end) and pd.Timestamp(coverage_end) >= required_end
    _check(checks, rule, "coverage_start", bool(start_ok), f"actual={coverage_start};required<={required_start.date()}")
    _check(checks, rule, "coverage_end", bool(end_ok), f"actual={coverage_end};required>={required_end.date()}")
    assets = len(assets_seen)
    _check(checks, rule, "minimum_assets", assets >= int(rule["minimum_assets"]), f"assets={assets};min={rule['minimum_assets']}")

    for column, invalid in allowed_invalid.items():
        _check(checks, rule, f"allowed_values:{column}", not invalid, ",".join(sorted(invalid)[:20]))
    for column, invalid in finite_invalid.items():
        _check(checks, rule, f"finite_numeric:{column}", invalid == 0, f"bad_rows={invalid}")
    for (_, column), invalid in conditional_finite_invalid.items():
        _check(checks, rule, f"conditional_finite:{column}", invalid == 0, f"bad_rows={invalid}")
    for (_, column), invalid in conditional_null_invalid.items():
        _check(checks, rule, f"conditional_null:{column}", invalid == 0, f"bad_rows={invalid}")
    for label, invalid in relation_invalid.items():
        _check(checks, rule, f"numeric_relation:{label}", invalid == 0, f"bad_rows={invalid}")
    for index, requirement in enumerate(rule.get("value_ratio_limits", [])):
        ratio = ratio_counts[index] / rows if rows else math.nan
        minimum = float(requirement.get("minimum", 0.0))
        maximum = float(requirement.get("maximum", 1.0))
        valid = math.isfinite(ratio) and minimum <= ratio <= maximum
        _check(
            checks,
            rule,
            f"value_ratio:{requirement['column']}",
            valid,
            f"ratio={ratio:.8f};required=[{minimum:.8f},{maximum:.8f}]",
        )
    for column, invalid in numeric_bound_invalid.items():
        _check(checks, rule, f"numeric_bounds:{column}", invalid == 0, f"bad_rows={invalid}")
    for index, requirement in enumerate(rule.get("require_any_values", [])):
        count = required_population[index]
        accepted = {str(value).lower().strip() for value in requirement["values"]}
        _check(
            checks,
            rule,
            f"required_population:{requirement['column']}",
            count > 0,
            f"matching_rows={count};values={sorted(accepted)}",
        )
    _check(checks, rule, "effective_intervals_non_overlapping", True, "streaming dataset has no interval contract")
    _check(
        checks,
        rule,
        "source_lineage_complete",
        source_bad_values == 0 and bool(source_vintages),
        f"source_vintages={len(source_vintages)};bad_values={source_bad_values}",
    )

    dataset_sha256 = _sha256(path)
    lineage_paths = _validate_external_evidence(
        checks,
        root,
        rule,
        path,
        dataset_sha256=dataset_sha256,
        source_vintages=source_vintages,
    )
    check_frame = pd.DataFrame(checks, columns=CHECK_COLUMNS)
    failed_checks = check_frame.loc[check_frame["status"].eq("fail"), "check"].tolist()
    summary = {
        "dataset_id": rule["dataset_id"],
        "sleeve": rule["sleeve"],
        "priority": rule["priority"],
        "path": rule["path"],
        "status": "pass" if not failed_checks else "blocked",
        "rows": rows,
        "assets": assets,
        "coverage_start": str(pd.Timestamp(coverage_start).date()) if pd.notna(coverage_start) else "",
        "coverage_end": str(pd.Timestamp(coverage_end).date()) if pd.notna(coverage_end) else "",
        "failed_checks": ";".join(failed_checks),
    }
    return check_frame, summary, lineage_paths


def validate_dataset(root: Path, rule: dict[str, Any], as_of: pd.Timestamp) -> tuple[pd.DataFrame, dict[str, Any], list[Path]]:
    if rule.get("streaming_csv") is True:
        return _validate_dataset_streaming(root, rule, as_of)
    checks: list[dict[str, str]] = []
    path = _safe_path(root, str(rule["path"]))
    exists = path.is_file()
    _check(checks, rule, "file_exists", exists, str(rule["path"]))
    empty_summary = {
        "dataset_id": rule["dataset_id"],
        "sleeve": rule["sleeve"],
        "priority": rule["priority"],
        "path": rule["path"],
        "status": "blocked",
        "rows": 0,
        "assets": 0,
        "coverage_start": "",
        "coverage_end": "",
        "failed_checks": "file_exists",
    }
    if not exists:
        return pd.DataFrame(checks, columns=CHECK_COLUMNS), empty_summary, []

    try:
        frame = pd.read_csv(path, dtype={"asset": str}, low_memory=False)
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        _check(checks, rule, "csv_readable", False, repr(exc))
        empty_summary["failed_checks"] = "csv_readable"
        return pd.DataFrame(checks, columns=CHECK_COLUMNS), empty_summary, [path]

    _check(checks, rule, "csv_readable", True, f"rows={len(frame)}")
    _check(checks, rule, "minimum_rows", len(frame) >= int(rule["minimum_rows"]), f"rows={len(frame)};min={rule['minimum_rows']}")
    required_columns = set(rule["required_columns"])
    missing_columns = sorted(required_columns.difference(frame.columns))
    _check(checks, rule, "required_columns", not missing_columns, ",".join(missing_columns))
    if missing_columns:
        failed = ",".join(row["check"] for row in checks if row["status"] == "fail")
        empty_summary.update({"rows": len(frame), "failed_checks": failed})
        return pd.DataFrame(checks, columns=CHECK_COLUMNS), empty_summary, [path]

    primary_key = list(rule["primary_key"])
    missing_key = int(frame[primary_key].isna().any(axis=1).sum())
    duplicates = int(frame.duplicated(primary_key, keep=False).sum())
    _check(checks, rule, "primary_key_complete", missing_key == 0, f"bad_rows={missing_key}")
    _check(checks, rule, "primary_key_unique", duplicates == 0, f"duplicate_rows={duplicates}")

    date_columns = list(dict.fromkeys([*rule["date_columns"], rule["available_date_column"], rule["coverage_column"]]))
    lag_reference_column = str(rule.get("availability_lag_reference_column", "")).strip()
    if lag_reference_column:
        date_columns = list(dict.fromkeys([*date_columns, lag_reference_column]))
    parsed, bad_dates = _parse_dates(frame, date_columns)
    _check(checks, rule, "dates_parseable", sum(bad_dates.values()) == 0, json.dumps(bad_dates, ensure_ascii=False, sort_keys=True))

    available = parsed[str(rule["available_date_column"])]
    future_available = int((available > as_of).sum())
    _check(checks, rule, "available_date_not_future", future_available == 0, f"bad_rows={future_available}")

    floor_columns = list(rule.get("availability_floor_columns", []))
    floor_bad = 0
    if floor_columns:
        floor = pd.concat([parsed[column] for column in floor_columns], axis=1).max(axis=1)
        floor_bad = int((available.notna() & floor.notna() & (available < floor)).sum())
    _check(checks, rule, "available_date_respects_events", floor_bad == 0, f"bad_rows={floor_bad}")
    maximum_lag = rule.get("maximum_availability_lag_days")
    if maximum_lag is not None:
        lag_reference = parsed[lag_reference_column]
        lag_days = (available - lag_reference).dt.days
        lag_bad = int((available.notna() & lag_reference.notna() & (lag_days > int(maximum_lag))).sum())
        _check(
            checks,
            rule,
            "availability_lag_within_limit",
            lag_bad == 0,
            f"bad_rows={lag_bad};max_days={int(maximum_lag)}",
        )

    coverage = parsed[str(rule["coverage_column"])].dropna()
    coverage_start = coverage.min() if not coverage.empty else pd.NaT
    coverage_end = coverage.max() if not coverage.empty else pd.NaT
    required_start = pd.Timestamp(rule["minimum_start_date"])
    required_end = pd.Timestamp(rule["minimum_end_date"])
    start_ok = pd.notna(coverage_start) and pd.Timestamp(coverage_start) <= required_start
    end_ok = pd.notna(coverage_end) and pd.Timestamp(coverage_end) >= required_end
    _check(checks, rule, "coverage_start", bool(start_ok), f"actual={coverage_start};required<={required_start.date()}")
    _check(checks, rule, "coverage_end", bool(end_ok), f"actual={coverage_end};required>={required_end.date()}")

    asset_column = str(rule.get("asset_column", "asset"))
    assets = int(frame[asset_column].dropna().astype(str).nunique()) if asset_column in frame.columns else 0
    _check(checks, rule, "minimum_assets", assets >= int(rule["minimum_assets"]), f"assets={assets};min={rule['minimum_assets']}")

    for column, accepted in rule.get("allowed_values", {}).items():
        actual = set(frame[column].dropna().astype(str).str.lower().str.strip())
        allowed = {str(value).lower().strip() for value in accepted}
        invalid = sorted(actual.difference(allowed))
        _check(checks, rule, f"allowed_values:{column}", not invalid, ",".join(invalid[:20]))

    for column in rule.get("finite_numeric_columns", []):
        values = pd.to_numeric(frame[column], errors="coerce")
        invalid = int((values.isna() | ~values.map(math.isfinite)).sum())
        _check(checks, rule, f"finite_numeric:{column}", invalid == 0, f"bad_rows={invalid}")

    for requirement in rule.get("conditional_finite_numeric_columns", []):
        condition_column = str(requirement["when_column"])
        accepted = {str(value).lower().strip() for value in requirement["when_values"]}
        condition = frame[condition_column].astype(str).str.lower().str.strip().isin(accepted)
        for column in requirement["columns"]:
            values = pd.to_numeric(frame[column], errors="coerce")
            invalid = int((condition & (values.isna() | ~values.map(math.isfinite))).sum())
            _check(checks, rule, f"conditional_finite:{column}", invalid == 0, f"bad_rows={invalid}")

    for requirement in rule.get("conditional_null_columns", []):
        condition_column = str(requirement["when_column"])
        accepted = {str(value).lower().strip() for value in requirement["when_values"]}
        condition = frame[condition_column].astype(str).str.lower().str.strip().isin(accepted)
        for column in requirement["columns"]:
            invalid = int((condition & frame[column].notna()).sum())
            _check(checks, rule, f"conditional_null:{column}", invalid == 0, f"bad_rows={invalid}")

    for requirement in rule.get("conditional_numeric_relations", []):
        condition_column = str(requirement["when_column"])
        accepted = {str(value).lower().strip() for value in requirement["when_values"]}
        condition = frame[condition_column].astype(str).str.lower().str.strip().isin(accepted)
        left = pd.to_numeric(frame[str(requirement["left_column"])], errors="coerce")
        right = pd.to_numeric(frame[str(requirement["right_column"])], errors="coerce")
        operator = str(requirement["operator"])
        if operator == ">":
            valid = left > right
        elif operator == ">=":
            valid = left >= right
        elif operator == "<":
            valid = left < right
        elif operator == "<=":
            valid = left <= right
        else:
            raise ValueError(f"unsupported numeric relation operator: {operator}")
        invalid = int((condition & ~valid).sum())
        label = f"numeric_relation:{requirement['left_column']}{operator}{requirement['right_column']}"
        _check(checks, rule, label, invalid == 0, f"bad_rows={invalid}")

    for requirement in rule.get("value_ratio_limits", []):
        column = str(requirement["column"])
        accepted = {str(value).lower().strip() for value in requirement["values"]}
        values = frame[column].astype(str).str.lower().str.strip()
        ratio = float(values.isin(accepted).mean()) if len(values) else math.nan
        minimum = float(requirement.get("minimum", 0.0))
        maximum = float(requirement.get("maximum", 1.0))
        valid = math.isfinite(ratio) and minimum <= ratio <= maximum
        _check(
            checks,
            rule,
            f"value_ratio:{column}",
            valid,
            f"ratio={ratio:.8f};required=[{minimum:.8f},{maximum:.8f}]",
        )

    for column, bounds in rule.get("numeric_bounds", {}).items():
        values = pd.to_numeric(frame[column], errors="coerce")
        valid = values.notna() & values.map(math.isfinite)
        if "minimum" in bounds:
            valid &= values >= float(bounds["minimum"])
        if "exclusive_minimum" in bounds:
            valid &= values > float(bounds["exclusive_minimum"])
        if "maximum" in bounds:
            valid &= values <= float(bounds["maximum"])
        if "exclusive_maximum" in bounds:
            valid &= values < float(bounds["exclusive_maximum"])
        invalid = int((~valid).sum())
        _check(checks, rule, f"numeric_bounds:{column}", invalid == 0, f"bad_rows={invalid}")

    for requirement in rule.get("require_any_values", []):
        column = str(requirement["column"])
        accepted = {str(value).lower().strip() for value in requirement["values"]}
        actual = frame[column].dropna().astype(str).str.lower().str.strip()
        count = int(actual.isin(accepted).sum())
        _check(checks, rule, f"required_population:{column}", count > 0, f"matching_rows={count};values={sorted(accepted)}")

    overlaps = _overlap_count(frame, rule, parsed)
    _check(checks, rule, "effective_intervals_non_overlapping", overlaps == 0, f"overlaps_or_invalid={overlaps}")

    source_columns = [column for column in ("data_source", "source_vintage") if column in frame.columns]
    empty_sources = sum(int(frame[column].isna().sum() + frame[column].astype(str).str.strip().eq("").sum()) for column in source_columns)
    _check(checks, rule, "source_lineage_complete", len(source_columns) == 2 and empty_sources == 0, f"columns={source_columns};bad_values={empty_sources}")

    lineage_paths = [path]
    lineage_relative = rule.get("lineage_manifest")
    if lineage_relative:
        lineage_path = _safe_path(root, str(lineage_relative))
        lineage_exists = lineage_path.is_file()
        _check(checks, rule, "lineage_manifest_exists", lineage_exists, str(lineage_relative))
        if lineage_exists:
            lineage_paths.append(lineage_path)
            try:
                lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                _check(checks, rule, "lineage_manifest_readable", False, repr(exc))
            else:
                _check(checks, rule, "lineage_manifest_readable", True, str(lineage_relative))
                required_output_role = rule.get("lineage_required_output_role")
                if required_output_role is None:
                    output_path = str(lineage.get("output_path", "")).replace("\\", "/")
                    output_hash = str(lineage.get("output_sha256", ""))
                else:
                    output_items = lineage.get("outputs")
                    output_items = output_items if isinstance(output_items, list) else []
                    roles = [str(item.get("role", "")) for item in output_items if isinstance(item, dict)]
                    _check(
                        checks,
                        rule,
                        "lineage_output_roles_unique",
                        len(roles) == len(set(roles)),
                        f"roles={roles}",
                    )
                    role_item = next(
                        (
                            item
                            for item in output_items
                            if isinstance(item, dict) and str(item.get("role", "")) == str(required_output_role)
                        ),
                        {},
                    )
                    output_path = str(role_item.get("path", "")).replace("\\", "/")
                    output_hash = str(role_item.get("sha256", ""))
                output_ok = output_path == str(rule["path"]).replace("\\", "/") and output_hash == _sha256(path)
                _check(checks, rule, "lineage_output_hash_match", output_ok, f"output_path={output_path}")
                code_relative = str(lineage.get("code_path", ""))
                try:
                    code_path = _safe_path(root, code_relative) if code_relative else None
                except ValueError:
                    code_path = None
                code_ok = bool(code_path and code_path.is_file() and _sha256(code_path) == str(lineage.get("code_sha256", "")))
                _check(checks, rule, "lineage_code_hash_match", code_ok, code_relative)
                if code_path is not None and code_path.is_file():
                    lineage_paths.append(code_path)

                inputs = lineage.get("inputs")
                if not isinstance(inputs, list):
                    input_relative = lineage.get("input_path")
                    input_hash = lineage.get("input_sha256")
                    inputs = [{"path": input_relative, "sha256": input_hash}] if input_relative and input_hash else []
                input_failures: list[str] = []
                for item in inputs:
                    if not isinstance(item, dict):
                        input_failures.append("invalid_manifest_item")
                        continue
                    input_relative = str(item.get("path", ""))
                    try:
                        input_path = _safe_path(root, input_relative) if input_relative else None
                    except ValueError:
                        input_path = None
                    expected_hash = str(item.get("sha256", ""))
                    if not input_path or not input_path.is_file() or not expected_hash or _sha256(input_path) != expected_hash:
                        input_failures.append(input_relative or "missing_path")
                        continue
                    lineage_paths.append(input_path)
                _check(
                    checks,
                    rule,
                    "lineage_inputs_hash_match",
                    bool(inputs) and not input_failures,
                    f"inputs={len(inputs)};failures={input_failures[:10]}",
                )

                source_vintages = set(
                    frame["source_vintage"].dropna().astype(str).str.strip().loc[lambda item: item.ne("")].unique()
                )
                vintage_mode = str(rule.get("lineage_source_vintage_mode", "single"))
                if vintage_mode == "set_sha256":
                    expected_vintage_hash = str(lineage.get("source_vintage_set_sha256", ""))
                    expected_vintage_count = lineage.get("source_vintage_count")
                    actual_vintage_hash = _value_set_sha256(source_vintages)
                    vintage_ok = bool(
                        expected_vintage_hash
                        and expected_vintage_hash == actual_vintage_hash
                        and expected_vintage_count == len(source_vintages)
                    )
                    vintage_detail = (
                        f"mode=set_sha256;count={len(source_vintages)};"
                        f"expected_count={expected_vintage_count};actual_hash={actual_vintage_hash}"
                    )
                else:
                    expected_vintage = str(lineage.get("source_vintage", ""))
                    if not expected_vintage and lineage.get("input_sha256"):
                        expected_vintage = f"macro_pit_panel_sha256:{lineage['input_sha256']}"
                    vintage_ok = bool(expected_vintage and source_vintages == {expected_vintage})
                    vintage_detail = expected_vintage
                _check(checks, rule, "lineage_source_vintage_match", vintage_ok, vintage_detail)
                _check(
                    checks,
                    rule,
                    "lineage_historical_use_approved",
                    lineage.get("historical_backtest_allowed") is True,
                    f"historical_backtest_allowed={lineage.get('historical_backtest_allowed')!r}",
                )
                for field, expected in rule.get("lineage_required_values", {}).items():
                    actual = lineage.get(field)
                    _check(
                        checks,
                        rule,
                        f"lineage_required_value:{field}",
                        actual == expected,
                        f"expected={expected!r};actual={actual!r}",
                    )

    validation_relative = rule.get("validation_manifest")
    if validation_relative:
        validation_path = _safe_path(root, str(validation_relative))
        validation_exists = validation_path.is_file()
        _check(checks, rule, "validation_manifest_exists", validation_exists, str(validation_relative))
        if validation_exists:
            lineage_paths.append(validation_path)
            try:
                validation = json.loads(validation_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                _check(checks, rule, "validation_manifest_readable", False, repr(exc))
            else:
                _check(checks, rule, "validation_manifest_readable", True, str(validation_relative))
                _check(
                    checks,
                    rule,
                    "validation_qualification_passed",
                    validation.get("qualification_status") == "PASS",
                    f"qualification_status={validation.get('qualification_status')!r}",
                )
                _check(
                    checks,
                    rule,
                    "validation_historical_use_approved",
                    validation.get("historical_backtest_allowed") is True,
                    f"historical_backtest_allowed={validation.get('historical_backtest_allowed')!r}",
                )
                _check(
                    checks,
                    rule,
                    "validation_cannot_promote_model",
                    validation.get("model_promotion_allowed") is False,
                    f"model_promotion_allowed={validation.get('model_promotion_allowed')!r}",
                )
                for field, expected in rule.get("validation_required_values", {}).items():
                    actual = validation.get(field)
                    _check(
                        checks,
                        rule,
                        f"validation_required_value:{field}",
                        actual == expected,
                        f"expected={expected!r};actual={actual!r}",
                    )
                expected_schema = rule.get("validation_schema")
                if expected_schema is not None:
                    _check(
                        checks,
                        rule,
                        "validation_schema_match",
                        validation.get("validation_schema") == expected_schema,
                        f"expected={expected_schema!r};actual={validation.get('validation_schema')!r}",
                    )

                zero_count_fields = rule.get("validation_required_zero_counts")
                if zero_count_fields is None:
                    zero_count_fields = ["continuous_trading_large_jump_count"]
                for field in zero_count_fields:
                    _check(
                        checks,
                        rule,
                        f"validation_zero_count:{field}",
                        validation.get(field) == 0,
                        f"count={validation.get(field)!r}",
                    )

                allowed_validation_values = rule.get("validation_allowed_values")
                if allowed_validation_values is None:
                    allowed_validation_values = {
                        "review_status": ["CLEAR", "LONG_SUSPENSION_REVIEW_REQUIRED"]
                    }
                for field, allowed in allowed_validation_values.items():
                    actual = validation.get(field)
                    _check(
                        checks,
                        rule,
                        f"validation_allowed_value:{field}",
                        actual in set(allowed),
                        f"value={actual!r};allowed={allowed!r}",
                    )

                code_relative = str(validation.get("code_path", ""))
                try:
                    validation_code = _safe_path(root, code_relative) if code_relative else None
                except ValueError:
                    validation_code = None
                code_ok = bool(
                    validation_code
                    and validation_code.is_file()
                    and _sha256(validation_code) == str(validation.get("code_sha256", ""))
                )
                _check(checks, rule, "validation_code_hash_match", code_ok, code_relative)
                if validation_code is not None and validation_code.is_file():
                    lineage_paths.append(validation_code)

                validation_inputs = validation.get("inputs")
                input_failures: list[str] = []
                input_paths: list[str] = []
                dataset_input_match = False
                if not isinstance(validation_inputs, list):
                    validation_inputs = []
                    input_failures.append("inputs_not_a_list")
                for item in validation_inputs:
                    if not isinstance(item, dict):
                        input_failures.append("invalid_manifest_item")
                        continue
                    input_relative = str(item.get("path", ""))
                    input_paths.append(input_relative.replace("\\", "/"))
                    try:
                        input_path = _safe_path(root, input_relative) if input_relative else None
                    except ValueError:
                        input_path = None
                    expected_hash = str(item.get("sha256", ""))
                    if not input_path or not input_path.is_file() or not expected_hash or _sha256(input_path) != expected_hash:
                        input_failures.append(input_relative or "missing_path")
                        continue
                    if input_relative.replace("\\", "/") == str(rule["path"]).replace("\\", "/"):
                        dataset_input_match = expected_hash == _sha256(path)
                minimum_inputs = int(rule.get("validation_minimum_input_count", 1))
                duplicate_inputs = len(input_paths) - len(set(input_paths))
                _check(
                    checks,
                    rule,
                    "validation_inputs_hash_match",
                    len(validation_inputs) >= minimum_inputs and not input_failures and duplicate_inputs == 0,
                    (
                        f"inputs={len(validation_inputs)};min={minimum_inputs};"
                        f"duplicates={duplicate_inputs};failures={input_failures[:10]}"
                    ),
                )
                require_dataset_input_match = bool(rule.get("validation_require_dataset_input_match", True))
                _check(
                    checks,
                    rule,
                    "validation_dataset_input_match",
                    dataset_input_match or not require_dataset_input_match,
                    (
                        str(rule["path"])
                        if require_dataset_input_match
                        else "disabled_by_explicit_post_validation_promotion_contract"
                    ),
                )

                required_output_roles = rule.get("validation_required_output_roles")
                if required_output_roles is not None:
                    output_items = validation.get("outputs")
                    output_items = output_items if isinstance(output_items, list) else []
                    roles = [str(item.get("role", "")) for item in output_items if isinstance(item, dict)]
                    role_map = {
                        str(item.get("role", "")): item
                        for item in output_items
                        if isinstance(item, dict) and item.get("role")
                    }
                    _check(
                        checks,
                        rule,
                        "validation_output_roles_unique",
                        len(roles) == len(set(roles)),
                        f"roles={roles}",
                    )
                    for role in required_output_roles:
                        item = role_map.get(str(role), {})
                        output_relative = str(item.get("path", ""))
                        try:
                            validation_output = _safe_path(root, output_relative) if output_relative else None
                        except ValueError:
                            validation_output = None
                        output_ok = bool(
                            validation_output
                            and validation_output.is_file()
                            and _sha256(validation_output) == str(item.get("sha256", ""))
                        )
                        _check(
                            checks,
                            rule,
                            f"validation_output_hash_match:{role}",
                            output_ok,
                            output_relative,
                        )
                        if validation_output is not None and validation_output.is_file():
                            lineage_paths.append(validation_output)
                else:
                    for path_key, hash_key in (
                        ("output_path", "output_sha256"),
                        ("exceptions_path", "exceptions_sha256"),
                        ("long_gap_review_path", "long_gap_review_sha256"),
                        ("report_path", "report_sha256"),
                    ):
                        output_relative = str(validation.get(path_key, ""))
                        try:
                            validation_output = _safe_path(root, output_relative) if output_relative else None
                        except ValueError:
                            validation_output = None
                        output_ok = bool(
                            validation_output
                            and validation_output.is_file()
                            and _sha256(validation_output) == str(validation.get(hash_key, ""))
                        )
                        _check(checks, rule, f"validation_output_hash_match:{path_key}", output_ok, output_relative)
                        if validation_output is not None and validation_output.is_file():
                            lineage_paths.append(validation_output)

                exceptions_relative = str(validation.get("exceptions_path", ""))
                try:
                    exceptions_path = _safe_path(root, exceptions_relative) if exceptions_relative else None
                    exceptions = pd.read_csv(exceptions_path) if exceptions_path and exceptions_path.is_file() else None
                except (OSError, UnicodeError, pd.errors.ParserError, ValueError):
                    exceptions = None
                _check(
                    checks,
                    rule,
                    "validation_exception_file_empty",
                    exceptions is not None and exceptions.empty,
                    f"rows={len(exceptions) if exceptions is not None else 'unreadable'}",
                )

    check_frame = pd.DataFrame(checks, columns=CHECK_COLUMNS)
    failed_checks = check_frame.loc[check_frame["status"].eq("fail"), "check"].tolist()
    summary = {
        "dataset_id": rule["dataset_id"],
        "sleeve": rule["sleeve"],
        "priority": rule["priority"],
        "path": rule["path"],
        "status": "pass" if not failed_checks else "blocked",
        "rows": int(len(frame)),
        "assets": assets,
        "coverage_start": str(pd.Timestamp(coverage_start).date()) if pd.notna(coverage_start) else "",
        "coverage_end": str(pd.Timestamp(coverage_end).date()) if pd.notna(coverage_end) else "",
        "failed_checks": ";".join(failed_checks),
    }
    return check_frame, summary, lineage_paths


def _sleeve_ready(summary: pd.DataFrame, sleeve: str) -> bool:
    required = summary[summary["sleeve"].isin({sleeve, "shared", "macro"})]
    return bool(not required.empty and required["status"].eq("pass").all())


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row.get(column, "")).replace("|", "/") for column in columns) + " |")
    if frame.empty:
        lines.append("| " + " | ".join([""] * len(columns)) + " |")
    return lines


def run_gate(root: Path, config: dict[str, Any], as_of: str | pd.Timestamp, config_path: Path | None = None) -> dict[str, Path]:
    as_of_ts = pd.Timestamp(as_of).normalize()
    output = _safe_path(root, str(config["output_directory"]))
    output.mkdir(parents=True, exist_ok=True)

    check_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    existing_paths: list[Path] = []
    for rule in config["datasets"]:
        checks, summary, source_paths = validate_dataset(root, rule, as_of_ts)
        check_frames.append(checks)
        summaries.append(summary)
        existing_paths.extend(source_paths)

    checks = pd.concat(check_frames, ignore_index=True) if check_frames else pd.DataFrame(columns=CHECK_COLUMNS)
    summary = pd.DataFrame(summaries, columns=SUMMARY_COLUMNS)
    stock_ready = _sleeve_ready(summary, "stock")
    etf_ready = _sleeve_ready(summary, "etf")
    macro_rows = summary[summary["sleeve"].isin({"macro", "shared"})]
    macro_ready = bool(not macro_rows.empty and macro_rows["status"].eq("pass").all())
    historical_inputs_ready = bool(stock_ready and etf_ready and macro_ready)
    blocked = summary[summary["status"].ne("pass")].copy()
    rule_map = {str(item["dataset_id"]): item for item in config["datasets"]}
    queue_rows = []
    for row in blocked.itertuples(index=False):
        rule = rule_map[str(row.dataset_id)]
        queue_rows.append(
            {
                "priority": row.priority,
                "dataset_id": row.dataset_id,
                "sleeve": row.sleeve,
                "target_path": row.path,
                "status": "missing" if "file_exists" in str(row.failed_checks) else "quality_blocked",
                "failed_checks": row.failed_checks,
                "provider_options": "; ".join(str(value) for value in rule["provider_options"]),
                "manual_action": str(rule.get("manual_action", "obtain and validate the declared PIT dataset")),
            }
        )
    queue = pd.DataFrame(queue_rows, columns=QUEUE_COLUMNS)
    readiness = {
        "model": config["model"],
        "as_of_date": str(as_of_ts.date()),
        "system_status": "PIT_INPUTS_READY_FOR_WALK_FORWARD" if historical_inputs_ready else "BLOCKED_MISSING_OR_INVALID_PIT_DATA",
        "stock_history_ready": stock_ready,
        "etf_history_ready": etf_ready,
        "macro_history_ready": macro_ready,
        "historical_inputs_ready": historical_inputs_ready,
        "dataset_count": int(len(summary)),
        "passed_dataset_count": int(summary["status"].eq("pass").sum()),
        "blocked_dataset_count": int(summary["status"].ne("pass").sum()),
        "walk_forward_completed": False,
        "promotion_allowed": False,
        "promotion_blockers": [
            *([] if historical_inputs_ready else ["historical_pit_inputs_not_ready"]),
            "walk_forward_validation_not_completed",
        ],
        "boundary": "data qualification only; no performance or investment claim",
    }

    paths = {
        "readiness": output / "readiness.json",
        "dataset_checks": output / "dataset_checks.csv",
        "dataset_summary": output / "dataset_summary.csv",
        "missing_data_queue": output / "missing_data_queue.csv",
        "report": output / "GATE_E2_REPORT.md",
        "manifest": output / "run_manifest.json",
    }
    checks.to_csv(paths["dataset_checks"], index=False, encoding="utf-8-sig")
    summary.to_csv(paths["dataset_summary"], index=False, encoding="utf-8-sig")
    queue.to_csv(paths["missing_data_queue"], index=False, encoding="utf-8-sig")
    paths["readiness"].write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")

    report_lines = [
        "# Long Hold V4 Gate E2 PIT Readiness",
        "",
        f"As of: `{as_of_ts.date()}`",
        "",
        f"- Status: `{readiness['system_status']}`",
        f"- Stock / ETF / Macro ready: `{stock_ready}` / `{etf_ready}` / `{macro_ready}`",
        f"- Datasets passed: `{readiness['passed_dataset_count']}/{readiness['dataset_count']}`",
        "- Walk-forward completed: `False`",
        "- Promotion allowed: `False`",
        "",
        "## Dataset Summary",
        "",
        *_markdown_table(summary, ["dataset_id", "sleeve", "priority", "status", "rows", "assets", "coverage_start", "coverage_end", "failed_checks"]),
        "",
        "## Acquisition Queue",
        "",
        *_markdown_table(queue, ["priority", "dataset_id", "sleeve", "status", "provider_options", "manual_action"]),
        "",
        "This gate validates historical input eligibility only. It does not run or promote a strategy.",
        "",
    ]
    paths["report"].write_text("\n".join(report_lines), encoding="utf-8")

    manifest_inputs = [path for path in [config_path, *existing_paths] if path is not None and path.is_file()]
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "as_of_date": str(as_of_ts.date()),
        "status": readiness["system_status"],
        "input_files": [
            {"path": str(path.resolve().relative_to(root.resolve())), "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in sorted(set(manifest_inputs))
        ],
        "code_files": [
            {
                "path": str(Path(__file__).resolve().relative_to(root.resolve())),
                "sha256": _sha256(Path(__file__)),
            }
        ],
        "environment": {"python": platform.python_version(), "pandas": pd.__version__},
        "historical_inputs_ready": historical_inputs_ready,
        "promotion_allowed": False,
    }
    paths["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--as-of", required=True)
    args = parser.parse_args()
    config = load_gate_config(args.config)
    paths = run_gate(ROOT, config, args.as_of, args.config)
    readiness = json.loads(paths["readiness"].read_text(encoding="utf-8"))
    print(json.dumps({"status": readiness["system_status"], "outputs": {key: str(value) for key, value in paths.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
