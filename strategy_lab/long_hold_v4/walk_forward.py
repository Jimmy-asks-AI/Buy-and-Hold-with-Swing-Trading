"""Purged/embargoed walk-forward governance and artifact helpers.

This module controls split construction, tuning boundaries, one-time holdout
use, execution-state normalization, sleeve attribution, and immutable window
artifacts. It does not auto-promote a model.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .backtest import run_weight_backtest
from .core import ContractError
from .pit_gate_v2 import (
    SAFE_ID_RE,
    SHA256_RE,
    canonical_json_bytes,
    sha256_file,
)


WALK_FORWARD_SCHEMA_VERSION = 1
ALLOWED_TUNING_ROLES = {"train", "validation"}
EXECUTION_STATE_COLUMNS = {
    "date",
    "asset",
    "asset_type",
    "open",
    "close",
    "return_basis",
    "price_basis",
    "available_date",
    "list_date",
    "delist_date",
    "has_market_data",
    "is_suspended",
    "is_limit_up",
    "is_limit_down",
    "is_delisted",
}


def hash_dataframe(frame: pd.DataFrame) -> str:
    data = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid JSON file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ContractError(f"JSON object required: {path}")
    return payload


def _verify_project_file_binding(
    root: Path, binding: dict[str, Any], label: str
) -> Path:
    relative = str(binding.get("path", ""))
    expected = str(binding.get("sha256", "")).lower()
    path = (root / relative).resolve()
    if root != path and root not in path.parents:
        raise ContractError(f"{label} path escapes project root")
    if (
        not path.is_file()
        or SHA256_RE.fullmatch(expected) is None
        or sha256_file(path) != expected
    ):
        raise ContractError(f"{label} hash mismatch")
    return path


def load_walk_forward_config(path: str | Path) -> dict[str, Any]:
    config = _read_json(Path(path))
    required = {
        "schema_version",
        "output_root",
        "method",
        "tuning",
        "cost_scenarios",
        "benchmark",
        "execution",
        "governance",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ContractError(f"walk-forward config missing fields: {missing}")
    if config["schema_version"] != WALK_FORWARD_SCHEMA_VERSION:
        raise ContractError(
            f"walk-forward config requires schema_version={WALK_FORWARD_SCHEMA_VERSION}"
        )

    method_required = {
        "training_window_sessions",
        "validation_window_sessions",
        "independent_test_window_sessions",
        "step_sessions",
        "purge_sessions",
        "embargo_sessions",
        "label_horizon_sessions",
        "training_mode",
    }
    method_missing = sorted(method_required.difference(config["method"]))
    if method_missing:
        raise ContractError(f"walk-forward method missing fields: {method_missing}")
    method = config["method"]
    for field in method_required.difference({"training_mode"}):
        if int(method[field]) < 1:
            raise ContractError(f"walk-forward {field} must be positive")
    if method["training_mode"] not in {"rolling", "expanding"}:
        raise ContractError("training_mode must be rolling or expanding")
    label_horizon = int(method["label_horizon_sessions"])
    if int(method["purge_sessions"]) < label_horizon:
        raise ContractError("purge_sessions must cover the label horizon")
    if int(method["embargo_sessions"]) < label_horizon:
        raise ContractError("embargo_sessions must cover the label horizon")

    tuning = config["tuning"]
    if set(tuning.get("allowed_split_roles", [])) != ALLOWED_TUNING_ROLES:
        raise ContractError("tuning roles must be exactly train and validation")
    if int(tuning.get("maximum_candidate_count", 0)) < 1:
        raise ContractError("maximum_candidate_count must be positive")
    familywise_alpha = float(tuning.get("familywise_alpha", float("nan")))
    if not 0.0 < familywise_alpha <= 0.10:
        raise ContractError("familywise_alpha must be in (0, 0.10]")
    significance_blocks = int(
        tuning.get("significance_block_sessions", 0)
    )
    if significance_blocks < int(method["label_horizon_sessions"]):
        raise ContractError(
            "significance_block_sessions must cover the label horizon"
        )
    pbo_blocks = int(tuning.get("pbo_blocks", 0))
    if pbo_blocks < 2 or pbo_blocks % 2 != 0:
        raise ContractError("pbo_blocks must be an even integer >= 2")
    maximum_pbo = float(tuning.get("maximum_pbo", float("nan")))
    if not 0.0 <= maximum_pbo <= 0.50:
        raise ContractError("maximum_pbo must be in [0, 0.50]")
    minimum_dsr = float(
        tuning.get(
            "minimum_deflated_sharpe_probability", float("nan")
        )
    )
    if not 0.50 <= minimum_dsr < 1.0:
        raise ContractError(
            "minimum_deflated_sharpe_probability must be in [0.50, 1)"
        )
    if tuning.get("multiple_testing_correction") not in {
        "holm",
        "bonferroni",
        "fdr_bh",
        "none_if_single",
    }:
        raise ContractError("unsupported multiple-testing correction")
    if tuning.get("candidate_registry_required") is not True:
        raise ContractError("candidate registry must be required")
    if tuning.get("independent_test_one_time") is not True:
        raise ContractError("independent test must be one-time")

    bps = [int(value) for value in config["cost_scenarios"].get("additional_slippage_bps", [])]
    if bps != [5, 10, 20]:
        raise ContractError("cost scenarios must be exactly 5, 10 and 20 bps")
    benchmark = config["benchmark"]
    if not isinstance(benchmark, dict):
        raise ContractError("benchmark config must be an object")
    if not str(benchmark.get("benchmark_id", "")).strip():
        raise ContractError("benchmark_id is required")
    if benchmark.get("return_basis") != "total_return":
        raise ContractError(
            "formal benchmark must use total_return basis"
        )
    if config["execution"].get("unfilled_targets_accrue_returns") is not False:
        raise ContractError("unfilled targets must not accrue returns")
    if config["governance"].get("promotion_allowed") is not False:
        raise ContractError("walk-forward config must keep promotion_allowed=false")
    if config["governance"].get("manual_review_required") is not True:
        raise ContractError("manual review must be required")
    return config


def _calendar_index(calendar: Iterable[Any]) -> pd.DatetimeIndex:
    values = pd.DatetimeIndex(pd.to_datetime(list(calendar), errors="coerce"))
    if values.isna().any():
        raise ContractError("walk-forward calendar contains invalid dates")
    values = pd.DatetimeIndex(values.normalize().unique()).sort_values()
    if len(values) < 2:
        raise ContractError("walk-forward calendar is too short")
    return values


def build_purged_embargoed_plan(
    calendar: Iterable[Any], config: dict[str, Any]
) -> dict[str, Any]:
    dates = _calendar_index(calendar)
    method = config["method"]
    train_sessions = int(method["training_window_sessions"])
    validation_sessions = int(method["validation_window_sessions"])
    test_sessions = int(method["independent_test_window_sessions"])
    step_sessions = int(method["step_sessions"])
    purge_sessions = int(method["purge_sessions"])
    embargo_sessions = int(method["embargo_sessions"])
    label_horizon = int(method["label_horizon_sessions"])
    minimum = (
        train_sessions
        + purge_sessions
        + validation_sessions
        + embargo_sessions
        + test_sessions
    )
    if len(dates) < minimum:
        raise ContractError(
            f"walk-forward calendar requires at least {minimum} sessions"
        )

    test_start_index = len(dates) - test_sessions
    embargo_start_index = test_start_index - embargo_sessions
    first_validation_start = train_sessions + purge_sessions
    folds: list[dict[str, Any]] = []
    validation_start = first_validation_start
    fold_number = 1
    while validation_start + validation_sessions <= embargo_start_index:
        train_end_exclusive = validation_start - purge_sessions
        if method["training_mode"] == "rolling":
            train_start = train_end_exclusive - train_sessions
        else:
            train_start = 0
        validation_end_exclusive = validation_start + validation_sessions
        fold = {
            "window_id": f"validation-{fold_number:02d}",
            "split_role": "validation",
            "train_start": dates[train_start].date().isoformat(),
            "train_end": dates[train_end_exclusive - 1].date().isoformat(),
            "train_sessions": train_end_exclusive - train_start,
            "purge_start": dates[train_end_exclusive].date().isoformat(),
            "purge_end": dates[validation_start - 1].date().isoformat(),
            "purge_sessions": purge_sessions,
            "validation_start": dates[validation_start].date().isoformat(),
            "validation_end": dates[validation_end_exclusive - 1].date().isoformat(),
            "validation_sessions": validation_sessions,
            "label_horizon_sessions": label_horizon,
            "_train_start_index": train_start,
            "_train_end_index": train_end_exclusive - 1,
            "_validation_start_index": validation_start,
            "_validation_end_index": validation_end_exclusive - 1,
        }
        folds.append(fold)
        validation_start += step_sessions
        fold_number += 1
    if not folds:
        raise ContractError("walk-forward plan produced no validation windows")

    plan = {
        "schema_version": WALK_FORWARD_SCHEMA_VERSION,
        "training_mode": method["training_mode"],
        "label_horizon_sessions": label_horizon,
        "purge_sessions": purge_sessions,
        "embargo_sessions": embargo_sessions,
        "validation_windows": folds,
        "embargo": {
            "start": dates[embargo_start_index].date().isoformat(),
            "end": dates[test_start_index - 1].date().isoformat(),
            "sessions": embargo_sessions,
            "_start_index": embargo_start_index,
            "_end_index": test_start_index - 1,
        },
        "independent_test": {
            "window_id": "independent-test",
            "split_role": "independent_test",
            "start": dates[test_start_index].date().isoformat(),
            "end": dates[-1].date().isoformat(),
            "sessions": test_sessions,
            "evaluation_limit": 1,
            "_start_index": test_start_index,
            "_end_index": len(dates) - 1,
        },
        "tuning_allowed_split_roles": sorted(ALLOWED_TUNING_ROLES),
        "promotion_allowed": False,
    }
    leakage = audit_split_leakage(dates, plan)
    if not leakage["passed"]:
        raise ContractError(f"walk-forward split leakage detected: {leakage['failures']}")
    plan["plan_sha256"] = hashlib.sha256(canonical_json_bytes(_public_plan(plan))).hexdigest()
    return plan


def _public_plan(plan: dict[str, Any]) -> dict[str, Any]:
    def strip_private(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: strip_private(item)
                for key, item in value.items()
                if not key.startswith("_") and key != "plan_sha256"
            }
        if isinstance(value, list):
            return [strip_private(item) for item in value]
        return value

    return strip_private(plan)


def audit_split_leakage(
    calendar: Iterable[Any], plan: dict[str, Any]
) -> dict[str, Any]:
    dates = _calendar_index(calendar)
    label_horizon = int(plan["label_horizon_sessions"])
    failures: list[str] = []
    for fold in plan["validation_windows"]:
        train_indices = set(
            range(fold["_train_start_index"], fold["_train_end_index"] + 1)
        )
        purge_indices = set(
            range(fold["_train_end_index"] + 1, fold["_validation_start_index"])
        )
        validation_indices = set(
            range(
                fold["_validation_start_index"],
                fold["_validation_end_index"] + 1,
            )
        )
        if train_indices & purge_indices:
            failures.append(f"{fold['window_id']}:train_purge_overlap")
        if train_indices & validation_indices:
            failures.append(f"{fold['window_id']}:train_validation_overlap")
        if purge_indices & validation_indices:
            failures.append(f"{fold['window_id']}:purge_validation_overlap")
        if fold["_train_end_index"] + label_horizon >= fold["_validation_start_index"]:
            failures.append(f"{fold['window_id']}:train_label_crosses_validation")

    embargo = set(
        range(plan["embargo"]["_start_index"], plan["embargo"]["_end_index"] + 1)
    )
    independent = set(
        range(
            plan["independent_test"]["_start_index"],
            plan["independent_test"]["_end_index"] + 1,
        )
    )
    if embargo & independent:
        failures.append("embargo_independent_test_overlap")
    latest_validation_end = max(
        fold["_validation_end_index"] for fold in plan["validation_windows"]
    )
    if latest_validation_end + label_horizon >= plan["independent_test"]["_start_index"]:
        failures.append("validation_label_crosses_independent_test")
    for fold in plan["validation_windows"]:
        validation = set(
            range(
                fold["_validation_start_index"],
                fold["_validation_end_index"] + 1,
            )
        )
        if validation & embargo:
            failures.append(f"{fold['window_id']}:validation_embargo_overlap")
        if validation & independent:
            failures.append(f"{fold['window_id']}:validation_test_overlap")
    return {
        "passed": not failures,
        "failures": failures,
        "calendar_start": dates[0].date().isoformat(),
        "calendar_end": dates[-1].date().isoformat(),
        "promotion_allowed": False,
    }


def assert_tuning_inputs(split_roles: Iterable[str]) -> None:
    roles = {str(value).strip() for value in split_roles}
    disallowed = sorted(roles.difference(ALLOWED_TUNING_ROLES))
    if disallowed:
        raise ContractError(
            f"independent test or unsupported data cannot participate in tuning: {disallowed}"
        )


def _adjust_p_values(values: pd.Series, method: str) -> pd.Series:
    p_values = pd.to_numeric(values, errors="coerce")
    if p_values.isna().any() or ((p_values < 0.0) | (p_values > 1.0)).any():
        raise ContractError("validation_p_value values must be in [0, 1]")
    count = len(p_values)
    if count == 1 and method == "none_if_single":
        return p_values.astype(float)
    if method == "none_if_single":
        raise ContractError("none_if_single is only valid for one candidate")
    if method == "bonferroni":
        return (p_values * count).clip(upper=1.0)

    order = p_values.sort_values(kind="mergesort").index.tolist()
    adjusted = pd.Series(index=p_values.index, dtype=float)
    if method == "holm":
        running = 0.0
        for rank, index in enumerate(order):
            candidate = float(p_values.loc[index]) * (count - rank)
            running = max(running, candidate)
            adjusted.loc[index] = min(running, 1.0)
        return adjusted
    if method == "fdr_bh":
        running = 1.0
        for reverse_rank, index in enumerate(reversed(order), start=1):
            rank = count - reverse_rank + 1
            candidate = float(p_values.loc[index]) * count / rank
            running = min(running, candidate)
            adjusted.loc[index] = min(running, 1.0)
        return adjusted
    raise ContractError(f"unsupported multiple-testing correction: {method}")


def select_frozen_candidate(
    candidate_scores: pd.DataFrame, config: dict[str, Any]
) -> dict[str, Any]:
    required = {
        "candidate_id",
        "parameters_json",
        "train_score",
        "validation_score",
        "validation_p_value",
        "split_roles_used",
    }
    missing = sorted(required.difference(candidate_scores.columns))
    if missing:
        raise ContractError(f"candidate registry missing fields: {missing}")
    if candidate_scores.empty:
        raise ContractError("candidate registry is empty")
    if len(candidate_scores) > int(config["tuning"]["maximum_candidate_count"]):
        raise ContractError("candidate registry exceeds the frozen tuning budget")
    if candidate_scores["candidate_id"].astype(str).duplicated().any():
        raise ContractError("candidate_id values must be unique")
    for value in candidate_scores["split_roles_used"]:
        assert_tuning_inputs(str(value).split("+"))
    forbidden = [
        column
        for column in candidate_scores.columns
        if re.search(r"(^|_)test(_|$)|holdout|independent", str(column), re.I)
    ]
    if forbidden:
        raise ContractError(
            f"candidate registry contains holdout-derived fields: {forbidden}"
        )
    scores = candidate_scores.copy()
    for column in ("train_score", "validation_score", "validation_p_value"):
        scores[column] = pd.to_numeric(scores[column], errors="coerce")
    if scores[["train_score", "validation_score", "validation_p_value"]].isna().any().any():
        raise ContractError("candidate registry contains invalid scores")
    correction = str(config["tuning"]["multiple_testing_correction"])
    alpha = float(config["tuning"]["familywise_alpha"])
    scores["adjusted_validation_p_value"] = _adjust_p_values(
        scores["validation_p_value"], correction
    )
    scores["survives_multiple_testing"] = (
        scores["adjusted_validation_p_value"] <= alpha
    )
    eligible = scores[scores["survives_multiple_testing"]].copy()
    if eligible.empty:
        raise ContractError(
            "no candidate survives the frozen multiple-testing correction"
        )
    selected = eligible.sort_values(
        ["validation_score", "train_score", "candidate_id"],
        ascending=[False, False, True],
    ).iloc[0]
    try:
        parameters = json.loads(str(selected["parameters_json"]))
    except json.JSONDecodeError as exc:
        raise ContractError("selected parameters_json is invalid") from exc
    return {
        "candidate_id": str(selected["candidate_id"]),
        "parameters": parameters,
        "train_score": float(selected["train_score"]),
        "validation_score": float(selected["validation_score"]),
        "validation_p_value": float(selected["validation_p_value"]),
        "adjusted_validation_p_value": float(
            selected["adjusted_validation_p_value"]
        ),
        "familywise_alpha": alpha,
        "selection_data_roles": sorted(ALLOWED_TUNING_ROLES),
        "multiple_testing_correction": correction,
        "candidate_count": len(scores),
        "surviving_candidate_count": len(eligible),
        "candidate_audit": scores[
            [
                "candidate_id",
                "validation_p_value",
                "adjusted_validation_p_value",
                "survives_multiple_testing",
            ]
        ].sort_values("candidate_id").to_dict(orient="records"),
        "frozen_before_independent_test": True,
        "promotion_allowed": False,
    }


def consume_independent_test_once(
    ledger_path: str | Path,
    *,
    run_id: str,
    plan_sha256: str,
    data_manifest_sha256: str,
) -> dict[str, Any]:
    path = Path(ledger_path)
    if not SAFE_ID_RE.fullmatch(run_id):
        raise ContractError(f"unsafe independent-test run_id: {run_id!r}")
    if not SHA256_RE.fullmatch(plan_sha256):
        raise ContractError("invalid walk-forward plan hash")
    if not SHA256_RE.fullmatch(data_manifest_sha256):
        raise ContractError("invalid independent-test data manifest hash")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": WALK_FORWARD_SCHEMA_VERSION,
        "run_id": run_id,
        "purpose": "FINAL_EVALUATION_ONLY",
        "plan_sha256": plan_sha256,
        "data_manifest_sha256": data_manifest_sha256,
        "consumed": True,
        "tuning_use_allowed": False,
        "promotion_allowed": False,
    }
    try:
        with path.open("xb") as handle:
            handle.write(canonical_json_bytes(payload))
    except FileExistsError as exc:
        raise ContractError("independent test has already been consumed") from exc
    return payload


def verify_pit_gate_binding(
    run_directory: str | Path, project_root: str | Path
) -> dict[str, Any]:
    directory = Path(run_directory)
    root = Path(project_root).resolve()
    manifest_path = directory / "pit_gate_manifest.json"
    seal_path = directory / "pit_gate_manifest_seal.json"
    if not manifest_path.is_file() or not seal_path.is_file():
        raise ContractError("PIT Gate manifest or seal is missing")
    manifest = _read_json(manifest_path)
    seal = _read_json(seal_path)
    actual_hash = sha256_file(manifest_path)
    if seal.get("pit_gate_manifest_sha256") != actual_hash:
        raise ContractError("PIT Gate manifest hash mismatch")
    if manifest.get("run_id") != seal.get("run_id"):
        raise ContractError("PIT Gate run_id mismatch")
    if (
        manifest.get("status") != "PASS_PIT_GATE"
        or manifest.get("formal_backtest_allowed") is not True
    ):
        raise ContractError("formal walk-forward is blocked by the PIT Gate")
    if manifest.get("promotion_allowed") is not False:
        raise ContractError("PIT Gate must not promote a model")

    _verify_project_file_binding(root, manifest["target_manifest"], "target manifest")
    usage_binding = manifest.get("point_in_time_usage")
    if not isinstance(usage_binding, dict):
        raise ContractError("PIT usage ledger binding is missing")
    _verify_project_file_binding(root, usage_binding, "PIT usage ledger")
    formal_inputs = manifest.get("formal_inputs")
    if not isinstance(formal_inputs, list) or not formal_inputs:
        raise ContractError("formal input bindings are missing")
    formal_roles = {
        "validation_execution_states",
        "validation_target_weights",
        "validation_benchmark_returns",
        "independent_execution_states",
        "independent_target_weights",
        "independent_benchmark_returns",
        "trading_calendar",
        "candidate_registry",
    }
    actual_roles = {
        str(item.get("role", "")).strip()
        for item in formal_inputs
        if isinstance(item, dict)
    }
    if actual_roles != formal_roles or len(formal_inputs) != len(formal_roles):
        raise ContractError("formal input binding roles are incomplete")
    for item in formal_inputs:
        _verify_project_file_binding(
            root, item, f"formal input {item.get('role')}"
        )
    for dataset in manifest.get("input_datasets", []):
        _verify_project_file_binding(
            root,
            {
                "path": dataset.get("file_path"),
                "sha256": dataset.get("file_sha256"),
            },
            f"dataset file {dataset.get('dataset_id')}",
        )
        _verify_project_file_binding(
            root,
            {
                "path": dataset.get("manifest_path"),
                "sha256": dataset.get("manifest_sha256"),
            },
            f"dataset manifest {dataset.get('dataset_id')}",
        )
    for generation_label in ("target_generation", "gate_generation"):
        generation = manifest.get(generation_label)
        if not isinstance(generation, dict):
            raise ContractError(f"{generation_label} binding is missing")
        for item in generation.get("code_files", []):
            _verify_project_file_binding(
                root, item, f"{generation_label} code"
            )
        _verify_project_file_binding(
            root,
            generation.get("config", {}), f"{generation_label} config"
        )
    for output in manifest.get("outputs", []):
        relative = str(output.get("path", ""))
        expected = str(output.get("sha256", "")).lower()
        path = (directory / relative).resolve()
        if directory.resolve() != path and directory.resolve() not in path.parents:
            raise ContractError("PIT Gate output path escapes run directory")
        if (
            not path.is_file()
            or SHA256_RE.fullmatch(expected) is None
            or sha256_file(path) != expected
        ):
            raise ContractError(f"PIT Gate output hash mismatch: {relative}")
    return {
        "pit_gate_run_id": str(manifest["run_id"]),
        "pit_gate_manifest_sha256": actual_hash,
        "target_manifest_sha256": str(manifest["target_manifest"]["sha256"]),
        "point_in_time_usage": usage_binding,
        "formal_input_bindings": formal_inputs,
        "input_datasets": manifest["input_datasets"],
        "promotion_allowed": False,
    }


def normalize_execution_states(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = sorted(EXECUTION_STATE_COLUMNS.difference(prices.columns))
    if missing:
        raise ContractError(f"walk-forward prices missing execution fields: {missing}")
    state = prices.copy()
    state["date"] = pd.to_datetime(state["date"], errors="coerce").dt.normalize()
    state["available_date"] = pd.to_datetime(
        state["available_date"], errors="coerce"
    ).dt.normalize()
    if state[["date", "available_date"]].isna().any().any():
        raise ContractError("execution states contain invalid dates")
    if (state["available_date"] > state["date"]).any():
        raise ContractError("execution states contain future available_date values")
    if state.duplicated(["date", "asset"]).any():
        raise ContractError("execution states contain duplicate date-asset rows")
    boolean_columns = [
        "has_market_data",
        "is_suspended",
        "is_limit_up",
        "is_limit_down",
        "is_delisted",
    ]
    for column in boolean_columns:
        if not state[column].map(lambda value: isinstance(value, bool)).all():
            raise ContractError(f"execution state {column} must be boolean")
    delist_dates = pd.to_datetime(
        state["delist_date"], errors="coerce"
    ).dt.normalize()
    stated_delisted = state["is_delisted"]
    expected_delisted = delist_dates.notna() & state["date"].ge(delist_dates)
    if (stated_delisted != expected_delisted).any():
        raise ContractError(
            "is_delisted must match the asset-level delist_date boundary"
        )
    contradictory = (
        state["is_suspended"]
        & (state["is_limit_up"] | state["is_limit_down"])
    )
    if contradictory.any():
        raise ContractError("suspended rows cannot also declare price-limit locks")
    state["is_tradable"] = (
        state["has_market_data"]
        & ~state["is_suspended"]
        & ~state["is_limit_up"]
        & ~state["is_limit_down"]
        & ~state["is_delisted"]
    )
    numeric = state[["open", "close"]].apply(pd.to_numeric, errors="coerce")
    valid_market = (
        ~numeric.isna().any(axis=1)
        & (numeric["open"] > 0)
        & (numeric["close"] > 0)
    )
    if (state["has_market_data"] & ~valid_market).any():
        raise ContractError("market-data rows require positive open and close")
    state["open"] = numeric["open"]
    state["close"] = numeric["close"]
    backtest_prices = state[state["has_market_data"]].copy()
    return (
        backtest_prices.sort_values(["date", "asset"]).reset_index(drop=True),
        state.sort_values(["date", "asset"]).reset_index(drop=True),
    )


def _unfilled_reason(attempt: Any) -> str:
    if not bool(attempt.has_market_data):
        return "missing_market_data"
    if bool(attempt.is_suspended):
        return "suspended"
    if bool(attempt.is_limit_up):
        return "limit_up_conservative_no_trade"
    if bool(attempt.is_limit_down):
        return "limit_down_conservative_no_trade"
    if bool(attempt.is_delisted):
        return "delisted"
    return "not_executable"


def build_order_attempt_audit(
    execution_states: pd.DataFrame,
    fills: pd.DataFrame,
    pending_targets: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "order_id",
        "signal_date",
        "attempt_date",
        "fill_date",
        "asset",
        "sleeve",
        "side",
        "status",
        "reason",
        "notional",
        "cost",
    ]
    states = execution_states.copy()
    states["date"] = pd.to_datetime(states["date"]).dt.normalize()
    rows: list[dict[str, Any]] = []
    for fill in fills.itertuples(index=False):
        signal_date = pd.Timestamp(fill.signal_date).normalize()
        fill_date = pd.Timestamp(fill.execution_date).normalize()
        order_key = (
            f"{signal_date.date()}|{fill.asset}|{fill.sleeve}|{fill.side}|"
            f"{fill_date.date()}|{float(fill.notional):.12f}"
        )
        order_id = hashlib.sha256(order_key.encode("utf-8")).hexdigest()[:24]
        attempts = states[
            states["asset"].astype(str).eq(str(fill.asset))
            & states["date"].gt(signal_date)
            & states["date"].le(fill_date)
        ].sort_values("date")
        for attempt in attempts.itertuples(index=False):
            attempt_date = pd.Timestamp(attempt.date).normalize()
            is_fill = attempt_date == fill_date
            if is_fill:
                reason = "executed_at_asset_level_next_tradable_open"
            else:
                reason = _unfilled_reason(attempt)
            rows.append(
                {
                    "order_id": order_id,
                    "signal_date": signal_date,
                    "attempt_date": attempt_date,
                    "fill_date": fill_date if is_fill else pd.NaT,
                    "asset": str(fill.asset),
                    "sleeve": str(fill.sleeve),
                    "side": str(fill.side),
                    "status": "FILLED" if is_fill else "UNFILLED",
                    "reason": reason,
                    "notional": float(fill.notional) if is_fill else 0.0,
                    "cost": float(fill.cost) if is_fill else 0.0,
                }
            )
    if not pending_targets.empty:
        for target in pending_targets.itertuples(index=False):
            signal_date = pd.Timestamp(target.signal_date).normalize()
            order_key = (
                f"{signal_date.date()}|{target.asset}|portfolio|rebalance|PENDING"
            )
            order_id = hashlib.sha256(order_key.encode("utf-8")).hexdigest()[:24]
            attempts = states[
                states["asset"].astype(str).eq(str(target.asset))
                & states["date"].gt(signal_date)
            ].sort_values("date")
            if attempts.empty:
                rows.append(
                    {
                        "order_id": order_id,
                        "signal_date": signal_date,
                        "attempt_date": pd.NaT,
                        "fill_date": pd.NaT,
                        "asset": str(target.asset),
                        "sleeve": "portfolio",
                        "side": "rebalance",
                        "status": "PENDING_UNFILLED",
                        "reason": "no_asset_state_after_signal",
                        "notional": 0.0,
                        "cost": 0.0,
                    }
                )
                continue
            for attempt in attempts.itertuples(index=False):
                rows.append(
                    {
                        "order_id": order_id,
                        "signal_date": signal_date,
                        "attempt_date": pd.Timestamp(attempt.date).normalize(),
                        "fill_date": pd.NaT,
                        "asset": str(target.asset),
                        "sleeve": "portfolio",
                        "side": "rebalance",
                        "status": "PENDING_UNFILLED",
                        "reason": _unfilled_reason(attempt),
                        "notional": 0.0,
                        "cost": 0.0,
                    }
                )
    return pd.DataFrame(rows, columns=columns)


def build_sleeve_attribution(
    nav: pd.DataFrame, fills: pd.DataFrame, *, initial_cash: float
) -> pd.DataFrame:
    required_nav = {"date", "nav", "core_value", "t_value"}
    missing = sorted(required_nav.difference(nav.columns))
    if missing:
        raise ContractError(f"NAV is missing attribution fields: {missing}")
    result = nav.copy()
    result["date"] = pd.to_datetime(result["date"]).dt.normalize()
    for sleeve in ("core", "t"):
        sleeve_fills = fills[fills["sleeve"].eq(sleeve)].copy() if not fills.empty else fills.copy()
        if sleeve_fills.empty:
            flow = pd.DataFrame(
                {
                    "date": result["date"],
                    f"{sleeve}_buy_notional": 0.0,
                    f"{sleeve}_sell_notional": 0.0,
                    f"{sleeve}_trading_cost": 0.0,
                }
            )
        else:
            sleeve_fills["date"] = pd.to_datetime(
                sleeve_fills["execution_date"]
            ).dt.normalize()
            sleeve_fills["buy_notional"] = sleeve_fills["notional"].where(
                sleeve_fills["side"].eq("buy"), 0.0
            )
            sleeve_fills["sell_notional"] = sleeve_fills["notional"].where(
                sleeve_fills["side"].eq("sell"), 0.0
            )
            flow = (
                sleeve_fills.groupby("date", as_index=False)
                .agg(
                    buy_notional=("buy_notional", "sum"),
                    sell_notional=("sell_notional", "sum"),
                    trading_cost=("cost", "sum"),
                )
                .rename(
                    columns={
                        "buy_notional": f"{sleeve}_buy_notional",
                        "sell_notional": f"{sleeve}_sell_notional",
                        "trading_cost": f"{sleeve}_trading_cost",
                    }
                )
            )
        result = result.merge(flow, on="date", how="left")
        for column in (
            f"{sleeve}_buy_notional",
            f"{sleeve}_sell_notional",
            f"{sleeve}_trading_cost",
        ):
            result[column] = result[column].fillna(0.0)
            result[f"cumulative_{column}"] = result[column].cumsum()
        value_column = "core_value" if sleeve == "core" else "t_value"
        result[f"{sleeve}_gross_pnl"] = (
            result[value_column]
            + result[f"cumulative_{sleeve}_sell_notional"]
            - result[f"cumulative_{sleeve}_buy_notional"]
        )
        result[f"{sleeve}_net_pnl"] = (
            result[f"{sleeve}_gross_pnl"]
            - result[f"cumulative_{sleeve}_trading_cost"]
        )

    result["core_only_nav"] = initial_cash + result["core_net_pnl"]
    result["t_gross_return"] = result["t_gross_pnl"] / initial_cash
    result["t_net_gain"] = result["t_net_pnl"]
    result["core_plus_t_nav"] = (
        initial_cash + result["core_net_pnl"] + result["t_net_pnl"]
    )
    result["nav_reconciliation_difference"] = (
        result["core_plus_t_nav"] - result["nav"]
    )
    if result["nav_reconciliation_difference"].abs().max() > 1e-7:
        raise ContractError("core/T attribution does not reconcile to account NAV")
    return result


def build_cost_scenarios(
    attribution: pd.DataFrame,
    fills: pd.DataFrame,
    *,
    initial_cash: float,
    additional_slippage_bps: Iterable[int],
) -> pd.DataFrame:
    if attribution.empty:
        raise ContractError("cost scenarios require non-empty attribution")
    final = attribution.iloc[-1]
    total_notional = float(fills["notional"].sum()) if not fills.empty else 0.0
    t_notional = (
        float(fills.loc[fills["sleeve"].eq("t"), "notional"].sum())
        if not fills.empty
        else 0.0
    )
    recorded_cost = float(fills["cost"].sum()) if not fills.empty else 0.0
    recorded_t_cost = (
        float(fills.loc[fills["sleeve"].eq("t"), "cost"].sum())
        if not fills.empty
        else 0.0
    )
    gross_total = float(final["core_gross_pnl"] + final["t_gross_pnl"])
    t_gross = float(final["t_gross_pnl"])
    rows: list[dict[str, Any]] = []
    for bps in additional_slippage_bps:
        extra_cost = total_notional * int(bps) / 10000.0
        extra_t_cost = t_notional * int(bps) / 10000.0
        combined_net = gross_total - recorded_cost - extra_cost
        t_net = t_gross - recorded_t_cost - extra_t_cost
        rows.append(
            {
                "additional_slippage_bps": int(bps),
                "recorded_trading_cost": recorded_cost,
                "additional_slippage_cost": extra_cost,
                "total_cost": recorded_cost + extra_cost,
                "core_plus_t_net_pnl": combined_net,
                "core_plus_t_return": combined_net / initial_cash,
                "t_gross_pnl": t_gross,
                "t_recorded_trading_cost": recorded_t_cost,
                "t_additional_slippage_cost": extra_t_cost,
                "t_net_gain": t_net,
                "promotion_allowed": False,
            }
        )
    return pd.DataFrame(rows)


def run_audited_window_backtest(
    prices: pd.DataFrame,
    targets: pd.DataFrame,
    strategy_config: dict[str, Any],
    walk_forward_config: dict[str, Any],
    *,
    initial_cash: float,
) -> dict[str, Any]:
    backtest_prices, execution_states = normalize_execution_states(prices)
    result = run_weight_backtest(
        backtest_prices,
        targets,
        strategy_config,
        initial_cash=initial_cash,
        mode="formal",
    )
    fills = result["trades"].copy()
    pending_targets = result["pending_targets"].copy()
    orders = build_order_attempt_audit(
        execution_states, fills, pending_targets
    )
    attribution = build_sleeve_attribution(
        result["nav"], fills, initial_cash=initial_cash
    )
    cost_scenarios = build_cost_scenarios(
        attribution,
        fills,
        initial_cash=initial_cash,
        additional_slippage_bps=walk_forward_config["cost_scenarios"][
            "additional_slippage_bps"
        ],
    )
    account = result["nav"][
        ["date", "cash", "core_value", "t_value", "nav"]
    ].copy()
    risk_exposures = result["nav"][
        [
            "date",
            "core_weight",
            "t_weight",
            "cash_weight",
            "holding_count",
            "one_way_turnover",
        ]
    ].copy()
    return {
        "target_weights": targets.copy(),
        "orders": orders,
        "fills": fills,
        "pending_targets": pending_targets,
        "account": account,
        "nav": result["nav"],
        "attribution": attribution,
        "cost_scenarios": cost_scenarios,
        "risk_exposures": risk_exposures,
        "metrics": {
            **result["metrics"],
            "promotion_allowed": False,
            "unfilled_targets_accrue_returns": False,
            "limit_execution_policy": "conservative_no_trade_on_limit_up_or_limit_down",
        },
        "input_hashes": {
            "execution_states_sha256": hash_dataframe(execution_states),
            "backtest_prices_sha256": hash_dataframe(backtest_prices),
            "target_weights_sha256": hash_dataframe(targets),
        },
    }


def write_window_bundle(
    output_root: str | Path,
    *,
    project_root: str | Path,
    pit_gate_run_directory: str | Path,
    run_id: str,
    window_id: str,
    split_role: str,
    context: dict[str, Any],
    artifacts: dict[str, Any],
) -> dict[str, Path]:
    if not SAFE_ID_RE.fullmatch(run_id) or not SAFE_ID_RE.fullmatch(window_id):
        raise ContractError("unsafe walk-forward run_id or window_id")
    if split_role not in {"validation", "independent_test"}:
        raise ContractError("window split_role must be validation or independent_test")
    required_context = {
        "pit_gate_run_id",
        "pit_gate_manifest_sha256",
        "target_manifest_sha256",
        "code_commit",
        "code_files",
        "config_bindings",
        "formal_input_bindings",
        "training_parameters",
        "data_manifest",
        "cost_assumptions",
    }
    missing_context = sorted(required_context.difference(context))
    if missing_context:
        raise ContractError(f"window context missing fields: {missing_context}")
    for field in (
        "pit_gate_manifest_sha256",
        "target_manifest_sha256",
    ):
        if SHA256_RE.fullmatch(str(context[field])) is None:
            raise ContractError(f"window context has invalid {field}")
    if re.fullmatch(r"[0-9a-f]{40}", str(context["code_commit"]).lower()) is None:
        raise ContractError("window context has invalid code_commit")
    code_files = context["code_files"]
    if not isinstance(code_files, list) or not code_files:
        raise ContractError("window context requires code_files")
    if any(
        not isinstance(item, dict)
        or not str(item.get("path", "")).strip()
        or SHA256_RE.fullmatch(str(item.get("sha256", "")).lower()) is None
        for item in code_files
    ):
        raise ContractError("window context has invalid code file binding")
    root = Path(project_root).resolve()
    gate_binding = verify_pit_gate_binding(pit_gate_run_directory, root)
    if (
        str(context["pit_gate_run_id"]) != gate_binding["pit_gate_run_id"]
        or str(context["pit_gate_manifest_sha256"])
        != gate_binding["pit_gate_manifest_sha256"]
        or str(context["target_manifest_sha256"])
        != gate_binding["target_manifest_sha256"]
    ):
        raise ContractError("window context does not match verified PIT Gate")
    for item in code_files:
        _verify_project_file_binding(root, item, "window code")
    config_bindings = context["config_bindings"]
    if not isinstance(config_bindings, list) or not config_bindings:
        raise ContractError("window context requires config bindings")
    for item in config_bindings:
        if not isinstance(item, dict) or not str(item.get("role", "")).strip():
            raise ContractError("window config binding is invalid")
        _verify_project_file_binding(
            root, item, f"window config {item.get('role')}"
        )
    formal_inputs = context["formal_input_bindings"]
    if not isinstance(formal_inputs, list) or not formal_inputs:
        raise ContractError("window context requires formal input bindings")
    for item in formal_inputs:
        if not isinstance(item, dict) or not str(item.get("role", "")).strip():
            raise ContractError("window formal input binding is invalid")
        _verify_project_file_binding(
            root, item, f"formal input {item.get('role')}"
        )
    if not isinstance(context["training_parameters"], dict):
        raise ContractError("window training_parameters must be an object")
    if not isinstance(context["data_manifest"], dict):
        raise ContractError("window data_manifest must be an object")
    if not isinstance(context["cost_assumptions"], dict):
        raise ContractError("window cost_assumptions must be an object")
    if context["data_manifest"] != artifacts["input_hashes"]:
        raise ContractError("window data manifest does not match audited inputs")
    if split_role == "independent_test":
        holdout_binding = context.get("holdout_consumption_binding")
        if not isinstance(holdout_binding, dict):
            raise ContractError(
                "independent-test window must bind holdout consumption"
            )
        _verify_project_file_binding(
            root, holdout_binding, "holdout consumption ledger"
        )

    required_artifacts = {
        "target_weights",
        "orders",
        "fills",
        "pending_targets",
        "account",
        "nav",
        "attribution",
        "cost_scenarios",
        "risk_exposures",
        "metrics",
        "input_hashes",
    }
    missing_artifacts = sorted(required_artifacts.difference(artifacts))
    if missing_artifacts:
        raise ContractError(f"window artifacts missing: {missing_artifacts}")

    root = Path(output_root)
    final = root / run_id / window_id
    temporary = root / run_id / f".{window_id}.tmp-{uuid.uuid4().hex}"
    if final.exists():
        raise ContractError(f"walk-forward window is immutable: {run_id}/{window_id}")
    temporary.mkdir(parents=True)
    outputs: list[dict[str, Any]] = []
    for name in (
        "target_weights",
        "orders",
        "fills",
        "pending_targets",
        "account",
        "nav",
        "attribution",
        "cost_scenarios",
        "risk_exposures",
    ):
        path = temporary / f"{name}.csv"
        artifacts[name].to_csv(path, index=False, encoding="utf-8")
        outputs.append(
            {"path": path.name, "sha256": sha256_file(path), "schema_version": 1}
        )
    for name, payload in (
        ("metrics", artifacts["metrics"]),
        ("input_hashes", artifacts["input_hashes"]),
        ("context", context),
    ):
        path = temporary / f"{name}.json"
        path.write_bytes(canonical_json_bytes(payload))
        outputs.append(
            {"path": path.name, "sha256": sha256_file(path), "schema_version": 1}
        )

    manifest = {
        "schema_version": WALK_FORWARD_SCHEMA_VERSION,
        "run_id": run_id,
        "window_id": window_id,
        "split_role": split_role,
        "pit_gate_run_id": context["pit_gate_run_id"],
        "pit_gate_manifest_sha256": context["pit_gate_manifest_sha256"],
        "target_manifest_sha256": context["target_manifest_sha256"],
        "code_commit": context["code_commit"],
        "code_files": context["code_files"],
        "config_bindings": context["config_bindings"],
        "formal_input_bindings": context["formal_input_bindings"],
        "training_parameters": context["training_parameters"],
        "data_manifest": context["data_manifest"],
        "cost_assumptions": context["cost_assumptions"],
        "outputs": outputs,
        "promotion_allowed": False,
        "manual_review_required": True,
        "manual_review_signed": False,
    }
    manifest_path = temporary / "window_manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    seal_path = temporary / "window_manifest_seal.json"
    seal_path.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": WALK_FORWARD_SCHEMA_VERSION,
                "run_id": run_id,
                "window_id": window_id,
                "window_manifest_sha256": sha256_file(manifest_path),
                "promotion_allowed": False,
            }
        )
    )
    temporary.rename(final)
    return {
        "directory": final,
        "manifest": final / manifest_path.name,
        "seal": final / seal_path.name,
    }


def promotion_decision(
    *,
    pit_gate_passed: bool = False,
    all_windows_completed: bool = False,
    independent_test_completed: bool = False,
    cost_adjusted_results_explained: bool = False,
    failed_windows_disclosed: bool = False,
    manual_review_signed: bool = False,
) -> dict[str, Any]:
    conditions = {
        "pit_gate_passed": bool(pit_gate_passed),
        "all_windows_completed": bool(all_windows_completed),
        "independent_test_completed": bool(independent_test_completed),
        "cost_adjusted_results_explained": bool(
            cost_adjusted_results_explained
        ),
        "failed_windows_disclosed": bool(failed_windows_disclosed),
        "manual_review_signed": bool(manual_review_signed),
    }
    blockers = sorted(key for key, value in conditions.items() if not value)
    if not blockers:
        blockers.append("manual_promotion_action_required")
    return {
        "schema_version": WALK_FORWARD_SCHEMA_VERSION,
        "conditions": conditions,
        "promotion_allowed": False,
        "automatic_promotion_allowed": False,
        "promotion_blocking_reasons": blockers,
        "evidence_status": (
            "BLOCKED" if blockers != ["manual_promotion_action_required"] else "REVIEW_PENDING"
        ),
    }


def build_bias_audit(
    *,
    historical_universe_verified: bool,
    available_dates_verified: bool,
    tuning_split_roles: Iterable[str],
    independent_test_access_count: int,
    registered_candidate_count: int,
    maximum_candidate_count: int,
    multiple_testing_correction: str,
    adjusted_p_values_verified: bool,
    candidate_registry_frozen_before_holdout: bool,
) -> dict[str, Any]:
    roles = {str(value).strip() for value in tuning_split_roles}
    checks = {
        "survivorship_bias": {
            "passed": bool(historical_universe_verified),
            "requirement": "historical constituents and lifecycle assets only",
        },
        "lookahead": {
            "passed": bool(available_dates_verified),
            "requirement": "available_date <= decision_date for every input row",
        },
        "repeated_tuning": {
            "passed": roles.issubset(ALLOWED_TUNING_ROLES)
            and int(independent_test_access_count) <= 1,
            "requirement": "tuning uses train/validation only; independent test is evaluated once",
        },
        "multiple_testing": {
            "passed": 1
            <= int(registered_candidate_count)
            <= int(maximum_candidate_count)
            and multiple_testing_correction
            in {"holm", "bonferroni", "fdr_bh", "none_if_single"},
            "requirement": "candidate family and correction method are registered",
        },
        "multiple_testing_applied": {
            "passed": bool(adjusted_p_values_verified),
            "requirement": "adjusted validation p-values were computed and verified",
        },
        "candidate_registry_frozen": {
            "passed": bool(candidate_registry_frozen_before_holdout),
            "requirement": "candidate registry was sealed before holdout consumption",
        },
    }
    failures = sorted(key for key, value in checks.items() if not value["passed"])
    return {
        "schema_version": WALK_FORWARD_SCHEMA_VERSION,
        "checks": checks,
        "passed": not failures,
        "failures": failures,
        "promotion_allowed": False,
    }


def build_window_status_registry(
    plan: dict[str, Any],
    *,
    completed_window_ids: Iterable[str] = (),
    failed_windows: dict[str, str] | None = None,
    blocked_reason: str | None = None,
) -> pd.DataFrame:
    completed = set(str(value) for value in completed_window_ids)
    failures = {str(key): str(value) for key, value in (failed_windows or {}).items()}
    windows = [
        {
            "window_id": fold["window_id"],
            "split_role": "validation",
            "start": fold["validation_start"],
            "end": fold["validation_end"],
        }
        for fold in plan["validation_windows"]
    ]
    test = plan["independent_test"]
    windows.append(
        {
            "window_id": test["window_id"],
            "split_role": "independent_test",
            "start": test["start"],
            "end": test["end"],
        }
    )
    rows: list[dict[str, Any]] = []
    for window in windows:
        window_id = window["window_id"]
        if window_id in failures:
            status = "FAILED"
            reason = failures[window_id]
        elif window_id in completed:
            status = "COMPLETED"
            reason = ""
        elif blocked_reason:
            status = "BLOCKED_NOT_RUN"
            reason = blocked_reason
        else:
            status = "PENDING"
            reason = ""
        rows.append(
            {
                **window,
                "status": status,
                "reason": reason,
                "promotion_allowed": False,
            }
        )
    return pd.DataFrame(rows)
