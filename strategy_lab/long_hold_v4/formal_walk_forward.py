"""Governed entry point for formal Long Hold V4 walk-forward evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import uuid
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

from strategy_lab.backtest_overfit_pbo import cscv_pbo

from .core import ContractError, load_config
from .pit_gate_v2 import canonical_json_bytes, sha256_file
from .walk_forward import (
    assert_tuning_inputs,
    build_bias_audit,
    build_purged_embargoed_plan,
    build_window_status_registry,
    consume_independent_test_once,
    load_walk_forward_config,
    run_audited_window_backtest,
    select_frozen_candidate,
    verify_pit_gate_binding,
    write_window_bundle,
)


FORMAL_RUN_SCHEMA_VERSION = 1
RUNNER_CODE_PATHS = (
    "strategy_lab/long_hold_v4/formal_walk_forward.py",
    "strategy_lab/long_hold_v4/walk_forward.py",
    "strategy_lab/long_hold_v4/pit_gate_v2.py",
    "strategy_lab/long_hold_v4/backtest.py",
    "strategy_lab/long_hold_v4/core.py",
    "strategy_lab/backtest_overfit_pbo.py",
)


def _git_head(root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    commit = completed.stdout.strip().lower()
    if completed.returncode != 0 or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ContractError("formal walk-forward requires a verifiable Git commit")
    return commit


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ContractError(f"formal input is outside project root: {path}") from exc


def _binding(root: Path, path: Path, role: str) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ContractError(f"formal {role} file is missing: {resolved}")
    return {
        "role": role,
        "path": _relative(root, resolved),
        "sha256": sha256_file(resolved),
    }


def _verify_binding(
    root: Path, binding: dict[str, Any], label: str
) -> Path:
    path = (root / str(binding.get("path", ""))).resolve()
    if root != path and root not in path.parents:
        raise ContractError(f"{label} path escapes project root")
    expected = str(binding.get("sha256", "")).lower()
    if (
        not path.is_file()
        or re.fullmatch(r"[0-9a-f]{64}", expected) is None
        or sha256_file(path) != expected
    ):
        raise ContractError(f"{label} hash mismatch")
    return path


def _read_bound_inputs(
    root: Path, gate_binding: dict[str, Any]
) -> tuple[dict[str, Path], list[dict[str, str]]]:
    bindings = gate_binding["formal_input_bindings"]
    paths: dict[str, Path] = {}
    normalized: list[dict[str, str]] = []
    for item in bindings:
        role = str(item["role"])
        path = (root / str(item["path"])).resolve()
        paths[role] = path
        normalized.append(
            {
                "role": role,
                "path": _relative(root, path),
                "sha256": str(item["sha256"]),
            }
        )
    return paths, sorted(normalized, key=lambda item: item["role"])


def _load_calendar(path: Path) -> pd.DatetimeIndex:
    frame = pd.read_csv(path, dtype=str)
    if set(frame.columns) != {"date"}:
        raise ContractError("trading calendar must contain only the date column")
    dates = pd.to_datetime(frame["date"], errors="coerce")
    if dates.isna().any() or dates.duplicated().any():
        raise ContractError("trading calendar contains invalid or duplicate dates")
    dates = pd.DatetimeIndex(dates.dt.normalize())
    if not dates.is_monotonic_increasing:
        raise ContractError("trading calendar must be strictly increasing")
    return dates


def _load_benchmark_returns(
    path: Path, walk_forward_config: dict[str, Any]
) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str)
    required = {
        "date",
        "benchmark_id",
        "total_return",
        "available_date",
        "return_basis",
        "historical_backtest_allowed",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ContractError(
            f"benchmark returns missing columns: {missing}"
        )
    if frame.empty:
        raise ContractError("benchmark returns are empty")
    frame["date"] = pd.to_datetime(
        frame["date"], errors="coerce"
    ).dt.normalize()
    frame["available_date"] = pd.to_datetime(
        frame["available_date"], errors="coerce"
    ).dt.normalize()
    frame["total_return"] = pd.to_numeric(
        frame["total_return"], errors="coerce"
    )
    expected_id = str(
        walk_forward_config["benchmark"]["benchmark_id"]
    )
    allowed = (
        frame["historical_backtest_allowed"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("true")
    )
    if (
        frame[["date", "available_date", "total_return"]]
        .isna()
        .any()
        .any()
        or frame["date"].duplicated().any()
        or not frame["benchmark_id"].astype(str).eq(expected_id).all()
        or not frame["return_basis"].astype(str).eq("total_return").all()
        or not allowed.all()
        or (frame["available_date"] > frame["date"]).any()
        or (frame["total_return"] <= -1.0).any()
        or not np.isfinite(frame["total_return"]).all()
    ):
        raise ContractError(
            "benchmark returns violate the PIT total-return contract"
        )
    return frame.sort_values("date").reset_index(drop=True)


def _benchmark_for_dates(
    benchmark_returns: pd.DataFrame, dates: pd.Series
) -> pd.Series:
    normalized = pd.to_datetime(dates, errors="coerce").dt.normalize()
    lookup = benchmark_returns.set_index("date")["total_return"]
    matched = normalized.map(lookup)
    if normalized.isna().any() or matched.isna().any():
        raise ContractError(
            "benchmark returns do not cover every strategy valuation date"
        )
    return matched.astype(float)


def _window_frames(
    prices: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    window_id: str,
    start: str,
    end: str,
    candidate_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    price_dates = pd.to_datetime(prices["date"], errors="coerce").dt.normalize()
    signal_dates = pd.to_datetime(
        targets["signal_date"], errors="coerce"
    ).dt.normalize()
    start_date = pd.Timestamp(start)
    end_date = pd.Timestamp(end)
    window_prices = prices[
        price_dates.between(start_date, end_date, inclusive="both")
    ].copy()
    window_targets = targets[
        targets["window_id"].astype(str).eq(window_id)
        & targets["candidate_id"].astype(str).eq(candidate_id)
        & signal_dates.between(start_date, end_date, inclusive="both")
    ].copy()
    if window_prices.empty:
        raise ContractError(f"{window_id} has no execution-state rows")
    if window_targets.empty:
        raise ContractError(f"{window_id} has no frozen-candidate targets")
    return window_prices, window_targets


def _validate_target_window_contract(
    targets: pd.DataFrame,
    windows: list[dict[str, Any]],
    *,
    split_role: str,
) -> None:
    required = {"window_id", "signal_date"}
    missing = sorted(required.difference(targets.columns))
    if missing:
        raise ContractError(
            f"{split_role} targets missing window columns: {missing}"
        )
    bounds = {
        str(window["window_id"]): (
            pd.Timestamp(
                window.get("validation_start", window.get("start"))
            ),
            pd.Timestamp(window.get("validation_end", window.get("end"))),
        )
        for window in windows
    }
    actual_ids = set(targets["window_id"].astype(str))
    unknown = sorted(actual_ids.difference(bounds))
    missing_ids = sorted(set(bounds).difference(actual_ids))
    if unknown or missing_ids:
        raise ContractError(
            f"{split_role} target windows mismatch: "
            f"unknown={unknown};missing={missing_ids}"
        )
    signal_dates = pd.to_datetime(
        targets["signal_date"], errors="coerce"
    ).dt.normalize()
    invalid = signal_dates.isna()
    for window_id, (start, end) in bounds.items():
        belongs = targets["window_id"].astype(str).eq(window_id)
        invalid |= belongs & ~signal_dates.between(
            start, end, inclusive="both"
        )
    if invalid.any():
        raise ContractError(
            f"{split_role} target rows fall outside their declared windows"
        )


def _validate_candidate_target_coverage(
    targets: pd.DataFrame,
    candidate_ids: list[str],
    windows: list[dict[str, Any]],
) -> None:
    expected = {
        (candidate_id, str(window["window_id"]))
        for candidate_id in candidate_ids
        for window in windows
    }
    actual = set(
        targets[["candidate_id", "window_id"]]
        .astype(str)
        .itertuples(index=False, name=None)
    )
    missing = sorted(expected.difference(actual))
    extra = sorted(actual.difference(expected))
    if missing or extra:
        raise ContractError(
            "candidate target coverage is incomplete: "
            f"missing={missing[:10]};extra={extra[:10]}"
        )


def _daily_nav_returns(
    nav: pd.DataFrame, initial_cash: float
) -> pd.Series:
    values = pd.to_numeric(nav["nav"], errors="coerce")
    if values.isna().any() or (values <= 0).any():
        raise ContractError("candidate validation NAV contains invalid values")
    previous = values.shift(1)
    previous.iloc[0] = float(initial_cash)
    returns = values / previous - 1.0
    if not np.isfinite(returns).all():
        raise ContractError("candidate validation returns are not finite")
    return returns.astype(float)


def _cost_stressed_returns(
    nav: pd.DataFrame,
    fills: pd.DataFrame,
    *,
    initial_cash: float,
    additional_slippage_bps: int,
) -> pd.Series:
    values = pd.to_numeric(nav["nav"], errors="coerce")
    dates = pd.to_datetime(nav["date"], errors="coerce").dt.normalize()
    if values.isna().any() or dates.isna().any():
        raise ContractError("cost-stressed NAV contains invalid rows")
    additional_cost = pd.Series(0.0, index=nav.index)
    if not fills.empty:
        fill_dates = pd.to_datetime(
            fills["execution_date"], errors="coerce"
        ).dt.normalize()
        notionals = pd.to_numeric(fills["notional"], errors="coerce")
        if fill_dates.isna().any() or notionals.isna().any():
            raise ContractError("candidate fills contain invalid cost inputs")
        daily_cost = (
            pd.DataFrame(
                {
                    "date": fill_dates,
                    "extra_cost": notionals
                    * int(additional_slippage_bps)
                    / 10000.0,
                }
            )
            .groupby("date")["extra_cost"]
            .sum()
        )
        additional_cost = dates.map(daily_cost).fillna(0.0)
    stressed_nav = values - additional_cost.cumsum()
    if (stressed_nav <= 0).any():
        raise ContractError("cost stress makes candidate NAV non-positive")
    previous = stressed_nav.shift(1)
    previous.iloc[0] = float(initial_cash)
    returns = stressed_nav / previous - 1.0
    if not np.isfinite(returns).all():
        raise ContractError("cost-stressed candidate returns are not finite")
    return returns.astype(float)


def _return_statistics(returns: pd.Series) -> dict[str, float]:
    clean = returns.dropna().astype(float)
    if clean.empty or (clean <= -1.0).any():
        raise ContractError("performance returns are invalid")
    total_return = float((1.0 + clean).prod() - 1.0)
    annualized_return = float(
        (1.0 + total_return) ** (252.0 / len(clean)) - 1.0
    )
    volatility = float(clean.std(ddof=1) * math.sqrt(252.0))
    sharpe = (
        float(clean.mean() / clean.std(ddof=1) * math.sqrt(252.0))
        if len(clean) > 1 and float(clean.std(ddof=1)) > 0
        else float("nan")
    )
    nav = (1.0 + clean).cumprod()
    maximum_drawdown = float((nav / nav.cummax() - 1.0).min())
    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": volatility,
        "sharpe": sharpe,
        "maximum_drawdown": maximum_drawdown,
    }


def _block_normal_p_value(
    returns: pd.Series, block_sessions: int
) -> tuple[float, int]:
    clean = returns.dropna().astype(float)
    blocks = (
        (1.0 + clean)
        .groupby(np.arange(len(clean)) // int(block_sessions))
        .prod()
        .sub(1.0)
    )
    if len(blocks) < 8:
        return 1.0, len(blocks)
    standard_deviation = float(blocks.std(ddof=1))
    mean_return = float(blocks.mean())
    if not math.isfinite(standard_deviation) or standard_deviation <= 0:
        return (0.0 if mean_return > 0 else 1.0), len(blocks)
    z_score = mean_return / (
        standard_deviation / math.sqrt(len(blocks))
    )
    return (
        float(0.5 * math.erfc(z_score / math.sqrt(2.0))),
        len(blocks),
    )


def _deflated_sharpe_probability(
    returns: pd.Series,
    *,
    trial_count: int,
    trial_sharpe_standard_deviation: float,
) -> tuple[float, float, float]:
    clean = returns.dropna().astype(float)
    if len(clean) < 3:
        return 0.0, float("nan"), float("nan")
    volatility = float(clean.std(ddof=1))
    if not math.isfinite(volatility) or volatility <= 0:
        probability = 1.0 if float(clean.mean()) > 0 else 0.0
        return probability, float("inf"), 0.0
    daily_sharpe = float(clean.mean() / volatility)
    benchmark = 0.0
    if trial_count > 1 and trial_sharpe_standard_deviation > 0:
        euler_gamma = 0.5772156649015329
        normal = NormalDist()
        benchmark = trial_sharpe_standard_deviation * (
            (1.0 - euler_gamma)
            * normal.inv_cdf(1.0 - 1.0 / trial_count)
            + euler_gamma
            * normal.inv_cdf(1.0 - 1.0 / (trial_count * math.e))
        )
    skewness = float(clean.skew()) if len(clean) >= 3 else 0.0
    pearson_kurtosis = (
        float(clean.kurt()) + 3.0 if len(clean) >= 4 else 3.0
    )
    variance_term = (
        1.0
        - skewness * daily_sharpe
        + ((pearson_kurtosis - 1.0) / 4.0) * daily_sharpe**2
    )
    if not math.isfinite(variance_term) or variance_term <= 0:
        return 0.0, daily_sharpe, benchmark
    z_score = (
        (daily_sharpe - benchmark)
        * math.sqrt(len(clean) - 1)
        / math.sqrt(variance_term)
    )
    return float(NormalDist().cdf(z_score)), daily_sharpe, benchmark


def _evaluate_validation_candidates(
    prices: pd.DataFrame,
    targets: pd.DataFrame,
    benchmark_returns: pd.DataFrame,
    registry: pd.DataFrame,
    plan: dict[str, Any],
    strategy_config: dict[str, Any],
    walk_forward_config: dict[str, Any],
    *,
    initial_cash: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    forbidden_claims = sorted(
        {"validation_score", "validation_p_value"}.intersection(
            registry.columns
        )
    )
    if forbidden_claims:
        raise ContractError(
            "candidate registry must not contain self-reported validation "
            f"fields: {forbidden_claims}"
        )
    candidate_ids = registry["candidate_id"].astype(str).tolist()
    selection_cost_bps = max(
        int(value)
        for value in walk_forward_config["cost_scenarios"][
            "additional_slippage_bps"
        ]
    )
    _validate_candidate_target_coverage(
        targets, candidate_ids, plan["validation_windows"]
    )
    return_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        candidate_returns: list[pd.DataFrame] = []
        for fold in plan["validation_windows"]:
            window_prices, window_targets = _window_frames(
                prices,
                targets,
                window_id=str(fold["window_id"]),
                start=str(fold["validation_start"]),
                end=str(fold["validation_end"]),
                candidate_id=candidate_id,
            )
            artifacts = run_audited_window_backtest(
                window_prices,
                window_targets,
                strategy_config,
                walk_forward_config,
                initial_cash=initial_cash,
            )
            nav = artifacts["nav"].copy()
            recorded_returns = _daily_nav_returns(nav, initial_cash)
            selection_returns = _cost_stressed_returns(
                nav,
                artifacts["fills"],
                initial_cash=initial_cash,
                additional_slippage_bps=selection_cost_bps,
            )
            benchmark = _benchmark_for_dates(
                benchmark_returns, nav["date"]
            )
            active_returns = (
                (1.0 + selection_returns)
                / (1.0 + benchmark)
                - 1.0
            )
            frame = pd.DataFrame(
                {
                    "date": pd.to_datetime(nav["date"]).dt.normalize(),
                    "window_id": str(fold["window_id"]),
                    "candidate_id": candidate_id,
                    "recorded_net_return": recorded_returns,
                    "selection_additional_slippage_bps": (
                        selection_cost_bps
                    ),
                    "selection_net_return": selection_returns,
                    "benchmark_total_return": benchmark,
                    "selection_active_return": active_returns,
                }
            )
            candidate_returns.append(frame)
        combined = pd.concat(candidate_returns, ignore_index=True)
        daily = combined["selection_active_return"]
        volatility = float(daily.std(ddof=1))
        if not math.isfinite(volatility) or volatility <= 0:
            raise ContractError(
                f"candidate {candidate_id} has invalid validation volatility"
            )
        annualized_sharpe = float(
            daily.mean() / volatility * math.sqrt(252.0)
        )
        p_value, block_count = _block_normal_p_value(
            daily,
            int(
                walk_forward_config["tuning"][
                    "significance_block_sessions"
                ]
            ),
        )
        metric_rows.append(
            {
                "candidate_id": candidate_id,
                "validation_observations": len(daily),
                "validation_score": annualized_sharpe,
                "validation_p_value": p_value,
                "significance_block_count": block_count,
                "selection_additional_slippage_bps": selection_cost_bps,
                "selection_metric": "active_sharpe_vs_total_return_benchmark",
            }
        )
        return_rows.append(combined)

    validation_returns = pd.concat(return_rows, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    return_matrix = validation_returns.pivot(
        index=["date", "window_id"],
        columns="candidate_id",
        values="selection_active_return",
    ).sort_index()
    if return_matrix.isna().any().any():
        raise ContractError(
            "candidate validation returns do not share an identical calendar"
        )

    daily_sharpes = (
        return_matrix.mean()
        / return_matrix.std(ddof=1).replace(0.0, np.nan)
    )
    trial_std = (
        float(daily_sharpes.std(ddof=1))
        if len(daily_sharpes.dropna()) > 1
        else 0.0
    )
    dsr_rows: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        probability, daily_sharpe, benchmark = (
            _deflated_sharpe_probability(
                return_matrix[candidate_id],
                trial_count=len(candidate_ids),
                trial_sharpe_standard_deviation=trial_std,
            )
        )
        dsr_rows.append(
            {
                "candidate_id": candidate_id,
                "deflated_sharpe_probability": probability,
                "daily_sharpe": daily_sharpe,
                "multiple_trial_benchmark_daily_sharpe": benchmark,
                "trial_count": len(candidate_ids),
            }
        )
    dsr_report = pd.DataFrame(dsr_rows)
    metrics = metrics.merge(dsr_report, on="candidate_id", how="left")

    if len(candidate_ids) == 1:
        pbo_report = pd.DataFrame(
            [
                {
                    "pbo": np.nan,
                    "n_splits": 0,
                    "status": "NOT_APPLICABLE_SINGLE_CANDIDATE",
                }
            ]
        )
    else:
        requested_blocks = int(
            walk_forward_config["tuning"]["pbo_blocks"]
        )
        maximum_even_blocks = len(return_matrix) - len(return_matrix) % 2
        blocks = min(requested_blocks, maximum_even_blocks)
        if blocks < 2:
            raise ContractError("insufficient validation rows for CSCV/PBO")
        pbo_result = cscv_pbo(
            return_matrix.reset_index(drop=True),
            n_blocks=blocks,
            metric="sharpe",
            min_periods=max(5, len(return_matrix) // blocks // 2),
        )
        pbo_report = pbo_result["summary"].copy()
        pbo_value = float(pbo_report.iloc[0]["pbo"])
        pbo_report["status"] = (
            "PASS"
            if pbo_value
            <= float(walk_forward_config["tuning"]["maximum_pbo"])
            else "FAIL"
        )

    scored_registry = registry.merge(
        metrics[
            ["candidate_id", "validation_score", "validation_p_value"]
        ],
        on="candidate_id",
        how="left",
        validate="one_to_one",
    )
    return scored_registry, validation_returns, metrics, pbo_report


def run_formal_walk_forward(
    project_root: str | Path,
    *,
    pit_gate_run_directory: str | Path,
    strategy_config_path: str | Path,
    walk_forward_config_path: str | Path,
    run_id: str,
    initial_cash: float,
    consume_independent_test: bool,
) -> dict[str, Path]:
    root = Path(project_root).resolve()
    if re.fullmatch(r"[A-Za-z0-9._-]+", run_id) is None:
        raise ContractError(f"unsafe formal walk-forward run_id: {run_id!r}")
    if initial_cash <= 0:
        raise ContractError("formal walk-forward initial_cash must be positive")

    strategy_config_file = (root / strategy_config_path).resolve()
    walk_forward_config_file = (root / walk_forward_config_path).resolve()
    strategy_config = load_config(strategy_config_file)
    walk_forward_config = load_walk_forward_config(walk_forward_config_file)
    gate_binding = verify_pit_gate_binding(pit_gate_run_directory, root)
    input_paths, formal_input_bindings = _read_bound_inputs(root, gate_binding)

    calendar = _load_calendar(input_paths["trading_calendar"])
    validation_prices = pd.read_csv(
        input_paths["validation_execution_states"], dtype={"asset": str}
    )
    validation_targets = pd.read_csv(
        input_paths["validation_target_weights"], dtype={"asset": str}
    )
    validation_benchmark_returns = _load_benchmark_returns(
        input_paths["validation_benchmark_returns"],
        walk_forward_config,
    )
    candidate_registry = pd.read_csv(
        input_paths["candidate_registry"], dtype=str
    )
    required_candidate_columns = {
        "candidate_id",
        "parameters_json",
        "train_score",
        "split_roles_used",
    }
    missing = sorted(
        required_candidate_columns.difference(candidate_registry.columns)
    )
    if missing:
        raise ContractError(
            f"candidate registry missing pre-registration fields: {missing}"
        )
    if (
        candidate_registry.empty
        or candidate_registry["candidate_id"].astype(str).duplicated().any()
    ):
        raise ContractError(
            "candidate registry must be non-empty with unique candidate_id"
        )
    if len(candidate_registry) > int(
        walk_forward_config["tuning"]["maximum_candidate_count"]
    ):
        raise ContractError(
            "candidate registry exceeds the frozen tuning budget"
        )
    for row in candidate_registry.itertuples(index=False):
        assert_tuning_inputs(str(row.split_roles_used).split("+"))
        try:
            parameters = json.loads(str(row.parameters_json))
        except json.JSONDecodeError as exc:
            raise ContractError(
                f"candidate parameters_json is invalid: {row.candidate_id}"
            ) from exc
        if not isinstance(parameters, dict):
            raise ContractError(
                "candidate parameters_json must contain an object"
            )
    required_target_columns = {"window_id", "candidate_id"}
    missing = sorted(
        required_target_columns.difference(validation_targets.columns)
    )
    if missing:
        raise ContractError(
            f"formal target weights missing governance columns: {missing}"
        )

    plan = build_purged_embargoed_plan(calendar, walk_forward_config)
    independent_start = pd.Timestamp(plan["independent_test"]["start"])
    validation_price_dates = pd.to_datetime(
        validation_prices["date"], errors="coerce"
    ).dt.normalize()
    validation_signal_dates = pd.to_datetime(
        validation_targets["signal_date"], errors="coerce"
    ).dt.normalize()
    if (
        validation_price_dates.isna().any()
        or validation_signal_dates.isna().any()
        or (validation_price_dates >= independent_start).any()
        or (validation_signal_dates >= independent_start).any()
        or (
            validation_benchmark_returns["date"] >= independent_start
        ).any()
    ):
        raise ContractError(
            "validation inputs contain invalid dates or independent-test rows"
        )
    unknown_candidates = sorted(
        set(validation_targets["candidate_id"].astype(str)).difference(
            candidate_registry["candidate_id"].astype(str)
        )
    )
    if unknown_candidates:
        raise ContractError(
            f"target weights contain unregistered candidates: {unknown_candidates}"
        )
    _validate_target_window_contract(
        validation_targets,
        plan["validation_windows"],
        split_role="validation",
    )
    (
        scored_candidate_registry,
        candidate_validation_returns,
        candidate_validation_metrics,
        pbo_report,
    ) = _evaluate_validation_candidates(
        validation_prices,
        validation_targets,
        validation_benchmark_returns,
        candidate_registry,
        plan,
        strategy_config,
        walk_forward_config,
        initial_cash=initial_cash,
    )
    selected = select_frozen_candidate(
        scored_candidate_registry, walk_forward_config
    )
    selected_id = str(selected["candidate_id"])
    selected_dsr = float(
        candidate_validation_metrics.loc[
            candidate_validation_metrics["candidate_id"].astype(str).eq(
                selected_id
            ),
            "deflated_sharpe_probability",
        ].iloc[0]
    )
    pbo_status = str(pbo_report.iloc[0]["status"])
    validation_robustness_passed = (
        pbo_status in {"PASS", "NOT_APPLICABLE_SINGLE_CANDIDATE"}
        and selected_dsr
        >= float(
            walk_forward_config["tuning"][
                "minimum_deflated_sharpe_probability"
            ]
        )
    )

    commit = _git_head(root)
    code_bindings = [
        _binding(root, root / relative, "runner_code")
        for relative in RUNNER_CODE_PATHS
    ]
    config_bindings = [
        _binding(root, strategy_config_file, "strategy_config"),
        _binding(root, walk_forward_config_file, "walk_forward_config"),
    ]
    output_root = (root / walk_forward_config["output_root"]).resolve()
    final_run = output_root / run_id
    if final_run.exists():
        raise ContractError(f"formal walk-forward run is immutable: {run_id}")
    _relative(root, output_root)
    stage_root = output_root / f".{run_id}.tmp-{uuid.uuid4().hex}"
    stage_run = stage_root / run_id
    stage_run.mkdir(parents=True)

    completed: list[str] = []
    window_manifests: list[dict[str, str]] = []
    performance_rows: list[dict[str, Any]] = []
    validation_windows: list[
        tuple[str, str, str, str, pd.DataFrame, pd.DataFrame]
    ] = [
        (
            fold["window_id"],
            "validation",
            fold["validation_start"],
            fold["validation_end"],
            validation_prices,
            validation_targets,
        )
        for fold in plan["validation_windows"]
    ]
    holdout_binding: dict[str, str] | None = None

    def execute_window(
        window_id: str,
        split_role: str,
        start: str,
        end: str,
        prices: pd.DataFrame,
        targets: pd.DataFrame,
        benchmark_returns: pd.DataFrame,
    ) -> None:
        window_prices, window_targets = _window_frames(
            prices,
            targets,
            window_id=window_id,
            start=start,
            end=end,
            candidate_id=selected_id,
        )
        artifacts = run_audited_window_backtest(
            window_prices,
            window_targets,
            strategy_config,
            walk_forward_config,
            initial_cash=initial_cash,
        )
        benchmark = _benchmark_for_dates(
            benchmark_returns, artifacts["nav"]["date"]
        )
        benchmark_stats = _return_statistics(benchmark)
        stress_levels = [0] + [
            int(value)
            for value in walk_forward_config["cost_scenarios"][
                "additional_slippage_bps"
            ]
        ]
        for bps in stress_levels:
            strategy_returns = (
                _daily_nav_returns(artifacts["nav"], initial_cash)
                if bps == 0
                else _cost_stressed_returns(
                    artifacts["nav"],
                    artifacts["fills"],
                    initial_cash=initial_cash,
                    additional_slippage_bps=bps,
                )
            )
            active_returns = (
                (1.0 + strategy_returns) / (1.0 + benchmark) - 1.0
            )
            strategy_stats = _return_statistics(strategy_returns)
            active_stats = _return_statistics(active_returns)
            performance_rows.append(
                {
                    "window_id": window_id,
                    "split_role": split_role,
                    "benchmark_id": walk_forward_config["benchmark"][
                        "benchmark_id"
                    ],
                    "additional_slippage_bps": bps,
                    **{
                        f"strategy_{key}": value
                        for key, value in strategy_stats.items()
                    },
                    **{
                        f"benchmark_{key}": value
                        for key, value in benchmark_stats.items()
                    },
                    **{
                        f"active_{key}": value
                        for key, value in active_stats.items()
                    },
                }
            )
        context: dict[str, Any] = {
            "pit_gate_run_id": gate_binding["pit_gate_run_id"],
            "pit_gate_manifest_sha256": gate_binding[
                "pit_gate_manifest_sha256"
            ],
            "target_manifest_sha256": gate_binding[
                "target_manifest_sha256"
            ],
            "code_commit": commit,
            "code_files": code_bindings,
            "config_bindings": config_bindings,
            "formal_input_bindings": formal_input_bindings,
            "training_parameters": selected,
            "data_manifest": artifacts["input_hashes"],
            "cost_assumptions": {
                "recorded_costs": True,
                "additional_slippage_bps": walk_forward_config[
                    "cost_scenarios"
                ]["additional_slippage_bps"],
            },
        }
        if split_role == "independent_test":
            context["holdout_consumption_binding"] = holdout_binding
        paths = write_window_bundle(
            stage_root,
            project_root=root,
            pit_gate_run_directory=pit_gate_run_directory,
            run_id=run_id,
            window_id=window_id,
            split_role=split_role,
            context=context,
            artifacts=artifacts,
        )
        completed.append(window_id)
        window_manifests.append(
            {
                "window_id": window_id,
                "path": f"{window_id}/window_manifest.json",
                "sha256": sha256_file(paths["manifest"]),
            }
        )

    for window in validation_windows:
        execute_window(*window, validation_benchmark_returns)

    consume_holdout = (
        consume_independent_test and validation_robustness_passed
    )
    if consume_holdout:
        holdout_payload_hash = hashlib.sha256(
            canonical_json_bytes(
                {
                    "pit_gate_manifest_sha256": gate_binding[
                        "pit_gate_manifest_sha256"
                    ],
                    "plan_sha256": plan["plan_sha256"],
                    "selected_candidate": selected,
                    "independent_inputs": [
                        item
                        for item in formal_input_bindings
                        if str(item["role"]).startswith("independent_")
                    ],
                }
            )
        ).hexdigest()
        ledger_path = (
            output_root
            / "holdout_consumption"
            / "independent-test.json"
        )
        consume_independent_test_once(
            ledger_path,
            run_id=run_id,
            plan_sha256=plan["plan_sha256"],
            data_manifest_sha256=holdout_payload_hash,
        )
        holdout_binding = _binding(
            root, ledger_path, "holdout_consumption"
        )

        independent_prices = pd.read_csv(
            input_paths["independent_execution_states"],
            dtype={"asset": str},
        )
        independent_targets = pd.read_csv(
            input_paths["independent_target_weights"],
            dtype={"asset": str},
        )
        independent_benchmark_returns = _load_benchmark_returns(
            input_paths["independent_benchmark_returns"],
            walk_forward_config,
        )
        missing = sorted(
            required_target_columns.difference(independent_targets.columns)
        )
        if missing:
            raise ContractError(
                f"independent target weights missing governance columns: {missing}"
            )
        independent_candidate_ids = set(
            independent_targets["candidate_id"].astype(str)
        )
        if independent_candidate_ids != {selected_id}:
            raise ContractError(
                "independent targets must contain only the frozen candidate"
            )
        test = plan["independent_test"]
        _validate_target_window_contract(
            independent_targets,
            [test],
            split_role="independent_test",
        )
        independent_price_dates = pd.to_datetime(
            independent_prices["date"], errors="coerce"
        ).dt.normalize()
        independent_signal_dates = pd.to_datetime(
            independent_targets["signal_date"], errors="coerce"
        ).dt.normalize()
        test_start = pd.Timestamp(test["start"])
        test_end = pd.Timestamp(test["end"])
        if (
            independent_price_dates.isna().any()
            or independent_signal_dates.isna().any()
            or not independent_price_dates.between(
                test_start, test_end, inclusive="both"
            ).all()
            or not independent_signal_dates.between(
                test_start, test_end, inclusive="both"
            ).all()
            or not independent_benchmark_returns["date"].between(
                test_start, test_end, inclusive="both"
            ).all()
        ):
            raise ContractError(
                "independent inputs contain invalid or out-of-window dates"
            )
        execute_window(
            test["window_id"],
            "independent_test",
            test["start"],
            test["end"],
            independent_prices,
            independent_targets,
            independent_benchmark_returns,
        )

    registry = build_window_status_registry(
        plan, completed_window_ids=completed
    )
    registry_path = stage_run / "window_status_registry.csv"
    registry.to_csv(registry_path, index=False, encoding="utf-8")
    plan_path = stage_run / "plan.json"
    public_plan = json.loads(
        json.dumps(plan, default=str)
    )
    for fold in public_plan["validation_windows"]:
        for key in list(fold):
            if key.startswith("_"):
                fold.pop(key)
    for section in ("embargo", "independent_test"):
        for key in list(public_plan[section]):
            if key.startswith("_"):
                public_plan[section].pop(key)
    plan_path.write_bytes(canonical_json_bytes(public_plan))
    selected_path = stage_run / "selected_candidate.json"
    selected_path.write_bytes(canonical_json_bytes(selected))
    candidate_registry_path = (
        stage_run / "candidate_validation_registry.csv"
    )
    scored_candidate_registry.to_csv(
        candidate_registry_path, index=False, encoding="utf-8"
    )
    candidate_returns_path = (
        stage_run / "candidate_validation_returns.csv"
    )
    candidate_validation_returns.to_csv(
        candidate_returns_path, index=False, encoding="utf-8"
    )
    candidate_metrics_path = (
        stage_run / "candidate_validation_metrics.csv"
    )
    candidate_validation_metrics.to_csv(
        candidate_metrics_path, index=False, encoding="utf-8"
    )
    pbo_path = stage_run / "pbo_report.csv"
    pbo_report.to_csv(pbo_path, index=False, encoding="utf-8")
    performance_path = stage_run / "window_performance.csv"
    pd.DataFrame(performance_rows).to_csv(
        performance_path, index=False, encoding="utf-8"
    )
    bias_audit = build_bias_audit(
        historical_universe_verified=True,
        available_dates_verified=True,
        tuning_split_roles=["train", "validation"],
        independent_test_access_count=1 if consume_holdout else 0,
        registered_candidate_count=len(candidate_registry),
        maximum_candidate_count=int(
            walk_forward_config["tuning"]["maximum_candidate_count"]
        ),
        multiple_testing_correction=walk_forward_config["tuning"][
            "multiple_testing_correction"
        ],
        adjusted_p_values_verified=True,
        candidate_registry_frozen_before_holdout=True,
    )
    bias_path = stage_run / "bias_audit.json"
    bias_path.write_bytes(canonical_json_bytes(bias_audit))

    top_level_outputs = [
        {
            "path": path.name,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in (plan_path, selected_path, registry_path, bias_path)
        + (
            candidate_registry_path,
            candidate_returns_path,
            candidate_metrics_path,
            pbo_path,
            performance_path,
        )
    ]
    manifest = {
        "schema_version": FORMAL_RUN_SCHEMA_VERSION,
        "run_id": run_id,
        "status": (
            "FORMAL_EVALUATION_COMPLETE_REVIEW_REQUIRED"
            if consume_holdout
            else (
                "VALIDATION_BLOCKED_HOLDOUT_NOT_CONSUMED"
                if consume_independent_test
                and not validation_robustness_passed
                else "VALIDATION_COMPLETE_HOLDOUT_NOT_CONSUMED"
            )
        ),
        "pit_gate_run_id": gate_binding["pit_gate_run_id"],
        "pit_gate_manifest_sha256": gate_binding[
            "pit_gate_manifest_sha256"
        ],
        "target_manifest_sha256": gate_binding["target_manifest_sha256"],
        "code_commit": commit,
        "code_files": code_bindings,
        "config_bindings": config_bindings,
        "formal_input_bindings": formal_input_bindings,
        "selected_candidate": selected,
        "validation_robustness": {
            "passed": validation_robustness_passed,
            "pbo_status": pbo_status,
            "selected_deflated_sharpe_probability": selected_dsr,
            "minimum_deflated_sharpe_probability": walk_forward_config[
                "tuning"
            ]["minimum_deflated_sharpe_probability"],
            "selection_additional_slippage_bps": max(
                walk_forward_config["cost_scenarios"][
                    "additional_slippage_bps"
                ]
            ),
            "selection_metric": "active_sharpe_vs_total_return_benchmark",
        },
        "window_manifests": window_manifests,
        "outputs": top_level_outputs,
        "holdout_consumption_binding": holdout_binding,
        "independent_test_requested": bool(consume_independent_test),
        "independent_test_consumed": bool(consume_holdout),
        "promotion_allowed": False,
        "live_trading_allowed": False,
        "manual_review_required": True,
    }
    manifest_path = stage_run / "formal_run_manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    seal_path = stage_run / "formal_run_manifest_seal.json"
    seal_path.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": FORMAL_RUN_SCHEMA_VERSION,
                "run_id": run_id,
                "formal_run_manifest_sha256": sha256_file(manifest_path),
                "promotion_allowed": False,
                "live_trading_allowed": False,
            }
        )
    )
    output_root.mkdir(parents=True, exist_ok=True)
    os.replace(stage_run, final_run)
    stage_root.rmdir()
    return {
        "run_directory": final_run,
        "manifest": final_run / manifest_path.name,
        "seal": final_run / seal_path.name,
    }


def verify_formal_run(
    run_directory: str | Path, project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    directory = Path(run_directory).resolve()
    manifest_path = directory / "formal_run_manifest.json"
    seal_path = directory / "formal_run_manifest_seal.json"
    if not manifest_path.is_file() or not seal_path.is_file():
        raise ContractError("formal run manifest or seal is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if (
        seal.get("formal_run_manifest_sha256") != sha256_file(manifest_path)
        or seal.get("run_id") != manifest.get("run_id")
    ):
        raise ContractError("formal run manifest seal mismatch")
    if (
        manifest.get("promotion_allowed") is not False
        or manifest.get("live_trading_allowed") is not False
        or manifest.get("manual_review_required") is not True
    ):
        raise ContractError("formal run contains an unsafe promotion state")
    consumed = manifest.get("independent_test_consumed")
    holdout_binding = manifest.get("holdout_consumption_binding")
    if consumed is True:
        if not isinstance(holdout_binding, dict):
            raise ContractError(
                "consumed independent test lacks a ledger binding"
            )
        _verify_binding(
            root, holdout_binding, "holdout consumption ledger"
        )
        if (
            manifest.get("status")
            != "FORMAL_EVALUATION_COMPLETE_REVIEW_REQUIRED"
        ):
            raise ContractError(
                "formal run status disagrees with holdout consumption"
            )
    elif consumed is False:
        if holdout_binding is not None:
            raise ContractError(
                "unconsumed independent test has a ledger binding"
            )
    else:
        raise ContractError(
            "formal run independent_test_consumed flag is invalid"
        )
    robustness = manifest.get("validation_robustness")
    if (
        not isinstance(robustness, dict)
        or (
            robustness.get("passed") is not True
            and consumed is True
        )
    ):
        raise ContractError(
            "independent test was consumed without validation robustness"
        )

    for group, label in (
        (manifest.get("code_files"), "code"),
        (manifest.get("config_bindings"), "config"),
        (manifest.get("formal_input_bindings"), "formal input"),
    ):
        if not isinstance(group, list) or not group:
            raise ContractError(f"formal run {label} bindings are missing")
        for item in group:
            path = (root / str(item.get("path", ""))).resolve()
            if (
                (root != path and root not in path.parents)
                or not path.is_file()
                or sha256_file(path) != str(item.get("sha256", ""))
            ):
                raise ContractError(
                    f"formal run {label} binding mismatch: {item.get('path')}"
                )

    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise ContractError("formal run output bindings are missing")
    output_paths = [str(item.get("path", "")) for item in outputs]
    if len(output_paths) != len(set(output_paths)):
        raise ContractError("formal run output bindings contain duplicates")
    for item in outputs:
        path = (directory / str(item.get("path", ""))).resolve()
        if (
            (directory != path and directory not in path.parents)
            or not path.is_file()
            or path.stat().st_size != int(item.get("bytes", -1))
            or sha256_file(path) != str(item.get("sha256", ""))
        ):
            raise ContractError(
                f"formal run output mismatch: {item.get('path')}"
            )
    window_bindings = manifest.get("window_manifests")
    if not isinstance(window_bindings, list) or not window_bindings:
        raise ContractError("formal run window bindings are missing")
    window_ids = [str(item.get("window_id", "")) for item in window_bindings]
    if len(window_ids) != len(set(window_ids)):
        raise ContractError("formal run window bindings contain duplicates")
    independent_window_count = 0
    for item in window_bindings:
        window_manifest_path = (
            directory / str(item.get("path", ""))
        ).resolve()
        if (
            directory not in window_manifest_path.parents
            or not window_manifest_path.is_file()
            or sha256_file(window_manifest_path)
            != str(item.get("sha256", ""))
        ):
            raise ContractError(
                f"formal window manifest mismatch: {item.get('window_id')}"
            )
        window_manifest = json.loads(
            window_manifest_path.read_text(encoding="utf-8")
        )
        if window_manifest.get("split_role") == "independent_test":
            independent_window_count += 1
        window_directory = window_manifest_path.parent
        for output in window_manifest.get("outputs", []):
            output_path = (
                window_directory / str(output.get("path", ""))
            ).resolve()
            if (
                window_directory not in output_path.parents
                or not output_path.is_file()
                or sha256_file(output_path) != str(output.get("sha256", ""))
            ):
                raise ContractError(
                    f"formal window output mismatch: {output.get('path')}"
                )
    if independent_window_count != (1 if consumed else 0):
        raise ContractError(
            "independent window count disagrees with holdout consumption"
        )
    return {
        "run_id": str(manifest["run_id"]),
        "status": str(manifest["status"]),
        "window_count": len(manifest.get("window_manifests", [])),
        "promotion_allowed": False,
        "live_trading_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Operate PIT-bound Long Hold V4 formal walk-forward"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--project-root", default=".")
    run_parser.add_argument("--pit-gate-run-directory", required=True)
    run_parser.add_argument(
        "--strategy-config", default="configs/long_hold_v4.json"
    )
    run_parser.add_argument(
        "--walk-forward-config",
        default="configs/long_hold_v4_work_package_5_walk_forward.json",
    )
    run_parser.add_argument("--run-id", required=True)
    run_parser.add_argument("--initial-cash", type=float, default=500_000.0)
    run_parser.add_argument(
        "--consume-independent-test",
        action="store_true",
        help="Irreversibly consume the one-time independent test",
    )
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("--project-root", default=".")
    verify_parser.add_argument("--run-directory", required=True)
    args = parser.parse_args()
    if args.command == "verify":
        result = verify_formal_run(args.run_directory, args.project_root)
        print(json.dumps(result, ensure_ascii=False))
        return
    paths = run_formal_walk_forward(
        args.project_root,
        pit_gate_run_directory=args.pit_gate_run_directory,
        strategy_config_path=args.strategy_config,
        walk_forward_config_path=args.walk_forward_config,
        run_id=args.run_id,
        initial_cash=args.initial_cash,
        consume_independent_test=args.consume_independent_test,
    )
    manifest = json.loads(
        paths["manifest"].read_text(encoding="utf-8")
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "run_directory": str(paths["run_directory"]),
                "independent_test_consumed": manifest[
                    "independent_test_consumed"
                ],
                "promotion_allowed": False,
                "live_trading_allowed": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
