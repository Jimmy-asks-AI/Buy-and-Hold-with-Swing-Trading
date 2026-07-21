#!/usr/bin/env python
"""Data-quality reports for quant factor-factory panels."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

import a_share_low_cost_factor_builder as lowcost
import csv_io
import factor_factory_walk_forward as wf


def _coverage(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return float(series.notna().mean())


def _column_role(column: str, registry: pd.DataFrame, config: wf.WalkForwardConfig) -> str:
    factor_cols = set(registry.get("column", pd.Series(dtype=str)).dropna().astype(str))
    if column == config.date_col:
        return "date"
    if column == config.asset_col:
        return "asset"
    if column == config.forward_return_col:
        return "label"
    if column == config.industry_col:
        return "industry"
    if column in config.control_cols:
        return "control"
    if column == config.tradeable_col:
        return "tradeability"
    if column == config.amount_col:
        return "capacity"
    if column in factor_cols:
        return "registered_factor"
    return "extra"


def _numeric_range(series: pd.Series) -> tuple[Any, Any]:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == 0:
        return "", ""
    return float(numeric.min()), float(numeric.max())


def _clean_optional_col(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null"} else text


def required_columns(registry: pd.DataFrame, config: wf.WalkForwardConfig) -> list[str]:
    cols = [config.date_col, config.asset_col, config.forward_return_col]
    if config.industry_col:
        cols.append(config.industry_col)
    cols.extend(config.control_cols)
    if config.tradeable_col:
        cols.append(config.tradeable_col)
    if config.amount_col:
        cols.append(config.amount_col)
    cols.extend(registry.get("column", pd.Series(dtype=str)).dropna().astype(str).tolist())
    return list(dict.fromkeys(cols))


def build_quality_report(panel: pd.DataFrame, registry: pd.DataFrame, config: wf.WalkForwardConfig) -> dict[str, pd.DataFrame]:
    data = panel.copy()
    invalid_date_rate = 1.0
    if config.date_col in data.columns:
        parsed_dates = pd.to_datetime(data[config.date_col], errors="coerce")
        invalid_date_rate = float(parsed_dates.isna().mean()) if len(parsed_dates) else 0.0
        data[config.date_col] = parsed_dates
    missing_required = [col for col in [config.date_col, config.asset_col, config.forward_return_col] if col not in data.columns]
    has_date_asset = config.date_col in data.columns and config.asset_col in data.columns
    duplicate_count = int(data.duplicated([config.date_col, config.asset_col]).sum()) if has_date_asset else 0
    available_factor_cols = [col for col in registry.get("column", pd.Series(dtype=str)).dropna().astype(str) if col in data.columns]

    summary = pd.DataFrame(
        [
            {
                "metric": "rows",
                "value": int(data.shape[0]),
                "status": "pass" if data.shape[0] > 0 else "fail",
            },
            {
                "metric": "assets",
                "value": int(data[config.asset_col].nunique()) if config.asset_col in data.columns else 0,
                "status": "pass" if config.asset_col in data.columns and data[config.asset_col].nunique() >= config.min_assets else "fail",
            },
            {
                "metric": "dates",
                "value": int(data[config.date_col].nunique()) if config.date_col in data.columns else 0,
                "status": "pass" if config.date_col in data.columns and data[config.date_col].nunique() >= config.train_periods + config.test_periods else "fail",
            },
            {
                "metric": "duplicate_date_asset",
                "value": duplicate_count,
                "status": "pass" if duplicate_count == 0 else "fail",
            },
            {
                "metric": "invalid_date_rate",
                "value": invalid_date_rate,
                "status": "pass" if invalid_date_rate == 0.0 else "fail",
            },
            {
                "metric": "label_coverage",
                "value": _coverage(data[config.forward_return_col]) if config.forward_return_col in data.columns else 0.0,
                "status": "pass" if config.forward_return_col in data.columns and _coverage(data[config.forward_return_col]) >= 0.5 else "warn",
            },
            {
                "metric": "available_registered_factors",
                "value": len(available_factor_cols),
                "status": "pass" if available_factor_cols else "fail",
            },
        ]
    )

    rows = []
    for col in required_columns(registry, config):
        present = col in data.columns
        min_value, max_value = _numeric_range(data[col]) if present else ("", "")
        rows.append(
            {
                "column": col,
                "role": _column_role(col, registry, config),
                "present": bool(present),
                "non_null_rate": _coverage(data[col]) if present else 0.0,
                "dtype": str(data[col].dtype) if present else "",
                "min": min_value,
                "max": max_value,
            }
        )
    column_coverage = pd.DataFrame(rows)

    factor_rows = []
    availability_rows = []
    for row in registry.itertuples(index=False):
        factor = str(getattr(row, "factor_id", ""))
        col = str(getattr(row, "column", ""))
        family = str(getattr(row, "family", ""))
        availability_col = _clean_optional_col(getattr(row, "availability_col", ""))
        present = col in data.columns
        availability_present = bool(not availability_col or availability_col in data.columns)
        invalid_availability_rate = 0.0
        future_availability_rate = 0.0
        future_availability_count = 0
        if availability_col and availability_col in data.columns and config.date_col in data.columns:
            available = pd.to_datetime(data[availability_col], errors="coerce")
            valid_factor_rows = data[col].notna() if present else pd.Series(False, index=data.index)
            rows_with_factor = int(valid_factor_rows.sum())
            invalid_availability_rate = float((valid_factor_rows & available.isna()).sum() / rows_with_factor) if rows_with_factor else 0.0
            future = valid_factor_rows & available.notna() & (available > data[config.date_col])
            future_availability_count = int(future.sum())
            future_availability_rate = float(future.sum() / rows_with_factor) if rows_with_factor else 0.0
            availability_rows.append(
                {
                    "factor_id": factor,
                    "availability_col": availability_col,
                    "rows_with_factor": rows_with_factor,
                    "invalid_availability_rate": invalid_availability_rate,
                    "future_availability_count": future_availability_count,
                    "future_availability_rate": future_availability_rate,
                }
            )
        factor_rows.append(
            {
                "factor_id": factor,
                "family": family,
                "column": col,
                "present": bool(present),
                "coverage": _coverage(data[col]) if present else 0.0,
                "availability_col": availability_col,
                "availability_present": availability_present,
                "invalid_availability_rate": invalid_availability_rate,
                "future_availability_count": future_availability_count,
                "future_availability_rate": future_availability_rate,
            }
        )
    factor_coverage = pd.DataFrame(factor_rows)
    availability_audit = pd.DataFrame(availability_rows)

    if config.date_col in data.columns and config.asset_col in data.columns:
        grouped = data.groupby(config.date_col, sort=True)
        date_health = grouped.agg(rows=(config.asset_col, "size"), assets=(config.asset_col, "nunique")).reset_index()
        if config.forward_return_col in data.columns:
            date_health["label_coverage"] = grouped[config.forward_return_col].apply(_coverage).values
        if config.tradeable_col and config.tradeable_col in data.columns:
            data["_tradeable_bool"] = csv_io.coerce_bool_series(data[config.tradeable_col], default=None).astype(float)
            date_health["tradeable_rate"] = data.groupby(config.date_col, sort=True)["_tradeable_bool"].mean().values
        if config.amount_col and config.amount_col in data.columns:
            date_health["amount_sum"] = grouped[config.amount_col].sum(min_count=1).values
    else:
        date_health = pd.DataFrame()

    gates = []
    for row in summary.itertuples(index=False):
        gates.append({"gate": row.metric, "status": row.status, "detail": row.value})
    missing_cols = column_coverage.loc[~column_coverage["present"], "column"].tolist()
    gates.append({"gate": "missing_required_or_registered_columns", "status": "pass" if not missing_cols else "warn", "detail": "|".join(missing_cols[:50])})
    low_coverage = factor_coverage.loc[(factor_coverage["present"]) & (factor_coverage["coverage"] < 0.5), "factor_id"].tolist()
    gates.append({"gate": "low_factor_coverage", "status": "pass" if not low_coverage else "warn", "detail": "|".join(low_coverage[:50])})
    missing_availability = factor_coverage.loc[
        (factor_coverage["present"]) & (factor_coverage["availability_col"] != "") & (~factor_coverage["availability_present"]),
        "factor_id",
    ].tolist()
    gates.append({"gate": "missing_availability_columns", "status": "pass" if not missing_availability else "fail", "detail": "|".join(missing_availability[:50])})
    future_availability = factor_coverage.loc[factor_coverage["future_availability_count"] > 0, "factor_id"].tolist()
    gates.append({"gate": "future_availability_leakage", "status": "pass" if not future_availability else "fail", "detail": "|".join(future_availability[:50])})
    if config.amount_col and config.amount_col in data.columns:
        amount = pd.to_numeric(data[config.amount_col], errors="coerce")
        negative_amount_rate = float((amount < 0).mean()) if len(amount) else 0.0
        gates.append({"gate": "negative_amount", "status": "pass" if negative_amount_rate == 0.0 else "fail", "detail": f"{negative_amount_rate:.6f}"})
    if "market_cap" in data.columns:
        market_cap = pd.to_numeric(data["market_cap"], errors="coerce")
        nonpositive_market_cap_rate = float((market_cap <= 0).mean()) if len(market_cap) else 0.0
        gates.append({"gate": "nonpositive_market_cap", "status": "pass" if nonpositive_market_cap_rate == 0.0 else "fail", "detail": f"{nonpositive_market_cap_rate:.6f}"})
    if config.tradeable_col and config.tradeable_col in data.columns:
        tradeable = csv_io.coerce_bool_series(data[config.tradeable_col], default=None)
        unknown_tradeable_rate = float(tradeable.isna().mean()) if len(tradeable) else 0.0
        gates.append({"gate": "unknown_tradeable_values", "status": "pass" if unknown_tradeable_rate == 0.0 else "warn", "detail": f"{unknown_tradeable_rate:.6f}"})

    return {
        "summary": summary,
        "column_coverage": column_coverage,
        "factor_coverage": factor_coverage,
        "availability_audit": availability_audit,
        "date_health": date_health,
        "gates": pd.DataFrame(gates),
    }


def save_report(report: dict[str, pd.DataFrame], output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, table in report.items():
        table.to_csv(out / f"{name}.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-csv")
    parser.add_argument("--registry-csv", default=str(Path("Introduction-to-Quantitative-Finance") / "data_catalog" / "a_share_factor_registry_v0.csv"))
    parser.add_argument("--config-json", default=str(Path("Introduction-to-Quantitative-Finance") / "configs" / "factor_factory_smoke.json"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--synthetic-demo", action="store_true")
    args = parser.parse_args()

    if args.synthetic_demo:
        panel = lowcost.add_forward_return(lowcost.build_low_cost_factors(lowcost.make_synthetic_low_cost_panel()))
    else:
        if not args.panel_csv:
            raise ValueError("Provide --panel-csv or use --synthetic-demo.")
        panel = csv_io.read_csv_robust(args.panel_csv)
    registry = csv_io.read_csv_robust(args.registry_csv)
    config = wf.load_config(args.config_json)
    report = build_quality_report(panel, registry, config)
    save_report(report, args.output_dir)
    print(report["summary"].to_string(index=False))


if __name__ == "__main__":
    main()
