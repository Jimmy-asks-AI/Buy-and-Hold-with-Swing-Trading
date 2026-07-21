#!/usr/bin/env python
"""Paper-trading state files for factor-factory model outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def latest_target_weights(weights: pd.DataFrame) -> pd.DataFrame:
    if weights.empty:
        raise ValueError("weights table is empty.")
    weights = weights.copy()
    weights["date"] = pd.to_datetime(weights["date"])
    latest_date = weights["date"].max()
    latest = weights.loc[weights["date"] == latest_date, ["date", "asset", "weight"]].copy()
    latest = latest.sort_values("weight", ascending=False).reset_index(drop=True)
    return latest


def paper_monitoring_checklist() -> pd.DataFrame:
    rows = [
        ("data_freshness", "Confirm all market and factor data are updated before target weights are used."),
        ("tradeability", "Check suspension, limit-up/down, ST, and new-listing filters before simulated trades."),
        ("turnover", "Compare current holdings to target weights and estimate one-way turnover."),
        ("cost", "Estimate commission, stamp duty, spread, slippage, and market impact."),
        ("exposure", "Check industry, size, beta, value, momentum, and liquidity exposures."),
        ("capacity", "Check participation rate versus latest trading amount."),
        ("drift", "Compare realized portfolio return with model expected and benchmark return."),
        ("decay", "Monitor selected factor quality and family distribution versus training windows."),
        ("exception", "Record missing data, stale factors, abnormal weights, and failed constraints."),
        ("human_review", "Require manual review before any transition from paper to live trading."),
    ]
    return pd.DataFrame(rows, columns=["monitor", "requirement"])


def normalize_current_holdings(holdings: pd.DataFrame) -> pd.DataFrame:
    """Convert current holdings to asset/current_weight format."""
    if "asset" not in holdings.columns:
        raise ValueError("current holdings must include asset.")
    current = holdings.copy()
    if "current_weight" in current.columns:
        current["current_weight"] = pd.to_numeric(current["current_weight"], errors="coerce").fillna(0.0)
    elif "weight" in current.columns:
        current["current_weight"] = pd.to_numeric(current["weight"], errors="coerce").fillna(0.0)
    elif "market_value" in current.columns:
        value = pd.to_numeric(current["market_value"], errors="coerce").fillna(0.0)
        total = float(value.abs().sum())
        current["current_weight"] = value / total if total > 0 else 0.0
    else:
        raise ValueError("current holdings must include current_weight, weight, or market_value.")
    return current[["asset", "current_weight"]].groupby("asset", as_index=False).sum()


def paper_drift_report(target_weights: pd.DataFrame, current_holdings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare current paper holdings with latest target weights."""
    if not {"asset", "weight"}.issubset(target_weights.columns):
        raise ValueError("target weights must include asset and weight.")
    target = target_weights[["asset", "weight"]].copy()
    target["target_weight"] = pd.to_numeric(target["weight"], errors="coerce").fillna(0.0)
    target = target[["asset", "target_weight"]].groupby("asset", as_index=False).sum()
    current = normalize_current_holdings(current_holdings)
    drift = target.merge(current, on="asset", how="outer").fillna(0.0)
    drift["trade_weight"] = drift["target_weight"] - drift["current_weight"]
    drift["abs_drift"] = drift["trade_weight"].abs()
    drift["drift_bucket"] = pd.cut(
        drift["abs_drift"],
        bins=[-0.0000001, 0.001, 0.005, 0.02, float("inf")],
        labels=["tiny", "small", "medium", "large"],
    )
    drift = drift.sort_values("abs_drift", ascending=False).reset_index(drop=True)
    summary = pd.DataFrame(
        [
            {
                "target_positions": int((drift["target_weight"].abs() > 0).sum()),
                "current_positions": int((drift["current_weight"].abs() > 0).sum()),
                "new_positions": int(((drift["target_weight"].abs() > 0) & (drift["current_weight"].abs() == 0)).sum()),
                "exit_positions": int(((drift["target_weight"].abs() == 0) & (drift["current_weight"].abs() > 0)).sum()),
                "gross_target": float(drift["target_weight"].abs().sum()),
                "gross_current": float(drift["current_weight"].abs().sum()),
                "rebalance_trade_weight": float(drift["abs_drift"].sum()),
                "max_abs_drift": float(drift["abs_drift"].max()) if not drift.empty else 0.0,
            }
        ]
    )
    return drift, summary


def save_paper_drift_report(paper_dir: str | Path, current_holdings_csv: str | Path) -> tuple[Path, Path]:
    """Save paper drift report for a paper-tracking directory."""
    paper = Path(paper_dir)
    target = pd.read_csv(paper / "target_weights_latest.csv", encoding="utf-8-sig")
    current = pd.read_csv(current_holdings_csv, encoding="utf-8-sig")
    drift, summary = paper_drift_report(target, current)
    drift_path = paper / "paper_drift_report.csv"
    summary_path = paper / "paper_drift_summary.csv"
    drift.to_csv(drift_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    return drift_path, summary_path


def initialize_paper_state(
    walk_forward_dir: str | Path,
    output_dir: str | Path,
    experiment_id: str,
    capital: float = 1_000_000.0,
) -> dict[str, object]:
    wf = Path(walk_forward_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    weights = pd.read_csv(wf / "weights.csv", encoding="utf-8-sig")
    latest = latest_target_weights(weights)
    latest.to_csv(out / "target_weights_latest.csv", index=False, encoding="utf-8-sig")
    checklist = paper_monitoring_checklist()
    checklist.to_csv(out / "paper_monitoring_checklist.csv", index=False, encoding="utf-8-sig")
    gross_exposure = float(latest["weight"].abs().sum())
    long_exposure = float(latest.loc[latest["weight"] > 0, "weight"].sum())
    short_exposure = float(latest.loc[latest["weight"] < 0, "weight"].sum())
    state = {
        "experiment_id": experiment_id,
        "status": "paper_tracking",
        "capital": float(capital),
        "latest_rebalance_date": str(pd.to_datetime(latest["date"].iloc[0]).date()),
        "positions": int(latest.shape[0]),
        "gross_exposure": gross_exposure,
        "long_exposure": long_exposure,
        "short_exposure": short_exposure,
        "source_walk_forward_dir": str(wf),
        "target_weights_file": "target_weights_latest.csv",
        "monitoring_checklist_file": "paper_monitoring_checklist.csv",
        "live_trading_allowed": False,
    }
    (out / "paper_state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--walk-forward-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--capital", type=float, default=1_000_000.0)
    parser.add_argument("--current-holdings-csv")
    args = parser.parse_args()
    state = initialize_paper_state(args.walk_forward_dir, args.output_dir, args.experiment_id, args.capital)
    if args.current_holdings_csv:
        save_paper_drift_report(args.output_dir, args.current_holdings_csv)
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
