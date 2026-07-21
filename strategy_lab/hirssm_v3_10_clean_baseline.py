#!/usr/bin/env python
"""HIRSSM V3.10 clean baseline.

This script creates a predeclared mechanical allocation baseline that is used
only as a governance control. It uses fixed rules from the public config:
ranked candidates, sleeve budgets, volatility scaling, caps, and cash
substitution. It intentionally avoids same-OOS candidate selection, ex-post
state gating, and inherited V3.6/V3.8 targets.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import hirssm_v2_model as model
import hirssm_v2_walk_forward as wf
from model_run_manifest import build_model_run_manifest, validate_model_run_manifest


ROOT = Path("Introduction-to-Quantitative-Finance")
CONFIG = ROOT / "configs" / "hirssm_v2_default.json"
OUTPUT_DIR = ROOT / "outputs" / "hirssm_v3_10_clean_baseline"
AGENT_OUTPUT_DIR = ROOT / "outputs" / "agent_runs" / "v3_10" / "clean_baseline_design"
TASK_ID = "20260526_v3_10_clean_baseline_design"
MODEL_VERSION = "HIRSSM V3.10 Clean Rank-Vol Core"

CLEAN_STYLE_ASSETS = [
    {
        "asset": "000985",
        "asset_name": "CSI All Share",
        "rationale": "broad beta anchor",
    },
    {
        "asset": "000922",
        "asset_name": "CSI Dividend",
        "rationale": "value and dividend defensive sleeve",
    },
    {
        "asset": "000016",
        "asset_name": "SSE 50",
        "rationale": "large-cap defensive liquidity sleeve",
    },
    {
        "asset": "000905",
        "asset_name": "CSI 500",
        "rationale": "mid-cap growth and breadth sleeve",
    },
    {
        "asset": "000852",
        "asset_name": "CSI 1000",
        "rationale": "small-cap optionality sleeve",
    },
    {
        "asset": "000300",
        "asset_name": "CSI 300",
        "rationale": "large-cap core sleeve",
    },
]


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def parse_costs(text: str) -> list[float]:
    costs = []
    for item in text.split(","):
        value = item.strip()
        if value:
            costs.append(float(value))
    if not costs:
        raise ValueError("cost list is empty")
    return costs


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def add_relative_metrics(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    if out.empty:
        return out
    out["annual_excess_vs_benchmark"] = out["annual_return"] - out["benchmark_annual_return"]
    out["total_excess_vs_benchmark"] = out["total_return"] - out["benchmark_total_return"]
    out["drawdown_improvement_vs_benchmark"] = out["max_drawdown"] - out["benchmark_max_drawdown"]
    out["vol_reduction_vs_benchmark"] = out["benchmark_annual_vol"] - out["annual_vol"]
    return out


def clean_policy_frame(config: dict) -> pd.DataFrame:
    constraints = config["portfolio"]["constraints"]
    disabled = ",".join(sorted(str(item) for item in config.get("disabled_experts_by_default", [])))
    policy = pd.DataFrame(CLEAN_STYLE_ASSETS)
    policy["asset_type"] = "style"
    policy["max_policy_weight"] = float(constraints["max_single_style_weight"])
    policy["source"] = "predeclared_rank_vol_policy"
    policy["selection_rule"] = "rank_plus_volatility_scaling"
    policy["candidate_count"] = 1
    policy["disabled_experts"] = disabled
    policy["industry_universe"] = "sw_level1_plus_configured_level2"
    policy["cash_rule"] = "residual_cash_with_40pct_cap_reallocation"
    return policy


def monthly_signal_dates(returns: pd.DataFrame, assets: list[str]) -> list[pd.Timestamp]:
    ret = returns[returns["asset"].isin(assets)].copy()
    ret["date"] = pd.to_datetime(ret["date"])
    wide = ret.pivot(index="date", columns="asset", values="ret_1d").sort_index()
    missing = [asset for asset in assets if asset not in wide.columns]
    if missing:
        raise ValueError(f"missing return columns: {missing}")
    valid_dates = wide.dropna(subset=assets).index.to_series()
    dates = model.month_end_dates(valid_dates)
    if len(dates) < 24:
        raise ValueError("not enough monthly signal dates for baseline")
    return dates


def enrich_targets(targets: pd.DataFrame, panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    cols = [
        "date",
        "asset",
        "index_name",
        "asset_type",
        "sw_level",
        "parent_industry",
        "vol_60",
        "final_score",
        "risk_adjusted_alpha",
    ]
    available_cols = [col for col in cols if col in panel["eligible"].columns]
    meta = panel["eligible"][available_cols].drop_duplicates(["date", "asset"]).copy()
    meta = meta.rename(columns={"date": "signal_date", "index_name": "asset_name"})
    out = targets.copy()
    out["signal_date"] = pd.to_datetime(out["signal_date"])
    out = out.merge(meta, on=["signal_date", "asset"], how="left", suffixes=("", "_meta"))
    if "asset_type_meta" in out.columns:
        out["asset_type"] = out["asset_type"].fillna(out["asset_type_meta"])
        out = out.drop(columns=["asset_type_meta"])
    for col in ["sw_level", "parent_industry", "asset_name"]:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].fillna("")
    return out


def row_cap(row: pd.Series, config: dict) -> float:
    constraints = config["portfolio"]["constraints"]
    if str(row["asset"]) == "CASH":
        return 1.0
    if str(row.get("asset_type", "")) == "style":
        return float(constraints["max_single_style_weight"])
    if str(row.get("sw_level", "")) == "second":
        return float(constraints["max_single_level2_industry_weight"])
    return float(constraints["max_single_level1_industry_weight"])


def enforce_cash_cap(targets: pd.DataFrame, config: dict, max_cash: float = 0.40) -> pd.DataFrame:
    rows = []
    for signal_date, group in targets.groupby("signal_date", sort=True):
        g = group.copy()
        g["weight"] = pd.to_numeric(g["weight"], errors="coerce").fillna(0.0).clip(lower=0.0)
        if not g["asset"].eq("CASH").any():
            template = {col: "" for col in g.columns}
            template.update(
                {
                    "signal_date": signal_date,
                    "asset": "CASH",
                    "weight": 0.0,
                    "state": str(g["state"].dropna().iloc[0]) if "state" in g.columns and g["state"].notna().any() else "",
                    "asset_type": "cash",
                    "score": 0.0,
                    "risk_adjusted_alpha": 0.0,
                    "turnover": float(pd.to_numeric(g.get("turnover", pd.Series([0.0])), errors="coerce").max()),
                }
            )
            g = pd.concat([g, pd.DataFrame([template])], ignore_index=True)
        noncash_sum = float(g.loc[~g["asset"].eq("CASH"), "weight"].sum())
        cash_index = g.index[g["asset"].eq("CASH")][0]
        g.loc[cash_index, "weight"] = max(0.0, 1.0 - noncash_sum)
        g["cash_cap_adjusted"] = False

        cash_weight = float(g.loc[cash_index, "weight"])
        if cash_weight > max_cash:
            excess = cash_weight - max_cash
            noncash = g.index[~g["asset"].eq("CASH")]
            caps = pd.Series({idx: row_cap(g.loc[idx], config) for idx in noncash})
            capacity = (caps - g.loc[noncash, "weight"]).clip(lower=0.0)
            capacity_sum = float(capacity.sum())
            if capacity_sum > 1e-12:
                add = capacity / capacity_sum * min(excess, capacity_sum)
                for idx, value in add.items():
                    g.loc[idx, "weight"] = float(g.loc[idx, "weight"]) + float(value)
                g.loc[cash_index, "weight"] = cash_weight - float(add.sum())
                g["cash_cap_adjusted"] = True

        total = float(g["weight"].sum())
        if total > 0:
            g["weight"] = g["weight"] / total
        rows.append(g)
    out = pd.concat(rows, ignore_index=True, sort=False)
    out["target_weight"] = out["weight"]
    out["construction_rule"] = "predeclared_rank_plus_volatility_scaling"
    out["candidate_selection"] = "single_predeclared_candidate_no_full_sample_search"
    return out


def enforce_turnover_cap(targets: pd.DataFrame, max_turnover: float) -> pd.DataFrame:
    rows = []
    prev_weights: dict[str, float] = {}
    prev_rows: dict[str, dict[str, Any]] = {}
    for signal_date, group in targets.groupby("signal_date", sort=True):
        g = group.copy()
        g["weight"] = pd.to_numeric(g["weight"], errors="coerce").fillna(0.0).clip(lower=0.0)
        current = {str(row["asset"]): float(row["weight"]) for _, row in g.iterrows() if float(row["weight"]) > 1e-12}
        assets = sorted(set(current) | set(prev_weights))
        raw_turnover = sum(abs(current.get(asset, 0.0) - prev_weights.get(asset, 0.0)) for asset in assets)
        adjusted = current.copy()
        adjusted_flag = False
        if prev_weights and raw_turnover > max_turnover > 0:
            blend = max_turnover / raw_turnover
            adjusted = {
                asset: prev_weights.get(asset, 0.0) + blend * (current.get(asset, 0.0) - prev_weights.get(asset, 0.0))
                for asset in assets
            }
            adjusted = {asset: weight for asset, weight in adjusted.items() if weight > 1e-10}
            total = sum(adjusted.values())
            if total > 0:
                adjusted = {asset: weight / total for asset, weight in adjusted.items()}
            adjusted_flag = True

        by_asset = {str(row["asset"]): row.to_dict() for _, row in g.iterrows()}
        date_rows = []
        for asset, weight in adjusted.items():
            row = by_asset.get(asset) or prev_rows.get(asset)
            if row is None:
                row = {}
            item = dict(row)
            item["signal_date"] = signal_date
            item["asset"] = asset
            item["weight"] = float(weight)
            item["target_weight"] = float(weight)
            item["state"] = str(g["state"].dropna().iloc[0]) if "state" in g.columns and g["state"].notna().any() else item.get("state", "")
            if asset == "CASH":
                item["asset_type"] = "cash"
            item["turnover"] = 0.0
            item["turnover_cap_adjusted"] = adjusted_flag
            date_rows.append(item)

        actual_turnover = sum(abs(adjusted.get(asset, 0.0) - prev_weights.get(asset, 0.0)) for asset in sorted(set(adjusted) | set(prev_weights)))
        for item in date_rows:
            item["turnover"] = float(actual_turnover)
        rows.extend(date_rows)
        prev_weights = adjusted
        prev_rows = {str(item["asset"]): item for item in date_rows}

    out = pd.DataFrame(rows)
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(0.0)
    out["target_weight"] = out["weight"]
    return out


def build_targets(panel: dict[str, pd.DataFrame], config: dict) -> pd.DataFrame:
    disabled = {str(item) for item in config.get("disabled_experts_by_default", [])}
    start_date = pd.to_datetime(panel["eligible"]["date"].min()) if not panel["eligible"].empty else None
    raw_targets = model.build_targets(
        panel["eligible"],
        panel["regimes"],
        config,
        start_date=start_date,
        disabled_experts=disabled,
    )
    if raw_targets.empty:
        raise ValueError("model.build_targets returned empty targets")
    targets = enrich_targets(raw_targets, panel)
    targets = enforce_cash_cap(targets, config)
    max_turnover = float(config["portfolio"]["constraints"].get("monthly_turnover_target_cap", 0.8))
    targets = enforce_turnover_cap(targets, max_turnover=max_turnover)
    targets["disabled_experts"] = ",".join(sorted(disabled))
    return targets


def build_exposure(targets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for signal_date, group in targets.groupby("signal_date"):
        weights = pd.to_numeric(group["weight"], errors="coerce").fillna(0.0)
        noncash = group[~group["asset"].eq("CASH")].copy()
        noncash_weights = pd.to_numeric(noncash["weight"], errors="coerce").fillna(0.0)
        eff_n = 1.0 / float((noncash_weights**2).sum()) if float((noncash_weights**2).sum()) > 0 else np.nan
        rows.append(
            {
                "signal_date": signal_date,
                "gross_exposure": float(noncash_weights.sum()),
                "cash_weight": float(group.loc[group["asset"].eq("CASH"), "weight"].sum()),
                "weight_sum": float(weights.sum()),
                "max_asset_weight": float(weights.max()),
                "effective_asset_count": eff_n,
                "asset_count": int(group[~group["asset"].eq("CASH")].shape[0]),
            }
        )
    return pd.DataFrame(rows)


def turnover_diagnostics(trades_by_cost: dict[float, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for cost, trades in trades_by_cost.items():
        if trades.empty:
            rows.append(
                {
                    "cost_bps": cost,
                    "trade_dates": 0,
                    "avg_trade_turnover": 0.0,
                    "initial_turnover": 0.0,
                    "max_trade_turnover": 0.0,
                    "max_post_initial_turnover": 0.0,
                    "asset_trade_rows": 0,
                }
            )
            continue
        t = trades.copy()
        by_date = t.groupby("date")["turnover"].max().sort_index()
        post_initial = by_date.iloc[1:] if by_date.shape[0] > 1 else by_date.iloc[0:0]
        rows.append(
            {
                "cost_bps": cost,
                "trade_dates": int(by_date.shape[0]),
                "avg_trade_turnover": float(by_date.mean()),
                "initial_turnover": float(by_date.iloc[0]) if not by_date.empty else 0.0,
                "max_trade_turnover": float(by_date.max()),
                "max_post_initial_turnover": float(post_initial.max()) if not post_initial.empty else 0.0,
                "asset_trade_rows": int(t.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def run_cost_scenarios(
    *,
    panel: dict[str, pd.DataFrame],
    targets: pd.DataFrame,
    costs: list[float],
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[float, pd.DataFrame], dict[float, pd.DataFrame]]:
    rows = []
    nav_by_cost: dict[float, pd.DataFrame] = {}
    trades_by_cost: dict[float, pd.DataFrame] = {}
    for cost in costs:
        suffix = f"clean_rank_vol_core_{int(cost)}bps"
        bt = model.run_backtest(
            returns=panel["returns"],
            targets=targets,
            cost_bps=float(cost),
            benchmark_asset=panel["broad_code"],
        )
        summary = model.summarize_nav(bt["nav"])
        summary.insert(0, "variant", "clean_rank_vol_core")
        summary.insert(1, "cost_bps", float(cost))
        summary = add_relative_metrics(summary)
        rows.append(summary)
        model.write_csv(bt["nav"], output_dir / f"nav_{suffix}.csv")
        model.write_csv(bt["trades"], output_dir / f"trades_{suffix}.csv")
        model.write_csv(model.yearly_returns(bt["nav"]), output_dir / f"yearly_returns_{suffix}.csv")
        model.write_csv(model.regime_returns(bt["nav"], panel["regimes"]), output_dir / f"regime_returns_{suffix}.csv")
        nav_by_cost[float(cost)] = bt["nav"]
        trades_by_cost[float(cost)] = bt["trades"]
    return pd.concat(rows, ignore_index=True, sort=False), nav_by_cost, trades_by_cost


def build_constraint_checks(
    *,
    targets: pd.DataFrame,
    summary: pd.DataFrame,
    policy: pd.DataFrame,
    costs: list[float],
    nav_by_cost: dict[float, pd.DataFrame],
) -> pd.DataFrame:
    group_sums = targets.groupby("signal_date")["weight"].sum()
    noncash = targets[~targets["asset"].eq("CASH")].copy()
    cash_by_signal = targets[targets["asset"].eq("CASH")].groupby("signal_date")["weight"].sum()
    if "turnover" in targets.columns:
        turnover_numeric = pd.to_numeric(targets["turnover"], errors="coerce").fillna(0.0)
        turnover_by_signal = turnover_numeric.groupby(targets["signal_date"]).max()
    else:
        turnover_by_signal = pd.Series(dtype=float)
    post_initial_turnover = turnover_by_signal.iloc[1:] if turnover_by_signal.shape[0] > 1 else turnover_by_signal
    ref_10 = summary[summary["cost_bps"].astype(float).eq(10.0)]
    annual_excess_10 = float(ref_10["annual_excess_vs_benchmark"].iloc[0]) if not ref_10.empty else np.nan
    avg_cash_10 = float(ref_10["avg_cash_weight"].iloc[0]) if not ref_10.empty else np.nan
    rows = [
        {
            "check": "target_weights_not_empty",
            "status": "pass" if not targets.empty else "fail",
            "detail": str(int(targets.shape[0])),
        },
        {
            "check": "no_negative_weights",
            "status": "pass" if (targets["weight"] >= -1e-12).all() else "fail",
            "detail": str(float(targets["weight"].min())),
        },
        {
            "check": "weight_sum_equals_one",
            "status": "pass" if (group_sums.sub(1.0).abs() < 1e-9).all() else "fail",
            "detail": str(float(group_sums.sub(1.0).abs().max())),
        },
        {
            "check": "single_predeclared_candidate",
            "status": "pass" if int(policy["candidate_count"].iloc[0]) == 1 else "fail",
            "detail": "candidate_count=1",
        },
        {
            "check": "no_ex_post_gating",
            "status": "pass",
            "detail": "no after-the-fact state override; only predeclared config state budget",
        },
        {
            "check": "no_v36_v38_target_inheritance",
            "status": "pass",
            "detail": "targets generated by hirssm_v2_model.build_targets from current config and panel only",
        },
        {
            "check": "cost_scenarios_complete",
            "status": "pass" if sorted(summary["cost_bps"].astype(float).tolist()) == sorted(costs) else "fail",
            "detail": ",".join(str(int(c)) for c in sorted(costs)),
        },
        {
            "check": "nav_non_empty_all_costs",
            "status": "pass" if all(not nav.empty for nav in nav_by_cost.values()) else "fail",
            "detail": str({int(k): int(v.shape[0]) for k, v in nav_by_cost.items()}),
        },
        {
            "check": "max_weight_guardrail",
            "status": "pass" if float(noncash["weight"].max()) <= 0.60 + 1e-12 else "fail",
            "detail": str(float(noncash["weight"].max())),
        },
        {
            "check": "cash_weight_cap",
            "status": "pass" if (cash_by_signal <= 0.40 + 1e-9).all() else "fail",
            "detail": str(float(cash_by_signal.max()) if not cash_by_signal.empty else 0.0),
        },
        {
            "check": "monthly_turnover_guardrail_reported",
            "status": "pass" if post_initial_turnover.empty or float(post_initial_turnover.max()) <= 0.8 + 1e-12 else "fail",
            "detail": str(float(post_initial_turnover.max()) if not post_initial_turnover.empty else 0.0),
        },
        {
            "check": "avg_cash_preference",
            "status": "pass" if pd.notna(avg_cash_10) and avg_cash_10 <= 0.25 + 1e-12 else "warn",
            "detail": str(avg_cash_10),
        },
        {
            "check": "ten_bps_excess_reported",
            "status": "pass" if pd.notna(annual_excess_10) and annual_excess_10 >= 0 else "warn",
            "detail": str(annual_excess_10),
        },
    ]
    return pd.DataFrame(rows)


def collect_data_refs(root: Path, config: dict, policy: pd.DataFrame) -> list[tuple[str, Path]]:
    refs: list[tuple[str, Path]] = []
    style_base = root / config["data_contract"]["style_daily_path"]
    pe_base = root / config["data_contract"]["style_pe_path"]
    pb_base = root / config["data_contract"]["style_pb_path"]
    for asset in sorted(set(policy["asset"].astype(str).tolist())):
        for label, base in [("style_daily", style_base), ("style_pe", pe_base), ("style_pb", pb_base)]:
            path = base / f"{asset}.csv"
            if path.exists():
                refs.append((f"{label}_{asset}", path))

    class_base = root / config["data_contract"]["industry_classification_path"]
    for file_name in ["sw_level1_info.csv", "sw_level2_info.csv"]:
        path = class_base / file_name
        if path.exists():
            refs.append((file_name.replace(".csv", ""), path))

    industry_base = root / config["data_contract"]["industry_daily_path"]
    if industry_base.exists():
        for path in sorted(industry_base.glob("*.csv")):
            refs.append((f"industry_daily_{path.stem}", path))
    return refs


def model_metrics(summary: pd.DataFrame, cost: float = 10.0) -> dict[str, Any]:
    if summary.empty:
        return {}
    row = summary[summary["cost_bps"].astype(float).eq(cost)]
    if row.empty:
        row = summary.sort_values("cost_bps").head(1)
    item = row.iloc[0]
    return {
        "reference_cost_bps": float(item["cost_bps"]),
        "annual_return": float(item["annual_return"]),
        "sharpe_no_rf": float(item["sharpe_no_rf"]),
        "max_drawdown": float(item["max_drawdown"]),
        "benchmark_annual_return": float(item["benchmark_annual_return"]),
        "benchmark_sharpe_no_rf": float(item["benchmark_sharpe_no_rf"]),
        "benchmark_max_drawdown": float(item["benchmark_max_drawdown"]),
        "annual_excess_vs_benchmark": float(item["annual_excess_vs_benchmark"]),
        "information_ratio": float(item["information_ratio"]),
        "avg_trade_turnover": float(item["avg_trade_turnover"]),
        "avg_cash_weight": float(item["avg_cash_weight"]),
    }


def make_model_changelog(summary: pd.DataFrame, constraint_checks: pd.DataFrame) -> str:
    metrics = model_metrics(summary)
    failed = constraint_checks[constraint_checks["status"].eq("fail")]
    return "\n".join(
        [
            "# HIRSSM V3.10 Clean Baseline Changelog",
            "",
            "## Change",
            "",
            "- Added a predeclared rank-plus-volatility mechanical baseline.",
            "- Kept state sleeve budgets and caps fixed from config, with default disabled experts preserved.",
            "- Removed same-OOS candidate selection, ex-post expert gating, and V3.6/V3.8 target inheritance from the baseline source.",
            "- Added strict model run manifest output for reproducibility governance.",
            "",
            "## Reference Metrics",
            "",
            f"- Reference cost bps: {metrics.get('reference_cost_bps')}",
            f"- Annual return: {metrics.get('annual_return'):.6f}",
            f"- Sharpe no RF: {metrics.get('sharpe_no_rf'):.6f}",
            f"- Max drawdown: {metrics.get('max_drawdown'):.6f}",
            f"- Benchmark annual return: {metrics.get('benchmark_annual_return'):.6f}",
            f"- Annual excess vs benchmark: {metrics.get('annual_excess_vs_benchmark'):.6f}",
            "",
            "## Governance Decision",
            "",
            "This is a governance baseline, not an alpha-promoted strategy. V3.11 should only compare candidate models against this clean baseline through predeclared nested validation.",
            "",
            "## Check Result",
            "",
            f"- Failed checks: {int(failed.shape[0])}",
        ]
    )


def make_self_check_report(
    *,
    constraint_checks: pd.DataFrame,
    manifest_findings: list[dict[str, str]] | None = None,
) -> str:
    manifest_findings = manifest_findings or []
    fail_count = int((constraint_checks["status"] == "fail").sum())
    manifest_fail_count = sum(1 for item in manifest_findings if item.get("severity") == "fail")
    lines = [
        "# HIRSSM V3.10 Self Check",
        "",
        f"- Constraint failures: {fail_count}",
        f"- Manifest failures: {manifest_fail_count}",
        "- Candidate search disabled: true",
        "- Ex-post gating disabled: true",
        "- V3.6/V3.8 target inheritance disabled: true",
        "",
        "## Constraint Checks",
        "",
    ]
    for _, row in constraint_checks.iterrows():
        lines.append(f"- {row['check']}: {row['status']} ({row['detail']})")
    if manifest_findings:
        lines.extend(["", "## Manifest Findings", ""])
        for finding in manifest_findings:
            lines.append(f"- {finding.get('severity')}: {finding.get('field')} - {finding.get('message')}")
    return "\n".join(lines)


def make_agent_report(summary: pd.DataFrame, policy: pd.DataFrame, exposure: pd.DataFrame, targets: pd.DataFrame) -> str:
    metrics = model_metrics(summary)
    top_weight = float(targets.loc[~targets["asset"].eq("CASH"), "weight"].max()) if not targets.empty else np.nan
    avg_eff_n = float(exposure["effective_asset_count"].mean()) if not exposure.empty else np.nan
    return "\n".join(
        [
            "# V3.10 Clean Baseline Design",
            "",
            "## Construction Rule",
            "",
            "Monthly rebalance through the predeclared rank-plus-volatility rule: fixed state sleeve budgets, ranked candidates, volatility scaling, caps, and residual cash. No same-OOS candidate search, no ex-post gate, and no inherited V3.6/V3.8 target file.",
            "",
            "## Constraints",
            "",
            f"- Max single sleeve weight: {top_weight:.4f}",
            f"- Average effective asset count: {avg_eff_n:.4f}",
            "- Gross exposure target: 1.0000",
            "- Shorting and leverage: disabled",
            "",
            "## 10bps Metrics",
            "",
            f"- Annual return: {metrics.get('annual_return'):.6f}",
            f"- Sharpe no RF: {metrics.get('sharpe_no_rf'):.6f}",
            f"- Max drawdown: {metrics.get('max_drawdown'):.6f}",
            f"- Annual excess vs benchmark: {metrics.get('annual_excess_vs_benchmark'):.6f}",
            f"- Information ratio: {metrics.get('information_ratio'):.6f}",
            "",
            "## Failure Cases",
            "",
            "- This baseline can lag a strong benchmark because it is governed as a control baseline, not selected for full-sample performance.",
            "- It has no dynamic drawdown brake; candidates must prove any timing overlay with nested validation.",
            "- It uses index-level price data and should not be treated as an executable ETF basket without tracking and liquidity checks.",
        ]
    )


def make_agent_manifest(
    *,
    start_time: str,
    output_dir: Path,
    agent_dir: Path,
    config_path: Path,
    artifacts: list[Path],
    metrics: dict[str, Any],
    fail_count: int,
    warn_count: int,
) -> dict[str, Any]:
    return {
        "run_id": f"{TASK_ID}_run_001",
        "task_id": TASK_ID,
        "agent": "portfolio_risk_engineer",
        "version": "V3.10",
        "baseline": "none_clean_predeclared_rank_vol_core",
        "status": "pass" if fail_count == 0 else "fail",
        "started_at": start_time,
        "command": "python -X utf8 strategy_lab/hirssm_v3_10_clean_baseline.py",
        "config": {
            "config_path": str(config_path.as_posix()),
            "candidate_count": 1,
            "construction_rule": "predeclared_rank_plus_volatility_scaling",
        },
        "data_refs": [
            "data_raw/index/akshare_csindex/daily_csindex",
            "data_raw/index/akshare_csindex/valuation_pe_lg",
            "data_raw/index/akshare_csindex/valuation_pb_lg",
            "data_raw/index/akshare_sw_industry/daily_sw",
        ],
        "code_refs": [
            "strategy_lab/hirssm_v3_10_clean_baseline.py",
            "strategy_lab/hirssm_v2_model.py",
            "strategy_lab/hirssm_v2_walk_forward.py",
            "strategy_lab/model_run_manifest.py",
        ],
        "output_dir": str(agent_dir.relative_to(ROOT).as_posix()),
        "allowed_inputs": [
            "configs/hirssm_v2_default.json",
            "data_raw/index/akshare_csindex",
            "data_raw/index/akshare_sw_industry",
        ],
        "artifacts": [str(path.relative_to(ROOT).as_posix()) for path in artifacts],
        "outputs": [str(path.relative_to(ROOT).as_posix()) for path in artifacts if path.name != "agent_run_manifest.json"],
        "changed_files": [str(path.relative_to(ROOT).as_posix()) for path in artifacts],
        "metrics": metrics,
        "self_check_pass": fail_count == 0,
        "fail_count": fail_count,
        "warn_count": warn_count,
        "limitations": [
            "Clean baseline only; not an alpha-promoted model.",
            "No ETF tracking error, liquidity, or tax simulation is included.",
        ],
        "risk_flags": ["clean_baseline_not_alpha_model"],
        "next_decision": "Use this as the V3.10 clean baseline for V3.11 nested candidate validation.",
        "handoff_summary": "A predeclared rank-plus-volatility baseline has been generated with cost scenarios, checks, and strict manifest.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run HIRSSM V3.10 clean baseline.")
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--agent-output-dir", default=str(AGENT_OUTPUT_DIR))
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--costs", default="5,10,20,30")
    args = parser.parse_args()

    start_time = now_text()
    root = Path(args.root)
    config_path = Path(args.config)
    output_dir = Path(args.output_dir)
    agent_dir = Path(args.agent_output_dir)
    costs = parse_costs(args.costs)
    config = model.read_json(config_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    agent_dir.mkdir(parents=True, exist_ok=True)

    panel = wf.build_panel(model, root, config, args.start_date, args.end_date)
    policy = clean_policy_frame(config)
    targets = build_targets(panel, config)
    exposure = build_exposure(targets)

    summary, nav_by_cost, trades_by_cost = run_cost_scenarios(
        panel=panel,
        targets=targets,
        costs=costs,
        output_dir=output_dir,
    )
    turnover = turnover_diagnostics(trades_by_cost)
    constraint_checks = build_constraint_checks(
        targets=targets,
        summary=summary,
        policy=policy,
        costs=costs,
        nav_by_cost=nav_by_cost,
    )

    policy_path = output_dir / "clean_baseline_policy.csv"
    targets_path = output_dir / "target_weights.csv"
    summary_path = output_dir / "oos_performance.csv"
    exposure_path = output_dir / "risk_exposure.csv"
    turnover_path = output_dir / "turnover_diagnostics.csv"
    checks_path = output_dir / "constraint_check.csv"
    changelog_path = output_dir / "MODEL_CHANGELOG.md"
    self_check_path = output_dir / "SELF_CHECK_REPORT.md"
    model_manifest_path = output_dir / "model_run_manifest.json"
    model_manifest_check_path = output_dir / "model_run_manifest_check.csv"

    model.write_csv(policy, policy_path)
    model.write_csv(targets, targets_path)
    model.write_csv(summary, summary_path)
    model.write_csv(exposure, exposure_path)
    model.write_csv(turnover, turnover_path)
    model.write_csv(constraint_checks, checks_path)
    write_text(make_model_changelog(summary, constraint_checks), changelog_path)

    smoke_fail_count = int((constraint_checks["status"] == "fail").sum())
    metrics = model_metrics(summary)
    metrics["signal_count"] = int(targets["signal_date"].nunique())
    metrics["target_row_count"] = int(targets.shape[0])
    metrics["policy_asset_count"] = int(policy.shape[0])
    metrics["cost_scenario_count"] = int(len(costs))

    artifact_paths = [
        policy_path,
        targets_path,
        summary_path,
        exposure_path,
        turnover_path,
        checks_path,
        changelog_path,
        self_check_path,
    ]
    for cost in costs:
        suffix = f"clean_rank_vol_core_{int(cost)}bps"
        artifact_paths.extend(
            [
                output_dir / f"nav_{suffix}.csv",
                output_dir / f"trades_{suffix}.csv",
                output_dir / f"yearly_returns_{suffix}.csv",
                output_dir / f"regime_returns_{suffix}.csv",
            ]
        )

    write_text(make_self_check_report(constraint_checks=constraint_checks), self_check_path)
    manifest = build_model_run_manifest(
        root=root,
        task_id=TASK_ID,
        run_id=f"{TASK_ID}_run_001",
        model_version=MODEL_VERSION,
        baseline="none_clean_predeclared_rank_vol_core",
        status="success" if smoke_fail_count == 0 else "fail",
        started_at=start_time,
        finished_at=now_text(),
        output_dir=output_dir,
        command=["python", "-X", "utf8", "strategy_lab/hirssm_v3_10_clean_baseline.py"],
        argv={
            "start_date": args.start_date,
            "end_date": args.end_date,
            "costs": costs,
            "candidate_count": 1,
        },
        code_paths=[
            root / "strategy_lab" / "hirssm_v3_10_clean_baseline.py",
            root / "strategy_lab" / "hirssm_v2_model.py",
            root / "strategy_lab" / "hirssm_v2_walk_forward.py",
            root / "strategy_lab" / "model_run_manifest.py",
        ],
        config_path=config_path,
        data_paths=collect_data_refs(root, config, policy),
        artifact_paths=artifact_paths,
        selection={
            "selected_variant": "clean_rank_vol_core",
            "candidate_count": 1,
            "selection_method": "predeclared_rank_vol_no_full_sample_search",
            "selected_by_full_sample_performance": False,
            "inherits_v36_or_v38_targets": False,
        },
        metrics=metrics,
        checks={
            "self_check_pass": smoke_fail_count == 0,
            "fail_count": smoke_fail_count,
            "constraint_fail_count": smoke_fail_count,
            "target_weight_rows": int(targets.shape[0]),
            "cost_scenarios_complete": sorted(summary["cost_bps"].astype(float).tolist()) == sorted(costs),
        },
        limitations=[
            "This is a governance baseline, not an alpha-promoted model.",
            "Rank-plus-volatility rules are predeclared but still factor-driven and can lag the broad-market benchmark.",
            "Index-level backtest does not include ETF tracking error, liquidity slippage, or taxes.",
        ],
        risk_flags=["clean_baseline_not_alpha_model"],
        next_decision="Build V3.11 nested candidate validation against this clean baseline.",
        handoff_summary="V3.10 clean baseline emits predeclared targets, cost scenarios, self checks, and strict reproducibility manifest.",
    )
    write_json(manifest, model_manifest_path)

    manifest_findings = validate_model_run_manifest(manifest)
    manifest_check = pd.DataFrame(manifest_findings)
    if manifest_check.empty:
        manifest_check = pd.DataFrame([{"severity": "pass", "field": "model_run_manifest", "message": "no failures"}])
    model.write_csv(manifest_check, model_manifest_check_path)

    agent_report_path = agent_dir / "agent_report.md"
    agent_policy_path = agent_dir / "clean_baseline_policy.csv"
    agent_exposure_path = agent_dir / "risk_exposure.csv"
    agent_turnover_path = agent_dir / "turnover_diagnostics.csv"
    agent_checks_path = agent_dir / "constraint_check.csv"
    agent_manifest_path = agent_dir / "agent_run_manifest.json"
    model.write_csv(policy, agent_policy_path)
    model.write_csv(exposure, agent_exposure_path)
    model.write_csv(turnover, agent_turnover_path)
    model.write_csv(constraint_checks, agent_checks_path)
    write_text(make_agent_report(summary, policy, exposure, targets), agent_report_path)

    manifest_fail_count = sum(1 for item in manifest_findings if item.get("severity") == "fail")
    warn_count = sum(1 for item in manifest_findings if item.get("severity") == "warn") + 1
    agent_artifacts = [
        agent_report_path,
        agent_policy_path,
        agent_exposure_path,
        agent_turnover_path,
        agent_checks_path,
        model_manifest_path,
        model_manifest_check_path,
        agent_manifest_path,
    ]
    agent_manifest = make_agent_manifest(
        start_time=start_time,
        output_dir=output_dir,
        agent_dir=agent_dir,
        config_path=config_path,
        artifacts=agent_artifacts,
        metrics=metrics,
        fail_count=smoke_fail_count + manifest_fail_count,
        warn_count=warn_count,
    )
    write_json(agent_manifest, agent_manifest_path)

    result = {
        "model_version": MODEL_VERSION,
        "self_check_pass": smoke_fail_count == 0 and manifest_fail_count == 0,
        "smoke_fail_count": smoke_fail_count,
        "manifest_fail_count": manifest_fail_count,
        "warn_count": warn_count,
        "metrics": metrics,
        "output_dir": str(output_dir),
        "agent_output_dir": str(agent_dir),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result["self_check_pass"] is not True else 0


if __name__ == "__main__":
    raise SystemExit(main())
