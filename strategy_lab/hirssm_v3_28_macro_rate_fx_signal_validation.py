#!/usr/bin/env python
"""HIRSSM V3.28 macro rate/FX signal validation.

V3.27 created point-in-time macro tables and allowed the
cn_us_rate_spread_risk_budget family to enter signal validation. V3.28 tests
predeclared rate/FX signals with full-sample diagnostics plus a holdout
implementation gate. This version does not run a strategy harness or promote a
model by itself.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = ROOT / "outputs" / "hirssm_v3_10_clean_baseline"
V320_DIR = ROOT / "outputs" / "agent_runs" / "v3_20" / "rescue_signal_research"
V324_DIR = ROOT / "outputs" / "agent_runs" / "v3_24" / "valuation_source_research"
V325_DIR = ROOT / "outputs" / "agent_runs" / "v3_25" / "industry_structure_source_research"
V327_DIR = ROOT / "outputs" / "agent_runs" / "v3_27" / "macro_data_ingestion"
MACRO_PANEL_PATH = ROOT / "data_raw" / "macro" / "macro_pit_panel.csv"
OUTPUT_DIR = ROOT / "outputs" / "agent_runs" / "v3_28" / "macro_rate_fx_signal_validation"
TASK_ID = "20260527_v3_28_macro_rate_fx_signal_validation"
BASELINE_VARIANT = "v3_10_clean_rank_vol_core"


SIGNAL_SPECS = [
    {
        "variant": "rate_fx_stress_defense",
        "role": "candidate",
        "description": "China-US rate spread is depressed and RMB is weakening, indicating external discount-rate and capital-flow stress.",
        "target": "broad_forward_63d",
        "direction": "negative",
        "trigger": "cn_us_rate_spread_z_252_lag1 <= -0.75 and usdcny_ret60_lag1 >= 0.01",
        "action": "move 5pct from equity risk budget to cash in a later harness",
    },
    {
        "variant": "us_rate_shock_fx_stress_defense",
        "role": "candidate",
        "description": "US 10Y yield rises quickly while RMB weakens, pressuring A-share valuation multiples.",
        "target": "broad_forward_63d",
        "direction": "negative",
        "trigger": "us_10y_chg60_lag1 >= 0.25 and usdcny_ret60_lag1 >= 0.005",
        "action": "move 4pct from high-beta/style sleeves to cash in a later harness",
    },
    {
        "variant": "spread_repair_risk_on",
        "role": "candidate",
        "description": "China-US spread repairs while RMB stabilizes, suggesting external pressure is easing.",
        "target": "broad_forward_63d",
        "direction": "positive",
        "trigger": "cn_us_rate_spread_chg60_lag1 >= 0.20 and usdcny_ret60_lag1 <= 0.0",
        "action": "release 4pct cash to broad/style sleeves in a later harness",
    },
]


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def rolling_zscore(series: pd.Series, window: int = 252, min_periods: int = 126) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    mean = values.rolling(window, min_periods=min_periods).mean()
    std = values.rolling(window, min_periods=min_periods).std()
    return (values - mean) / std.replace(0, np.nan)


def load_macro_wide(path: Path) -> pd.DataFrame:
    panel = read_csv(path)
    if panel.empty:
        raise FileNotFoundError(path)
    panel["available_date"] = pd.to_datetime(panel["available_date"], errors="coerce")
    panel["value"] = pd.to_numeric(panel["value"], errors="coerce")
    panel = panel.dropna(subset=["available_date", "series_id", "value"]).sort_values("available_date")
    wide = panel.pivot_table(index="available_date", columns="series_id", values="value", aggfunc="last").sort_index()
    wide = wide.ffill()
    required = ["cn_10y_gov_bond_yield", "us_10y_treasury_yield", "cn_us_10y_rate_spread", "usdcny"]
    missing = [col for col in required if col not in wide.columns]
    if missing:
        raise RuntimeError(f"missing required macro series: {missing}")
    return wide.reset_index().rename(columns={"available_date": "signal_date"})


def build_daily_feature_panel(macro_path: Path) -> pd.DataFrame:
    wide = load_macro_wide(macro_path)
    out = wide[["signal_date", "cn_10y_gov_bond_yield", "us_10y_treasury_yield", "cn_us_10y_rate_spread", "usdcny"]].copy()
    out["cn_us_rate_spread_z_252"] = rolling_zscore(out["cn_us_10y_rate_spread"])
    out["cn_us_rate_spread_chg20"] = out["cn_us_10y_rate_spread"] - out["cn_us_10y_rate_spread"].shift(20)
    out["cn_us_rate_spread_chg60"] = out["cn_us_10y_rate_spread"] - out["cn_us_10y_rate_spread"].shift(60)
    out["us_10y_chg20"] = out["us_10y_treasury_yield"] - out["us_10y_treasury_yield"].shift(20)
    out["us_10y_chg60"] = out["us_10y_treasury_yield"] - out["us_10y_treasury_yield"].shift(60)
    out["cn_10y_chg60"] = out["cn_10y_gov_bond_yield"] - out["cn_10y_gov_bond_yield"].shift(60)
    out["usdcny_ret20"] = out["usdcny"] / out["usdcny"].shift(20) - 1.0
    out["usdcny_ret60"] = out["usdcny"] / out["usdcny"].shift(60) - 1.0
    out["usdcny_z_252"] = rolling_zscore(out["usdcny"])
    return out


def add_broad_targets(panel: pd.DataFrame) -> pd.DataFrame:
    broad = read_csv(BASELINE_DIR / "nav_clean_rank_vol_core_10bps.csv")
    if broad.empty:
        raise FileNotFoundError(BASELINE_DIR / "nav_clean_rank_vol_core_10bps.csv")
    broad["date"] = pd.to_datetime(broad["date"])
    broad = broad.sort_values("date")
    broad["benchmark_nav"] = pd.to_numeric(broad["benchmark_nav"], errors="coerce")
    broad["broad_forward_63d"] = broad["benchmark_nav"].shift(-63) / broad["benchmark_nav"] - 1.0
    forward_min = broad["benchmark_nav"].rolling(63, min_periods=20).min().shift(-62)
    broad["broad_forward_63d_max_drawdown"] = forward_min / broad["benchmark_nav"] - 1.0
    targets = broad[["date", "broad_forward_63d", "broad_forward_63d_max_drawdown"]].rename(columns={"date": "signal_date"})
    return pd.merge_asof(panel.sort_values("signal_date"), targets, on="signal_date", direction="backward")


def monthly_signal_panel(daily_features: pd.DataFrame) -> pd.DataFrame:
    targets = read_csv(BASELINE_DIR / "target_weights.csv")
    if targets.empty:
        raise FileNotFoundError(BASELINE_DIR / "target_weights.csv")
    dates = pd.DataFrame({"signal_date": pd.to_datetime(sorted(targets["signal_date"].dropna().unique()))})
    panel = pd.merge_asof(dates.sort_values("signal_date"), daily_features.sort_values("signal_date"), on="signal_date", direction="backward")
    lag_cols = [
        "cn_us_10y_rate_spread",
        "cn_us_rate_spread_z_252",
        "cn_us_rate_spread_chg20",
        "cn_us_rate_spread_chg60",
        "us_10y_chg20",
        "us_10y_chg60",
        "cn_10y_chg60",
        "usdcny",
        "usdcny_ret20",
        "usdcny_ret60",
        "usdcny_z_252",
    ]
    for col in lag_cols:
        panel[f"{col}_lag1"] = panel[col].shift(1)
    return add_broad_targets(panel)


def apply_signal(spec: dict[str, str], panel: pd.DataFrame) -> pd.Series:
    if spec["variant"] == "rate_fx_stress_defense":
        return (
            (pd.to_numeric(panel["cn_us_rate_spread_z_252_lag1"], errors="coerce") <= -0.75)
            & (pd.to_numeric(panel["usdcny_ret60_lag1"], errors="coerce") >= 0.01)
        )
    if spec["variant"] == "us_rate_shock_fx_stress_defense":
        return (
            (pd.to_numeric(panel["us_10y_chg60_lag1"], errors="coerce") >= 0.25)
            & (pd.to_numeric(panel["usdcny_ret60_lag1"], errors="coerce") >= 0.005)
        )
    if spec["variant"] == "spread_repair_risk_on":
        return (
            (pd.to_numeric(panel["cn_us_rate_spread_chg60_lag1"], errors="coerce") >= 0.20)
            & (pd.to_numeric(panel["usdcny_ret60_lag1"], errors="coerce") <= 0.0)
        )
    return pd.Series(False, index=panel.index)


def pass_rule(spec: dict[str, str], fwd: pd.Series, unconditional: float, min_obs: int) -> bool:
    obs = int(fwd.shape[0])
    if obs < min_obs:
        return False
    mean_fwd = float(fwd.mean())
    positive_share = float((fwd > 0).mean())
    if spec["direction"] == "negative":
        return bool(mean_fwd < unconditional and positive_share < 0.48)
    return bool(mean_fwd > unconditional and positive_share > 0.52)


def full_sample_validation(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for spec in SIGNAL_SPECS:
        valid = panel.dropna(subset=[spec["target"]]).copy()
        mask = apply_signal(spec, valid).fillna(False)
        selected = valid[mask]
        fwd = pd.to_numeric(selected[spec["target"]], errors="coerce").dropna()
        unconditional = float(pd.to_numeric(valid[spec["target"]], errors="coerce").mean()) if not valid.empty else np.nan
        drawdown = pd.to_numeric(selected["broad_forward_63d_max_drawdown"], errors="coerce").dropna()
        unconditional_drawdown = float(pd.to_numeric(valid["broad_forward_63d_max_drawdown"], errors="coerce").mean()) if not valid.empty else np.nan
        obs = int(fwd.shape[0])
        rows.append(
            {
                "variant": spec["variant"],
                "role": spec["role"],
                "information_source": "macro_rate_fx_pit",
                "target": spec["target"],
                "direction": spec["direction"],
                "trigger": spec["trigger"],
                "observations": obs,
                "coverage": obs / max(int(valid.shape[0]), 1),
                "forward_63d_mean": float(fwd.mean()) if obs else np.nan,
                "unconditional_forward_63d_mean": unconditional,
                "positive_forward_share": float((fwd > 0).mean()) if obs else np.nan,
                "forward_63d_max_drawdown_mean": float(drawdown.mean()) if not drawdown.empty else np.nan,
                "unconditional_forward_63d_max_drawdown_mean": unconditional_drawdown,
                "pass_pre_backtest_gate": pass_rule(spec, fwd, unconditional, min_obs=12),
                "diagnostic_full_sample_only": True,
            }
        )
    return pd.DataFrame(rows)


def holdout_validation(panel: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    valid_dates = panel.dropna(subset=["broad_forward_63d"])["signal_date"].sort_values().reset_index(drop=True)
    split_idx = int(len(valid_dates) * 0.70)
    split_date = valid_dates.iloc[split_idx] if len(valid_dates) else pd.NaT
    rows = []
    for spec in SIGNAL_SPECS:
        full_pass = bool(validation.loc[validation["variant"].eq(spec["variant"]), "pass_pre_backtest_gate"].iloc[0])
        for split_name, split_panel in [
            ("train", panel[panel["signal_date"] < split_date]),
            ("holdout", panel[panel["signal_date"] >= split_date]),
        ]:
            valid = split_panel.dropna(subset=[spec["target"]]).copy()
            mask = apply_signal(spec, valid).fillna(False)
            selected = valid[mask]
            fwd = pd.to_numeric(selected[spec["target"]], errors="coerce").dropna()
            unconditional = float(pd.to_numeric(valid[spec["target"]], errors="coerce").mean()) if not valid.empty else np.nan
            pass_holdout = pass_rule(spec, fwd, unconditional, min_obs=5) if split_name == "holdout" else np.nan
            eligible = bool(split_name == "holdout" and full_pass and pass_holdout)
            reason = ""
            if split_name == "holdout" and not eligible:
                if not full_pass:
                    reason = "full_sample_gate_failed"
                elif not pass_holdout:
                    reason = "holdout_effect_or_sample_gate_failed"
            rows.append(
                {
                    "variant": spec["variant"],
                    "split": split_name,
                    "split_date": split_date,
                    "target": spec["target"],
                    "observations": int(fwd.shape[0]),
                    "forward_63d_mean": float(fwd.mean()) if not fwd.empty else np.nan,
                    "unconditional_forward_63d_mean": unconditional,
                    "positive_forward_share": float((fwd > 0).mean()) if not fwd.empty else np.nan,
                    "pass_full_sample_gate": full_pass,
                    "pass_holdout_gate": pass_holdout,
                    "eligible_for_implementation": eligible,
                    "failure_reason": reason,
                }
            )
    return pd.DataFrame(rows)


def holdout_pass_map(holdout: pd.DataFrame) -> dict[str, bool]:
    if holdout.empty:
        return {}
    rows = holdout[holdout["split"].astype(str).eq("holdout")]
    return {str(row["variant"]): bool(str(row["eligible_for_implementation"]).lower() == "true") for _, row in rows.iterrows()}


def candidate_registry(validation: pd.DataFrame, holdout: pd.DataFrame) -> pd.DataFrame:
    holdout_ok = holdout_pass_map(holdout)
    rows = [
        {
            "variant": BASELINE_VARIANT,
            "role": "control",
            "description": "Frozen V3.10 clean rank-vol baseline.",
            "multipliers_json": json.dumps({"mode": "baseline"}, sort_keys=True),
            "disabled_experts": "range_reversal,style_trend_continuation",
            "selection_source": "v3_10_clean_baseline",
            "diagnostic_full_sample_only": True,
            "eligible_for_default_promotion": False,
        }
    ]
    for spec in SIGNAL_SPECS:
        full_pass = bool(validation.loc[validation["variant"].eq(spec["variant"]), "pass_pre_backtest_gate"].iloc[0])
        passed = bool(full_pass and holdout_ok.get(spec["variant"], False))
        rows.append(
            {
                "variant": spec["variant"],
                "role": "candidate" if passed else "observation",
                "description": spec["description"],
                "multipliers_json": json.dumps({"trigger": spec["trigger"], "action": spec["action"], "target": spec["target"]}, sort_keys=True),
                "disabled_experts": "range_reversal,style_trend_continuation",
                "selection_source": "v3_28_macro_signal_holdout_gate",
                "diagnostic_full_sample_only": True,
                "eligible_for_default_promotion": False,
            }
        )
    return pd.DataFrame(rows)


def implementation_specs(validation: pd.DataFrame, holdout: pd.DataFrame) -> pd.DataFrame:
    holdout_ok = holdout_pass_map(holdout)
    rows = []
    for spec in SIGNAL_SPECS:
        full_pass = bool(validation.loc[validation["variant"].eq(spec["variant"]), "pass_pre_backtest_gate"].iloc[0])
        passed = bool(full_pass and holdout_ok.get(spec["variant"], False))
        rows.append(
            {
                "variant": spec["variant"],
                "implementation_allowed": passed,
                "information_source": "macro_rate_fx_pit",
                "weight_rule": spec["action"],
                "target": spec["target"],
                "acceptance_standard": "must pass full-sample diagnostic and holdout source gate; V3.28 does not authorize default promotion",
            }
        )
    return pd.DataFrame(rows)


def reference_signals(panel: pd.DataFrame) -> pd.DataFrame:
    ref = panel[["signal_date"]].copy()
    v320 = read_csv(V320_DIR / "signal_feature_panel.csv")
    if not v320.empty:
        v320["signal_date"] = pd.to_datetime(v320["signal_date"])
        ref_v320 = v320[["signal_date"]].copy()
        ref_v320["v3_20_vol_compression_reentry"] = (
            (pd.to_numeric(v320["vol_60"], errors="coerce") <= 0.85 * pd.to_numeric(v320["vol_120"], errors="coerce"))
            & (pd.to_numeric(v320["drawdown_252"], errors="coerce") <= -0.10)
            & (pd.to_numeric(v320["benchmark_ret_20"], errors="coerce") > 0)
        ).astype(float)
        ref = ref.merge(ref_v320, on="signal_date", how="left")
    else:
        ref["v3_20_vol_compression_reentry"] = np.nan
    v324 = read_csv(V324_DIR / "valuation_signal_feature_panel.csv")
    if not v324.empty:
        v324["signal_date"] = pd.to_datetime(v324["signal_date"])
        ref_v324 = v324[["signal_date"]].copy()
        ref_v324["v3_24_broad_market_deep_value_repair"] = (
            (pd.to_numeric(v324["broad_pe_pct"], errors="coerce") <= 0.30)
            & (pd.to_numeric(v324["broad_ret20"], errors="coerce") > 0)
        ).astype(float)
        ref = ref.merge(ref_v324, on="signal_date", how="left")
    else:
        ref["v3_24_broad_market_deep_value_repair"] = np.nan
    v325 = read_csv(V325_DIR / "industry_structure_signal_feature_panel.csv")
    if not v325.empty:
        v325["signal_date"] = pd.to_datetime(v325["signal_date"])
        ref_v325 = v325[["signal_date"]].copy()
        ref_v325["v3_25_industry_breadth_repair"] = (
            (pd.to_numeric(v325["above_ma60_ratio_lag1"], errors="coerce") <= 0.45)
            & (pd.to_numeric(v325["positive_ret20_ratio_lag1"], errors="coerce") >= 0.55)
            & (pd.to_numeric(v325["breadth_change_20_lag1"], errors="coerce") > 0.10)
        ).astype(float)
        ref = ref.merge(ref_v325, on="signal_date", how="left")
    else:
        ref["v3_25_industry_breadth_repair"] = np.nan
    cash = read_csv(BASELINE_DIR / "target_weights.csv")
    if not cash.empty:
        cash = cash[cash["asset"].astype(str).eq("CASH")][["signal_date", "weight"]].copy()
        cash["signal_date"] = pd.to_datetime(cash["signal_date"])
        cash = cash.rename(columns={"weight": "v3_10_cash_weight"})
        ref = ref.merge(cash, on="signal_date", how="left")
    else:
        ref["v3_10_cash_weight"] = np.nan
    return ref


def source_orthogonality(panel: pd.DataFrame) -> pd.DataFrame:
    refs = reference_signals(panel)
    rows = []
    references = [
        "v3_20_vol_compression_reentry",
        "v3_24_broad_market_deep_value_repair",
        "v3_25_industry_breadth_repair",
        "v3_10_cash_weight",
    ]
    for spec in SIGNAL_SPECS:
        signal = apply_signal(spec, panel).fillna(False).astype(float)
        for reference in references:
            ref = pd.to_numeric(refs[reference], errors="coerce")
            aligned = pd.DataFrame({"signal": signal, "ref": ref}).dropna()
            corr = float(aligned["signal"].corr(aligned["ref"])) if aligned["signal"].nunique() > 1 and aligned["ref"].nunique() > 1 else np.nan
            rows.append(
                {
                    "variant": spec["variant"],
                    "reference": reference,
                    "correlation": corr,
                    "pass_orthogonality_gate": bool(pd.isna(corr) or abs(corr) < 0.85),
                }
            )
    return pd.DataFrame(rows)


def data_quality(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    required = ["cn_10y_gov_bond_yield", "us_10y_treasury_yield", "cn_us_10y_rate_spread", "usdcny"]
    for col in required:
        rows.append(
            {
                "field": col,
                "nonnull_count": int(panel[col].notna().sum()) if col in panel.columns else 0,
                "start_signal_date": panel.loc[panel[col].notna(), "signal_date"].min() if col in panel.columns and panel[col].notna().any() else "",
                "end_signal_date": panel.loc[panel[col].notna(), "signal_date"].max() if col in panel.columns and panel[col].notna().any() else "",
            }
        )
    return pd.DataFrame(rows)


def make_report(validation: pd.DataFrame, holdout: pd.DataFrame, registry: pd.DataFrame) -> str:
    full_passed = validation[validation["pass_pre_backtest_gate"].astype(bool)]["variant"].astype(str).tolist() if not validation.empty else []
    holdout_passed = holdout[
        holdout["split"].astype(str).eq("holdout") & holdout["eligible_for_implementation"].astype(str).str.lower().eq("true")
    ]["variant"].astype(str).tolist() if not holdout.empty else []
    return "\n".join(
        [
            "# HIRSSM V3.28 Macro Rate/FX Signal Validation",
            "",
            "## Purpose",
            "",
            "Validate the V3.27 data-gated China-US rate spread and USDCNY macro branch.",
            "",
            "## Findings",
            "",
            f"- Signals tested: {int(validation.shape[0])}",
            f"- Passed full-sample diagnostic gate: {', '.join(full_passed) if full_passed else 'none'}",
            f"- Passed holdout implementation gate: {', '.join(holdout_passed) if holdout_passed else 'none'}",
            f"- Candidate rows: {int((registry['role'] == 'candidate').sum())}",
            "",
            "## Decision",
            "",
            "- V3.28 is macro signal validation only.",
            "- Only holdout-qualified macro signals may enter a later model harness.",
            "- No default model promotion is authorized in this version.",
        ]
    )


def failure_cases(holdout: pd.DataFrame) -> str:
    lines = [
        "# V3.28 Macro Rate/FX Failure Cases",
        "",
        "- Rate-spread stress can reflect already-priced policy divergence rather than future equity loss.",
        "- RMB depreciation can coincide with export-sector support, so broad-index sign may be unstable.",
        "- US-rate shocks may be offset by domestic liquidity easing.",
        "- Macro series are vendor snapshots, not full historical vintage databases.",
        "",
        "## Holdout Failures",
        "",
    ]
    failures = holdout[holdout["split"].astype(str).eq("holdout") & ~holdout["eligible_for_implementation"].astype(str).str.lower().eq("true")]
    if failures.empty:
        lines.append("- No holdout failures among implemented candidates.")
    else:
        for _, row in failures.iterrows():
            lines.append(f"- {row['variant']}: {row.get('failure_reason', '')}; observations={row.get('observations', '')}")
    return "\n".join(lines)


def self_check(validation: pd.DataFrame, holdout: pd.DataFrame, registry: pd.DataFrame, specs: pd.DataFrame, ortho: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    candidate_count = int((registry["role"] == "candidate").sum()) if not registry.empty else 0
    required_cols = ["cn_10y_gov_bond_yield", "us_10y_treasury_yield", "cn_us_10y_rate_spread", "usdcny"]
    required_ready = all(col in panel.columns and panel[col].notna().sum() >= 120 for col in required_cols)
    allowed_without_candidate = specs[specs["implementation_allowed"].astype(str).str.lower().eq("true")]
    candidate_variants = set(registry.loc[registry["role"].eq("candidate"), "variant"].astype(str)) if not registry.empty else set()
    implementation_has_candidate = set(allowed_without_candidate["variant"].astype(str)).issubset(candidate_variants)
    return pd.DataFrame(
        [
            {"check": "macro_required_fields_ready", "status": "pass" if required_ready else "fail", "detail": ",".join(required_cols)},
            {"check": "signal_validation_exists", "status": "pass" if not validation.empty else "fail", "detail": str(int(validation.shape[0]))},
            {"check": "signal_gate_holdout_exists", "status": "pass" if not holdout.empty else "fail", "detail": str(int(holdout.shape[0]))},
            {"check": "candidate_registry_exists", "status": "pass" if not registry.empty else "fail", "detail": str(int(registry.shape[0]))},
            {"check": "implementation_specs_exist", "status": "pass" if not specs.empty else "fail", "detail": str(int(specs.shape[0]))},
            {"check": "implementation_allowed_has_candidate_row", "status": "pass" if implementation_has_candidate else "fail", "detail": str(int(allowed_without_candidate.shape[0]))},
            {"check": "source_orthogonality_exists", "status": "pass" if not ortho.empty else "fail", "detail": str(int(ortho.shape[0]))},
            {"check": "at_least_one_candidate_for_harness", "status": "pass" if candidate_count >= 1 else "warn", "detail": str(candidate_count)},
            {"check": "research_only_no_default_promotion", "status": "pass", "detail": "macro signal validation only"},
        ]
    )


def make_manifest(start_time: str, output_dir: Path, artifacts: list[Path], metrics: dict[str, Any], fail_count: int, warn_count: int) -> dict[str, Any]:
    return {
        "run_id": f"{TASK_ID}_run_001",
        "task_id": TASK_ID,
        "agent": "factor_researcher",
        "version": "V3.28",
        "baseline": "HIRSSM V3.10 Clean Rank-Vol Core",
        "status": "pass" if fail_count == 0 else "fail",
        "started_at": start_time,
        "command": "python -X utf8 strategy_lab/hirssm_v3_28_macro_rate_fx_signal_validation.py",
        "config": {"research_only": True, "information_source": "macro_rate_fx_pit", "forward_window_days": 63},
        "data_refs": [
            "data_raw/macro/macro_pit_panel.csv",
            "outputs/hirssm_v3_10_clean_baseline",
            "outputs/agent_runs/v3_27/macro_data_ingestion",
        ],
        "code_refs": ["strategy_lab/hirssm_v3_28_macro_rate_fx_signal_validation.py"],
        "output_dir": rel(output_dir),
        "allowed_inputs": [
            "data_raw/macro/macro_pit_panel.csv",
            "outputs/hirssm_v3_10_clean_baseline",
            "outputs/agent_runs/v3_20/rescue_signal_research",
            "outputs/agent_runs/v3_24/valuation_source_research",
            "outputs/agent_runs/v3_25/industry_structure_source_research",
            "outputs/agent_runs/v3_27/macro_data_ingestion",
        ],
        "artifacts": [rel(path) for path in artifacts],
        "outputs": [rel(path) for path in artifacts if path.name != "agent_run_manifest.json"],
        "changed_files": [rel(path) for path in artifacts],
        "metrics": metrics,
        "self_check_pass": fail_count == 0,
        "fail_count": fail_count,
        "warn_count": warn_count,
        "limitations": [
            "Forward labels are used only for source validation with holdout gating.",
            "Macro event data are vendor snapshots rather than full vintage databases.",
            "V3.28 does not run a portfolio harness or authorize default promotion.",
        ],
        "risk_flags": [
            "macro_signal_instability",
            "rate_fx_regime_shift",
            "source_requires_later_nested_validation",
        ],
        "next_decision": "Implement only holdout-qualified macro rate/FX candidates in V3.29; if none pass, continue source research or strengthen data.",
        "handoff_summary": "V3.28 tested PIT-aligned China-US rate spread and USDCNY macro signals with holdout-gated implementation eligibility.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate V3.28 macro rate/FX signal validation artifacts.")
    parser.add_argument("--macro-panel", default=str(MACRO_PANEL_PATH))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    start_time = now_text()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    daily_features = build_daily_feature_panel(Path(args.macro_panel))
    panel = monthly_signal_panel(daily_features)
    validation = full_sample_validation(panel)
    holdout = holdout_validation(panel, validation)
    registry = candidate_registry(validation, holdout)
    specs = implementation_specs(validation, holdout)
    ortho = source_orthogonality(panel)
    quality = data_quality(panel)
    checks = self_check(validation, holdout, registry, specs, ortho, panel)

    panel_path = output_dir / "macro_rate_fx_signal_feature_panel.csv"
    validation_path = output_dir / "signal_validation.csv"
    holdout_path = output_dir / "signal_gate_holdout_validation.csv"
    registry_path = output_dir / "candidate_registry.csv"
    specs_path = output_dir / "implementation_candidate_spec.csv"
    ortho_path = output_dir / "source_orthogonality_check.csv"
    quality_path = output_dir / "data_quality_report.csv"
    failure_path = output_dir / "factor_failure_cases.md"
    report_path = output_dir / "agent_report.md"
    checks_path = output_dir / "self_check.csv"
    changed_path = output_dir / "changed_files.txt"
    manifest_path = output_dir / "agent_run_manifest.json"

    panel.to_csv(panel_path, index=False, encoding="utf-8-sig")
    validation.to_csv(validation_path, index=False, encoding="utf-8-sig")
    holdout.to_csv(holdout_path, index=False, encoding="utf-8-sig")
    registry.to_csv(registry_path, index=False, encoding="utf-8-sig")
    specs.to_csv(specs_path, index=False, encoding="utf-8-sig")
    ortho.to_csv(ortho_path, index=False, encoding="utf-8-sig")
    quality.to_csv(quality_path, index=False, encoding="utf-8-sig")
    write_text(failure_cases(holdout), failure_path)
    write_text(make_report(validation, holdout, registry), report_path)
    checks.to_csv(checks_path, index=False, encoding="utf-8-sig")

    artifacts = [
        panel_path,
        validation_path,
        holdout_path,
        registry_path,
        specs_path,
        ortho_path,
        quality_path,
        failure_path,
        report_path,
        checks_path,
        changed_path,
        manifest_path,
    ]
    write_text("\n".join(rel(path) for path in artifacts), changed_path)

    fail_count = int((checks["status"] == "fail").sum())
    warn_count = int((checks["status"] == "warn").sum())
    metrics = {
        "signals_tested": int(validation.shape[0]),
        "passed_signal_count": int(validation["pass_pre_backtest_gate"].astype(bool).sum()) if not validation.empty else 0,
        "holdout_passed_signal_count": int(holdout[holdout["split"].astype(str).eq("holdout")]["eligible_for_implementation"].astype(str).str.lower().eq("true").sum()) if not holdout.empty else 0,
        "candidate_count": int((registry["role"] == "candidate").sum()) if not registry.empty else 0,
    }
    write_json(make_manifest(start_time, output_dir, artifacts, metrics, fail_count, warn_count), manifest_path)
    print(json.dumps({"task_id": TASK_ID, "self_check_pass": fail_count == 0, "metrics": metrics}, ensure_ascii=False, indent=2))
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
