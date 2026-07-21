#!/usr/bin/env python
"""HIRSSM V3.25 industry-structure source research.

V3.25 switches to an industry-internal information source after V3.24 found no
holdout-qualified valuation-spread candidate. This is source research only: it
may authorize a later harness only when a signal passes both full-sample
diagnostic and holdout source gates.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("Introduction-to-Quantitative-Finance")
BASELINE_DIR = ROOT / "outputs" / "hirssm_v3_10_clean_baseline"
V320_DIR = ROOT / "outputs" / "agent_runs" / "v3_20" / "rescue_signal_research"
V324_DIR = ROOT / "outputs" / "agent_runs" / "v3_24" / "valuation_source_research"
INDUSTRY_DIR = ROOT / "data_raw" / "index" / "akshare_sw_industry" / "daily_sw"
CLASSIFICATION_PATH = ROOT / "data_raw" / "index" / "akshare_sw_industry" / "classification" / "sw_level1_info.csv"
OUTPUT_DIR = ROOT / "outputs" / "agent_runs" / "v3_25" / "industry_structure_source_research"
TASK_ID = "20260527_v3_25_industry_structure_source_research"
BASELINE_VARIANT = "v3_10_clean_rank_vol_core"


SIGNAL_SPECS = [
    {
        "variant": "industry_breadth_repair_thrust",
        "role": "candidate",
        "description": "Industry breadth repairs from weak levels while short-term industry momentum turns broadly positive.",
        "target": "broad_forward_63d",
        "direction": "positive",
        "trigger": "above_ma60_ratio_lag1 <= 0.45 and positive_ret20_ratio_lag1 >= 0.55 and breadth_change_20_lag1 > 0.10",
        "action": "release 5pct cash to broad/style sleeves in later harness",
    },
    {
        "variant": "narrow_leadership_overheat_defense",
        "role": "candidate",
        "description": "Breadth is narrow while turnover concentration is elevated, indicating fragile leadership.",
        "target": "broad_forward_63d",
        "direction": "negative",
        "trigger": "above_ma60_ratio_lag1 <= 0.35 and amount_concentration_z_lag1 >= 0.75 and dispersion_z_lag1 >= 0.50",
        "action": "move 5pct from industry sleeves to cash in later harness",
    },
    {
        "variant": "dispersion_rotation_opportunity",
        "role": "candidate",
        "description": "Industry return dispersion is high but breadth is not collapsing, suggesting rotation opportunity.",
        "target": "industry_equal_minus_broad_forward_63d",
        "direction": "positive",
        "trigger": "dispersion_z_lag1 >= 0.75 and above_ma60_ratio_lag1 >= 0.45 and positive_ret20_ratio_lag1 >= 0.45",
        "action": "increase active industry sleeve budget by 4pct in later harness",
    },
]


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


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


def load_level1_codes(classification_path: Path) -> list[str]:
    info = read_csv(classification_path)
    if info.empty:
        raise FileNotFoundError(classification_path)
    if "sw_level" in info.columns:
        info = info[info["sw_level"].astype(str).eq("first")]
    return sorted(info["index_code"].astype(str).unique().tolist())


def load_industry_returns(industry_dir: Path, classification_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    codes = load_level1_codes(classification_path)
    close_frames = []
    amount_frames = []
    quality_rows = []
    for code in codes:
        path = industry_dir / f"{code}.csv"
        df = read_csv(path)
        if df.empty:
            quality_rows.append({"index_code": code, "rows": 0, "status": "missing"})
            continue
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").drop_duplicates("date")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        close_frames.append(df[["date", "close"]].rename(columns={"close": code}))
        amount_frames.append(df[["date", "amount"]].rename(columns={"amount": code}))
        quality_rows.append(
            {
                "index_code": code,
                "rows": int(df.shape[0]),
                "start_date": df["date"].min(),
                "end_date": df["date"].max(),
                "close_nonnull": int(df["close"].notna().sum()),
                "amount_nonnull": int(df["amount"].notna().sum()),
                "status": "pass",
            }
        )
    if not close_frames:
        raise RuntimeError("no industry close data loaded")
    close_wide = close_frames[0]
    amount_wide = amount_frames[0]
    for frame in close_frames[1:]:
        close_wide = close_wide.merge(frame, on="date", how="outer")
    for frame in amount_frames[1:]:
        amount_wide = amount_wide.merge(frame, on="date", how="outer")
    close_wide = close_wide.sort_values("date").reset_index(drop=True)
    amount_wide = amount_wide.sort_values("date").reset_index(drop=True)
    close_wide[codes] = close_wide[codes].ffill()
    amount_wide[codes] = amount_wide[codes].fillna(0.0)
    quality = pd.DataFrame(quality_rows)
    return close_wide, amount_wide, quality


def build_daily_structure_panel(industry_dir: Path, classification_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    close_wide, amount_wide, quality = load_industry_returns(industry_dir, classification_path)
    codes = [col for col in close_wide.columns if col != "date"]
    close = close_wide[codes].astype(float)
    ret20 = close / close.shift(20) - 1.0
    ret60 = close / close.shift(60) - 1.0
    ma60 = close.rolling(60, min_periods=30).mean()
    above_ma60 = close > ma60
    positive_ret20 = ret20 > 0.0
    amount = amount_wide[codes].astype(float)
    amount_share = amount.div(amount.sum(axis=1).replace(0, np.nan), axis=0)
    amount_hhi = (amount_share**2).sum(axis=1)
    industry_equal_close = close.mean(axis=1)
    broad_proxy_ret = industry_equal_close / industry_equal_close.shift(1) - 1.0
    industry_equal_forward = industry_equal_close.shift(-63) / industry_equal_close - 1.0
    daily = pd.DataFrame(
        {
            "date": close_wide["date"],
            "above_ma60_ratio": above_ma60.mean(axis=1),
            "positive_ret20_ratio": positive_ret20.mean(axis=1),
            "industry_return_dispersion": ret20.std(axis=1),
            "amount_concentration": amount_hhi,
            "industry_equal_close": industry_equal_close,
            "industry_equal_ret20": industry_equal_close / industry_equal_close.shift(20) - 1.0,
            "industry_equal_forward_63d": industry_equal_forward,
            "broad_proxy_ret": broad_proxy_ret,
        }
    )
    daily["breadth_change_20"] = daily["above_ma60_ratio"] - daily["above_ma60_ratio"].shift(20)
    daily["dispersion_z"] = rolling_zscore(daily["industry_return_dispersion"])
    daily["amount_concentration_z"] = rolling_zscore(daily["amount_concentration"])
    return daily, quality


def add_broad_targets(panel: pd.DataFrame) -> pd.DataFrame:
    broad = read_csv(BASELINE_DIR / "nav_clean_rank_vol_core_10bps.csv")
    if broad.empty:
        return panel
    broad["date"] = pd.to_datetime(broad["date"])
    broad = broad.sort_values("date")
    broad["benchmark_nav"] = pd.to_numeric(broad["benchmark_nav"], errors="coerce")
    broad["broad_forward_63d"] = broad["benchmark_nav"].shift(-63) / broad["benchmark_nav"] - 1.0
    out = pd.merge_asof(panel.sort_values("signal_date"), broad[["date", "broad_forward_63d"]].rename(columns={"date": "signal_date"}), on="signal_date", direction="backward")
    out["industry_equal_minus_broad_forward_63d"] = out["industry_equal_forward_63d"] - out["broad_forward_63d"]
    return out


def monthly_signal_panel(daily: pd.DataFrame) -> pd.DataFrame:
    targets = read_csv(BASELINE_DIR / "target_weights.csv")
    if targets.empty:
        raise FileNotFoundError(BASELINE_DIR / "target_weights.csv")
    dates = pd.DataFrame({"signal_date": pd.to_datetime(sorted(targets["signal_date"].dropna().unique()))})
    features = daily.rename(columns={"date": "signal_date"}).sort_values("signal_date")
    panel = pd.merge_asof(dates.sort_values("signal_date"), features, on="signal_date", direction="backward")
    for col in ["above_ma60_ratio", "positive_ret20_ratio", "breadth_change_20", "dispersion_z", "amount_concentration_z"]:
        panel[f"{col}_lag1"] = panel[col].shift(1)
    return add_broad_targets(panel)


def apply_signal(spec: dict[str, str], panel: pd.DataFrame) -> pd.Series:
    if spec["variant"] == "industry_breadth_repair_thrust":
        return (
            (pd.to_numeric(panel["above_ma60_ratio_lag1"], errors="coerce") <= 0.45)
            & (pd.to_numeric(panel["positive_ret20_ratio_lag1"], errors="coerce") >= 0.55)
            & (pd.to_numeric(panel["breadth_change_20_lag1"], errors="coerce") > 0.10)
        )
    if spec["variant"] == "narrow_leadership_overheat_defense":
        return (
            (pd.to_numeric(panel["above_ma60_ratio_lag1"], errors="coerce") <= 0.35)
            & (pd.to_numeric(panel["amount_concentration_z_lag1"], errors="coerce") >= 0.75)
            & (pd.to_numeric(panel["dispersion_z_lag1"], errors="coerce") >= 0.50)
        )
    if spec["variant"] == "dispersion_rotation_opportunity":
        return (
            (pd.to_numeric(panel["dispersion_z_lag1"], errors="coerce") >= 0.75)
            & (pd.to_numeric(panel["above_ma60_ratio_lag1"], errors="coerce") >= 0.45)
            & (pd.to_numeric(panel["positive_ret20_ratio_lag1"], errors="coerce") >= 0.45)
        )
    return pd.Series(False, index=panel.index)


def pass_rule(spec: dict[str, str], fwd: pd.Series, unconditional: float, min_obs: int = 12) -> bool:
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
        target = spec["target"]
        valid = panel.dropna(subset=[target]).copy()
        mask = apply_signal(spec, valid).fillna(False)
        selected = valid[mask]
        fwd = pd.to_numeric(selected[target], errors="coerce").dropna()
        unconditional = float(pd.to_numeric(valid[target], errors="coerce").mean()) if not valid.empty else np.nan
        obs = int(fwd.shape[0])
        rows.append(
            {
                "variant": spec["variant"],
                "role": spec["role"],
                "information_source": "sw_industry_internal_structure",
                "target": target,
                "direction": spec["direction"],
                "trigger": spec["trigger"],
                "observations": obs,
                "coverage": obs / max(int(valid.shape[0]), 1),
                "forward_63d_mean": float(fwd.mean()) if obs else np.nan,
                "unconditional_forward_63d_mean": unconditional,
                "positive_forward_share": float((fwd > 0).mean()) if obs else np.nan,
                "pass_pre_backtest_gate": pass_rule(spec, fwd, unconditional, min_obs=12),
                "diagnostic_full_sample_only": True,
            }
        )
    return pd.DataFrame(rows)


def holdout_validation(panel: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    target_cols = sorted({spec["target"] for spec in SIGNAL_SPECS})
    valid = panel.dropna(subset=target_cols, how="all").copy()
    if valid.empty:
        return pd.DataFrame()
    split_date = pd.to_datetime(valid["signal_date"]).quantile(0.70)
    rows = []
    for spec in SIGNAL_SPECS:
        target = spec["target"]
        spec_valid = valid.dropna(subset=[target]).copy()
        mask = apply_signal(spec, spec_valid).fillna(False)
        full_pass = bool(validation.loc[validation["variant"].eq(spec["variant"]), "pass_pre_backtest_gate"].iloc[0])
        for split_name, split_mask in [
            ("train", pd.to_datetime(spec_valid["signal_date"]) <= split_date),
            ("holdout", pd.to_datetime(spec_valid["signal_date"]) > split_date),
        ]:
            subset = spec_valid[split_mask].copy()
            selected = subset[mask.loc[subset.index]]
            fwd = pd.to_numeric(selected[target], errors="coerce").dropna()
            unconditional = float(pd.to_numeric(subset[target], errors="coerce").mean()) if not subset.empty else np.nan
            passed = pass_rule(spec, fwd, unconditional, min_obs=5)
            rows.append(
                {
                    "variant": spec["variant"],
                    "split": split_name,
                    "split_date": split_date,
                    "target": target,
                    "observations": int(fwd.shape[0]),
                    "forward_63d_mean": float(fwd.mean()) if not fwd.empty else np.nan,
                    "unconditional_forward_63d_mean": unconditional,
                    "positive_forward_share": float((fwd > 0).mean()) if not fwd.empty else np.nan,
                    "pass_full_sample_gate": full_pass,
                    "pass_holdout_gate": passed if split_name == "holdout" else "",
                    "eligible_for_implementation": bool(full_pass and passed) if split_name == "holdout" else "",
                    "failure_reason": "" if split_name == "train" or passed else "holdout_effect_or_sample_gate_failed",
                }
            )
    return pd.DataFrame(rows)


def holdout_pass_map(holdout: pd.DataFrame) -> dict[str, bool]:
    if holdout.empty:
        return {}
    rows = holdout[holdout["split"].astype(str).eq("holdout")]
    return {str(row["variant"]): str(row["eligible_for_implementation"]).strip().lower() == "true" for _, row in rows.iterrows()}


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
                "selection_source": "v3_25_industry_structure_holdout_gate",
                "diagnostic_full_sample_only": True,
                "eligible_for_default_promotion": passed,
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
                "information_source": "sw_industry_internal_structure",
                "weight_rule": spec["action"],
                "target": spec["target"],
                "acceptance_standard": "must pass holdout source gate and V3.11-style nested/PBO gates; full-sample industry-structure signal validation is not promotion evidence",
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
    for spec in SIGNAL_SPECS:
        signal = apply_signal(spec, panel).fillna(False).astype(float)
        for reference in ["v3_20_vol_compression_reentry", "v3_24_broad_market_deep_value_repair", "v3_10_cash_weight"]:
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


def make_report(validation: pd.DataFrame, holdout: pd.DataFrame, registry: pd.DataFrame) -> str:
    full_passed = validation[validation["pass_pre_backtest_gate"].astype(bool)]["variant"].astype(str).tolist() if not validation.empty else []
    holdout_passed = holdout[
        holdout["split"].astype(str).eq("holdout") & holdout["eligible_for_implementation"].astype(str).str.lower().eq("true")
    ]["variant"].astype(str).tolist() if not holdout.empty else []
    return "\n".join(
        [
            "# HIRSSM V3.25 Industry Structure Source Research",
            "",
            "## Purpose",
            "",
            "Introduce SW industry internal breadth, dispersion, and turnover concentration as a new source.",
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
            "- V3.25 is source research only.",
            "- Only holdout-qualified industry-structure signals may enter a later model harness.",
        ]
    )


def failure_cases(holdout: pd.DataFrame) -> str:
    lines = [
        "# V3.25 Factor Failure Cases",
        "",
        "- Industry breadth can improve after sharp rebounds that are already mostly priced.",
        "- Turnover concentration can reflect legitimate institutional leadership rather than fragility.",
        "- Equal-weight industry targets may not map cleanly to the investable HIRSSM industry sleeve.",
        "- Later implementation must account for turnover and capacity before promotion.",
        "",
        "## Holdout Failures",
        "",
    ]
    if holdout.empty:
        lines.append("- Holdout validation missing.")
    else:
        failures = holdout[holdout["split"].astype(str).eq("holdout") & ~holdout["eligible_for_implementation"].astype(str).str.lower().eq("true")]
        for _, row in failures.iterrows():
            lines.append(f"- {row['variant']}: {row.get('failure_reason', '')}; observations={row.get('observations', '')}")
    return "\n".join(lines)


def self_check(validation: pd.DataFrame, holdout: pd.DataFrame, registry: pd.DataFrame, specs: pd.DataFrame, ortho: pd.DataFrame) -> pd.DataFrame:
    candidate_count = int((registry["role"] == "candidate").sum()) if not registry.empty else 0
    return pd.DataFrame(
        [
            {"check": "signal_validation_exists", "status": "pass" if not validation.empty else "fail", "detail": str(int(validation.shape[0]))},
            {"check": "signal_gate_holdout_exists", "status": "pass" if not holdout.empty else "fail", "detail": str(int(holdout.shape[0]))},
            {"check": "candidate_registry_exists", "status": "pass" if not registry.empty else "fail", "detail": str(int(registry.shape[0]))},
            {"check": "implementation_specs_exist", "status": "pass" if not specs.empty else "fail", "detail": str(int(specs.shape[0]))},
            {"check": "source_orthogonality_exists", "status": "pass" if not ortho.empty else "fail", "detail": str(int(ortho.shape[0]))},
            {"check": "at_least_one_candidate_for_harness", "status": "pass" if candidate_count >= 1 else "warn", "detail": str(candidate_count)},
            {"check": "research_only_no_promotion", "status": "pass", "detail": "source-gated signal validation only"},
        ]
    )


def manifest(start_time: str, output_dir: Path, artifacts: list[Path], metrics: dict[str, Any], fail_count: int, warn_count: int) -> dict[str, Any]:
    return {
        "run_id": f"{TASK_ID}_run_001",
        "task_id": TASK_ID,
        "agent": "factor_researcher",
        "version": "V3.25",
        "baseline": "HIRSSM V3.10 Clean Rank-Vol Core",
        "status": "pass" if fail_count == 0 else "fail",
        "started_at": start_time,
        "command": "python -X utf8 strategy_lab/hirssm_v3_25_industry_structure_source_research.py",
        "config": {"research_only": True, "information_source": "sw_industry_internal_structure", "forward_window_days": 63},
        "data_refs": ["data_raw/index/akshare_sw_industry/daily_sw", "outputs/hirssm_v3_10_clean_baseline", "outputs/agent_runs/v3_20/rescue_signal_research", "outputs/agent_runs/v3_24/valuation_source_research"],
        "code_refs": ["strategy_lab/hirssm_v3_25_industry_structure_source_research.py"],
        "output_dir": str(output_dir.relative_to(ROOT).as_posix()),
        "allowed_inputs": ["data_raw/index/akshare_sw_industry/daily_sw", "data_raw/index/akshare_sw_industry/classification", "outputs/hirssm_v3_10_clean_baseline", "outputs/agent_runs/v3_20/rescue_signal_research", "outputs/agent_runs/v3_24/valuation_source_research"],
        "artifacts": [str(path.relative_to(ROOT).as_posix()) for path in artifacts],
        "outputs": [str(path.relative_to(ROOT).as_posix()) for path in artifacts if path.name != "agent_run_manifest.json"],
        "changed_files": [str(path.relative_to(ROOT).as_posix()) for path in artifacts],
        "metrics": metrics,
        "self_check_pass": fail_count == 0,
        "fail_count": fail_count,
        "warn_count": warn_count,
        "limitations": ["Uses SW industry index history; mapping to investable sleeves requires later portfolio harness.", "Forward labels are used only for source validation with holdout gating."],
        "risk_flags": ["industry_breadth_signal_instability", "source_requires_later_nested_validation"],
        "next_decision": "Implement only holdout-qualified industry-structure candidates in V3.26; if none pass, run another source research task.",
        "handoff_summary": "V3.25 introduced SW industry breadth, dispersion, and turnover concentration signals with holdout-gated implementation eligibility.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate V3.25 industry-structure source research artifacts.")
    parser.add_argument("--industry-dir", default=str(INDUSTRY_DIR))
    parser.add_argument("--classification", default=str(CLASSIFICATION_PATH))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    start_time = now_text()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    daily, quality = build_daily_structure_panel(Path(args.industry_dir), Path(args.classification))
    panel = monthly_signal_panel(daily)
    validation = full_sample_validation(panel)
    holdout = holdout_validation(panel, validation)
    registry = candidate_registry(validation, holdout)
    specs = implementation_specs(validation, holdout)
    ortho = source_orthogonality(panel)
    checks = self_check(validation, holdout, registry, specs, ortho)

    panel_path = output_dir / "industry_structure_signal_feature_panel.csv"
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

    artifacts = [panel_path, validation_path, holdout_path, registry_path, specs_path, ortho_path, quality_path, failure_path, report_path, checks_path, changed_path, manifest_path]
    write_text("\n".join(str(path.relative_to(ROOT).as_posix()) for path in artifacts), changed_path)

    fail_count = int((checks["status"] == "fail").sum())
    warn_count = int((checks["status"] == "warn").sum())
    metrics = {
        "signals_tested": int(validation.shape[0]),
        "passed_signal_count": int(validation["pass_pre_backtest_gate"].astype(bool).sum()) if not validation.empty else 0,
        "holdout_passed_signal_count": int(holdout[holdout["split"].astype(str).eq("holdout")]["eligible_for_implementation"].astype(str).str.lower().eq("true").sum()) if not holdout.empty else 0,
        "candidate_count": int((registry["role"] == "candidate").sum()) if not registry.empty else 0,
    }
    write_json(manifest(start_time, output_dir, artifacts, metrics, fail_count, warn_count), manifest_path)
    print(json.dumps({"task_id": TASK_ID, "self_check_pass": fail_count == 0, "metrics": metrics}, ensure_ascii=False, indent=2))
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
