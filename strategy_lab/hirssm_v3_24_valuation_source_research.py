#!/usr/bin/env python
"""HIRSSM V3.24 valuation source research.

V3.24 introduces a genuinely new information source after the V3.20/V3.21
forward-label source-gate correction: historical index valuation spread. This
version is signal research only. It may authorize a later harness only when a
signal passes both full-sample diagnostic and holdout source gates.
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
DAILY_DIR = ROOT / "data_raw" / "index" / "akshare_csindex" / "daily_csindex"
OUTPUT_DIR = ROOT / "outputs" / "agent_runs" / "v3_24" / "valuation_source_research"
TASK_ID = "20260527_v3_24_valuation_source_research"
BASELINE_VARIANT = "v3_10_clean_rank_vol_core"


INDEX_CODES = {
    "dividend": "000922",
    "broad": "000985",
    "large": "000300",
    "mid": "000905",
    "small": "000852",
    "defensive_large": "000016",
}


SIGNAL_SPECS = [
    {
        "variant": "dividend_valuation_repair",
        "role": "candidate",
        "description": "Dividend index is cheap versus its own history and broad market while relative trend is not collapsing.",
        "target": "dividend_minus_broad_63d",
        "direction": "positive",
        "trigger": "dividend_pe_pct <= 0.35 and dividend_vs_broad_pe_spread_z >= 0.50 and dividend_rel_ret60 > -0.05",
        "action": "tilt 5pct from broad/style cash budget toward dividend sleeve in later harness",
    },
    {
        "variant": "broad_market_deep_value_repair",
        "role": "candidate",
        "description": "Broad market valuation is depressed and 20-day trend has repaired.",
        "target": "broad_63d",
        "direction": "positive",
        "trigger": "broad_pe_pct <= 0.30 and broad_ret20 > 0",
        "action": "release 4pct cash to broad-market sleeve in later harness",
    },
    {
        "variant": "large_vs_mid_valuation_spread",
        "role": "candidate",
        "description": "CSI 300 is cheap versus CSI 500 while large-cap relative momentum has stabilized.",
        "target": "large_minus_mid_63d",
        "direction": "positive",
        "trigger": "large_vs_mid_pe_spread_z >= 0.50 and large_rel_mid_ret60 > -0.03",
        "action": "tilt 4pct from mid-cap sleeve to large-cap sleeve in later harness",
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


def rolling_percentile(series: pd.Series, window: int = 756, min_periods: int = 252) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")

    def pct_rank(arr: np.ndarray) -> float:
        arr = arr[~np.isnan(arr)]
        if len(arr) < min_periods:
            return np.nan
        return float((arr <= arr[-1]).mean())

    return values.rolling(window, min_periods=min_periods).apply(pct_rank, raw=True)


def rolling_zscore(series: pd.Series, window: int = 756, min_periods: int = 252) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    mean = values.rolling(window, min_periods=min_periods).mean()
    std = values.rolling(window, min_periods=min_periods).std()
    return (values - mean) / std.replace(0, np.nan)


def load_index_daily(code: str, daily_dir: Path) -> pd.DataFrame:
    path = daily_dir / f"{code}.csv"
    df = read_csv(path)
    if df.empty:
        raise FileNotFoundError(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date")
    for col in ["close", "pe_ttm", "amount", "constituent_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["date", "close", "pe_ttm", "amount", "constituent_count"]].copy()


def build_daily_panel(daily_dir: Path) -> pd.DataFrame:
    frames = []
    for name, code in INDEX_CODES.items():
        df = load_index_daily(code, daily_dir)
        df = df.rename(
            columns={
                "close": f"{name}_close",
                "pe_ttm": f"{name}_pe",
                "amount": f"{name}_amount",
                "constituent_count": f"{name}_constituent_count",
            }
        )
        frames.append(df)
    panel = frames[0]
    for frame in frames[1:]:
        panel = panel.merge(frame, on="date", how="outer")
    panel = panel.sort_values("date").reset_index(drop=True)
    close_cols = [col for col in panel.columns if col.endswith("_close")]
    pe_cols = [col for col in panel.columns if col.endswith("_pe")]
    panel[close_cols + pe_cols] = panel[close_cols + pe_cols].ffill()

    for name in INDEX_CODES:
        close = pd.to_numeric(panel[f"{name}_close"], errors="coerce")
        panel[f"{name}_ret20"] = close / close.shift(20) - 1.0
        panel[f"{name}_ret60"] = close / close.shift(60) - 1.0
        panel[f"{name}_fwd63"] = close.shift(-63) / close - 1.0
        panel[f"{name}_pe_pct"] = rolling_percentile(panel[f"{name}_pe"])

    panel["dividend_rel_ret60"] = panel["dividend_ret60"] - panel["broad_ret60"]
    panel["large_rel_mid_ret60"] = panel["large_ret60"] - panel["mid_ret60"]
    panel["dividend_vs_broad_pe_spread"] = np.log(panel["broad_pe"] / panel["dividend_pe"])
    panel["large_vs_mid_pe_spread"] = np.log(panel["mid_pe"] / panel["large_pe"])
    panel["dividend_vs_broad_pe_spread_z"] = rolling_zscore(panel["dividend_vs_broad_pe_spread"])
    panel["large_vs_mid_pe_spread_z"] = rolling_zscore(panel["large_vs_mid_pe_spread"])
    panel["dividend_minus_broad_63d"] = panel["dividend_fwd63"] - panel["broad_fwd63"]
    panel["large_minus_mid_63d"] = panel["large_fwd63"] - panel["mid_fwd63"]
    panel["broad_63d"] = panel["broad_fwd63"]
    return panel


def monthly_signal_panel(baseline_dir: Path, daily_panel: pd.DataFrame) -> pd.DataFrame:
    targets = read_csv(baseline_dir / "target_weights.csv")
    if targets.empty:
        raise FileNotFoundError(baseline_dir / "target_weights.csv")
    dates = pd.DataFrame({"signal_date": pd.to_datetime(sorted(targets["signal_date"].dropna().unique()))})
    daily = daily_panel.rename(columns={"date": "signal_date"}).sort_values("signal_date")
    return pd.merge_asof(dates.sort_values("signal_date"), daily, on="signal_date", direction="backward")


def apply_signal(spec: dict[str, str], panel: pd.DataFrame) -> pd.Series:
    if spec["variant"] == "dividend_valuation_repair":
        return (
            (pd.to_numeric(panel["dividend_pe_pct"], errors="coerce") <= 0.35)
            & (pd.to_numeric(panel["dividend_vs_broad_pe_spread_z"], errors="coerce") >= 0.50)
            & (pd.to_numeric(panel["dividend_rel_ret60"], errors="coerce") > -0.05)
        )
    if spec["variant"] == "broad_market_deep_value_repair":
        return (
            (pd.to_numeric(panel["broad_pe_pct"], errors="coerce") <= 0.30)
            & (pd.to_numeric(panel["broad_ret20"], errors="coerce") > 0)
        )
    if spec["variant"] == "large_vs_mid_valuation_spread":
        return (
            (pd.to_numeric(panel["large_vs_mid_pe_spread_z"], errors="coerce") >= 0.50)
            & (pd.to_numeric(panel["large_rel_mid_ret60"], errors="coerce") > -0.03)
        )
    return pd.Series(False, index=panel.index)


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
        mean_fwd = float(fwd.mean()) if obs else np.nan
        positive_share = float((fwd > 0).mean()) if obs else np.nan
        pass_gate = bool(obs >= 12 and mean_fwd > unconditional and positive_share > 0.52)
        rows.append(
            {
                "variant": spec["variant"],
                "role": spec["role"],
                "information_source": "historical_index_valuation_spread",
                "target": target,
                "trigger": spec["trigger"],
                "observations": obs,
                "coverage": obs / max(int(valid.shape[0]), 1),
                "forward_63d_mean": mean_fwd,
                "unconditional_forward_63d_mean": unconditional,
                "positive_forward_share": positive_share,
                "pass_pre_backtest_gate": pass_gate,
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
            obs = int(fwd.shape[0])
            mean_fwd = float(fwd.mean()) if obs else np.nan
            positive_share = float((fwd > 0).mean()) if obs else np.nan
            passed = bool(obs >= 5 and mean_fwd > unconditional and positive_share > 0.52)
            rows.append(
                {
                    "variant": spec["variant"],
                    "split": split_name,
                    "split_date": split_date,
                    "target": target,
                    "observations": obs,
                    "forward_63d_mean": mean_fwd,
                    "unconditional_forward_63d_mean": unconditional,
                    "positive_forward_share": positive_share,
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
                "selection_source": "v3_24_valuation_source_holdout_gate",
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
                "information_source": "historical_index_valuation_spread",
                "weight_rule": spec["action"],
                "target": spec["target"],
                "acceptance_standard": "must pass holdout source gate and V3.11-style nested/PBO gates; full-sample valuation signal validation is not promotion evidence",
            }
        )
    return pd.DataFrame(rows)


def source_orthogonality(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    v320 = read_csv(V320_DIR / "signal_feature_panel.csv")
    if not v320.empty:
        v320["signal_date"] = pd.to_datetime(v320["signal_date"])
        v320_trigger = (
            (pd.to_numeric(v320["vol_60"], errors="coerce") <= 0.85 * pd.to_numeric(v320["vol_120"], errors="coerce"))
            & (pd.to_numeric(v320["drawdown_252"], errors="coerce") <= -0.10)
            & (pd.to_numeric(v320["benchmark_ret_20"], errors="coerce") > 0)
        )
        compare = v320[["signal_date"]].copy()
        compare["v3_20_vol_compression_reentry"] = v320_trigger.astype(float)
        merged = panel[["signal_date"]].merge(compare, on="signal_date", how="left")
    else:
        merged = panel[["signal_date"]].copy()
        merged["v3_20_vol_compression_reentry"] = np.nan
    baseline_targets = read_csv(BASELINE_DIR / "target_weights.csv")
    if not baseline_targets.empty:
        cash = baseline_targets[baseline_targets["asset"].astype(str).eq("CASH")][["signal_date", "weight"]].copy()
        cash["signal_date"] = pd.to_datetime(cash["signal_date"])
        cash = cash.rename(columns={"weight": "v3_10_cash_weight"})
        merged = merged.merge(cash, on="signal_date", how="left")
    else:
        merged["v3_10_cash_weight"] = np.nan
    for spec in SIGNAL_SPECS:
        signal = apply_signal(spec, panel).fillna(False).astype(float)
        for reference in ["v3_20_vol_compression_reentry", "v3_10_cash_weight"]:
            ref = pd.to_numeric(merged[reference], errors="coerce")
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


def data_quality_report(daily_panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, code in INDEX_CODES.items():
        rows.append(
            {
                "dataset": f"daily_csindex_{code}",
                "asset": name,
                "start_date": daily_panel["date"].min(),
                "end_date": daily_panel["date"].max(),
                "close_nonnull": int(daily_panel[f"{name}_close"].notna().sum()),
                "pe_nonnull": int(daily_panel[f"{name}_pe"].notna().sum()),
                "pit_status": "historical_daily_field_research_only",
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
            "# HIRSSM V3.24 Valuation Source Research",
            "",
            "## Purpose",
            "",
            "Introduce historical index valuation spread as a new source before any post-V3.23 model harness.",
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
            "- V3.24 is source research only.",
            "- Only holdout-qualified valuation signals may enter a later model harness.",
        ]
    )


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


def failure_cases(validation: pd.DataFrame, holdout: pd.DataFrame) -> str:
    lines = [
        "# V3.24 Factor Failure Cases",
        "",
        "- Historical valuation fields may be revised by the data vendor; treat source research as research-only until PIT provenance is stronger.",
        "- Cheap valuation can stay cheap in value traps and during earnings downcycles.",
        "- Holdout sample may be too small for rare deep-value states.",
        "- Later implementation must avoid using current index constituents or current weights for historical decisions.",
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


def manifest(start_time: str, output_dir: Path, artifacts: list[Path], metrics: dict[str, Any], fail_count: int, warn_count: int) -> dict[str, Any]:
    return {
        "run_id": f"{TASK_ID}_run_001",
        "task_id": TASK_ID,
        "agent": "factor_researcher",
        "version": "V3.24",
        "baseline": "HIRSSM V3.10 Clean Rank-Vol Core",
        "status": "pass" if fail_count == 0 else "fail",
        "started_at": start_time,
        "command": "python -X utf8 strategy_lab/hirssm_v3_24_valuation_source_research.py",
        "config": {"research_only": True, "information_source": "historical_index_valuation_spread", "forward_window_days": 63},
        "data_refs": ["data_raw/index/akshare_csindex/daily_csindex", "outputs/hirssm_v3_10_clean_baseline", "outputs/agent_runs/v3_20/rescue_signal_research"],
        "code_refs": ["strategy_lab/hirssm_v3_24_valuation_source_research.py"],
        "output_dir": str(output_dir.relative_to(ROOT).as_posix()),
        "allowed_inputs": ["data_raw/index/akshare_csindex/daily_csindex", "outputs/hirssm_v3_10_clean_baseline", "outputs/agent_runs/v3_20/rescue_signal_research"],
        "artifacts": [str(path.relative_to(ROOT).as_posix()) for path in artifacts],
        "outputs": [str(path.relative_to(ROOT).as_posix()) for path in artifacts if path.name != "agent_run_manifest.json"],
        "changed_files": [str(path.relative_to(ROOT).as_posix()) for path in artifacts],
        "metrics": metrics,
        "self_check_pass": fail_count == 0,
        "fail_count": fail_count,
        "warn_count": warn_count,
        "limitations": ["Uses historical valuation fields from index daily data; PIT provenance remains research-only.", "Forward labels are used only for source validation with holdout gating."],
        "risk_flags": ["valuation_data_revision_risk", "source_requires_later_nested_validation"],
        "next_decision": "Implement only holdout-qualified valuation candidates in V3.25; if none pass, run another source research task.",
        "handoff_summary": "V3.24 introduced historical valuation spread signals with holdout-gated implementation eligibility.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate V3.24 valuation source research artifacts.")
    parser.add_argument("--daily-dir", default=str(DAILY_DIR))
    parser.add_argument("--baseline-dir", default=str(BASELINE_DIR))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    start_time = now_text()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    daily_panel = build_daily_panel(Path(args.daily_dir))
    panel = monthly_signal_panel(Path(args.baseline_dir), daily_panel)
    validation = full_sample_validation(panel)
    holdout = holdout_validation(panel, validation)
    registry = candidate_registry(validation, holdout)
    specs = implementation_specs(validation, holdout)
    ortho = source_orthogonality(panel)
    quality = data_quality_report(daily_panel)
    checks = self_check(validation, holdout, registry, specs, ortho)

    panel_path = output_dir / "valuation_signal_feature_panel.csv"
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
    write_text(failure_cases(validation, holdout), failure_path)
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
