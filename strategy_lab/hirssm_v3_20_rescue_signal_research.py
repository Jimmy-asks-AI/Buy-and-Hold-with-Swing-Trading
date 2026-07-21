#!/usr/bin/env python
"""HIRSSM V3.20 rescue signal research.

After V3.19 blocks the duplicate no-trade path, V3.20 searches for simple
predeclared drawdown/reentry signals that are different from the V3.15 breadth
overlay. This is signal research only; it does not promote a model.
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
V319_DIR = ROOT / "outputs" / "agent_runs" / "v3_19" / "filtered_no_trade_candidate"
OUTPUT_DIR = ROOT / "outputs" / "agent_runs" / "v3_20" / "rescue_signal_research"
TASK_ID = "20260527_v3_20_rescue_signal_research"
BASELINE_VARIANT = "v3_10_clean_rank_vol_core"


SIGNAL_SPECS = [
    {
        "variant": "drawdown_reentry_cash_release",
        "role": "candidate",
        "description": "Release cash after deep drawdown when 20-day benchmark momentum turns positive.",
        "trigger": "drawdown_252 <= -0.20 and benchmark_ret_20 > 0",
        "action": "release up to 8pct cash to existing noncash sleeves",
        "mode": "risk_on_reentry",
    },
    {
        "variant": "vol_compression_reentry",
        "role": "candidate",
        "description": "Release cash when volatility compresses after stress and benchmark momentum is positive.",
        "trigger": "vol_60 <= 0.85 * vol_120 and drawdown_252 <= -0.10 and benchmark_ret_20 > 0",
        "action": "release up to 5pct cash to existing noncash sleeves",
        "mode": "risk_on_reentry",
    },
    {
        "variant": "stress_defensive_cash_buffer",
        "role": "candidate",
        "description": "Increase cash when drawdown is meaningful and 20-day benchmark momentum remains negative.",
        "trigger": "drawdown_252 <= -0.15 and benchmark_ret_20 <= 0",
        "action": "move 6pct noncash exposure to cash",
        "mode": "risk_off_buffer",
    },
]


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def daily_features(nav: pd.DataFrame) -> pd.DataFrame:
    out = nav[["date", "portfolio_return", "nav", "benchmark_return", "benchmark_nav"]].copy()
    out["date"] = pd.to_datetime(out["date"])
    for col in ["portfolio_return", "nav", "benchmark_return", "benchmark_nav"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.sort_values("date").reset_index(drop=True)
    out["benchmark_peak_252"] = out["benchmark_nav"].rolling(252, min_periods=60).max()
    out["drawdown_252"] = out["benchmark_nav"] / out["benchmark_peak_252"] - 1.0
    out["benchmark_ret_20"] = (1.0 + out["benchmark_return"].fillna(0.0)).rolling(20, min_periods=10).apply(np.prod, raw=True) - 1.0
    out["benchmark_ret_63_fwd"] = out["benchmark_nav"].shift(-63) / out["benchmark_nav"] - 1.0
    out["portfolio_ret_63_fwd"] = out["nav"].shift(-63) / out["nav"] - 1.0
    out["vol_60"] = out["benchmark_return"].rolling(60, min_periods=30).std() * np.sqrt(252)
    out["vol_120"] = out["benchmark_return"].rolling(120, min_periods=60).std() * np.sqrt(252)
    return out


def signal_dates(targets: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    dates = pd.DataFrame({"signal_date": pd.to_datetime(sorted(targets["signal_date"].dropna().unique()))})
    feat = features.rename(columns={"date": "signal_date"}).sort_values("signal_date")
    return pd.merge_asof(dates.sort_values("signal_date"), feat, on="signal_date", direction="backward")


def apply_signal(spec: dict[str, str], panel: pd.DataFrame) -> pd.Series:
    dd = pd.to_numeric(panel["drawdown_252"], errors="coerce")
    r20 = pd.to_numeric(panel["benchmark_ret_20"], errors="coerce")
    v60 = pd.to_numeric(panel["vol_60"], errors="coerce")
    v120 = pd.to_numeric(panel["vol_120"], errors="coerce")
    if spec["variant"] == "drawdown_reentry_cash_release":
        return (dd <= -0.20) & (r20 > 0)
    if spec["variant"] == "vol_compression_reentry":
        return (v60 <= 0.85 * v120) & (dd <= -0.10) & (r20 > 0)
    if spec["variant"] == "stress_defensive_cash_buffer":
        return (dd <= -0.15) & (r20 <= 0)
    return pd.Series(False, index=panel.index)


def validate_signals(panel: pd.DataFrame) -> pd.DataFrame:
    unconditional_mean = float(pd.to_numeric(panel["benchmark_ret_63_fwd"], errors="coerce").mean())
    rows = []
    for spec in SIGNAL_SPECS:
        mask = apply_signal(spec, panel).fillna(False)
        sample = panel[mask].copy()
        fwd = pd.to_numeric(sample["benchmark_ret_63_fwd"], errors="coerce").dropna()
        port_fwd = pd.to_numeric(sample["portfolio_ret_63_fwd"], errors="coerce").dropna()
        obs = int(fwd.shape[0])
        mean_fwd = float(fwd.mean()) if obs else np.nan
        positive_share = float((fwd > 0).mean()) if obs else np.nan
        if spec["mode"] == "risk_off_buffer":
            pass_gate = obs >= 10 and mean_fwd < unconditional_mean and positive_share < 0.55
            rationale = "risk-off signal should identify below-average forward benchmark windows"
        else:
            pass_gate = obs >= 10 and mean_fwd > unconditional_mean and positive_share > 0.52
            rationale = "risk-on reentry signal should identify above-average forward benchmark windows"
        rows.append(
            {
                "variant": spec["variant"],
                "role": spec["role"],
                "mode": spec["mode"],
                "trigger": spec["trigger"],
                "observations": obs,
                "coverage": obs / max(int(panel.shape[0]), 1),
                "forward_63d_benchmark_mean": mean_fwd,
                "forward_63d_portfolio_mean": float(port_fwd.mean()) if not port_fwd.empty else np.nan,
                "unconditional_forward_63d_benchmark_mean": unconditional_mean,
                "positive_forward_share": positive_share,
                "pass_pre_backtest_gate": bool(pass_gate),
                "gate_rationale": rationale,
                "diagnostic_full_sample_only": True,
            }
        )
    return pd.DataFrame(rows)


def signal_gate_holdout_validation(panel: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    valid = panel.dropna(subset=["benchmark_ret_63_fwd"]).copy()
    if valid.empty:
        return pd.DataFrame()
    split_date = pd.to_datetime(valid["signal_date"]).quantile(0.70)
    rows = []
    for spec in SIGNAL_SPECS:
        mask = apply_signal(spec, valid).fillna(False)
        full_pass = bool(validation.loc[validation["variant"].eq(spec["variant"]), "pass_pre_backtest_gate"].iloc[0])
        for split_name, split_mask in [
            ("train", pd.to_datetime(valid["signal_date"]) <= split_date),
            ("holdout", pd.to_datetime(valid["signal_date"]) > split_date),
        ]:
            subset = valid[split_mask].copy()
            selected = subset[mask.loc[subset.index]]
            fwd = pd.to_numeric(selected["benchmark_ret_63_fwd"], errors="coerce").dropna()
            unconditional = float(pd.to_numeric(subset["benchmark_ret_63_fwd"], errors="coerce").mean()) if not subset.empty else np.nan
            obs = int(fwd.shape[0])
            mean_fwd = float(fwd.mean()) if obs else np.nan
            positive_share = float((fwd > 0).mean()) if obs else np.nan
            if spec["mode"] == "risk_off_buffer":
                passed = obs >= 5 and mean_fwd < unconditional and positive_share < 0.55
            else:
                passed = obs >= 5 and mean_fwd > unconditional and positive_share > 0.52
            rows.append(
                {
                    "variant": spec["variant"],
                    "split": split_name,
                    "split_date": split_date,
                    "observations": obs,
                    "forward_63d_benchmark_mean": mean_fwd,
                    "unconditional_forward_63d_benchmark_mean": unconditional,
                    "positive_forward_share": positive_share,
                    "pass_full_sample_gate": full_pass,
                    "pass_holdout_gate": bool(passed) if split_name == "holdout" else "",
                    "eligible_for_implementation": bool(full_pass and passed) if split_name == "holdout" else "",
                    "failure_reason": "" if split_name == "train" or passed else "holdout_effect_or_sample_gate_failed",
                }
            )
    return pd.DataFrame(rows)


def holdout_pass_map(holdout: pd.DataFrame) -> dict[str, bool]:
    if holdout.empty or "split" not in holdout.columns:
        return {}
    rows = holdout[holdout["split"].astype(str).eq("holdout")]
    return {str(row["variant"]): bool(row["eligible_for_implementation"]) for _, row in rows.iterrows()}


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
        valid = validation[validation["variant"].eq(spec["variant"])]
        full_passed = bool(valid["pass_pre_backtest_gate"].iloc[0]) if not valid.empty else False
        passed = bool(full_passed and holdout_ok.get(spec["variant"], False))
        rows.append(
            {
                "variant": spec["variant"],
                "role": "candidate" if passed else "observation",
                "description": spec["description"],
                "multipliers_json": json.dumps({"mode": spec["mode"], "trigger": spec["trigger"], "action": spec["action"]}, sort_keys=True),
                "disabled_experts": "range_reversal,style_trend_continuation",
                "selection_source": "v3_20_pre_backtest_signal_validation",
                "diagnostic_full_sample_only": True,
                "eligible_for_default_promotion": passed,
            }
        )
    return pd.DataFrame(rows)


def implementation_specs(validation: pd.DataFrame, holdout: pd.DataFrame) -> pd.DataFrame:
    holdout_ok = holdout_pass_map(holdout)
    rows = []
    for spec in SIGNAL_SPECS:
        full_passed = bool(validation.loc[validation["variant"].eq(spec["variant"]), "pass_pre_backtest_gate"].iloc[0])
        passed = bool(full_passed and holdout_ok.get(spec["variant"], False))
        rows.append(
            {
                "variant": spec["variant"],
                "implementation_allowed": passed,
                "weight_rule": spec["action"],
                "no_trade_band": 0.03 if passed else np.nan,
                "acceptance_standard": "must pass holdout signal gate and V3.11-style nested/PBO gates; full-sample signal validation is not promotion evidence",
            }
        )
    return pd.DataFrame(rows)


def make_report(validation: pd.DataFrame, holdout: pd.DataFrame, registry: pd.DataFrame) -> str:
    passed = validation[validation["pass_pre_backtest_gate"].astype(bool)]["variant"].astype(str).tolist() if not validation.empty else []
    holdout_passed = holdout[
        holdout["split"].astype(str).eq("holdout") & holdout["eligible_for_implementation"].astype(str).str.lower().eq("true")
    ]["variant"].astype(str).tolist() if not holdout.empty else []
    return "\n".join(
        [
            "# HIRSSM V3.20 Rescue Signal Research",
            "",
            "## Purpose",
            "",
            "Find new drawdown/reentry candidates after V3.19 blocks the duplicate no-trade path.",
            "",
            "## Findings",
            "",
            f"- Signals tested: {int(validation.shape[0])}",
            f"- Passed pre-backtest gate: {', '.join(passed) if passed else 'none'}",
            f"- Passed holdout implementation gate: {', '.join(holdout_passed) if holdout_passed else 'none'}",
            f"- Candidate rows: {int((registry['role'] == 'candidate').sum())}",
            "",
            "## Decision",
            "",
            "- This is signal research only.",
            "- Only signals passing both full-sample diagnostic and holdout gates may enter a governed implementation harness.",
        ]
    )


def self_check(validation: pd.DataFrame, holdout: pd.DataFrame, registry: pd.DataFrame, specs: pd.DataFrame) -> pd.DataFrame:
    candidate_count = int((registry["role"] == "candidate").sum()) if not registry.empty else 0
    return pd.DataFrame(
        [
            {"check": "signal_validation_exists", "status": "pass" if not validation.empty else "fail", "detail": str(int(validation.shape[0]))},
            {"check": "signal_gate_holdout_exists", "status": "pass" if not holdout.empty else "fail", "detail": str(int(holdout.shape[0]))},
            {"check": "candidate_registry_exists", "status": "pass" if not registry.empty else "fail", "detail": str(int(registry.shape[0]))},
            {"check": "implementation_specs_exist", "status": "pass" if not specs.empty else "fail", "detail": str(int(specs.shape[0]))},
            {"check": "at_least_one_candidate_for_harness", "status": "pass" if candidate_count >= 1 else "warn", "detail": str(candidate_count)},
            {"check": "research_only_no_promotion", "status": "pass", "detail": "pre-backtest signal validation only"},
        ]
    )


def manifest(start_time: str, output_dir: Path, artifacts: list[Path], metrics: dict[str, Any], fail_count: int, warn_count: int) -> dict[str, Any]:
    return {
        "run_id": f"{TASK_ID}_run_001",
        "task_id": TASK_ID,
        "agent": "factor_researcher",
        "version": "V3.20",
        "baseline": "HIRSSM V3.10 Clean Rank-Vol Core",
        "status": "pass" if fail_count == 0 else "fail",
        "started_at": start_time,
        "command": "python -X utf8 strategy_lab/hirssm_v3_20_rescue_signal_research.py",
        "config": {"research_only": True, "forward_window_days": 63},
        "data_refs": ["outputs/hirssm_v3_10_clean_baseline", "outputs/agent_runs/v3_19/filtered_no_trade_candidate"],
        "code_refs": ["strategy_lab/hirssm_v3_20_rescue_signal_research.py"],
        "output_dir": str(output_dir.relative_to(ROOT).as_posix()),
        "allowed_inputs": ["outputs/hirssm_v3_10_clean_baseline", "outputs/agent_runs/v3_19/filtered_no_trade_candidate"],
        "artifacts": [str(path.relative_to(ROOT).as_posix()) for path in artifacts],
        "outputs": [str(path.relative_to(ROOT).as_posix()) for path in artifacts if path.name != "agent_run_manifest.json"],
        "changed_files": [str(path.relative_to(ROOT).as_posix()) for path in artifacts],
        "metrics": metrics,
        "self_check_pass": fail_count == 0,
        "fail_count": fail_count,
        "warn_count": warn_count,
        "limitations": ["Uses historical signal validation and cannot promote a default model.", "Forward labels are used only for pre-backtest triage.", "Implementation requires a separate holdout signal gate."],
        "risk_flags": ["full_sample_signal_diagnostic", "candidate_requires_nested_validation"],
        "next_decision": "Implement only signals passing both full-sample diagnostic and holdout signal gates.",
        "handoff_summary": "V3.20 produced drawdown/reentry diagnostics and holdout-gated implementation specs after V3.19 blocked duplicate candidates.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate V3.20 rescue signal research artifacts.")
    parser.add_argument("--baseline-dir", default=str(BASELINE_DIR))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    start_time = now_text()
    baseline_dir = Path(args.baseline_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nav = read_csv(baseline_dir / "nav_clean_rank_vol_core_10bps.csv")
    targets = read_csv(baseline_dir / "target_weights.csv")
    panel = signal_dates(targets, daily_features(nav))
    validation = validate_signals(panel)
    holdout = signal_gate_holdout_validation(panel, validation)
    registry = candidate_registry(validation, holdout)
    specs = implementation_specs(validation, holdout)
    checks = self_check(validation, holdout, registry, specs)

    panel_path = output_dir / "signal_feature_panel.csv"
    validation_path = output_dir / "signal_validation.csv"
    holdout_path = output_dir / "signal_gate_holdout_validation.csv"
    registry_path = output_dir / "candidate_registry.csv"
    specs_path = output_dir / "implementation_candidate_spec.csv"
    report_path = output_dir / "agent_report.md"
    checks_path = output_dir / "self_check.csv"
    changed_path = output_dir / "changed_files.txt"
    manifest_path = output_dir / "agent_run_manifest.json"
    panel.to_csv(panel_path, index=False, encoding="utf-8-sig")
    validation.to_csv(validation_path, index=False, encoding="utf-8-sig")
    holdout.to_csv(holdout_path, index=False, encoding="utf-8-sig")
    registry.to_csv(registry_path, index=False, encoding="utf-8-sig")
    specs.to_csv(specs_path, index=False, encoding="utf-8-sig")
    write_text(make_report(validation, holdout, registry), report_path)
    checks.to_csv(checks_path, index=False, encoding="utf-8-sig")
    artifacts = [panel_path, validation_path, holdout_path, registry_path, specs_path, report_path, checks_path, changed_path, manifest_path]
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
