#!/usr/bin/env python
"""HIRSSM V3.14 orthogonal candidate research.

This is a pre-backtest factor research task. It validates whether V3.13
candidate directions have measurable signal evidence and whether existing
candidate returns are too similar before any portfolio implementation.
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


ROOT = Path("Introduction-to-Quantitative-Finance")
CONFIG = ROOT / "configs" / "hirssm_v2_default.json"
V313_DIR = ROOT / "outputs" / "agent_runs" / "v3_13" / "failure_revision_design"
V312_DIR = ROOT / "outputs" / "hirssm_v3_12_candidate_implementation_harness"
OUTPUT_DIR = ROOT / "outputs" / "agent_runs" / "v3_14" / "orthogonal_candidate_research"
TASK_ID = "20260526_v3_14_orthogonal_candidate_research"
BASELINE_VARIANT = "v3_10_clean_rank_vol_core"
TRADING_DAYS = 252


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def future_returns(returns: pd.DataFrame, horizon: int = 21) -> pd.DataFrame:
    ret = returns.copy()
    ret["date"] = pd.to_datetime(ret["date"])
    ret["ret_1d"] = pd.to_numeric(ret["ret_1d"], errors="coerce").fillna(0.0)
    rows = []
    for asset, group in ret.groupby("asset", sort=False):
        g = group.sort_values("date").copy()
        # Forward return from the next trading day through horizon; no same-day look-ahead.
        fwd = (1.0 + g["ret_1d"]).shift(-1).rolling(horizon, min_periods=horizon).apply(np.prod, raw=True).shift(-(horizon - 1)) - 1.0
        rows.append(pd.DataFrame({"date": g["date"], "asset": asset, f"fwd_{horizon}d": fwd}))
    return pd.concat(rows, ignore_index=True, sort=False)


def spearman(x: pd.Series, y: pd.Series) -> float:
    frame = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if frame.shape[0] < 4:
        return np.nan
    xr = frame["x"].rank()
    yr = frame["y"].rank()
    if float(xr.std(ddof=0)) == 0.0 or float(yr.std(ddof=0)) == 0.0:
        return np.nan
    return float(xr.corr(yr))


def summarize_ic(rows: list[dict[str, Any]], signal_name: str, asset_scope: str, hypothesis: str) -> dict[str, Any]:
    df = pd.DataFrame(rows)
    if df.empty:
        return {
            "signal": signal_name,
            "asset_scope": asset_scope,
            "hypothesis": hypothesis,
            "observations": 0,
            "rank_ic_mean": np.nan,
            "rank_ic_std": np.nan,
            "rank_icir": np.nan,
            "positive_ic_rate": np.nan,
            "pass_pre_backtest_gate": False,
            "status": "blocked",
        }
    ic = pd.to_numeric(df["rank_ic"], errors="coerce").dropna()
    std = float(ic.std(ddof=1)) if ic.shape[0] > 1 else np.nan
    mean = float(ic.mean()) if not ic.empty else np.nan
    icir = float(mean / std) if pd.notna(std) and std > 0 else np.nan
    positive = float((ic > 0).mean()) if not ic.empty else np.nan
    passed = bool(pd.notna(mean) and mean > 0 and pd.notna(positive) and positive >= 0.52 and ic.shape[0] >= 24)
    return {
        "signal": signal_name,
        "asset_scope": asset_scope,
        "hypothesis": hypothesis,
        "observations": int(ic.shape[0]),
        "rank_ic_mean": mean,
        "rank_ic_std": std,
        "rank_icir": icir,
        "positive_ic_rate": positive,
        "pass_pre_backtest_gate": passed,
        "status": "candidate" if passed else "observation",
    }


def cross_sectional_signal_validation(panel: dict[str, pd.DataFrame], horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible = panel["eligible"].copy()
    eligible["date"] = pd.to_datetime(eligible["date"])
    fwd = future_returns(panel["returns"], horizon=horizon)
    data = eligible.merge(fwd, on=["date", "asset"], how="left")
    residual_cols = [col for col in ["residual_momentum_60_z", "residual_momentum_120_z", "residual_momentum_120", "excess_ret_60_z", "excess_ret_120_z"] if col in data.columns]
    style_cols = [col for col in ["valuation_repair_score", "risk_compression_score", "defensive_score"] if col in data.columns]
    data["residual_industry_momentum_signal"] = data[residual_cols].mean(axis=1) if residual_cols else np.nan
    data["style_value_quality_barbell_signal"] = data[style_cols].mean(axis=1) if style_cols else np.nan
    detail_rows = []
    summary_rows = []
    definitions = [
        (
            "residual_industry_momentum_low_corr",
            "industry",
            "residual_industry_momentum_signal",
            "Industry leadership should persist after removing broad beta when breadth confirms it.",
        ),
        (
            "value_quality_defensive_barbell",
            "style",
            "style_value_quality_barbell_signal",
            "Style valuation repair and defensive risk quality should help allocate between defense and recovery sleeves.",
        ),
    ]
    for signal_name, asset_scope, score_col, hypothesis in definitions:
        rows = []
        scoped = data[data["asset_type"].astype(str).eq(asset_scope)].copy()
        for signal_date, group in scoped.groupby("date", sort=True):
            g = group.dropna(subset=[score_col, f"fwd_{horizon}d"])
            if g.shape[0] < 4:
                continue
            ic = spearman(g[score_col], g[f"fwd_{horizon}d"])
            rows.append({"date": signal_date, "signal": signal_name, "asset_scope": asset_scope, "rank_ic": ic, "asset_count": int(g.shape[0])})
        detail_rows.extend(rows)
        summary_rows.append(summarize_ic(rows, signal_name, asset_scope, hypothesis))
    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)


def breadth_regime_validation(panel: dict[str, pd.DataFrame], horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    regimes = panel["regimes"].copy()
    regimes["date"] = pd.to_datetime(regimes["date"])
    regimes["breadth_repair_signal"] = (
        pd.to_numeric(regimes["industry_above_ma60_ratio"], errors="coerce")
        + pd.to_numeric(regimes["industry_positive_ret20_ratio"], errors="coerce")
    ) / 2.0
    fwd = future_returns(panel["returns"][panel["returns"]["asset"].astype(str).eq(panel["broad_code"])], horizon=horizon)
    broad = fwd.rename(columns={f"fwd_{horizon}d": "broad_fwd_return"})
    data = regimes.merge(broad[["date", "broad_fwd_return"]], on="date", how="left").dropna(subset=["breadth_repair_signal", "broad_fwd_return"])
    monthly_dates = set(model.month_end_dates(data["date"]))
    data = data[data["date"].isin(monthly_dates)].copy()
    data["breadth_bucket"] = pd.qcut(data["breadth_repair_signal"].rank(method="first"), 3, labels=["weak", "neutral", "strong"])
    corr = spearman(data["breadth_repair_signal"], data["broad_fwd_return"])
    bucket = data.groupby("breadth_bucket", observed=False).agg(
        observations=("broad_fwd_return", "size"),
        mean_fwd_return=("broad_fwd_return", "mean"),
        downside_rate=("broad_fwd_return", lambda x: float((pd.Series(x) < 0).mean())),
        avg_signal=("breadth_repair_signal", "mean"),
    ).reset_index()
    weak = bucket[bucket["breadth_bucket"].astype(str).eq("weak")]
    strong = bucket[bucket["breadth_bucket"].astype(str).eq("strong")]
    spread = float(strong["mean_fwd_return"].iloc[0] - weak["mean_fwd_return"].iloc[0]) if not weak.empty and not strong.empty else np.nan
    pass_gate = bool(pd.notna(corr) and corr > 0 and pd.notna(spread) and spread > 0 and data.shape[0] >= 24)
    summary = pd.DataFrame(
        [
            {
                "signal": "orthogonal_breadth_regime_overlay",
                "asset_scope": "time_series_broad_market",
                "hypothesis": "Breadth deterioration and repair should forecast broad-market risk budget better than minor expert multipliers.",
                "observations": int(data.shape[0]),
                "rank_ic_mean": float(corr) if pd.notna(corr) else np.nan,
                "rank_ic_std": np.nan,
                "rank_icir": np.nan,
                "positive_ic_rate": np.nan,
                "bucket_strong_minus_weak_return": spread,
                "pass_pre_backtest_gate": pass_gate,
                "status": "candidate" if pass_gate else "observation",
            }
        ]
    )
    return summary, bucket


def candidate_similarity_matrix(v312_dir: Path, cost_bps: int = 10) -> pd.DataFrame:
    frames = []
    for path in sorted(v312_dir.glob(f"nav_*_{cost_bps}bps.csv")):
        name = path.stem.removeprefix("nav_").removesuffix(f"_{cost_bps}bps")
        if name.startswith("pbo_stability_penalized_selector"):
            continue
        nav = pd.read_csv(path, encoding="utf-8-sig")
        if {"date", "portfolio_return", "benchmark_return"}.issubset(nav.columns):
            item = nav[["date", "portfolio_return", "benchmark_return"]].copy()
            item["date"] = pd.to_datetime(item["date"])
            item[name] = pd.to_numeric(item["portfolio_return"], errors="coerce") - pd.to_numeric(item["benchmark_return"], errors="coerce")
            frames.append(item[["date", name]])
    if not frames:
        return pd.DataFrame()
    wide = frames[0]
    for item in frames[1:]:
        wide = wide.merge(item, on="date", how="inner")
    corr = wide.drop(columns=["date"]).corr()
    corr.index.name = "variant"
    out = corr.reset_index()
    return out


def implementation_specs(v313_dir: Path, factor_validation: pd.DataFrame, similarity: pd.DataFrame) -> pd.DataFrame:
    registry = read_csv(v313_dir / "candidate_registry.csv")
    if registry.empty:
        return pd.DataFrame()
    validation_map = {
        str(row["signal"]): bool(row["pass_pre_backtest_gate"])
        for _, row in factor_validation.iterrows()
        if "pass_pre_backtest_gate" in factor_validation.columns
    }
    rows = []
    for _, row in registry[registry["role"].astype(str).eq("candidate")].iterrows():
        variant = str(row["variant"])
        if variant == "cost_aware_no_trade_band_overlay":
            gate = True
            next_action = "test as execution overlay only"
        elif variant == "candidate_diversity_selector":
            gate = not similarity.empty
            next_action = "filter near-duplicate candidate navs before PBO"
        else:
            gate = bool(validation_map.get(variant, False))
            next_action = "eligible for V3.15 implementation" if gate else "keep observation until signal evidence improves"
        rows.append(
            {
                "variant": variant,
                "pre_backtest_gate_pass": gate,
                "implementation_priority": "high" if gate else "observation",
                "next_action": next_action,
                "fixed_inputs": row.get("fixed_inputs", ""),
                "fixed_outputs": row.get("fixed_outputs", ""),
                "forbidden": row.get("forbidden", ""),
                "acceptance_standard": row.get("acceptance_standard", ""),
            }
        )
    return pd.DataFrame(rows)


def self_check(output_dir: Path, validation: pd.DataFrame, specs: pd.DataFrame, similarity: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "check": "factor_validation_exists",
            "status": "pass" if not validation.empty else "fail",
            "detail": str(int(validation.shape[0])),
        },
        {
            "check": "candidate_similarity_matrix_exists",
            "status": "pass" if not similarity.empty else "fail",
            "detail": str(int(similarity.shape[0])),
        },
        {
            "check": "implementation_specs_exist",
            "status": "pass" if not specs.empty else "fail",
            "detail": str(int(specs.shape[0])),
        },
        {
            "check": "no_full_backtest_in_v3_14",
            "status": "pass",
            "detail": "pre-backtest signal validation only",
        },
        {
            "check": "output_dir_scoped",
            "status": "pass" if str(output_dir.as_posix()).endswith("outputs/agent_runs/v3_14/orthogonal_candidate_research") else "fail",
            "detail": str(output_dir.as_posix()),
        },
    ]
    return pd.DataFrame(rows)


def make_report(validation: pd.DataFrame, specs: pd.DataFrame, similarity: pd.DataFrame) -> str:
    pass_count = int(pd.Series(validation.get("pass_pre_backtest_gate", [])).fillna(False).astype(bool).sum()) if not validation.empty else 0
    max_corr = np.nan
    if not similarity.empty:
        values = similarity.set_index("variant").to_numpy(dtype=float)
        if values.shape[0] > 1:
            mask = ~np.eye(values.shape[0], dtype=bool)
            max_corr = float(np.nanmax(np.abs(values[mask])))
    lines = [
        "# HIRSSM V3.14 Orthogonal Candidate Research",
        "",
        "## Purpose",
        "",
        "Validate V3.13 candidate signal directions before portfolio backtesting.",
        "",
        "## Findings",
        "",
        f"- Factor signals tested: {int(validation.shape[0])}",
        f"- Signals passing pre-backtest gate: {pass_count}",
        f"- Max absolute active-return correlation among V3.12 raw candidates: {max_corr:.6f}" if pd.notna(max_corr) else "- Candidate similarity unavailable.",
        "",
        "## Decision",
        "",
        "- V3.14 is accepted as signal research only.",
        "- Passing signals may be implemented in V3.15; failed signals remain observation.",
        "- No full-sample portfolio result is used for promotion.",
    ]
    if not specs.empty:
        lines.extend(["", "## Candidate Actions", ""])
        for _, row in specs.iterrows():
            lines.append(f"- {row['variant']}: {row['implementation_priority']} - {row['next_action']}")
    return "\n".join(lines)


def make_agent_manifest(start_time: str, output_dir: Path, artifacts: list[Path], metrics: dict[str, Any], fail_count: int, warn_count: int) -> dict[str, Any]:
    return {
        "run_id": f"{TASK_ID}_run_001",
        "task_id": TASK_ID,
        "agent": "factor_researcher",
        "version": "V3.14",
        "baseline": "HIRSSM V3.10 Clean Rank-Vol Core",
        "status": "pass" if fail_count == 0 else "fail",
        "started_at": start_time,
        "finished_at": now_text(),
        "command": "python -X utf8 strategy_lab/hirssm_v3_14_orthogonal_candidate_research.py",
        "config": {"config_path": str(CONFIG.as_posix()), "horizon_days": 21, "design_only": True},
        "data_refs": ["data_raw/index/akshare_csindex", "data_raw/index/akshare_sw_industry", "outputs/hirssm_v3_12_candidate_implementation_harness"],
        "code_refs": ["strategy_lab/hirssm_v3_14_orthogonal_candidate_research.py", "strategy_lab/hirssm_v2_model.py", "strategy_lab/hirssm_v2_walk_forward.py"],
        "output_dir": str(output_dir.relative_to(ROOT).as_posix()),
        "allowed_inputs": ["outputs/agent_runs/v3_13/failure_revision_design", "outputs/hirssm_v3_12_candidate_implementation_harness", "data_raw/index"],
        "artifacts": [str(path.relative_to(ROOT).as_posix()) for path in artifacts],
        "outputs": [str(path.relative_to(ROOT).as_posix()) for path in artifacts if path.name != "agent_run_manifest.json"],
        "changed_files": [str(path.relative_to(ROOT).as_posix()) for path in artifacts],
        "metrics": metrics,
        "self_check_pass": fail_count == 0,
        "fail_count": fail_count,
        "warn_count": warn_count,
        "limitations": ["Pre-backtest signal validation only; no portfolio promotion.", "RankIC evidence is historical and must be nested before promotion."],
        "risk_flags": ["signal_validation_not_full_backtest", "candidate_similarity_may_be_sample_specific"],
        "next_decision": "Implement only predeclared passing or execution-only candidates in V3.15.",
        "handoff_summary": "V3.14 validated orthogonal candidate signals and produced implementation specs for V3.15.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run HIRSSM V3.14 orthogonal candidate research.")
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--horizon", type=int, default=21)
    args = parser.parse_args()

    start_time = now_text()
    root = Path(args.root)
    config = model.read_json(Path(args.config))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    panel = wf.build_panel(model, root, config, None, None)
    cs_summary, cs_detail = cross_sectional_signal_validation(panel, horizon=args.horizon)
    breadth_summary, breadth_bucket = breadth_regime_validation(panel, horizon=args.horizon)
    validation = pd.concat([breadth_summary, cs_summary], ignore_index=True, sort=False)
    similarity = candidate_similarity_matrix(V312_DIR, cost_bps=10)
    specs = implementation_specs(V313_DIR, validation, similarity)
    checks = self_check(output_dir, validation, specs, similarity)

    factor_path = output_dir / "factor_validation.csv"
    factor_detail_path = output_dir / "factor_validation_detail.csv"
    breadth_path = output_dir / "breadth_bucket_validation.csv"
    similarity_path = output_dir / "candidate_similarity_matrix.csv"
    spec_path = output_dir / "candidate_implementation_spec.csv"
    report_path = output_dir / "agent_report.md"
    checks_path = output_dir / "self_check.csv"
    manifest_path = output_dir / "agent_run_manifest.json"
    changed_path = output_dir / "changed_files.txt"

    validation.to_csv(factor_path, index=False, encoding="utf-8-sig")
    cs_detail.to_csv(factor_detail_path, index=False, encoding="utf-8-sig")
    breadth_bucket.to_csv(breadth_path, index=False, encoding="utf-8-sig")
    similarity.to_csv(similarity_path, index=False, encoding="utf-8-sig")
    specs.to_csv(spec_path, index=False, encoding="utf-8-sig")
    write_text(make_report(validation, specs, similarity), report_path)
    checks.to_csv(checks_path, index=False, encoding="utf-8-sig")
    artifacts = [factor_path, factor_detail_path, breadth_path, similarity_path, spec_path, report_path, checks_path, changed_path, manifest_path]
    write_text("\n".join(str(path.relative_to(root).as_posix()) for path in artifacts), changed_path)

    fail_count = int((checks["status"] == "fail").sum())
    warn_count = int((checks["status"] == "warn").sum())
    metrics = {
        "factor_count": int(validation.shape[0]),
        "pre_backtest_pass_count": int(pd.Series(validation.get("pass_pre_backtest_gate", [])).fillna(False).astype(bool).sum()) if not validation.empty else 0,
        "implementation_spec_count": int(specs.shape[0]),
    }
    manifest = make_agent_manifest(start_time, output_dir, artifacts, metrics, fail_count, warn_count)
    write_json(manifest, manifest_path)
    result = {"task_id": TASK_ID, "self_check_pass": fail_count == 0, "fail_count": fail_count, "warn_count": warn_count, "metrics": metrics, "output_dir": str(output_dir)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
