#!/usr/bin/env python
"""Expert pruning study for HIRSSM V2.0.

The study reuses the HIRSSM feature pipeline once, then runs multiple expert
disable combinations through the same target construction and backtest logic.
It is intended for research governance, not live trading authorization.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import sys
from pathlib import Path

import pandas as pd


ROOT = Path("Introduction-to-Quantitative-Finance")
MODEL_PATH = ROOT / "strategy_lab" / "hirssm_v2_model.py"
CONFIG = ROOT / "configs" / "hirssm_v2_default.json"


def load_model():
    spec = importlib.util.spec_from_file_location("hirssm_v2_model", MODEL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {MODEL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_panel(model, root: Path, config: dict, start_date: str | None, end_date: str | None) -> dict:
    style = model.load_style_daily(root, config)
    industry = model.load_industry_daily(root, config)
    if start_date:
        start = pd.to_datetime(start_date)
        style = style[style["date"] >= start]
        industry = industry[industry["date"] >= start]
    if end_date:
        end = pd.to_datetime(end_date)
        style = style[style["date"] <= end]
        industry = industry[industry["date"] <= end]

    broad_code = model.normalize_code(config["asset_universe"]["style"].get("broad_market", "000985"))
    style_raw = style.sort_values(["asset", "date"]).copy()
    style_raw["ret_1d"] = style_raw.groupby("asset")["close"].pct_change()
    market_returns = style_raw[style_raw["asset"].eq(broad_code)].set_index("date")["ret_1d"]

    style_features = model.add_features(style, market_returns=market_returns)
    industry_features = model.add_features(industry, market_returns=market_returns)
    valuation = model.load_style_valuation(root, config, style_features["asset"].drop_duplicates())
    style_features = model.add_valuation_scores(style_features, valuation)
    industry_features["valuation_score"] = 0.0

    style_scores = model.score_assets(style_features, is_style=True)
    industry_scores = model.score_assets(industry_features, is_style=False)
    scored = pd.concat([style_scores, industry_scores], ignore_index=True, sort=False)
    regimes = model.assign_regime(style_scores, industry_scores, config)

    min_history_days = int(config["feature_pipeline"].get("min_history_days", 504))
    eligible = scored.copy()
    eligible["history_count"] = eligible.groupby("asset").cumcount() + 1
    eligible = eligible[eligible["history_count"] >= min_history_days]
    returns = scored[["date", "asset", "ret_1d"]].dropna().copy()
    expert_ic = model.expert_rank_ic_report(scored, regimes)
    return {
        "scored": scored,
        "eligible": eligible,
        "regimes": regimes,
        "returns": returns,
        "broad_code": broad_code,
        "expert_ic": expert_ic,
    }


def variant_name(extra_disabled: tuple[str, ...]) -> str:
    if not extra_disabled:
        return "base_default"
    return "disable_" + "_".join(extra_disabled)


def run_variant(model, panel: dict, config: dict, active_disabled: set[str], cost_bps: float) -> dict:
    targets = model.build_targets(
        panel["eligible"],
        panel["regimes"],
        config,
        start_date=panel["eligible"]["date"].min(),
        disabled_experts=active_disabled,
    )
    bt = model.run_backtest(panel["returns"], targets, cost_bps, panel["broad_code"])
    summary = model.summarize_nav(bt["nav"])
    if summary.empty:
        return {}
    row = summary.iloc[0].to_dict()
    row["target_rows"] = int(targets.shape[0])
    row["latest_signal_date"] = str(targets["signal_date"].max()) if not targets.empty else ""
    return row


def make_pruning_report(
    output_dir: Path,
    variants: pd.DataFrame,
    expert_ic: pd.DataFrame,
    default_disabled: set[str],
    cost_bps: float,
) -> None:
    base = variants[variants["variant"].eq("base_default")].head(1)
    best_return = variants.sort_values("annual_return", ascending=False).head(8)
    best_sharpe = variants.sort_values("sharpe_no_rf", ascending=False).head(8)
    lines = [
        "# HIRSSM V2.0 Expert Pruning Study",
        "",
        f"Cost bps: {cost_bps:g}",
        f"Default disabled experts: `{', '.join(sorted(default_disabled)) if default_disabled else 'none'}`",
        "",
        "## Base",
        "",
        base.to_markdown(index=False) if not base.empty else "No base result.",
        "",
        "## Top By Annual Return",
        "",
        best_return.to_markdown(index=False),
        "",
        "## Top By Sharpe",
        "",
        best_sharpe.to_markdown(index=False),
        "",
        "## Expert RankIC",
        "",
        expert_ic.sort_values(["asset_type", "rank_ic_mean"], ascending=[True, False]).to_markdown(index=False),
        "",
        "## Notes",
        "",
        "- This is an in-sample pruning diagnostic. It can reject clearly bad experts, but cannot promote a variant to production without walk-forward validation.",
        "- A higher full-sample return after disabling an expert is a warning signal, not sufficient proof that the expert should be permanently removed.",
    ]
    (output_dir / "HIRSSM_V2_EXPERT_PRUNING_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "hirssm_v2_pruning"))
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument(
        "--candidate-experts",
        nargs="*",
        default=["trend_continuation", "liquidity_overlay", "valuation_repair", "risk_compression", "defensive"],
    )
    parser.add_argument("--max-combo-size", type=int, default=3)
    args = parser.parse_args()

    model = load_model()
    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = model.read_json(Path(args.config))
    default_disabled = {str(item) for item in config.get("disabled_experts_by_default", [])}
    panel = build_panel(model, root, config, args.start_date, args.end_date)

    rows = []
    candidates = [item for item in args.candidate_experts if item not in default_disabled]
    for size in range(0, min(args.max_combo_size, len(candidates)) + 1):
        for combo in itertools.combinations(candidates, size):
            active_disabled = default_disabled | set(combo)
            row = run_variant(model, panel, config, active_disabled, args.cost_bps)
            if not row:
                continue
            row["variant"] = variant_name(combo)
            row["extra_disabled"] = ",".join(combo)
            row["active_disabled"] = ",".join(sorted(active_disabled))
            rows.append(row)

    variants = pd.DataFrame(rows)
    if not variants.empty:
        base = variants[variants["variant"].eq("base_default")].iloc[0]
        for col in ["annual_return", "sharpe_no_rf", "max_drawdown", "total_return", "avg_cash_weight"]:
            variants[f"delta_{col}_vs_base"] = variants[col] - base[col]
        variants = variants[
            [
                "variant",
                "extra_disabled",
                "active_disabled",
                "total_return",
                "annual_return",
                "annual_vol",
                "sharpe_no_rf",
                "max_drawdown",
                "calmar",
                "avg_cash_weight",
                "avg_trade_turnover",
                "trade_count",
                "benchmark_annual_return",
                "benchmark_max_drawdown",
                "delta_annual_return_vs_base",
                "delta_sharpe_no_rf_vs_base",
                "delta_max_drawdown_vs_base",
                "delta_total_return_vs_base",
                "delta_avg_cash_weight_vs_base",
                "target_rows",
                "latest_signal_date",
            ]
        ].sort_values(["annual_return", "sharpe_no_rf"], ascending=False)

    model.write_csv(variants, output_dir / "expert_pruning_variants.csv")
    model.write_csv(panel["expert_ic"], output_dir / "expert_rank_ic.csv")
    make_pruning_report(output_dir, variants, panel["expert_ic"], default_disabled, args.cost_bps)
    print(
        {
            "output_dir": str(output_dir.resolve()),
            "variant_rows": int(variants.shape[0]),
            "default_disabled": sorted(default_disabled),
        }
    )


if __name__ == "__main__":
    main()
