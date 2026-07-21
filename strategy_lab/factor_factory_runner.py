#!/usr/bin/env python
"""Autonomous factor factory runner.

This runner turns the multi-factor research framework into a reproducible
experiment loop. It is intentionally data-vendor neutral: pass a point-in-time
panel and a factor registry, then receive audit tables, diagnostics, selected
factors, scores, weights, and performance summaries.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import multi_factor_research_framework as mf


@dataclass(frozen=True)
class FactoryConfig:
    date_col: str = "date"
    asset_col: str = "asset"
    forward_return_col: str = "fwd_return"
    industry_col: str | None = "industry"
    control_cols: tuple[str, ...] = ("log_mkt_cap",)
    weight_col: str | None = None
    min_assets: int = 30
    groups: int = 5
    corr_threshold: float = 0.85
    min_ic_periods: int = 12
    min_abs_rank_ic: float = 0.01
    min_icir: float = 0.10
    top_quantile: float = 0.20
    max_weight: float = 0.05
    max_per_family: int = 30
    periods_per_year: int = 12
    long_only: bool = True
    output_score_col: str = "multi_factor_score"


def load_config(path: str | Path | None = None) -> FactoryConfig:
    if path is None:
        return FactoryConfig()
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if "control_cols" in raw:
        raw["control_cols"] = tuple(raw["control_cols"])
    return FactoryConfig(**raw)


def write_table(output_dir: Path, name: str, table: pd.DataFrame | pd.Series) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(table, pd.Series):
        table = table.to_frame("value").reset_index(names="metric")
    table.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")


def _as_validation_config(config: FactoryConfig) -> mf.ValidationConfig:
    return mf.ValidationConfig(
        min_assets=config.min_assets,
        groups=config.groups,
        corr_threshold=config.corr_threshold,
        min_ic_periods=config.min_ic_periods,
        min_abs_rank_ic=config.min_abs_rank_ic,
        min_icir=config.min_icir,
        top_quantile=config.top_quantile,
        max_weight=config.max_weight,
    )


def run_factor_factory(
    panel: pd.DataFrame,
    registry: pd.DataFrame,
    config: FactoryConfig = FactoryConfig(),
) -> dict[str, Any]:
    """Run one full factor-factory experiment."""
    data = panel.copy()
    data[config.date_col] = pd.to_datetime(data[config.date_col])
    registry = registry.copy()

    audit = mf.check_availability_dates(data, config.date_col, registry)
    data, alpha_cols = mf.apply_factor_directions(data, registry)
    if not alpha_cols:
        raise ValueError("No factor columns from registry were found in panel.")

    data, win_cols = mf.winsorize_by_date(data, config.date_col, alpha_cols)
    data, z_cols = mf.zscore_by_date(data, config.date_col, win_cols)
    neutralize_controls = [col for col in config.control_cols if col in data.columns]
    data, model_factor_cols = mf.neutralize_by_date(
        data,
        config.date_col,
        z_cols,
        control_cols=neutralize_controls,
        industry_col=config.industry_col if config.industry_col in data.columns else None,
        weight_col=config.weight_col if config.weight_col and config.weight_col in data.columns else None,
    )

    ic = mf.factor_ic_long(
        data,
        config.date_col,
        model_factor_cols,
        config.forward_return_col,
        min_assets=config.min_assets,
    )
    ic_stats = mf.ic_summary(ic)
    quantile_returns = mf.quantile_spread_long(
        data,
        config.date_col,
        model_factor_cols,
        config.forward_return_col,
        groups=config.groups,
        min_assets=config.min_assets,
    )
    turnover = mf.top_quantile_turnover(
        data,
        config.date_col,
        config.asset_col,
        model_factor_cols,
        top_quantile=config.top_quantile,
        min_assets=config.min_assets,
    )
    corr = mf.average_factor_correlation(
        data,
        config.date_col,
        model_factor_cols,
        min_assets=config.min_assets,
    )
    clusters = mf.correlation_clusters(corr, threshold=config.corr_threshold)
    quality = mf.factor_quality_table(ic_stats, turnover, registry=registry)
    selected = mf.select_factors(
        quality,
        clusters,
        config=_as_validation_config(config),
        max_per_family=config.max_per_family,
    )
    if selected.empty:
        raise ValueError("No factors passed selection gates. Loosen config or improve factor library.")

    scored, family_cols = mf.combine_by_family(data, config.date_col, selected)
    scored = mf.combine_families(scored, config.date_col, family_cols, output_col=config.output_score_col)
    score_col = f"{config.output_score_col}_z"
    weights = mf.portfolio_weights_from_scores(
        scored,
        config.date_col,
        config.asset_col,
        score_col,
        long_only=config.long_only,
        top_quantile=config.top_quantile,
        max_weight=config.max_weight,
    )
    portfolio_returns = mf.portfolio_forward_returns(
        weights,
        scored,
        config.date_col,
        config.asset_col,
        config.forward_return_col,
    )
    performance = mf.performance_summary(portfolio_returns, periods_per_year=config.periods_per_year)

    meta = pd.DataFrame(
        [
            {
                "n_rows": int(panel.shape[0]),
                "n_assets": int(panel[config.asset_col].nunique()),
                "n_dates": int(pd.to_datetime(panel[config.date_col]).nunique()),
                "registered_factors": int(registry.shape[0]),
                "available_factors": int(len(alpha_cols)),
                "model_factors": int(len(model_factor_cols)),
                "selected_factors": int(selected.shape[0]),
                "families": int(len(family_cols)),
                "score_col": score_col,
            }
        ]
    )

    return {
        "meta": meta,
        "availability_audit": audit,
        "prepared_panel": scored,
        "ic_by_date": ic,
        "ic_summary": ic_stats,
        "quantile_returns": quantile_returns,
        "turnover": turnover,
        "factor_correlation": corr.reset_index(names="factor"),
        "clusters": clusters,
        "quality": quality,
        "selected": selected,
        "weights": weights,
        "portfolio_returns": portfolio_returns.rename("return").reset_index().rename(columns={"index": "date"}),
        "performance": performance,
        "github_lessons": mf.github_quant_project_lessons(),
        "quality_gates": mf.hundred_factor_model_checklist(),
    }


def save_results(results: dict[str, Any], output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, value in results.items():
        if isinstance(value, (pd.DataFrame, pd.Series)):
            write_table(out, name, value)
    manifest = {
        "tables": sorted([f"{name}.csv" for name, value in results.items() if isinstance(value, (pd.DataFrame, pd.Series))]),
        "created_by": "factor_factory_runner.py",
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def make_synthetic_panel(
    n_dates: int = 48,
    n_assets: int = 120,
    n_signal_factors: int = 12,
    n_noise_factors: int = 24,
    seed: int = 20260523,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate a point-in-time synthetic panel for smoke tests and demos."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-31", periods=n_dates, freq="ME")
    assets = [f"S{i:04d}" for i in range(n_assets)]
    families = ["value", "quality", "momentum", "liquidity", "technical", "macro"]
    rows: list[dict[str, Any]] = []
    factor_records: list[dict[str, Any]] = []
    factor_ids = [f"sig_{i:02d}" for i in range(n_signal_factors)] + [f"noise_{i:02d}" for i in range(n_noise_factors)]
    true_betas = {fid: rng.normal(0.006, 0.004) if fid.startswith("sig") else 0.0 for fid in factor_ids}
    for i, fid in enumerate(factor_ids):
        factor_records.append(
            {
                "factor_id": fid,
                "column": f"{fid}_raw",
                "family": families[i % len(families)],
                "direction": 1.0,
                "horizon": 20,
                "data_type": "synthetic",
                "availability_col": None,
                "cost_tier": "low" if fid.startswith("sig") else "medium",
                "description": "synthetic smoke-test factor",
            }
        )
    for date in dates:
        market_shock = rng.normal(0, 0.02)
        for asset_idx, asset in enumerate(assets):
            row: dict[str, Any] = {
                "date": date,
                "asset": asset,
                "industry": f"industry_{asset_idx % 10}",
                "log_mkt_cap": rng.normal(10, 1),
            }
            fwd = market_shock + rng.normal(0, 0.08)
            common_style = rng.normal()
            for fid in factor_ids:
                value = 0.35 * common_style + rng.normal()
                row[f"{fid}_raw"] = value
                fwd += true_betas[fid] * value
            row["fwd_return"] = fwd
            rows.append(row)
    return pd.DataFrame(rows), mf.registry_from_records(factor_records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-csv")
    parser.add_argument("--registry-csv")
    parser.add_argument("--config-json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--synthetic-demo", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config_json)
    if args.synthetic_demo:
        panel, registry = make_synthetic_panel()
    else:
        if not args.panel_csv or not args.registry_csv:
            raise ValueError("Provide --panel-csv and --registry-csv, or use --synthetic-demo.")
        panel = pd.read_csv(args.panel_csv, encoding="utf-8-sig")
        registry = pd.read_csv(args.registry_csv, encoding="utf-8-sig")

    results = run_factor_factory(panel, registry, config)
    save_results(results, args.output_dir)
    print(results["meta"])
    print(results["performance"])
    print(f"saved={Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
