#!/usr/bin/env python
"""Walk-forward factor factory runner with turnover and transaction costs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import csv_io
import factor_factory_runner as runner
import multi_factor_research_framework as mf


@dataclass(frozen=True)
class WalkForwardConfig:
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
    periods_per_year: int = 252
    long_only: bool = True
    output_score_col: str = "multi_factor_score"
    train_periods: int = 120
    test_periods: int = 20
    step_periods: int = 20
    one_way_cost_bps: float = 20.0
    tradeable_col: str | None = "is_tradeable"
    amount_col: str | None = "amount"
    fund_size: float | None = None
    max_participation_rate: float = 0.05


def load_config(path: str | Path | None = None) -> WalkForwardConfig:
    if path is None:
        return WalkForwardConfig()
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if "control_cols" in raw:
        raw["control_cols"] = tuple(raw["control_cols"])
    return WalkForwardConfig(**raw)


def _factory_config(config: WalkForwardConfig) -> runner.FactoryConfig:
    fields = set(runner.FactoryConfig.__dataclass_fields__)
    values = {k: v for k, v in asdict(config).items() if k in fields}
    return runner.FactoryConfig(**values)


def _validation_config(config: WalkForwardConfig) -> mf.ValidationConfig:
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


def prepare_factor_panel(
    panel: pd.DataFrame,
    registry: pd.DataFrame,
    config: WalkForwardConfig,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Point-in-time safe cross-sectional factor preparation."""
    data = panel.copy()
    data[config.date_col] = pd.to_datetime(data[config.date_col])
    audit = mf.check_availability_dates(data, config.date_col, registry)
    data, alpha_cols = mf.apply_factor_directions(data, registry)
    if not alpha_cols:
        raise ValueError("No registered factor columns were found in panel.")
    data, win_cols = mf.winsorize_by_date(data, config.date_col, alpha_cols)
    data, z_cols = mf.zscore_by_date(data, config.date_col, win_cols)
    neutralize_controls = [col for col in config.control_cols if col in data.columns]
    data, model_factor_cols = mf.neutralize_by_date(
        data,
        config.date_col,
        z_cols,
        control_cols=neutralize_controls,
        industry_col=config.industry_col if config.industry_col and config.industry_col in data.columns else None,
        weight_col=config.weight_col if config.weight_col and config.weight_col in data.columns else None,
    )
    return data, model_factor_cols, audit


def split_boundaries(dates: pd.Series, train_periods: int, test_periods: int, step_periods: int) -> pd.DataFrame:
    unique_dates = pd.Index(sorted(pd.to_datetime(dates).dropna().unique()))
    rows = []
    split = 0
    train_end = train_periods
    while train_end + test_periods <= len(unique_dates):
        train = unique_dates[train_end - train_periods : train_end]
        test = unique_dates[train_end : train_end + test_periods]
        rows.append(
            {
                "split": split,
                "train_start": train[0],
                "train_end": train[-1],
                "test_start": test[0],
                "test_end": test[-1],
                "n_train_dates": len(train),
                "n_test_dates": len(test),
            }
        )
        split += 1
        train_end += step_periods
    return pd.DataFrame(rows)


def select_on_train(
    train: pd.DataFrame,
    registry: pd.DataFrame,
    model_factor_cols: list[str],
    config: WalkForwardConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ic = mf.factor_ic_long(train, config.date_col, model_factor_cols, config.forward_return_col, min_assets=config.min_assets)
    ic_stats = mf.ic_summary(ic)
    turnover = mf.top_quantile_turnover(
        train,
        config.date_col,
        config.asset_col,
        model_factor_cols,
        top_quantile=config.top_quantile,
        min_assets=config.min_assets,
    )
    corr = mf.average_factor_correlation(train, config.date_col, model_factor_cols, min_assets=config.min_assets)
    clusters = mf.correlation_clusters(corr, threshold=config.corr_threshold)
    quality = mf.factor_quality_table(ic_stats, turnover, registry=registry)
    selected = mf.select_factors(quality, clusters, config=_validation_config(config), max_per_family=config.max_per_family)
    return selected, quality, ic, turnover, clusters


def apply_selected_to_test(
    test: pd.DataFrame,
    selected: pd.DataFrame,
    config: WalkForwardConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    scored, family_cols = mf.combine_by_family(test, config.date_col, selected)
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
    weights = apply_execution_constraints(weights, scored, config)
    returns = mf.portfolio_forward_returns(weights, scored, config.date_col, config.asset_col, config.forward_return_col)
    return scored, weights, returns


def apply_execution_constraints(
    weights: pd.DataFrame,
    panel: pd.DataFrame,
    config: WalkForwardConfig,
) -> pd.DataFrame:
    """Apply tradeability and simple amount-capacity constraints to target weights."""
    if weights.empty:
        return weights
    out = weights.copy()
    extra_cols = [config.date_col, config.asset_col]
    if config.tradeable_col and config.tradeable_col in panel.columns:
        extra_cols.append(config.tradeable_col)
    if config.amount_col and config.amount_col in panel.columns:
        extra_cols.append(config.amount_col)
    extra = panel[extra_cols].drop_duplicates([config.date_col, config.asset_col])
    out["date"] = pd.to_datetime(out["date"])
    extra[config.date_col] = pd.to_datetime(extra[config.date_col])
    out = out.merge(extra, left_on=["date", "asset"], right_on=[config.date_col, config.asset_col], how="left")
    if config.tradeable_col and config.tradeable_col in out.columns:
        tradeable = csv_io.coerce_bool_series(out[config.tradeable_col], default=True).fillna(True).astype(bool)
        out.loc[~tradeable, "weight"] = 0.0
    has_capacity = bool(config.fund_size and config.amount_col and config.amount_col in out.columns)
    if config.fund_size and config.amount_col and config.amount_col in out.columns:
        cap = pd.to_numeric(out[config.amount_col], errors="coerce").clip(lower=0.0) * config.max_participation_rate / float(config.fund_size)
        out["_capacity_cap"] = cap.fillna(0.0)
    normalized = []
    for date, group in out.groupby("date", sort=True):
        g = group.copy()
        if config.long_only:
            raw = g["weight"].clip(lower=0.0)
            if has_capacity:
                caps = g["_capacity_cap"].clip(lower=0.0)
                target_total = min(1.0, float(caps.sum()))
                g["weight"] = _scale_weights_to_caps(raw, caps, target_total)
            else:
                total = raw.sum()
                g["weight"] = raw / total if total > mf.EPS else 0.0
        else:
            raw_abs = g["weight"].abs()
            sign = np.sign(g["weight"].astype(float))
            if has_capacity:
                caps = g["_capacity_cap"].clip(lower=0.0)
                target_total = min(1.0, float(caps.sum()))
                g["weight"] = sign * _scale_weights_to_caps(raw_abs, caps, target_total)
            else:
                total = raw_abs.sum()
                g["weight"] = g["weight"] / total if total > mf.EPS else 0.0
        normalized.append(g)
    clean = pd.concat(normalized, ignore_index=True) if normalized else out
    return clean[["date", "asset", "weight"]]


def _scale_weights_to_caps(values: pd.Series, caps: pd.Series, target_total: float) -> pd.Series:
    """Scale non-negative preferences without exceeding per-asset caps."""
    values = pd.to_numeric(values, errors="coerce").fillna(0.0).clip(lower=0.0)
    caps = pd.to_numeric(caps, errors="coerce").fillna(0.0).clip(lower=0.0)
    result = pd.Series(0.0, index=values.index, dtype=float)
    target_total = min(float(target_total), float(caps.sum()))
    if target_total <= mf.EPS or values.sum() <= mf.EPS or caps.sum() <= mf.EPS:
        return result

    remaining = caps > mf.EPS
    remaining_target = target_total
    for _ in range(len(values) + 1):
        if remaining_target <= mf.EPS or not remaining.any():
            break
        base_sum = float(values.loc[remaining].sum())
        if base_sum <= mf.EPS:
            break
        proposed = values.loc[remaining] * (remaining_target / base_sum)
        capped = proposed >= caps.loc[remaining] - mf.EPS
        if not capped.any():
            result.loc[remaining] = proposed
            break
        capped_index = capped[capped].index
        result.loc[capped_index] = caps.loc[capped_index]
        remaining_target -= float(caps.loc[capped_index].sum())
        remaining.loc[capped_index] = False
    return result.clip(upper=caps)


def portfolio_turnover(weights: pd.DataFrame) -> pd.DataFrame:
    """One-way turnover from date-asset weights."""
    if weights.empty:
        return pd.DataFrame(columns=["date", "turnover"])
    wide = weights.pivot_table(index="date", columns="asset", values="weight", aggfunc="sum").fillna(0.0).sort_index()
    turnover = wide.diff().abs().sum(axis=1)
    if not turnover.empty:
        turnover.iloc[0] = wide.iloc[0].abs().sum()
    return turnover.rename("turnover").reset_index()


def apply_transaction_costs(
    returns: pd.DataFrame,
    turnover: pd.DataFrame,
    one_way_cost_bps: float,
) -> pd.DataFrame:
    out = returns.copy()
    if out.empty:
        return pd.DataFrame(columns=["date", "gross_return", "turnover", "cost", "net_return"])
    out = out.rename(columns={"return": "gross_return"})
    out = out.merge(turnover, on="date", how="left")
    out["turnover"] = out["turnover"].fillna(0.0)
    out["cost"] = out["turnover"] * one_way_cost_bps / 10000.0
    out["net_return"] = out["gross_return"] - out["cost"]
    return out


def run_walk_forward(
    panel: pd.DataFrame,
    registry: pd.DataFrame,
    config: WalkForwardConfig = WalkForwardConfig(),
) -> dict[str, Any]:
    data, model_factor_cols, audit = prepare_factor_panel(panel, registry, config)
    splits = split_boundaries(data[config.date_col], config.train_periods, config.test_periods, config.step_periods)
    if splits.empty:
        raise ValueError("Not enough dates for requested walk-forward windows.")

    selected_rows = []
    quality_rows = []
    weight_rows = []
    return_rows = []
    split_rows = []
    scored_frames = []

    for split in splits.itertuples(index=False):
        train_mask = (data[config.date_col] >= split.train_start) & (data[config.date_col] <= split.train_end)
        test_mask = (data[config.date_col] >= split.test_start) & (data[config.date_col] <= split.test_end)
        train = data.loc[train_mask].copy()
        test = data.loc[test_mask].copy()
        selected, quality, _, _, _ = select_on_train(train, registry, model_factor_cols, config)
        if selected.empty:
            split_rows.append({**split._asdict(), "selected_factors": 0, "families": 0, "status": "no_selected_factors"})
            continue
        scored, weights, gross_returns = apply_selected_to_test(test, selected, config)
        turnover = portfolio_turnover(weights)
        return_table = gross_returns.rename("return").reset_index().rename(columns={"index": "date"})
        net = apply_transaction_costs(return_table, turnover, config.one_way_cost_bps)

        selected_rows.append(selected.assign(split=split.split))
        quality_rows.append(quality.assign(split=split.split))
        weight_rows.append(weights.assign(split=split.split))
        return_rows.append(net.assign(split=split.split))
        scored_frames.append(scored.assign(split=split.split))
        split_rows.append(
            {
                **split._asdict(),
                "selected_factors": int(selected.shape[0]),
                "families": int(selected["family"].nunique()) if "family" in selected.columns else 0,
                "status": "ok",
            }
        )

    selected_all = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()
    quality_all = pd.concat(quality_rows, ignore_index=True) if quality_rows else pd.DataFrame()
    weights_all = pd.concat(weight_rows, ignore_index=True) if weight_rows else pd.DataFrame()
    returns_all = pd.concat(return_rows, ignore_index=True) if return_rows else pd.DataFrame()
    scored_all = pd.concat(scored_frames, ignore_index=True) if scored_frames else pd.DataFrame()
    split_summary = pd.DataFrame(split_rows)

    perf_gross = mf.performance_summary(returns_all.set_index("date")["gross_return"], config.periods_per_year) if not returns_all.empty else pd.Series(dtype=float)
    perf_net = mf.performance_summary(returns_all.set_index("date")["net_return"], config.periods_per_year) if not returns_all.empty else pd.Series(dtype=float)
    performance = pd.concat(
        [
            perf_gross.rename("gross"),
            perf_net.rename("net"),
        ],
        axis=1,
    ).reset_index(names="metric")

    meta = pd.DataFrame(
        [
            {
                "n_rows": int(panel.shape[0]),
                "n_assets": int(panel[config.asset_col].nunique()),
                "n_dates": int(pd.to_datetime(panel[config.date_col]).nunique()),
                "registered_factors": int(registry.shape[0]),
                "available_factors": int(len(model_factor_cols)),
                "splits": int(splits.shape[0]),
                "ok_splits": int((split_summary["status"] == "ok").sum()) if not split_summary.empty else 0,
                "one_way_cost_bps": float(config.one_way_cost_bps),
                "fund_size": float(config.fund_size) if config.fund_size else np.nan,
                "max_participation_rate": float(config.max_participation_rate),
                "tradeable_col": config.tradeable_col or "",
                "amount_col": config.amount_col or "",
            }
        ]
    )
    return {
        "meta": meta,
        "availability_audit": audit,
        "splits": splits,
        "split_summary": split_summary,
        "selected": selected_all,
        "quality": quality_all,
        "weights": weights_all,
        "walk_forward_returns": returns_all,
        "walk_forward_performance": performance,
        "scored_panel": scored_all,
    }


def save_results(results: dict[str, Any], output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tables = []
    for name, value in results.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(out / f"{name}.csv", index=False, encoding="utf-8-sig")
            tables.append(f"{name}.csv")
    (out / "run_manifest.json").write_text(
        json.dumps({"created_by": "factor_factory_walk_forward.py", "tables": sorted(tables)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


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
        panel, registry = runner.make_synthetic_panel(n_dates=180, n_assets=140)
    else:
        if not args.panel_csv or not args.registry_csv:
            raise ValueError("Provide --panel-csv and --registry-csv, or use --synthetic-demo.")
        panel = csv_io.read_csv_robust(args.panel_csv)
        registry = csv_io.read_csv_robust(args.registry_csv)
    results = run_walk_forward(panel, registry, config)
    save_results(results, args.output_dir)
    print(results["meta"])
    print(results["walk_forward_performance"])
    print(f"saved={Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
