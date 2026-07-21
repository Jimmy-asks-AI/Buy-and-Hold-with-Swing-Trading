#!/usr/bin/env python
"""Factor crowding monitor.

The monitor estimates factor crowding from valuation spread, pairwise
correlation, long-term factor return, and volatility ratio. It is a risk and
factor-timing research tool, not a standalone trading signal.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


EPS = 1e-12


def safe_log_ratio(numerator: float, denominator: float) -> float:
    if numerator <= 0 or denominator <= 0 or pd.isna(numerator) or pd.isna(denominator):
        return float("nan")
    return float(math.log(numerator / denominator))


def ts_zscore(series: pd.Series, window: int | None = None, min_periods: int = 12) -> pd.Series:
    """Time-series z-score, optionally rolling."""
    s = series.astype(float)
    if window is None:
        std = s.std(ddof=1)
        if pd.isna(std) or std <= EPS:
            return pd.Series(0.0, index=s.index)
        return (s - s.mean()) / std
    mean = s.rolling(window, min_periods=min_periods).mean()
    std = s.rolling(window, min_periods=min_periods).std(ddof=1)
    return (s - mean) / std.replace(0.0, np.nan)


def tail_assets(
    factor_snapshot: pd.DataFrame,
    asset_col: str,
    factor_col: str,
    quantile: float = 0.10,
    long_high: bool = True,
) -> tuple[pd.Index, pd.Index]:
    """Return long and short asset lists from one cross-sectional factor snapshot."""
    if not 0 < quantile < 0.5:
        raise ValueError("quantile must be in (0, 0.5)")
    clean = factor_snapshot[[asset_col, factor_col]].dropna().copy()
    if clean.empty:
        return pd.Index([]), pd.Index([])
    n = max(1, int(math.floor(clean.shape[0] * quantile)))
    ordered = clean.sort_values(factor_col, ascending=True)
    if long_high:
        long_assets = ordered.tail(n)[asset_col]
        short_assets = ordered.head(n)[asset_col]
    else:
        long_assets = ordered.head(n)[asset_col]
        short_assets = ordered.tail(n)[asset_col]
    return pd.Index(long_assets.astype(str)), pd.Index(short_assets.astype(str))


def portfolio_return(returns_wide: pd.DataFrame, assets: pd.Index | list[str]) -> pd.Series:
    cols = [asset for asset in assets if asset in returns_wide.columns]
    if not cols:
        return pd.Series(dtype="float64", index=returns_wide.index)
    return returns_wide[cols].astype(float).mean(axis=1)


def average_corr_to_portfolio(returns_wide: pd.DataFrame, assets: pd.Index | list[str]) -> float:
    cols = [asset for asset in assets if asset in returns_wide.columns]
    if len(cols) < 2:
        return float("nan")
    sub = returns_wide[cols].astype(float).dropna(how="all")
    port = sub.mean(axis=1)
    corr = sub.corrwith(port).dropna()
    return float(corr.mean()) if not corr.empty else float("nan")


def pairwise_correlation_indicator(
    returns_wide: pd.DataFrame,
    long_assets: pd.Index | list[str],
    short_assets: pd.Index | list[str] | None = None,
    mode: str = "long_minus_short",
) -> float:
    """Compute a pairwise-correlation crowding indicator.

    Modes:
    - `long_minus_short`: mean corr(long) - mean corr(short)
    - `long_plus_short`: mean corr(long) + mean corr(short)
    - `long_only`: mean corr(long)
    """
    long_corr = average_corr_to_portfolio(returns_wide, long_assets)
    if mode == "long_only":
        return long_corr
    if short_assets is None:
        raise ValueError("short_assets required unless mode is long_only")
    short_corr = average_corr_to_portfolio(returns_wide, short_assets)
    if mode == "long_minus_short":
        return float(long_corr - short_corr)
    if mode == "long_plus_short":
        return float(long_corr + short_corr)
    raise ValueError("mode must be long_minus_short, long_plus_short, or long_only")


def valuation_spread_snapshot(
    factor_snapshot: pd.DataFrame,
    asset_col: str,
    factor_col: str,
    valuation_col: str,
    quantile: float = 0.10,
    long_high: bool = True,
    agg: str = "median",
) -> float:
    long_assets, short_assets = tail_assets(factor_snapshot, asset_col, factor_col, quantile, long_high)
    data = factor_snapshot.set_index(asset_col)
    data.index = data.index.astype(str)
    long_val = data.reindex(long_assets)[valuation_col].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    short_val = data.reindex(short_assets)[valuation_col].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    long_val = long_val[long_val > 0]
    short_val = short_val[short_val > 0]
    if long_val.empty or short_val.empty:
        return float("nan")
    if agg == "mean":
        return safe_log_ratio(float(long_val.mean()), float(short_val.mean()))
    if agg == "median":
        return safe_log_ratio(float(long_val.median()), float(short_val.median()))
    raise ValueError("agg must be mean or median")


def long_term_factor_return(
    returns_wide: pd.DataFrame,
    long_assets: pd.Index | list[str],
    short_assets: pd.Index | list[str],
) -> float:
    long_ret = portfolio_return(returns_wide, long_assets)
    short_ret = portfolio_return(returns_wide, short_assets)
    factor_ret = (long_ret - short_ret).dropna()
    if factor_ret.empty:
        return float("nan")
    return float((1.0 + factor_ret).prod() - 1.0)


def realized_vol(series: pd.Series, annualize: bool = True, periods_per_year: int = 252) -> float:
    clean = series.dropna().astype(float)
    if clean.shape[0] < 2:
        return float("nan")
    vol = float(clean.std(ddof=1))
    if annualize:
        vol *= math.sqrt(periods_per_year)
    return vol


def volatility_ratio_indicator(
    returns_wide: pd.DataFrame,
    long_assets: pd.Index | list[str],
    short_assets: pd.Index | list[str] | None = None,
    market_return: pd.Series | None = None,
    mode: str = "long_short",
    annualize: bool = True,
) -> float:
    """Compute factor-volatility ratio.

    Modes:
    - `long_short`: vol(long portfolio) / vol(short portfolio)
    - `long_market`: vol(long portfolio) / vol(market)
    - `ls_market`: vol(long-short factor) / vol(market)
    """
    long_ret = portfolio_return(returns_wide, long_assets)
    if mode == "long_short":
        if short_assets is None:
            raise ValueError("short_assets required for long_short mode")
        denominator = realized_vol(portfolio_return(returns_wide, short_assets), annualize=annualize)
        numerator = realized_vol(long_ret, annualize=annualize)
    elif mode == "long_market":
        if market_return is None:
            raise ValueError("market_return required for long_market mode")
        denominator = realized_vol(market_return.reindex(returns_wide.index), annualize=annualize)
        numerator = realized_vol(long_ret, annualize=annualize)
    elif mode == "ls_market":
        if short_assets is None or market_return is None:
            raise ValueError("short_assets and market_return required for ls_market mode")
        factor_ret = long_ret - portfolio_return(returns_wide, short_assets)
        denominator = realized_vol(market_return.reindex(returns_wide.index), annualize=annualize)
        numerator = realized_vol(factor_ret, annualize=annualize)
    else:
        raise ValueError("mode must be long_short, long_market, or ls_market")
    if pd.isna(numerator) or pd.isna(denominator) or denominator <= EPS:
        return float("nan")
    return float(numerator / denominator)


def absorption_ratio(returns_wide: pd.DataFrame, n_components: int = 1) -> float:
    """Share of covariance explained by the top principal components."""
    clean = returns_wide.astype(float).dropna(axis=1, how="all").dropna(axis=0, how="any")
    if clean.shape[0] < 2 or clean.shape[1] < 2:
        return float("nan")
    cov = np.cov(clean.to_numpy(), rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(np.maximum(eigvals, 0.0))[::-1]
    total = eigvals.sum()
    if total <= EPS:
        return float("nan")
    n = min(max(1, n_components), eigvals.shape[0])
    return float(eigvals[:n].sum() / total)


def pca_asset_centrality(returns_wide: pd.DataFrame, n_components: int = 1) -> pd.Series:
    """Estimate each column's contribution to the dominant covariance components.

    This is a practical PCA centrality proxy. Use it consistently across factors
    and validate against future returns or drawdowns before promotion.
    """
    clean = returns_wide.astype(float).dropna(axis=1, how="all").dropna(axis=0, how="any")
    if clean.shape[0] < 2 or clean.shape[1] < 2:
        return pd.Series(dtype="float64")
    cov = np.cov(clean.to_numpy(), rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    eigvecs = eigvecs[:, order]
    total = eigvals.sum()
    if total <= EPS:
        return pd.Series(dtype="float64")
    n = min(max(1, n_components), eigvals.shape[0])
    component_weights = eigvals[:n] / eigvals[:n].sum()
    centrality = (eigvecs[:, :n] ** 2) @ component_weights
    return pd.Series(centrality, index=clean.columns, name="pca_asset_centrality")


def institutional_holding_metrics(
    factor_snapshot: pd.DataFrame,
    assets: pd.Index | list[str],
    asset_col: str,
    holding_amount_col: str,
    market_cap_col: str | None = None,
    total_holding_amount: float | None = None,
) -> dict[str, float]:
    """Compute institution holding crowding proxies for selected assets."""
    data = factor_snapshot.set_index(asset_col)
    data.index = data.index.astype(str)
    selected = data.reindex(pd.Index(assets).astype(str))
    holding_amount = selected[holding_amount_col].astype(float).clip(lower=0.0)
    all_holding = data[holding_amount_col].astype(float).clip(lower=0.0)
    denominator = float(total_holding_amount) if total_holding_amount else float(all_holding.sum())
    metrics = {
        "institution_holding_amount_ratio": float(holding_amount.sum() / denominator)
        if denominator > EPS
        else float("nan")
    }
    if market_cap_col:
        market_cap = selected[market_cap_col].astype(float).clip(lower=0.0)
        market_cap_sum = float(market_cap.sum())
        metrics["institution_holding_market_cap_ratio"] = (
            float(holding_amount.sum() / market_cap_sum) if market_cap_sum > EPS else float("nan")
        )
    return metrics


def crowding_snapshot(
    factor_snapshot: pd.DataFrame,
    returns_wide: pd.DataFrame,
    asset_col: str,
    factor_col: str,
    valuation_col: str | None = None,
    market_return: pd.Series | None = None,
    quantile: float = 0.10,
    long_high: bool = True,
    pairwise_mode: str = "long_minus_short",
    volatility_mode: str = "long_short",
    include_asset_centrality: bool = False,
    pca_components: int = 1,
    holding_amount_col: str | None = None,
    market_cap_col: str | None = None,
    total_holding_amount: float | None = None,
) -> pd.DataFrame:
    long_assets, short_assets = tail_assets(factor_snapshot, asset_col, factor_col, quantile, long_high)
    metrics: dict[str, float | int | str] = {
        "factor": factor_col,
        "long_count": int(len(long_assets)),
        "short_count": int(len(short_assets)),
        "pairwise_correlation": pairwise_correlation_indicator(
            returns_wide,
            long_assets,
            short_assets,
            mode=pairwise_mode,
        ),
        "long_term_factor_return": long_term_factor_return(returns_wide, long_assets, short_assets),
        "volatility_ratio": volatility_ratio_indicator(
            returns_wide,
            long_assets,
            short_assets,
            market_return=market_return,
            mode=volatility_mode,
        ),
    }
    if valuation_col:
        metrics["valuation_spread"] = valuation_spread_snapshot(
            factor_snapshot,
            asset_col=asset_col,
            factor_col=factor_col,
            valuation_col=valuation_col,
            quantile=quantile,
            long_high=long_high,
        )
    if include_asset_centrality:
        long_cols = [asset for asset in long_assets if asset in returns_wide.columns]
        metrics["long_absorption_ratio"] = absorption_ratio(returns_wide[long_cols], n_components=pca_components)
    if holding_amount_col:
        metrics.update(
            institutional_holding_metrics(
                factor_snapshot=factor_snapshot,
                assets=long_assets,
                asset_col=asset_col,
                holding_amount_col=holding_amount_col,
                market_cap_col=market_cap_col,
                total_holding_amount=total_holding_amount,
            )
        )
    return pd.DataFrame([metrics])


def composite_crowding(indicators: pd.DataFrame, direction: dict[str, float] | None = None) -> pd.Series:
    """Create an equal-weight composite crowding score from time series indicators."""
    direction = direction or {}
    parts = []
    for col in indicators.columns:
        s = indicators[col].astype(float)
        sign = float(direction.get(col, 1.0))
        parts.append(sign * ts_zscore(s))
    if not parts:
        return pd.Series(dtype="float64", index=indicators.index)
    return pd.concat(parts, axis=1).mean(axis=1).rename("composite_crowding")


def forward_cumulative_return(returns: pd.Series, horizon: int) -> pd.Series:
    future = returns.shift(-1)
    return (1.0 + future).rolling(horizon).apply(np.prod, raw=True).shift(-(horizon - 1)) - 1.0


def forward_realized_vol(returns: pd.Series, horizon: int, periods_per_year: int = 252) -> pd.Series:
    future = returns.shift(-1)
    return future.rolling(horizon).std(ddof=1).shift(-(horizon - 1)) * math.sqrt(periods_per_year)


def crowding_validation_table(
    crowding: pd.Series,
    factor_returns: pd.Series,
    horizons: list[int],
) -> pd.DataFrame:
    """Validate crowding against future factor return and future volatility."""
    rows = []
    aligned = pd.concat([crowding.rename("crowding"), factor_returns.rename("factor_return")], axis=1)
    for horizon in horizons:
        fwd_ret = forward_cumulative_return(aligned["factor_return"], horizon)
        fwd_vol = forward_realized_vol(aligned["factor_return"], horizon)
        rows.append(
            {
                "horizon": horizon,
                "corr_future_return": float(aligned["crowding"].corr(fwd_ret)),
                "corr_future_vol": float(aligned["crowding"].corr(fwd_vol)),
                "n_return": int(pd.concat([aligned["crowding"], fwd_ret], axis=1).dropna().shape[0]),
                "n_vol": int(pd.concat([aligned["crowding"], fwd_vol], axis=1).dropna().shape[0]),
            }
        )
    return pd.DataFrame(rows)


def load_returns_wide(path: str, date_col: str, encoding: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding=encoding)
    df[date_col] = pd.to_datetime(df[date_col])
    return df.set_index(date_col).sort_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor-csv", required=True, help="One factor snapshot with asset and factor columns.")
    parser.add_argument("--returns-csv", required=True, help="Wide return matrix: date column plus asset columns.")
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument("--date-col", default="date")
    parser.add_argument("--asset-col", required=True)
    parser.add_argument("--factor-col", required=True)
    parser.add_argument("--valuation-col")
    parser.add_argument("--holding-amount-col")
    parser.add_argument("--market-cap-col")
    parser.add_argument("--total-holding-amount", type=float)
    parser.add_argument("--quantile", type=float, default=0.10)
    parser.add_argument("--long-low", action="store_true", help="Use low factor values as long leg.")
    parser.add_argument("--pairwise-mode", default="long_minus_short", choices=["long_minus_short", "long_plus_short", "long_only"])
    parser.add_argument("--volatility-mode", default="long_short", choices=["long_short", "long_market", "ls_market"])
    parser.add_argument("--include-asset-centrality", action="store_true")
    parser.add_argument("--pca-components", type=int, default=1)
    parser.add_argument("--market-return-csv")
    parser.add_argument("--market-return-col", default="market_return")
    parser.add_argument("--output-dir", default="factor_crowding_output")
    args = parser.parse_args()

    factor_snapshot = pd.read_csv(args.factor_csv, encoding=args.encoding)
    returns_wide = load_returns_wide(args.returns_csv, args.date_col, args.encoding)
    market_return = None
    if args.market_return_csv:
        market_df = pd.read_csv(args.market_return_csv, encoding=args.encoding)
        market_df[args.date_col] = pd.to_datetime(market_df[args.date_col])
        market_return = market_df.set_index(args.date_col)[args.market_return_col]

    result = crowding_snapshot(
        factor_snapshot=factor_snapshot,
        returns_wide=returns_wide,
        asset_col=args.asset_col,
        factor_col=args.factor_col,
        valuation_col=args.valuation_col,
        market_return=market_return,
        quantile=args.quantile,
        long_high=not args.long_low,
        pairwise_mode=args.pairwise_mode,
        volatility_mode=args.volatility_mode,
        include_asset_centrality=args.include_asset_centrality,
        pca_components=args.pca_components,
        holding_amount_col=args.holding_amount_col,
        market_cap_col=args.market_cap_col,
        total_holding_amount=args.total_holding_amount,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "crowding_snapshot.csv", index=False, encoding="utf-8-sig")
    print(result)
    print(f"saved={output_dir.resolve()}")


if __name__ == "__main__":
    main()
