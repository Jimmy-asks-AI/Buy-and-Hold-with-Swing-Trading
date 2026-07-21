#!/usr/bin/env python
"""Backtest overfitting diagnostics with CSCV/PBO.

The core input is a T x N return matrix: rows are time periods, columns are
strategy variants or parameter sets. CSCV repeatedly selects the best in-sample
variant and checks where that selected variant ranks out of sample.
"""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd


EPS = 1e-12


def _as_return_matrix(returns: pd.DataFrame | pd.Series) -> pd.DataFrame:
    if isinstance(returns, pd.Series):
        matrix = returns.to_frame("strategy")
    else:
        matrix = returns.copy()
    matrix = matrix.astype(float).replace([np.inf, -np.inf], np.nan).dropna(how="all")
    if matrix.shape[1] < 2:
        raise ValueError("CSCV/PBO needs at least two strategy variants.")
    return matrix


def contiguous_blocks(n_obs: int, n_blocks: int) -> list[np.ndarray]:
    """Split observations into contiguous, near-equal blocks."""
    if n_blocks < 2 or n_blocks % 2 != 0:
        raise ValueError("n_blocks must be an even integer >= 2.")
    if n_obs < n_blocks:
        raise ValueError("n_obs must be >= n_blocks.")
    return [np.asarray(block, dtype=int) for block in np.array_split(np.arange(n_obs), n_blocks)]


def performance_metric(
    returns: pd.Series,
    metric: str = "sharpe",
    periods: int = 252,
    risk_free_per_period: float = 0.0,
    min_periods: int = 20,
) -> float:
    """Compute a scalar performance metric for one return series."""
    clean = returns.dropna().astype(float)
    if clean.shape[0] < min_periods:
        return float("nan")
    metric = metric.lower()
    excess = clean - risk_free_per_period
    if metric in {"mean", "avg_return"}:
        return float(clean.mean())
    if metric in {"cumulative", "total_return"}:
        return float((1.0 + clean).prod() - 1.0)
    if metric in {"annual_return", "ann_return"}:
        total = float((1.0 + clean).prod())
        if total <= 0:
            return float("nan")
        return float(total ** (periods / clean.shape[0]) - 1.0)
    if metric == "sharpe":
        vol = excess.std(ddof=1)
        if pd.isna(vol) or vol <= EPS:
            return float("nan")
        return float(excess.mean() / vol * math.sqrt(periods))
    if metric == "sortino":
        downside = excess[excess < 0].std(ddof=1)
        if pd.isna(downside) or downside <= EPS:
            return float("nan")
        return float(excess.mean() / downside * math.sqrt(periods))
    if metric == "calmar":
        total = float((1.0 + clean).prod())
        if total <= 0:
            return float("nan")
        ann_ret = total ** (periods / clean.shape[0]) - 1.0
        nav = (1.0 + clean).cumprod()
        drawdown = nav / nav.cummax() - 1.0
        max_dd = abs(float(drawdown.min()))
        if max_dd <= EPS:
            return float("nan")
        return float(ann_ret / max_dd)
    raise ValueError("metric must be one of: mean, cumulative, annual_return, sharpe, sortino, calmar.")


def strategy_metric_table(
    returns: pd.DataFrame,
    metric: str = "sharpe",
    periods: int = 252,
    risk_free_per_period: float = 0.0,
    min_periods: int = 20,
) -> pd.Series:
    """Compute one metric for every strategy column."""
    matrix = _as_return_matrix(returns)
    scores = {
        col: performance_metric(
            matrix[col],
            metric=metric,
            periods=periods,
            risk_free_per_period=risk_free_per_period,
            min_periods=min_periods,
        )
        for col in matrix.columns
    }
    return pd.Series(scores, dtype="float64", name=metric)


def cscv_pbo(
    returns: pd.DataFrame | pd.Series,
    n_blocks: int = 16,
    metric: str = "sharpe",
    periods: int = 252,
    risk_free_per_period: float = 0.0,
    min_periods: int = 20,
    higher_is_better: bool = True,
    max_combinations: int | None = None,
    random_state: int = 42,
) -> dict[str, pd.DataFrame]:
    """Estimate Probability of Backtest Overfitting with CSCV.

    Ranking convention in this module is explicit: test_rank=1 is the best
    out-of-sample rank. A split is marked as overfit when the in-sample winner
    lands in the worse half out of sample, i.e. relative_rank_best1 > 0.5.
    """
    matrix = _as_return_matrix(returns)
    blocks = contiguous_blocks(matrix.shape[0], n_blocks)
    all_combos = list(itertools.combinations(range(n_blocks), n_blocks // 2))
    if max_combinations is not None and len(all_combos) > max_combinations:
        rng = np.random.default_rng(random_state)
        picked = rng.choice(len(all_combos), size=max_combinations, replace=False)
        combos = [all_combos[int(i)] for i in np.sort(picked)]
    else:
        combos = all_combos

    rows: list[dict[str, object]] = []
    for combo_id, train_blocks in enumerate(combos):
        train_set = set(train_blocks)
        test_blocks = [idx for idx in range(n_blocks) if idx not in train_set]
        train_idx = np.concatenate([blocks[idx] for idx in range(n_blocks) if idx in train_set])
        test_idx = np.concatenate([blocks[idx] for idx in test_blocks])

        train_scores = strategy_metric_table(
            matrix.iloc[train_idx],
            metric=metric,
            periods=periods,
            risk_free_per_period=risk_free_per_period,
            min_periods=min_periods,
        ).dropna()
        test_scores = strategy_metric_table(
            matrix.iloc[test_idx],
            metric=metric,
            periods=periods,
            risk_free_per_period=risk_free_per_period,
            min_periods=min_periods,
        ).dropna()
        common = train_scores.index.intersection(test_scores.index)
        if len(common) < 2:
            continue
        train_scores = train_scores.loc[common]
        test_scores = test_scores.loc[common]

        if higher_is_better:
            selected = str(train_scores.idxmax())
            train_rank = train_scores.rank(ascending=False, method="min")
            test_rank = test_scores.rank(ascending=False, method="min")
        else:
            selected = str(train_scores.idxmin())
            train_rank = train_scores.rank(ascending=True, method="min")
            test_rank = test_scores.rank(ascending=True, method="min")

        rank_best1 = float(test_rank.loc[selected])
        relative_rank = rank_best1 / (len(common) + 1.0)
        relative_rank = min(max(relative_rank, EPS), 1.0 - EPS)
        rows.append(
            {
                "combo_id": combo_id,
                "train_blocks": ",".join(map(str, train_blocks)),
                "test_blocks": ",".join(map(str, test_blocks)),
                "selected_strategy": selected,
                "n_strategies": int(len(common)),
                "train_rank_best1": float(train_rank.loc[selected]),
                "test_rank_best1": rank_best1,
                "relative_rank_best1": relative_rank,
                "lambda_logit_best1": float(math.log(relative_rank / (1.0 - relative_rank))),
                "overfit_worse_half": bool(relative_rank > 0.5),
                "train_metric": float(train_scores.loc[selected]),
                "test_metric": float(test_scores.loc[selected]),
                "test_metric_median": float(test_scores.median()),
                "test_metric_best": float(test_scores.max() if higher_is_better else test_scores.min()),
            }
        )

    splits = pd.DataFrame(rows)
    return {
        "summary": pbo_summary(splits),
        "splits": splits,
        "selection_frequency": selection_frequency(splits),
    }


def pbo_summary(splits: pd.DataFrame) -> pd.DataFrame:
    """Summarize CSCV split diagnostics."""
    if splits.empty:
        return pd.DataFrame(
            [
                {
                    "pbo": np.nan,
                    "n_splits": 0,
                    "median_relative_rank_best1": np.nan,
                    "median_lambda_logit_best1": np.nan,
                    "mean_train_metric": np.nan,
                    "mean_test_metric": np.nan,
                    "selection_concentration_top1": np.nan,
                }
            ]
        )
    frequency = splits["selected_strategy"].value_counts(normalize=True)
    return pd.DataFrame(
        [
            {
                "pbo": float(splits["overfit_worse_half"].mean()),
                "n_splits": int(splits.shape[0]),
                "median_relative_rank_best1": float(splits["relative_rank_best1"].median()),
                "median_lambda_logit_best1": float(splits["lambda_logit_best1"].median()),
                "mean_train_metric": float(splits["train_metric"].mean()),
                "mean_test_metric": float(splits["test_metric"].mean()),
                "selection_concentration_top1": float(frequency.iloc[0]),
            }
        ]
    )


def selection_frequency(splits: pd.DataFrame) -> pd.DataFrame:
    """Count which strategy variants are repeatedly selected in sample."""
    if splits.empty:
        return pd.DataFrame(columns=["selected_strategy", "count", "frequency"])
    counts = splits["selected_strategy"].value_counts().rename_axis("selected_strategy").reset_index(name="count")
    counts["frequency"] = counts["count"] / splits.shape[0]
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument("--date-col")
    parser.add_argument("--strategy-cols", nargs="*")
    parser.add_argument("--n-blocks", type=int, default=16)
    parser.add_argument("--metric", default="sharpe")
    parser.add_argument("--periods", type=int, default=252)
    parser.add_argument("--min-periods", type=int, default=20)
    parser.add_argument("--max-combinations", type=int)
    parser.add_argument("--output-dir", default="pbo_output")
    args = parser.parse_args()

    df = pd.read_csv(args.csv, encoding=args.encoding)
    if args.date_col:
        df = df.sort_values(args.date_col)
        df[args.date_col] = pd.to_datetime(df[args.date_col])
    strategy_cols = args.strategy_cols or [col for col in df.columns if col != args.date_col]
    result = cscv_pbo(
        df[strategy_cols],
        n_blocks=args.n_blocks,
        metric=args.metric,
        periods=args.periods,
        min_periods=args.min_periods,
        max_combinations=args.max_combinations,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in result.items():
        table.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    print(result["summary"])
    print(f"saved={output_dir.resolve()}")


if __name__ == "__main__":
    main()
