#!/usr/bin/env python
"""Reusable signal validation template.

This script validates whether a discrete signal is followed by positive or
negative forward returns. It is signal validation, not a full portfolio
backtest. Use it before building a complete strategy.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_HORIZONS = (1, 5, 10, 20)


def compute_kdj_signal(
    df: pd.DataFrame,
    low_col: str,
    high_col: str,
    close_col: str,
    window: int = 40,
    smooth_span: int = 2,
    signal_col: str = "signal",
) -> pd.DataFrame:
    """Add KDJ-style K/D indicators and cross signals.

    signal = 1: K crosses above D.
    signal = 0: K crosses below D.
    """
    out = df.copy()
    low_n = out[low_col].rolling(window).min()
    high_n = out[high_col].rolling(window).max()
    denominator = high_n - low_n
    out["LOW_N"] = low_n
    out["HIGH_N"] = high_n
    out["RSV"] = (out[close_col] - low_n) / denominator.replace(0, pd.NA) * 100
    out["K"] = out["RSV"].ewm(span=smooth_span, adjust=False).mean()
    out["D"] = out["K"].ewm(span=smooth_span, adjust=False).mean()
    out[signal_col] = pd.NA
    out.loc[(out["K"].shift(1) <= out["D"].shift(1)) & (out["K"] > out["D"]), signal_col] = 1
    out.loc[(out["K"].shift(1) >= out["D"].shift(1)) & (out["K"] < out["D"]), signal_col] = 0
    return out


def add_forward_returns(
    df: pd.DataFrame,
    price_col: str,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    """Add forward returns using future prices as labels only."""
    out = df.copy()
    for horizon in horizons:
        out[f"fwd_ret_{horizon}d"] = out[price_col].shift(-horizon) / out[price_col] - 1
        out[f"fwd_up_{horizon}d"] = out[f"fwd_ret_{horizon}d"] > 0
    return out


def summarize_signal_forward_returns(
    df: pd.DataFrame,
    signal_col: str = "signal",
    horizons: Iterable[int] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    """Summarize forward-return distribution by signal value."""
    rows = []
    clean = df.dropna(subset=[signal_col]).copy()
    for signal_value, group in clean.groupby(signal_col, dropna=True):
        for horizon in horizons:
            ret_col = f"fwd_ret_{horizon}d"
            series = group[ret_col].dropna()
            if series.empty:
                continue
            rows.append(
                {
                    "signal": signal_value,
                    "horizon": horizon,
                    "count": int(series.shape[0]),
                    "mean": float(series.mean()),
                    "median": float(series.median()),
                    "std": float(series.std(ddof=1)) if series.shape[0] > 1 else 0.0,
                    "win_rate": float((series > 0).mean()),
                    "p25": float(series.quantile(0.25)),
                    "p75": float(series.quantile(0.75)),
                }
            )
    return pd.DataFrame(rows)


def read_one_csv(path: Path, encoding: str) -> pd.DataFrame:
    return pd.read_csv(path, encoding=encoding)


def load_inputs(csv_path: str | None, folder: str | None, encoding: str) -> pd.DataFrame:
    if csv_path:
        return read_one_csv(Path(csv_path), encoding)
    if not folder:
        raise ValueError("Provide --csv or --folder.")
    frames = [read_one_csv(path, encoding) for path in Path(folder).glob("*.csv")]
    if not frames:
        raise ValueError(f"No CSV files found in {folder}")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", help="Single CSV file.")
    parser.add_argument("--folder", help="Folder of CSV files.")
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument("--date-col")
    parser.add_argument("--close-col", required=True)
    parser.add_argument("--high-col")
    parser.add_argument("--low-col")
    parser.add_argument("--signal-col", default="signal")
    parser.add_argument("--signal-mode", choices=["existing", "kdj"], default="existing")
    parser.add_argument("--horizons", default="1,5,10,20")
    parser.add_argument("--output", default="signal_validation_summary.csv")
    args = parser.parse_args()

    horizons = tuple(int(x.strip()) for x in args.horizons.split(",") if x.strip())
    df = load_inputs(args.csv, args.folder, args.encoding)

    if args.date_col and args.date_col in df.columns:
        df[args.date_col] = pd.to_datetime(df[args.date_col])
        df = df.sort_values(args.date_col)

    if args.signal_mode == "kdj":
        if not args.high_col or not args.low_col:
            raise ValueError("--high-col and --low-col are required for KDJ mode.")
        df = compute_kdj_signal(
            df,
            low_col=args.low_col,
            high_col=args.high_col,
            close_col=args.close_col,
            signal_col=args.signal_col,
        )
    elif args.signal_col not in df.columns:
        raise ValueError(f"Missing signal column: {args.signal_col}")

    df = add_forward_returns(df, price_col=args.close_col, horizons=horizons)
    summary = summarize_signal_forward_returns(df, signal_col=args.signal_col, horizons=horizons)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False, encoding="utf-8-sig")
    print(summary)
    print(f"saved={output.resolve()}")


if __name__ == "__main__":
    main()
