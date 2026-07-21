#!/usr/bin/env python
"""Cross-sectional factor neutralization and orthogonalization.

Use this before multi-factor combination when factor exposures are correlated.
The script keeps original factors and appends orthogonalized factor columns.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def zscore(s: pd.Series) -> pd.Series:
    std = s.std(ddof=1)
    if pd.isna(std) or std == 0:
        return s * 0
    return (s - s.mean()) / std


def winsorize(s: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    return s.clip(lower=s.quantile(lower_q), upper=s.quantile(upper_q))


def regress_residual(y: pd.Series, x: pd.DataFrame) -> pd.Series:
    clean = pd.concat([y.rename("y"), x], axis=1).dropna()
    residual = pd.Series(index=y.index, dtype="float64")
    if clean.shape[0] <= x.shape[1] + 1:
        return residual
    yv = clean["y"].to_numpy(dtype=float)
    xv = clean.drop(columns=["y"]).to_numpy(dtype=float)
    xv = np.column_stack([np.ones(xv.shape[0]), xv])
    beta, *_ = np.linalg.lstsq(xv, yv, rcond=None)
    fitted = xv @ beta
    residual.loc[clean.index] = yv - fitted
    return residual


def preprocess_by_date(
    df: pd.DataFrame,
    date_col: str,
    factor_cols: list[str],
    winsor: bool = True,
    standardize: bool = True,
) -> pd.DataFrame:
    out = df.copy()
    for col in factor_cols:
        processed = out.groupby(date_col)[col].transform(lambda s: winsorize(s) if winsor else s)
        if standardize:
            processed = processed.groupby(out[date_col]).transform(zscore)
        out[f"{col}_prep"] = processed
    return out


def sequential_orthogonalize_one_date(group: pd.DataFrame, prepared_cols: list[str]) -> pd.DataFrame:
    out = group.copy()
    orth_cols: list[str] = []
    for idx, col in enumerate(prepared_cols):
        output_col = col.replace("_prep", "_orth")
        if idx == 0:
            out[output_col] = out[col]
        else:
            x = out[orth_cols]
            out[output_col] = regress_residual(out[col], x)
            out[output_col] = zscore(out[output_col])
        orth_cols.append(output_col)
    return out


def sequential_orthogonalize(
    df: pd.DataFrame,
    date_col: str,
    factor_cols: list[str],
    winsor: bool = True,
    standardize: bool = True,
) -> pd.DataFrame:
    out = preprocess_by_date(df, date_col, factor_cols, winsor=winsor, standardize=standardize)
    prepared_cols = [f"{col}_prep" for col in factor_cols]
    return out.groupby(date_col, group_keys=False).apply(lambda g: sequential_orthogonalize_one_date(g, prepared_cols))


def correlation_summary(
    df: pd.DataFrame,
    date_col: str,
    cols: list[str],
) -> pd.DataFrame:
    rows = []
    for date, group in df.groupby(date_col):
        corr = group[cols].corr()
        values = []
        for i, col_i in enumerate(cols):
            for j, col_j in enumerate(cols):
                if j <= i:
                    continue
                values.append(corr.loc[col_i, col_j])
        clean = pd.Series(values).dropna()
        rows.append(
            {
                "date": date,
                "mean_abs_corr": float(clean.abs().mean()) if not clean.empty else pd.NA,
                "max_abs_corr": float(clean.abs().max()) if not clean.empty else pd.NA,
                "pair_count": int(clean.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument("--date-col", required=True)
    parser.add_argument("--asset-col", required=True)
    parser.add_argument("--factor-cols", required=True, help="Comma-separated factor columns in orthogonalization order.")
    parser.add_argument("--no-winsor", action="store_true")
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--output-dir", default="factor_orthogonalization_output")
    args = parser.parse_args()

    df = pd.read_csv(args.csv, encoding=args.encoding)
    df[args.date_col] = pd.to_datetime(df[args.date_col])
    factor_cols = [col.strip() for col in args.factor_cols.split(",") if col.strip()]
    missing = [col for col in [args.date_col, args.asset_col, *factor_cols] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    result = sequential_orthogonalize(
        df,
        date_col=args.date_col,
        factor_cols=factor_cols,
        winsor=not args.no_winsor,
        standardize=not args.no_standardize,
    )
    prep_cols = [f"{col}_prep" for col in factor_cols]
    orth_cols = [f"{col}_orth" for col in factor_cols]
    raw_corr = correlation_summary(result, args.date_col, prep_cols)
    orth_corr = correlation_summary(result, args.date_col, orth_cols)
    raw_corr["type"] = "prepared"
    orth_corr["type"] = "orthogonalized"
    corr_summary = pd.concat([raw_corr, orth_corr], ignore_index=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "orthogonalized_factors.csv", index=False, encoding="utf-8-sig")
    corr_summary.to_csv(output_dir / "correlation_summary.csv", index=False, encoding="utf-8-sig")
    print(corr_summary.groupby("type")[["mean_abs_corr", "max_abs_corr"]].mean())
    print(f"saved={output_dir.resolve()}")


if __name__ == "__main__":
    main()
