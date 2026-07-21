#!/usr/bin/env python
"""Canonicalize real A-share source tables using a field-mapping template."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import csv_io


DEFAULT_MAPPING = Path("Introduction-to-Quantitative-Finance") / "data_catalog" / "a_share_real_data_field_mapping_template.csv"


def _split_aliases(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [item.strip() for item in str(value).split("|") if item.strip()]


def _coerce_dtype(series: pd.Series, dtype: str) -> pd.Series:
    dtype = str(dtype).strip().lower()
    if dtype == "date":
        return pd.to_datetime(series, errors="coerce")
    if dtype == "float":
        return pd.to_numeric(series, errors="coerce")
    if dtype == "bool":
        return csv_io.coerce_bool_series(series)
    return series.astype(str).str.strip()


def load_mapping(path: str | Path = DEFAULT_MAPPING) -> pd.DataFrame:
    mapping = csv_io.read_csv_robust(path)
    required = {"source_table", "required", "canonical_col", "dtype", "accepted_aliases"}
    missing = required - set(mapping.columns)
    if missing:
        raise ValueError(f"mapping missing columns: {sorted(missing)}")
    return mapping


def canonicalize_table(
    table: pd.DataFrame,
    mapping: pd.DataFrame,
    source_table: str,
    strict_required: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rename and type-coerce a raw source table to canonical columns."""
    rules = mapping.loc[mapping["source_table"].astype(str).str.lower() == source_table.lower()].copy()
    if rules.empty:
        raise ValueError(f"no mapping rows for source_table={source_table}")
    data = table.copy()
    lower_to_original = {str(col).strip().lower(): col for col in data.columns}
    report_rows = []
    used_source_cols: set[str] = set()
    out = pd.DataFrame(index=data.index)

    for row in rules.itertuples(index=False):
        canonical = str(row.canonical_col).strip()
        candidates = [canonical, *_split_aliases(row.accepted_aliases)]
        found = None
        for candidate in candidates:
            key = candidate.strip().lower()
            if key in lower_to_original:
                found = lower_to_original[key]
                break
        required = str(row.required).strip().lower() == "yes"
        status = "mapped" if found is not None else ("missing_required" if required else "missing_optional")
        report_rows.append(
            {
                "source_table": source_table,
                "canonical_col": canonical,
                "source_col": found or "",
                "required": required,
                "dtype": row.dtype,
                "status": status,
            }
        )
        if found is not None:
            out[canonical] = _coerce_dtype(data[found], row.dtype)
            used_source_cols.add(found)
        elif required and strict_required:
            continue

    for col in data.columns:
        if col not in used_source_cols and col not in out.columns:
            out[col] = data[col]

    report = pd.DataFrame(report_rows)
    missing_required = report.loc[report["status"] == "missing_required", "canonical_col"].tolist()
    if missing_required and strict_required:
        raise ValueError(f"{source_table} missing required columns after mapping: {missing_required}")
    return out, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--source-table", required=True, choices=["market", "financial", "industry"])
    parser.add_argument("--mapping-csv", default=str(DEFAULT_MAPPING))
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--report-csv")
    parser.add_argument("--allow-missing-required", action="store_true")
    args = parser.parse_args()

    table = csv_io.read_csv_robust(args.input_csv)
    mapping = load_mapping(args.mapping_csv)
    out, report = canonicalize_table(table, mapping, args.source_table, strict_required=not args.allow_missing_required)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    if args.report_csv:
        Path(args.report_csv).parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(args.report_csv, index=False, encoding="utf-8-sig")
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
