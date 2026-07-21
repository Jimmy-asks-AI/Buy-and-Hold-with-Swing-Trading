#!/usr/bin/env python
"""Audit factor registry completeness and governance rules."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "factor_id",
    "column",
    "family",
    "direction",
    "horizon",
    "data_type",
    "availability_col",
    "cost_tier",
    "description",
]

VALID_FAMILIES = {
    "value",
    "dividend",
    "quality",
    "profitability",
    "growth",
    "investment",
    "leverage",
    "momentum",
    "reversal",
    "liquidity",
    "volatility",
    "technical",
    "intraday",
    "level2",
    "analyst",
    "behavior",
    "fund_flow",
    "macro",
    "alternative",
    "llm_text",
    "ml_deep",
}

DATA_TYPES_REQUIRING_AVAILABILITY = {
    "financial",
    "corporate_action",
    "analyst",
    "margin",
    "intraday",
    "level2",
    "fund",
    "macro",
    "relationship",
    "text",
    "model",
}


def audit_registry(registry: pd.DataFrame) -> dict[str, pd.DataFrame]:
    rows = []
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in registry.columns]
    if missing_columns:
        rows.append({"gate": "required_columns", "status": "fail", "detail": "|".join(missing_columns)})
        return {"summary": pd.DataFrame(rows), "family_distribution": pd.DataFrame(), "issues": pd.DataFrame(rows)}

    issues = []
    duplicate = registry[registry["factor_id"].duplicated(keep=False)]
    for _, row in duplicate.iterrows():
        issues.append({"factor_id": row["factor_id"], "issue": "duplicated_factor_id", "severity": "fail"})

    for _, row in registry.iterrows():
        factor_id = row["factor_id"]
        for col in REQUIRED_COLUMNS:
            if col == "availability_col":
                continue
            value = row[col]
            if pd.isna(value) or (isinstance(value, str) and not value.strip()):
                issues.append({"factor_id": factor_id, "issue": f"missing_{col}", "severity": "fail"})
        if row["family"] not in VALID_FAMILIES:
            issues.append({"factor_id": factor_id, "issue": f"unknown_family:{row['family']}", "severity": "warn"})
        try:
            direction = float(row["direction"])
            if direction not in {-1.0, 1.0}:
                issues.append({"factor_id": factor_id, "issue": "direction_not_plus_or_minus_one", "severity": "fail"})
        except Exception:
            issues.append({"factor_id": factor_id, "issue": "direction_not_numeric", "severity": "fail"})
        try:
            horizon = int(row["horizon"])
            if horizon <= 0:
                issues.append({"factor_id": factor_id, "issue": "horizon_not_positive", "severity": "fail"})
        except Exception:
            issues.append({"factor_id": factor_id, "issue": "horizon_not_integer", "severity": "fail"})
        data_type = str(row["data_type"])
        availability_col = "" if pd.isna(row["availability_col"]) else str(row["availability_col"]).strip()
        if data_type in DATA_TYPES_REQUIRING_AVAILABILITY and not availability_col:
            issues.append({"factor_id": factor_id, "issue": f"missing_availability_for_{data_type}", "severity": "warn"})

    issues_df = pd.DataFrame(issues)
    family_distribution = (
        registry.groupby("family")
        .agg(factors=("factor_id", "count"), horizons=("horizon", lambda s: ",".join(map(str, sorted(set(s))))))
        .reset_index()
        .sort_values("factors", ascending=False)
    )
    rows = [
        {"gate": "factor_count", "status": "pass" if registry.shape[0] >= 50 else "warn", "detail": str(registry.shape[0])},
        {"gate": "family_count", "status": "pass" if registry["family"].nunique() >= 10 else "warn", "detail": str(registry["family"].nunique())},
        {"gate": "duplicate_factor_id", "status": "pass" if duplicate.empty else "fail", "detail": str(duplicate.shape[0])},
        {
            "gate": "fail_issues",
            "status": "pass" if issues_df.empty or (issues_df["severity"] == "fail").sum() == 0 else "fail",
            "detail": str(0 if issues_df.empty else int((issues_df["severity"] == "fail").sum())),
        },
        {
            "gate": "warn_issues",
            "status": "pass" if issues_df.empty or (issues_df["severity"] == "warn").sum() == 0 else "warn",
            "detail": str(0 if issues_df.empty else int((issues_df["severity"] == "warn").sum())),
        },
    ]
    return {"summary": pd.DataFrame(rows), "family_distribution": family_distribution, "issues": issues_df}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    registry = pd.read_csv(args.registry, encoding="utf-8-sig")
    result = audit_registry(registry)
    for name, table in result.items():
        print(f"\n[{name}]")
        print(table)
    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, table in result.items():
            table.to_csv(out / f"{name}.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
