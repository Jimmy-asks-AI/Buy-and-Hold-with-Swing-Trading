from __future__ import annotations

import pandas as pd

from strategy_lab.long_hold_v4 import pit_etf_eastmoney_nav_validator as validator
from strategy_lab.long_hold_v4.pit_etf_eastmoney_nav_validator import (
    assess_terminal_boundary,
    compare_nav_sources,
)


def test_compare_nav_sources_counts_missing_rows_as_failed_coverage() -> None:
    eastmoney = pd.DataFrame(
        {
            "asset": ["159207", "159207"],
            "date": ["2025-05-06", "2025-05-07"],
            "unit_nav": [1.0026, 1.0126],
            "cumulative_nav": [1.0026, None],
        }
    )
    joinquant = pd.DataFrame(
        {
            "asset": ["159207", "159207", "159207"],
            "date": ["2025-05-06", "2025-05-07", "2025-05-08"],
            "unit_nav": [1.0026, 1.0126, 1.0169],
            "cumulative_nav": [1.0026, 1.0126, 1.0169],
        }
    )
    asset_summary, mismatches, totals = compare_nav_sources(eastmoney, joinquant)
    assert totals["joinquant_rows"] == 3
    assert totals["overlap_rows"] == 2
    assert totals["row_coverage"] == 2 / 3
    assert totals["unit_nav_within_tolerance_ratio"] == 1.0
    assert totals["cumulative_nav_comparable_rows"] == 1
    assert totals["joinquant_cumulative_nav_rows"] == 3
    assert totals["cumulative_nav_comparable_coverage"] == 1 / 3
    assert asset_summary.loc[0, "row_coverage"] == 2 / 3
    assert mismatches[["asset", "date"]].to_dict("records") == [
        {"asset": "159207", "date": pd.Timestamp("2025-05-08")}
    ]


def test_compare_nav_sources_flags_numeric_disagreement() -> None:
    eastmoney = pd.DataFrame(
        {
            "asset": ["510880"],
            "date": ["2025-05-06"],
            "unit_nav": [1.0001],
            "cumulative_nav": [2.0],
        }
    )
    joinquant = pd.DataFrame(
        {
            "asset": ["510880"],
            "date": ["2025-05-06"],
            "unit_nav": [1.0002],
            "cumulative_nav": [2.0001],
        }
    )
    _, mismatches, totals = compare_nav_sources(eastmoney, joinquant)
    assert len(mismatches) == 1
    assert totals["unit_nav_within_tolerance_ratio"] == 0.0
    assert totals["cumulative_nav_within_tolerance_ratio"] == 0.0


def test_assess_terminal_boundary_keeps_liquidation_cash_separate() -> None:
    nav = pd.DataFrame(
        {
            "asset": ["511210", "511210"],
            "date": ["2017-11-06", "2017-11-07"],
            "unit_nav": [113.2064, 112.6195],
        }
    )
    event = pd.Series(
        {
            "last_operation_date": "2017-11-07",
            "liquidation_start_date": "2017-11-08",
            "liquidation_nav": 112.6579,
            "cash_per_share": 112.79,
        }
    )
    result = assess_terminal_boundary(nav, event)
    assert result["boundary_passed"]
    assert result["post_operation_nav_rows"] == 0
    assert result["last_unit_nav"] == 112.6195
    assert result["cash_distribution_per_share"] == 112.79


def test_assess_terminal_boundary_rejects_synthetic_post_operation_nav() -> None:
    nav = pd.DataFrame(
        {
            "asset": ["511210", "511210"],
            "date": ["2017-11-07", "2018-01-23"],
            "unit_nav": [112.6195, 112.79],
        }
    )
    event = pd.Series(
        {
            "last_operation_date": "2017-11-07",
            "liquidation_start_date": "2017-11-08",
            "liquidation_nav": 112.6579,
            "cash_per_share": 112.79,
        }
    )
    result = assess_terminal_boundary(nav, event)
    assert not result["boundary_passed"]
    assert result["post_operation_nav_rows"] == 1


def test_declared_upstream_code_authenticates_primary_and_dependencies(tmp_path, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(validator, "ROOT", tmp_path)
    monkeypatch.setattr(
        validator,
        "authenticate_current_or_archive",
        lambda path, digest: calls.append((path.name, digest)),
    )
    manifest = {
        "code_path": "collector.py",
        "code_sha256": "a" * 64,
        "code_files": [
            {"path": "collector.py", "sha256": "a" * 64},
            {"path": "dependency.py", "sha256": "b" * 64},
        ],
    }
    validator._authenticate_declared_code(manifest, "fixture")
    assert calls == [
        ("collector.py", "a" * 64),
        ("collector.py", "a" * 64),
        ("dependency.py", "b" * 64),
    ]
