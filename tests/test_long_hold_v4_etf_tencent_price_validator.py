from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from strategy_lab.long_hold_v4 import pit_etf_tencent_price_validator as validator
from strategy_lab.long_hold_v4.pit_etf_tencent_price_validator import (
    compare_asset_prices,
    identify_material_close_mismatches,
)


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_lifecycle_fixture(tmp_path, *, duplicate_asset: bool = False, divergent_immutable: bool = False):
    raw_path = tmp_path / "raw.csv.gz"
    raw_path.write_bytes(b"raw-price-evidence")
    latest_path = tmp_path / "latest.json"
    immutable_path = tmp_path / "immutable.json"
    inputs = [
        {"role": "etf_raw_price", "asset": "510050", "path": raw_path.name, "sha256": _sha256(raw_path)}
    ]
    if duplicate_asset:
        inputs.append(dict(inputs[0]))
    payload = {
        "run_id": "fixture",
        "immutable_manifest_path": immutable_path.name,
        "code_files": [{"path": "collector.py", "sha256": "a" * 64}],
        "historical_backtest_allowed": False,
        "selected_assets": 1,
        "inputs": inputs,
    }
    encoded = json.dumps(payload, sort_keys=True)
    latest_path.write_text(encoded, encoding="utf-8")
    immutable_path.write_text(encoded + ("\n" if divergent_immutable else ""), encoding="utf-8")
    return latest_path


def test_compare_asset_prices_applies_lot_conversion_before_volume_check() -> None:
    tencent = pd.DataFrame(
        {
            "date": ["2020-01-02", "2020-01-03"],
            "open": [1.0, 1.1],
            "high": [1.1, 1.2],
            "low": [0.9, 1.0],
            "close": [1.05, 1.15],
            "volume_shares": [1000.0, 2000.0],
        }
    )
    sina = pd.DataFrame(
        {
            "date": ["2020-01-02", "2020-01-03"],
            "open": [1.0, 1.1],
            "high": [1.1, 1.2],
            "low": [0.9, 1.0],
            "close": [1.05, 1.15],
            "volume": [1000.0, 2000.0],
        }
    )
    summary, mismatches, counts, errors = compare_asset_prices(tencent, sina, asset="510050")
    assert summary["primary_row_coverage"] == 1.0
    assert summary["close_exact_ratio"] == 1.0
    assert summary["volume_relative_error_p95"] == 0.0
    assert counts["overlap_rows"] == 2
    assert errors["volume_relative"].tolist() == [0.0, 0.0]
    assert mismatches.empty


def test_compare_asset_prices_discloses_source_only_dates_and_tick_differences() -> None:
    tencent = pd.DataFrame(
        {
            "date": ["2020-01-02", "2020-01-03"],
            "open": [1.0, 1.1],
            "high": [1.1, 1.2],
            "low": [0.9, 1.0],
            "close": [1.051, 1.15],
            "volume_shares": [1000.0, 2000.0],
        }
    )
    sina = pd.DataFrame(
        {
            "date": ["2020-01-02", "2020-01-04"],
            "open": [1.0, 1.2],
            "high": [1.1, 1.3],
            "low": [0.9, 1.1],
            "close": [1.05, 1.25],
            "volume": [1000.0, 3000.0],
        }
    )
    summary, mismatches, counts, _ = compare_asset_prices(tencent, sina, asset="510050")
    assert summary["primary_row_coverage"] == 0.5
    assert summary["close_exact_ratio"] == 0.0
    assert summary["close_within_one_tick_ratio"] == 1.0
    assert counts["primary_only_rows"] == 1
    assert counts["independent_only_rows"] == 1
    assert set(mismatches["_merge"].astype(str)) == {"both", "left_only", "right_only"}


def test_compare_asset_prices_rejects_duplicate_source_dates() -> None:
    tencent = pd.DataFrame(
        {
            "date": ["2020-01-02", "2020-01-02"],
            "open": [1.0, 1.0],
            "high": [1.1, 1.1],
            "low": [0.9, 0.9],
            "close": [1.0, 1.0],
            "volume_shares": [1000.0, 1000.0],
        }
    )
    sina = tencent.rename(columns={"volume_shares": "volume"}).iloc[:1]
    with pytest.raises(ValueError, match="duplicate dates"):
        compare_asset_prices(tencent, sina, asset="510050")


def test_material_close_mismatches_use_absolute_or_relative_threshold() -> None:
    mismatches = pd.DataFrame(
        {
            "asset": ["510180", "511260", "510050", "510060"],
            "date": ["2008-01-02", "2018-01-25", "2020-01-02", "2020-01-02"],
            "close_tencent": [4.151, 97.816, 1.01, 1.00],
            "close_sina": [11.868, 97.502, 1.00, 1.20],
            "close_absolute_error": [7.717, 0.314, 0.01, 0.20],
            "volume_relative_error": [0.7, 0.001, 0.0, 0.0],
            "_merge": ["both", "both", "both", "left_only"],
        }
    )
    material = identify_material_close_mismatches(mismatches)
    assert material["asset"].tolist() == ["510180"]
    assert material.loc[0, "close_relative_error"] == pytest.approx(7.717 / 11.868)


def test_sina_price_map_authenticates_immutable_manifest_and_code_bundle(tmp_path, monkeypatch) -> None:
    latest_path = _write_lifecycle_fixture(tmp_path)
    calls = []
    monkeypatch.setattr(validator, "ROOT", tmp_path)
    monkeypatch.setattr(
        validator,
        "authenticate_current_or_archive",
        lambda path, digest: calls.append((path.name, digest)),
    )
    price_map, manifest = validator._authenticate_sina_price_map(latest_path)
    assert set(price_map) == {"510050"}
    assert manifest["run_id"] == "fixture"
    assert calls == [("collector.py", "a" * 64)]


def test_sina_price_map_rejects_divergent_latest_pointer(tmp_path, monkeypatch) -> None:
    latest_path = _write_lifecycle_fixture(tmp_path, divergent_immutable=True)
    monkeypatch.setattr(validator, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="latest and immutable manifests differ"):
        validator._authenticate_sina_price_map(latest_path)


def test_sina_price_map_rejects_duplicate_asset_inputs(tmp_path, monkeypatch) -> None:
    latest_path = _write_lifecycle_fixture(tmp_path, duplicate_asset=True)
    monkeypatch.setattr(validator, "ROOT", tmp_path)
    monkeypatch.setattr(validator, "authenticate_current_or_archive", lambda path, digest: path)
    with pytest.raises(ValueError, match="duplicate raw prices"):
        validator._authenticate_sina_price_map(latest_path)
