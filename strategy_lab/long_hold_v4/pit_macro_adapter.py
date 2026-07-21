"""Adapt the governed local macro PIT panel to the V4 Gate E2 rate contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data_raw" / "macro" / "macro_pit_panel.csv"
DEFAULT_OUTPUT = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "macro_rate_history.csv"
DEFAULT_MANIFEST = ROOT / "data_raw" / "long_hold_v4" / "manifests" / "macro_pit_adapter_latest.json"
SERIES_MAP = {
    "cn_10y_gov_bond_yield": "CN10Y",
    "us_10y_treasury_yield": "US10Y",
    "cn_us_10y_rate_spread": "CN_US_10Y_SPREAD",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_macro_rate_history(source: pd.DataFrame, source_vintage: str, as_of: str | pd.Timestamp) -> pd.DataFrame:
    required = {"date", "available_date", "series_id", "value", "source"}
    missing = sorted(required.difference(source.columns))
    if missing:
        raise ValueError(f"macro PIT panel missing columns: {missing}")
    out = source[source["series_id"].astype(str).isin(SERIES_MAP)].copy()
    out["observation_date"] = pd.to_datetime(out["date"], errors="coerce")
    out["available_date"] = pd.to_datetime(out["available_date"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out["series_id"] = out["series_id"].map(SERIES_MAP)
    out["data_source"] = out["source"].astype(str)
    out["source_vintage"] = source_vintage
    out = out[
        ["observation_date", "series_id", "value", "available_date", "data_source", "source_vintage"]
    ].dropna(subset=["observation_date", "series_id", "value", "available_date"])
    out = out[out["available_date"] <= pd.Timestamp(as_of).normalize()].copy()
    if out.empty or set(out["series_id"]) != set(SERIES_MAP.values()):
        raise ValueError("macro PIT panel does not contain all required rate series")
    if (out["available_date"] < out["observation_date"]).any():
        raise ValueError("macro rate available_date precedes observation_date")
    if out.duplicated(["observation_date", "series_id", "source_vintage"]).any():
        raise ValueError("macro rate source contains duplicate PIT keys")
    return out.sort_values(["series_id", "observation_date"]).reset_index(drop=True)


def run(input_path: Path, output_path: Path, manifest_path: Path, as_of: str | pd.Timestamp) -> dict[str, object]:
    input_hash = _sha256(input_path)
    source = pd.read_csv(input_path, low_memory=False)
    vintage = f"macro_pit_panel_sha256:{input_hash}"
    output = build_macro_rate_history(source, vintage, as_of)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    output.to_csv(temporary, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    temporary.replace(output_path)
    result = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "as_of_date": str(pd.Timestamp(as_of).date()),
        "input_path": str(input_path.resolve().relative_to(ROOT.resolve())),
        "input_sha256": input_hash,
        "inputs": [
            {
                "path": str(input_path.resolve().relative_to(ROOT.resolve())),
                "sha256": input_hash,
            }
        ],
        "source_vintage": vintage,
        "output_path": str(output_path.resolve().relative_to(ROOT.resolve())),
        "output_sha256": _sha256(output_path),
        "code_path": str(Path(__file__).resolve().relative_to(ROOT.resolve())),
        "code_sha256": _sha256(Path(__file__)),
        "rows": int(len(output)),
        "series": sorted(output["series_id"].unique().tolist()),
        "coverage_start": str(output["observation_date"].min().date()),
        "coverage_end": str(output["observation_date"].max().date()),
        "historical_backtest_allowed": True,
        "model_promotion_allowed": False,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--as-of", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output, args.manifest, args.as_of), ensure_ascii=False))


if __name__ == "__main__":
    main()
