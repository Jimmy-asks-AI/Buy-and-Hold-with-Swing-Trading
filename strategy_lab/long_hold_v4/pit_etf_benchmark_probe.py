"""Capture the accessible ETF benchmark-history slice without qualifying it for backtests."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "raw_etf_benchmark"
DEFAULT_OUTPUT = (
    ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "observations" / "etf_benchmark_history_joinquant_limited.csv"
)
DEFAULT_MANIFEST = ROOT / "data_raw" / "long_hold_v4" / "manifests" / "etf_benchmark_probe_latest.json"
SOURCE_URL = "https://www.joinquant.com/help/api/doc?id=10672&name=JQDatadoc"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def build_etf_benchmark_observation(
    frame: pd.DataFrame,
    source_vintage: str,
    as_of: str | pd.Timestamp,
) -> pd.DataFrame:
    required = {"code", "pub_date", "start_date", "end_date", "traced_index_name", "traced_index_code"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"FUND_INVEST_TARGET missing columns: {missing}")

    as_of_date = pd.Timestamp(as_of).normalize()
    asset = frame["code"].astype(str).str.strip().str[:6]
    raw_index_code = frame["traced_index_code"].fillna("").astype(str).str.strip().str.split(".").str[0]
    name_fallback = frame["traced_index_name"].fillna("").astype(str).str.extract(r"\(([^()]+)\)\s*$", expand=False).fillna("")
    index_code = raw_index_code.mask(raw_index_code.eq(""), name_fallback)
    output = pd.DataFrame(
        {
            "asset": asset,
            "index_code": index_code,
            "index_name": frame["traced_index_name"].fillna("").astype(str).str.strip(),
            "effective_from": pd.to_datetime(frame["start_date"], errors="coerce"),
            "effective_to": pd.to_datetime(frame["end_date"], errors="coerce"),
            "announcement_date": pd.to_datetime(frame["pub_date"], errors="coerce"),
        }
    )
    valid = (
        output["asset"].str.fullmatch(r"\d{6}")
        & output["index_code"].ne("")
        & output["effective_from"].notna()
        & output["announcement_date"].notna()
        & output["effective_from"].le(as_of_date)
        & output["announcement_date"].le(as_of_date)
    )
    output = output[valid].copy()
    output["available_date"] = output["announcement_date"]
    output["data_source"] = "joinquant.finance.FUND_INVEST_TARGET"
    output["source_vintage"] = source_vintage
    output = output.sort_values(["asset", "effective_from", "announcement_date", "index_code"]).drop_duplicates(
        ["asset", "effective_from", "index_code"], keep="last"
    )
    if output.empty:
        raise ValueError("FUND_INVEST_TARGET produced no usable as-of rows")
    if output.duplicated(["asset", "effective_from", "index_code"]).any():
        raise ValueError("ETF benchmark observation contains duplicate keys")
    if (output["available_date"] > as_of_date).any():
        raise ValueError("ETF benchmark observation contains future announcements")
    return output.reset_index(drop=True)


def _load_joinquant() -> pd.DataFrame:
    credential_path = ROOT / "configs" / "data_credentials.json"
    if not credential_path.is_file():
        raise ValueError("JoinQuant credentials are required for the ETF benchmark probe")
    credentials = json.loads(credential_path.read_text(encoding="utf-8-sig")).get("joinquant", {})
    username = str(credentials.get("username", "")).strip()
    password = str(credentials.get("password", "")).strip()
    if not username or not password:
        raise ValueError("JoinQuant credentials are incomplete")

    import jqdatasdk as jq
    from jqdatasdk import finance, query

    jq.auth(username, password)
    frame = finance.run_query(query(finance.FUND_INVEST_TARGET))
    if frame.empty:
        raise ValueError("JoinQuant returned no FUND_INVEST_TARGET rows")
    return frame


def run_probe(as_of: str, raw_dir: Path, output_path: Path, manifest_path: Path) -> dict[str, Any]:
    as_of_date = pd.Timestamp(as_of).normalize()
    raw = _load_joinquant().sort_values("id").reset_index(drop=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    payload = raw.to_csv(index=False, lineterminator="\n").encode("utf-8-sig")
    raw_hash = hashlib.sha256(payload).hexdigest()
    raw_path = raw_dir / f"joinquant_fund_invest_target_{raw_hash[:16]}.csv"
    if not raw_path.exists():
        raw_path.write_bytes(payload)
    source_vintage = f"joinquant_fund_invest_target_sha256:{raw_hash}"
    output = build_etf_benchmark_observation(raw, source_vintage, as_of_date)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")

    master_path = ROOT / "data_raw" / "long_hold_v4" / "pit_history" / "etf_security_master.csv"
    master_assets = 0
    if master_path.is_file():
        master_assets = pd.read_csv(master_path, dtype={"asset": str})["asset"].nunique()
    code_path = Path(__file__).resolve()
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "as_of_date": str(as_of_date.date()),
        "inputs": [{"source_id": "joinquant_fund_invest_target", "source_url": SOURCE_URL, "path": _relative(raw_path), "sha256": raw_hash}],
        "source_vintage": source_vintage,
        "output_path": _relative(output_path),
        "output_sha256": _sha256(output_path),
        "code_path": _relative(code_path),
        "code_sha256": _sha256(code_path),
        "provider_rows": int(len(raw)),
        "usable_rows": int(len(output)),
        "assets": int(output["asset"].nunique()),
        "master_assets": int(master_assets),
        "asset_coverage": round(output["asset"].nunique() / master_assets, 6) if master_assets else None,
        "publication_start": str(output["announcement_date"].min().date()),
        "publication_end": str(output["announcement_date"].max().date()),
        "effective_start": str(output["effective_from"].min().date()),
        "effective_end": str(output["effective_from"].max().date()),
        "qualification_status": "BLOCKED_INSUFFICIENT_HISTORY_AND_UNIVERSE_COVERAGE",
        "limitations": [
            "The accessible account slice starts in 2025 and does not cover historical ETF benchmark changes",
            "Delisted ETF benchmark mappings are not comprehensively represented",
            "This observation is stored outside the formal Gate E2 target path",
        ],
        "historical_backtest_allowed": False,
        "model_promotion_allowed": False,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_probe(args.as_of, args.raw_dir, args.output, args.manifest)
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
