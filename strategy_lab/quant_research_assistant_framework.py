from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def rel_path(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_price_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"])
    for col in ["open", "high", "low", "close", "volume", "amount", "pe_ttm", "pct_chg"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["close"].notna() & (df["close"] > 0)].sort_values("date").reset_index(drop=True)
    return df


def last_non_empty(values: pd.Series) -> Any:
    cleaned = values.dropna()
    return cleaned.iloc[-1] if not cleaned.empty else None


def object_source_path(obj: dict[str, Any]) -> Path:
    if obj["object_type"] == "market_index":
        return ROOT / "data_raw" / "index" / "akshare_csindex" / "daily_csindex" / f"{obj['object_id']}.csv"
    if obj["object_type"] == "industry_index":
        return ROOT / "data_raw" / "index" / "akshare_sw_industry" / "daily_sw" / f"{obj['object_id']}.csv"
    raise ValueError(f"unsupported object_type: {obj['object_type']}")


def industry_display_name(object_id: str, fallback: str) -> str:
    path = ROOT / "data_raw" / "index" / "akshare_sw_industry" / "analysis_latest_sw" / "analysis_latest_combined.csv"
    if not path.exists():
        return fallback
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return fallback
    row = df[df["index_code"].astype(str).eq(str(object_id))]
    if row.empty or "index_name" not in row.columns:
        return fallback
    name = row.iloc[-1]["index_name"]
    return str(name) if pd.notna(name) and str(name).strip() else fallback


def build_research_universe(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for obj in config["research_objects"]:
        path = object_source_path(obj)
        status = "missing"
        row_count = 0
        start_date = ""
        end_date = ""
        data_source = ""
        resolved_name = obj.get("object_name", obj["object_id"])
        if path.exists():
            df = load_price_frame(path)
            row_count = len(df)
            status = "usable" if row_count >= config["minimum_price_rows"] else "limited_history"
            if row_count:
                start_date = df["date"].min().date().isoformat()
                end_date = df["date"].max().date().isoformat()
                if "index_name" in df.columns:
                    name = last_non_empty(df["index_name"])
                    if name:
                        resolved_name = str(name)
                elif "index_short_name" in df.columns:
                    name = last_non_empty(df["index_short_name"])
                    if name:
                        resolved_name = str(name)
                data_source = str(last_non_empty(df.get("data_source", pd.Series(dtype=object))) or "")
        if obj["object_type"] == "industry_index" and resolved_name == obj["object_id"]:
            resolved_name = industry_display_name(obj["object_id"], obj.get("object_name", obj["object_id"]))
        rows.append(
            {
                "object_id": obj["object_id"],
                "object_name": resolved_name,
                "object_type": obj["object_type"],
                "sleeve": obj.get("sleeve", ""),
                "horizon": obj.get("horizon", obj.get("primary_horizon", "1m_6m")),
                "asof_date": "",
                "price_path": rel_path(path),
                "price_status": status,
                "row_count": row_count,
                "start_date": start_date,
                "end_date": end_date,
                "data_source": data_source,
            }
        )
    return rows


def pct_rank(series: pd.Series, value: float | None) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if value is None or pd.isna(value) or clean.empty:
        return None
    return float((clean <= value).mean())


def bounded(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def score_to_view(score: float, bullish_cut: float = 0.58, bearish_cut: float = 0.42) -> str:
    if score >= bullish_cut:
        return "bullish"
    if score <= bearish_cut:
        return "bearish"
    return "neutral"


def pct_to_volatility_state(value: Any) -> str:
    if value is None or pd.isna(value):
        return "unknown"
    numeric = float(value)
    if numeric >= 0.85:
        return "high"
    if numeric >= 0.70:
        return "elevated"
    if numeric <= 0.20:
        return "low"
    return "normal"


def truthy_cell(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def compute_technical_panel(universe: list[dict[str, Any]], config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    latest_rows: list[dict[str, Any]] = []
    panel_rows: list[dict[str, Any]] = []
    keep_tail = int(config.get("technical_panel_tail_rows", 260))
    for obj in universe:
        path = ROOT / obj["price_path"]
        if not path.exists() or obj["price_status"] == "missing":
            latest_rows.append(
                {
                    "object_id": obj["object_id"],
                    "object_name": obj["object_name"],
                    "object_type": obj["object_type"],
                    "date": "",
                    "technical_status": "missing_price_data",
                    "technical_score": None,
                    "technical_view": "blocked",
                    "confidence": 0.0,
                    "evidence": "price file missing",
                    "failure_scenario": "cannot judge trend without price history",
                }
            )
            continue
        df = load_price_frame(path)
        df["ret_1d"] = df["close"].pct_change()
        for window in [20, 60, 120]:
            df[f"ma_{window}"] = df["close"].rolling(window).mean()
            df[f"ret_{window}d"] = df["close"] / df["close"].shift(window) - 1.0
        df["vol_20d_ann"] = df["ret_1d"].rolling(20).std() * (252 ** 0.5)
        df["volatility_percentile_252"] = df["vol_20d_ann"].rolling(252, min_periods=60).apply(
            lambda window: float((window <= window.iloc[-1]).mean()),
            raw=False,
        )
        df["volatility_percentile_252"] = df["volatility_percentile_252"].fillna(0.5)
        df["volatility_state"] = df["volatility_percentile_252"].map(pct_to_volatility_state)
        df["drawdown_120d"] = df["close"] / df["close"].rolling(120).max() - 1.0
        if "amount" in df.columns:
            df["amount_ratio_20_120"] = df["amount"].rolling(20).mean() / df["amount"].rolling(120).mean()
        else:
            df["amount_ratio_20_120"] = 1.0
        if "is_full_ohlc_bar" in df.columns:
            df["has_full_ohlc_bar"] = df["is_full_ohlc_bar"].map(truthy_cell)
        else:
            df["has_full_ohlc_bar"] = False
        if "is_close_only_bar" in df.columns:
            df["is_close_only_bar_flag"] = df["is_close_only_bar"].map(truthy_cell)
        else:
            df["is_close_only_bar_flag"] = ~df["has_full_ohlc_bar"]
        df["data_quality_status"] = df.apply(
            lambda row: "full_ohlc" if row["has_full_ohlc_bar"] else "close_only" if row["is_close_only_bar_flag"] else "unknown",
            axis=1,
        )
        df["ma_stack"] = (
            (df["close"] > df["ma_20"]).astype(float)
            + (df["ma_20"] > df["ma_60"]).astype(float)
            + (df["ma_60"] > df["ma_120"]).astype(float)
        ) / 3.0
        df["momentum_score"] = (
            df["ret_20d"].fillna(0.0).clip(-0.12, 0.12) / 0.24
            + df["ret_60d"].fillna(0.0).clip(-0.24, 0.24) / 0.48
            + 1.0
        ) / 2.0
        df["drawdown_repair_score"] = (1.0 + df["drawdown_120d"].fillna(0.0).clip(-0.35, 0.0) / 0.35).clip(0.0, 1.0)
        df["liquidity_score"] = df["amount_ratio_20_120"].replace([float("inf"), -float("inf")], pd.NA).fillna(1.0).clip(0.5, 1.5) - 0.5
        df["technical_score"] = (
            0.38 * df["ma_stack"]
            + 0.34 * df["momentum_score"]
            + 0.18 * df["drawdown_repair_score"]
            + 0.10 * df["liquidity_score"]
        ).clip(0.0, 1.0)
        df["score_saturation_flag"] = (df["technical_score"] >= 0.95) | (df["technical_score"] <= 0.05)
        df["technical_view"] = df["technical_score"].map(lambda value: score_to_view(float(value)))
        df["volatility_confidence_penalty"] = (df["volatility_percentile_252"] - 0.75).clip(lower=0.0) * 0.35
        df["saturation_confidence_penalty"] = df["score_saturation_flag"].astype(float) * 0.08
        df["bar_quality_confidence_penalty"] = df["is_close_only_bar_flag"].astype(float) * 0.10
        df["confidence"] = (
            0.45
            + 0.25 * float(len(df) >= 500)
            + 0.10 * df["ma_120"].notna().astype(float)
            + 0.10 * df["has_full_ohlc_bar"].astype(float)
            - df["volatility_confidence_penalty"]
            - df["saturation_confidence_penalty"]
            - df["bar_quality_confidence_penalty"]
        ).clip(0.25, 0.9)
        tail = df.tail(keep_tail)
        for _, row in tail.iterrows():
            panel_rows.append(
                {
                    "object_id": obj["object_id"],
                    "date": row["date"].date().isoformat(),
                    "close": round(float(row["close"]), 6),
                    "ret_20d": round(float(row["ret_20d"]), 6) if pd.notna(row["ret_20d"]) else None,
                    "ret_60d": round(float(row["ret_60d"]), 6) if pd.notna(row["ret_60d"]) else None,
                    "vol_20d_ann": round(float(row["vol_20d_ann"]), 6) if pd.notna(row["vol_20d_ann"]) else None,
                    "volatility_percentile_252": round(float(row["volatility_percentile_252"]), 6) if pd.notna(row["volatility_percentile_252"]) else None,
                    "volatility_state": row["volatility_state"],
                    "drawdown_120d": round(float(row["drawdown_120d"]), 6) if pd.notna(row["drawdown_120d"]) else None,
                    "ma_stack": round(float(row["ma_stack"]), 6) if pd.notna(row["ma_stack"]) else None,
                    "momentum_score": round(float(row["momentum_score"]), 6) if pd.notna(row["momentum_score"]) else None,
                    "drawdown_repair_score": round(float(row["drawdown_repair_score"]), 6) if pd.notna(row["drawdown_repair_score"]) else None,
                    "liquidity_score": round(float(row["liquidity_score"]), 6) if pd.notna(row["liquidity_score"]) else None,
                    "technical_score": round(float(row["technical_score"]), 6) if pd.notna(row["technical_score"]) else None,
                    "technical_view": row["technical_view"],
                    "data_quality_status": row["data_quality_status"],
                    "score_saturation_flag": bool(row["score_saturation_flag"]),
                }
            )
        last = df.iloc[-1]
        cap_reasons = []
        if str(last["volatility_state"]) in {"high", "elevated"}:
            cap_reasons.append(f"volatility_{last['volatility_state']}")
        if bool(last["score_saturation_flag"]):
            cap_reasons.append("score_saturation")
        if bool(last["is_close_only_bar_flag"]):
            cap_reasons.append("close_only_bar")
        latest_rows.append(
            {
                "object_id": obj["object_id"],
                "object_name": obj["object_name"],
                "object_type": obj["object_type"],
                "date": last["date"].date().isoformat(),
                "close": round(float(last["close"]), 6),
                "ret_20d": round(float(last["ret_20d"]), 6) if pd.notna(last["ret_20d"]) else None,
                "ret_60d": round(float(last["ret_60d"]), 6) if pd.notna(last["ret_60d"]) else None,
                "ret_120d": round(float(last["ret_120d"]), 6) if pd.notna(last["ret_120d"]) else None,
                "vol_20d_ann": round(float(last["vol_20d_ann"]), 6) if pd.notna(last["vol_20d_ann"]) else None,
                "volatility_percentile_252": round(float(last["volatility_percentile_252"]), 6) if pd.notna(last["volatility_percentile_252"]) else None,
                "volatility_state": str(last["volatility_state"]),
                "drawdown_120d": round(float(last["drawdown_120d"]), 6) if pd.notna(last["drawdown_120d"]) else None,
                "ma_stack": round(float(last["ma_stack"]), 6) if pd.notna(last["ma_stack"]) else None,
                "momentum_score": round(float(last["momentum_score"]), 6) if pd.notna(last["momentum_score"]) else None,
                "drawdown_repair_score": round(float(last["drawdown_repair_score"]), 6) if pd.notna(last["drawdown_repair_score"]) else None,
                "liquidity_score": round(float(last["liquidity_score"]), 6) if pd.notna(last["liquidity_score"]) else None,
                "data_quality_status": str(last["data_quality_status"]),
                "score_saturation_flag": bool(last["score_saturation_flag"]),
                "technical_status": "usable",
                "technical_score": round(float(last["technical_score"]), 6),
                "technical_view": str(last["technical_view"]),
                "confidence": round(float(last["confidence"]), 6),
                "confidence_cap_reason": ";".join(cap_reasons) if cap_reasons else "none",
                "evidence": "ma stack, 20/60/120d momentum, 120d drawdown repair, amount activity, volatility context",
                "failure_scenario": "range-bound whipsaw, volatility shock, score saturation, or policy shock can break trend/momentum signals",
            }
        )
    return latest_rows, panel_rows


def load_latest_macro_snapshot(asof_date: str) -> list[dict[str, Any]]:
    path = ROOT / "data_raw" / "macro" / "macro_pit_panel.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["available_date"] = pd.to_datetime(df["available_date"])
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    asof = pd.to_datetime(asof_date)
    df = df[df["available_date"] <= asof].sort_values(["series_id", "available_date", "date"])
    rows: list[dict[str, Any]] = []
    for series_id, group in df.groupby("series_id"):
        last = group.iloc[-1]
        stale_days = int((asof.normalize() - last["available_date"].normalize()).days)
        rows.append(
            {
                "series_id": series_id,
                "asof_date": asof.date().isoformat(),
                "date": last["date"].date().isoformat(),
                "available_date": last["available_date"].date().isoformat(),
                "stale_days": stale_days,
                "is_available_asof": bool(last["available_date"] <= asof),
                "value": round(float(last["value"]), 6) if pd.notna(last["value"]) else None,
                "source": last.get("source", ""),
                "pit_quality": last.get("pit_quality", ""),
                "macro_usage": "latest_context_only; historical use requires available_date asof join",
            }
        )
    return rows


def load_index_valuation(object_id: str) -> dict[str, Any]:
    pe_path = ROOT / "data_raw" / "index" / "akshare_csindex" / "valuation_pe_lg" / f"{object_id}.csv"
    pb_path = ROOT / "data_raw" / "index" / "akshare_csindex" / "valuation_pb_lg" / f"{object_id}.csv"
    latest_path = ROOT / "data_raw" / "index" / "akshare_csindex" / "valuation_latest_csindex" / f"{object_id}.csv"
    daily_path = ROOT / "data_raw" / "index" / "akshare_csindex" / "daily_csindex" / f"{object_id}.csv"
    data: dict[str, Any] = {
        "source_status": "missing",
        "pe_ttm": None,
        "pb": None,
        "dividend_yield": None,
        "pe_percentile": None,
        "pb_percentile": None,
        "pe_source": "",
        "pb_source": "",
        "valuation_source": "",
    }
    if pe_path.exists():
        pe = pd.read_csv(pe_path, encoding="utf-8-sig")
        pe["date"] = pd.to_datetime(pe["date"])
        pe["pe_ttm"] = pd.to_numeric(pe.get("pe_ttm"), errors="coerce")
        clean = pe[pe["pe_ttm"].notna()].sort_values("date")
        if not clean.empty:
            value = float(clean.iloc[-1]["pe_ttm"])
            data["pe_ttm"] = value
            data["pe_percentile"] = pct_rank(clean["pe_ttm"], value)
            data["source_status"] = "historical_pe"
            data["pe_source"] = rel_path(pe_path)
            data["valuation_source"] = rel_path(pe_path)
    elif daily_path.exists():
        daily = pd.read_csv(daily_path, encoding="utf-8-sig")
        if "pe_ttm" in daily.columns:
            daily["date"] = pd.to_datetime(daily["date"])
            daily["pe_ttm"] = pd.to_numeric(daily["pe_ttm"], errors="coerce")
            clean_daily = daily[daily["pe_ttm"].notna()].sort_values("date")
            if not clean_daily.empty:
                value = float(clean_daily.iloc[-1]["pe_ttm"])
                data["pe_ttm"] = value
                data["pe_percentile"] = pct_rank(clean_daily["pe_ttm"], value)
                data["source_status"] = "historical_pe_daily"
                data["pe_source"] = rel_path(daily_path)
                data["valuation_source"] = rel_path(daily_path)
    if pb_path.exists():
        pb = pd.read_csv(pb_path, encoding="utf-8-sig")
        pb["date"] = pd.to_datetime(pb["date"])
        pb["pb"] = pd.to_numeric(pb.get("pb"), errors="coerce")
        clean_pb = pb[pb["pb"].notna()].sort_values("date")
        if not clean_pb.empty:
            value = float(clean_pb.iloc[-1]["pb"])
            data["pb"] = value
            data["pb_percentile"] = pct_rank(clean_pb["pb"], value)
            data["source_status"] = "historical_pe_pb" if data["pe_ttm"] is not None else "historical_pb"
            data["pb_source"] = rel_path(pb_path)
            data["valuation_source"] = ";".join(source for source in [data.get("pe_source"), data.get("pb_source")] if source)
    if latest_path.exists():
        latest = pd.read_csv(latest_path, encoding="utf-8-sig").tail(1)
        for col in ["dividend_yield1", "dividend_yield2"]:
            if col in latest.columns and pd.notna(latest.iloc[0][col]):
                data["dividend_yield"] = float(latest.iloc[0][col])
                break
    return data


def load_industry_snapshot(object_id: str) -> dict[str, Any]:
    path = ROOT / "data_raw" / "index" / "akshare_sw_industry" / "analysis_latest_sw" / "analysis_latest_combined.csv"
    if not path.exists():
        return {
            "source_status": "missing",
            "pe_ttm": None,
            "pb": None,
            "dividend_yield": None,
            "pe_percentile": None,
            "pb_percentile": None,
            "valuation_source": "",
        }
    df = pd.read_csv(path, encoding="utf-8-sig")
    row = df[df["index_code"].astype(str).eq(str(object_id))]
    if row.empty:
        return {
            "source_status": "missing",
            "pe_ttm": None,
            "pb": None,
            "dividend_yield": None,
            "pe_percentile": None,
            "pb_percentile": None,
            "valuation_source": "",
        }
    last = row.iloc[-1]
    fetched_at = str(last.get("fetched_at", "") or "")
    available_date = fetched_at[:10] if fetched_at else str(last.get("resolved_analysis_date", "") or "")
    return {
        "source_status": "current_snapshot_only",
        "snapshot_date": str(last.get("date", "") or ""),
        "available_date": available_date,
        "pe_ttm": float(last["pe"]) if pd.notna(last.get("pe")) else None,
        "pb": float(last["pb"]) if pd.notna(last.get("pb")) else None,
        "dividend_yield": float(last["dividend_yield"]) if pd.notna(last.get("dividend_yield")) else None,
        "pe_percentile": None,
        "pb_percentile": None,
        "valuation_source": rel_path(path),
    }


def compute_fundamental_latest(universe: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    for obj in universe:
        valuation_repair_score = None
        dividend_score = None
        pe_inverse_score = None
        pb_inverse_score = None
        available_component_count = 0
        historical_backtest_allowed = False
        latest_research_allowed = False
        valuation_history_backtest_allowed = False
        confidence_cap_reason = "missing_or_unsupported"
        score_usage = "blocked"
        evidence = "missing PIT fundamental data"
        if obj["object_type"] == "market_index":
            data = load_index_valuation(obj["object_id"])
            pit_status = "historical_available" if "historical" in data["source_status"] else "missing"
            valuation_components: list[float] = []
            if data["pe_percentile"] is not None:
                pe_inverse_score = 1.0 - float(data["pe_percentile"])
                valuation_components.append(pe_inverse_score)
            if data["pb_percentile"] is not None:
                pb_inverse_score = 1.0 - float(data["pb_percentile"])
                valuation_components.append(pb_inverse_score)
            available_component_count = len(valuation_components)
            valuation_repair_score = sum(valuation_components) / len(valuation_components) if valuation_components else None
            dividend_score = bounded(float(data["dividend_yield"] or 0.0) / 5.0)
            valuation_history_backtest_allowed = bool(valuation_repair_score is not None)
            historical_backtest_allowed = False
            latest_research_allowed = bool(valuation_repair_score is not None)
            if valuation_repair_score is None:
                fundamental_score = None
                view = "blocked"
                confidence = 0.0
                confidence_cap_reason = "missing_valuation_history"
            else:
                fundamental_score = 0.72 * valuation_repair_score + 0.28 * dividend_score
                view = score_to_view(fundamental_score)
                confidence = 0.78 if available_component_count >= 2 else 0.68
                confidence_cap_reason = "none" if available_component_count >= 2 else "partial_valuation_history"
                score_usage = "latest_research_score; do not use this latest row in historical backtests; reconstruct PE/PB components by date and available_date"
                evidence_parts = []
                if pe_inverse_score is not None:
                    evidence_parts.append("historical PE percentile")
                if pb_inverse_score is not None:
                    evidence_parts.append("historical PB percentile")
                if data.get("dividend_yield") is not None:
                    evidence_parts.append("latest dividend yield")
                evidence = " plus ".join(evidence_parts)
            missing_components = []
            if data.get("pe_percentile") is None:
                missing_components.append(("valuation_pe_history_missing", "historical PE percentile missing"))
            if data.get("pb_percentile") is None:
                missing_components.append(("valuation_pb_history_missing", "historical PB percentile missing"))
            for gap_type, impact in missing_components:
                gaps.append(
                    {
                        "object_id": obj["object_id"],
                        "object_name": obj["object_name"],
                        "gap_type": gap_type,
                        "severity": "medium" if available_component_count else "high",
                        "impact": f"confidence capped; {impact}; score uses remaining valuation components only",
                        "repair": "ingest historical PE/PB valuation series with available_date",
                    }
                )
        elif obj["object_type"] == "industry_index":
            data = load_industry_snapshot(obj["object_id"])
            pit_status = data["source_status"]
            pe = data.get("pe_ttm")
            pb = data.get("pb")
            dy = data.get("dividend_yield")
            pe_inverse_score = bounded(30.0 / pe) if pe and pe > 0 else 0.35
            pb_inverse_score = bounded(3.0 / pb) if pb and pb > 0 else 0.35
            dividend_score = bounded(float(dy or 0.0) / 5.0)
            available_component_count = sum(value is not None for value in [pe, pb, dy])
            valuation_repair_score = 0.5 * pe_inverse_score + 0.5 * pb_inverse_score
            fundamental_score = 0.45 * pe_inverse_score + 0.35 * pb_inverse_score + 0.20 * dividend_score
            view = score_to_view(fundamental_score, bullish_cut=0.62, bearish_cut=0.38)
            confidence = 0.38 if data["source_status"] == "current_snapshot_only" else 0.0
            historical_backtest_allowed = False
            valuation_history_backtest_allowed = False
            latest_research_allowed = bool(data["source_status"] == "current_snapshot_only")
            confidence_cap_reason = "current_snapshot_only" if data["source_status"] == "current_snapshot_only" else "missing_industry_snapshot"
            score_usage = "latest_research_only; not allowed as historical backtest feature"
            evidence = "current industry valuation snapshot only"
        else:
            data = {"source_status": "unsupported"}
            pit_status = "missing"
            fundamental_score = None
            view = "blocked"
            confidence = 0.0
            dividend_score = None
            pe_inverse_score = None
            pb_inverse_score = None
        if pit_status in {"missing", "current_snapshot_only"}:
            gaps.append(
                {
                    "object_id": obj["object_id"],
                    "object_name": obj["object_name"],
                    "gap_type": "pit_history_limited" if pit_status == "current_snapshot_only" else "fundamental_data_missing",
                    "severity": "medium" if pit_status == "current_snapshot_only" else "high",
                    "impact": "confidence capped; do not use as historical backtest feature" if pit_status == "current_snapshot_only" else "fundamental view blocked",
                    "repair": "ingest historical industry valuation with available_date" if obj["object_type"] == "industry_index" else "ingest PIT fundamental or valuation history",
                }
            )
        rows.append(
            {
                "object_id": obj["object_id"],
                "object_name": obj["object_name"],
                "object_type": obj["object_type"],
                "fundamental_status": pit_status,
                "pe_ttm": round(float(data["pe_ttm"]), 6) if data.get("pe_ttm") is not None else None,
                "pb": round(float(data["pb"]), 6) if data.get("pb") is not None else None,
                "dividend_yield": round(float(data["dividend_yield"]), 6) if data.get("dividend_yield") is not None else None,
                "pe_percentile": round(float(data["pe_percentile"]), 6) if data.get("pe_percentile") is not None else None,
                "pb_percentile": round(float(data["pb_percentile"]), 6) if data.get("pb_percentile") is not None else None,
                "pe_inverse_score": round(float(pe_inverse_score), 6) if pe_inverse_score is not None else None,
                "pb_inverse_score": round(float(pb_inverse_score), 6) if pb_inverse_score is not None else None,
                "valuation_repair_score": round(float(valuation_repair_score), 6) if valuation_repair_score is not None else None,
                "dividend_score": round(float(dividend_score), 6) if dividend_score is not None else None,
                "available_valuation_component_count": int(available_component_count),
                "fundamental_score": round(float(fundamental_score), 6) if fundamental_score is not None else None,
                "fundamental_view": view,
                "confidence": round(float(confidence), 6),
                "confidence_cap_reason": confidence_cap_reason,
                "historical_backtest_allowed": bool(historical_backtest_allowed),
                "valuation_history_backtest_allowed": bool(valuation_history_backtest_allowed),
                "latest_research_allowed": bool(latest_research_allowed),
                "score_usage": score_usage,
                "valuation_source": data.get("valuation_source", ""),
                "evidence": evidence,
                "failure_scenario": "valuation stays cheap during earnings downgrade or policy shock",
            }
        )
    return rows, gaps


def build_fundamental_formula_spec_rows() -> list[dict[str, Any]]:
    return [
        {
            "formula_id": "market_index_fundamental_v1",
            "object_type": "market_index",
            "component": "valuation_repair_score",
            "weight": 0.72,
            "definition": "average of available inverted historical PE and PB percentiles",
            "source_boundary": "historical PE/PB series; backtests must join by date and available_date",
            "usage": "fundamental_score",
        },
        {
            "formula_id": "market_index_fundamental_v1",
            "object_type": "market_index",
            "component": "dividend_score",
            "weight": 0.28,
            "definition": "bounded dividend yield divided by 5",
            "source_boundary": "latest research context unless PIT dividend history is available",
            "usage": "fundamental_score",
        },
        {
            "formula_id": "industry_snapshot_fundamental_v1",
            "object_type": "industry_index",
            "component": "pe_inverse_score",
            "weight": 0.45,
            "definition": "bounded 30 divided by current PE",
            "source_boundary": "current industry snapshot only; not a historical backtest feature",
            "usage": "fundamental_score",
        },
        {
            "formula_id": "industry_snapshot_fundamental_v1",
            "object_type": "industry_index",
            "component": "pb_inverse_score",
            "weight": 0.35,
            "definition": "bounded 3 divided by current PB",
            "source_boundary": "current industry snapshot only; not a historical backtest feature",
            "usage": "fundamental_score",
        },
        {
            "formula_id": "industry_snapshot_fundamental_v1",
            "object_type": "industry_index",
            "component": "dividend_score",
            "weight": 0.20,
            "definition": "bounded dividend yield divided by 5",
            "source_boundary": "current industry snapshot only; not a historical backtest feature",
            "usage": "fundamental_score",
        },
        {
            "formula_id": "confidence_policy_v1",
            "object_type": "all",
            "component": "confidence_cap_reason",
            "weight": 0.0,
            "definition": "full market PE/PB history caps at 0.78, partial valuation history at 0.68, current snapshots at 0.38",
            "source_boundary": "governance policy",
            "usage": "confidence_cap",
        },
    ]


def build_fundamental_score_reconciliation_rows(fundamental_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in fundamental_rows:
        expected = None
        if row["object_type"] == "market_index" and row.get("valuation_repair_score") is not None:
            expected = 0.72 * float(row["valuation_repair_score"]) + 0.28 * float(row["dividend_score"] or 0.0)
            formula_id = "market_index_fundamental_v1"
        elif row["object_type"] == "industry_index":
            expected = (
                0.45 * float(row["pe_inverse_score"] or 0.0)
                + 0.35 * float(row["pb_inverse_score"] or 0.0)
                + 0.20 * float(row["dividend_score"] or 0.0)
            )
            formula_id = "industry_snapshot_fundamental_v1"
        else:
            formula_id = "unsupported"
        actual = row.get("fundamental_score")
        if expected is None and actual is None:
            abs_diff = 0.0
            status = "pass"
        elif expected is None or actual is None:
            abs_diff = None
            status = "fail"
        else:
            abs_diff = abs(round(float(expected), 6) - float(actual))
            status = "pass" if abs_diff <= 1e-6 else "fail"
        rows.append(
            {
                "object_id": row["object_id"],
                "object_type": row["object_type"],
                "formula_id": formula_id,
                "reported_score": actual,
                "recomputed_score": round(float(expected), 6) if expected is not None else None,
                "abs_diff": round(float(abs_diff), 9) if abs_diff is not None else None,
                "status": status,
            }
        )
    return rows


def build_macro_pit_check_rows(macro_rows: list[dict[str, Any]], asof_date: str) -> list[dict[str, Any]]:
    if not macro_rows:
        return [
            {
                "series_id": "all",
                "check": "macro_rows_present",
                "status": "fail",
                "detail": "no macro rows available as of requested date",
            }
        ]
    asof = pd.to_datetime(asof_date)
    rows: list[dict[str, Any]] = []
    for row in macro_rows:
        available_date = pd.to_datetime(row.get("available_date"), errors="coerce")
        stale_days = row.get("stale_days")
        is_available = pd.notna(available_date) and available_date <= asof
        has_staleness = stale_days is not None and float(stale_days) >= 0
        rows.append(
            {
                "series_id": row.get("series_id", ""),
                "asof_date": asof.date().isoformat(),
                "available_date": row.get("available_date", ""),
                "stale_days": stale_days,
                "is_available_asof": bool(is_available),
                "check": "available_date_not_after_asof_and_staleness_recorded",
                "status": "pass" if is_available and has_staleness else "fail",
                "detail": "macro snapshot can be used as latest context only with available_date boundary",
            }
        )
    return rows


def view_direction(view: Any) -> int:
    value = str(view).strip().lower()
    if value == "bullish":
        return 1
    if value == "bearish":
        return -1
    return 0


def synthesize_views(
    universe: list[dict[str, Any]],
    technical_rows: list[dict[str, Any]],
    fundamental_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tech_by_id = {row["object_id"]: row for row in technical_rows}
    fund_by_id = {row["object_id"]: row for row in fundamental_rows}
    views: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    for obj in universe:
        tech = tech_by_id[obj["object_id"]]
        fund = fund_by_id[obj["object_id"]]
        tech_score = tech.get("technical_score")
        fund_score = fund.get("fundamental_score")
        fund_conf = float(fund.get("confidence") or 0.0)
        tech_conf = float(tech.get("confidence") or 0.0)
        tech_dir = view_direction(tech.get("technical_view"))
        fund_dir = view_direction(fund.get("fundamental_view"))
        hard_conflict = tech_dir * fund_dir == -1
        conflict_score_cap_applied = False
        if tech_score is None:
            final_score = fund_score
            confidence = fund_conf * 0.65
            synthesis_rule = "fundamental_only_due_missing_technical"
        elif fund_score is None:
            final_score = tech_score
            confidence = min(tech_conf, 0.42)
            synthesis_rule = "technical_only_due_missing_fundamental"
        elif hard_conflict and fund_conf >= 0.45:
            final_score = 0.5 + 0.25 * (float(tech_score) - 0.5) + 0.25 * (float(fund_score) - 0.5)
            confidence = min(0.55, 0.45 * tech_conf + 0.45 * fund_conf)
            synthesis_rule = "conflict_penalty_balanced"
        elif hard_conflict:
            final_score = 0.5 + 0.35 * (float(tech_score) - 0.5) + 0.10 * (float(fund_score) - 0.5)
            confidence = min(0.48, 0.65 * tech_conf + 0.20 * fund_conf)
            synthesis_rule = "conflict_penalty_fundamental_low_confidence"
        elif fund_conf < 0.45:
            final_score = 0.72 * float(tech_score) + 0.28 * float(fund_score)
            confidence = min(0.55, 0.72 * tech_conf + 0.28 * fund_conf)
            synthesis_rule = "technical_primary_fundamental_low_confidence"
        else:
            final_score = 0.55 * float(tech_score) + 0.45 * float(fund_score)
            confidence = min(0.82, 0.55 * tech_conf + 0.45 * fund_conf)
            synthesis_rule = "balanced_technical_fundamental"
        raw_synthesis_score = final_score
        if hard_conflict and final_score is not None:
            capped_score = min(0.59, max(0.41, float(final_score)))
            conflict_score_cap_applied = abs(capped_score - float(final_score)) > 1e-12
            final_score = capped_score
        final_view = score_to_view(float(final_score), bullish_cut=0.60, bearish_cut=0.40) if final_score is not None else "blocked"
        action = {
            "bullish": "research_positive_bias_watchlist_only",
            "neutral": "research_wait_for_confirmation_only",
            "bearish": "research_negative_bias_review_only",
            "blocked": "data_repair_first",
        }[final_view]
        cap_reasons = []
        if hard_conflict:
            cap_reasons.append("technical_fundamental_conflict")
        if fund_conf < 0.45:
            cap_reasons.append("fundamental_low_confidence")
        fund_cap_reason = str(fund.get("confidence_cap_reason", "") or "")
        if fund_cap_reason and fund_cap_reason != "none":
            cap_reasons.append(f"fundamental:{fund_cap_reason}")
        tech_cap_reason = str(tech.get("confidence_cap_reason", "") or "")
        if tech_cap_reason and tech_cap_reason != "none":
            cap_reasons.append(f"technical:{tech_cap_reason}")
        confidence_cap_reason = ";".join(cap_reasons) if cap_reasons else "none"
        source_boundary_summary = (
            "research-only latest view; do not use synthesized rows as historical backtest features; "
            "reconstruct governed technical/fundamental components with PIT rules before any model validation"
        )
        views.append(
            {
                "object_id": obj["object_id"],
                "object_name": obj["object_name"],
                "object_type": obj["object_type"],
                "date": tech.get("date", ""),
                "raw_synthesis_score": round(float(raw_synthesis_score), 6) if raw_synthesis_score is not None else None,
                "final_score": round(float(final_score), 6) if final_score is not None else None,
                "final_view": final_view,
                "confidence": round(float(confidence), 6),
                "confidence_cap_reason": confidence_cap_reason,
                "action_bias": action,
                "action_bias_usage": "research_only_not_order_instruction",
                "technical_view": tech.get("technical_view"),
                "technical_score": tech_score,
                "technical_confidence": round(float(tech_conf), 6),
                "technical_confidence_cap_reason": tech.get("confidence_cap_reason", ""),
                "fundamental_view": fund.get("fundamental_view"),
                "fundamental_score": fund_score,
                "fundamental_confidence": round(float(fund_conf), 6),
                "fundamental_confidence_cap_reason": fund.get("confidence_cap_reason", ""),
                "fundamental_score_usage": fund.get("score_usage", ""),
                "fundamental_latest_backtest_allowed": bool(fund.get("historical_backtest_allowed", False)),
                "valuation_history_backtest_allowed": bool(fund.get("valuation_history_backtest_allowed", False)),
                "hard_conflict": hard_conflict,
                "conflict_score_cap_applied": bool(conflict_score_cap_applied),
                "synthesis_rule": synthesis_rule,
                "view_scope": "research_only",
                "historical_backtest_allowed": False,
                "order_instruction_allowed": False,
                "portfolio_weight_allowed": False,
                "strategy_promotion_allowed": False,
                "requires_validation_before_trade": True,
                "source_boundary_summary": source_boundary_summary,
                "invalidation": "break below 60d trend support or valuation thesis receives PIT downgrade",
            }
        )
        trace.append(
            {
                "object_id": obj["object_id"],
                "technical_score": tech_score,
                "technical_view": tech.get("technical_view", ""),
                "technical_confidence": round(float(tech_conf), 6),
                "technical_evidence": tech.get("evidence", ""),
                "fundamental_score": fund_score,
                "fundamental_view": fund.get("fundamental_view", ""),
                "fundamental_confidence": round(float(fund_conf), 6),
                "fundamental_confidence_cap_reason": fund.get("confidence_cap_reason", ""),
                "fundamental_evidence": fund.get("evidence", ""),
                "hard_conflict": hard_conflict,
                "raw_synthesis_score": round(float(raw_synthesis_score), 6) if raw_synthesis_score is not None else None,
                "final_score_after_caps": round(float(final_score), 6) if final_score is not None else None,
                "conflict_score_cap_applied": bool(conflict_score_cap_applied),
                "synthesis_rule": synthesis_rule,
                "confidence_cap_reason": confidence_cap_reason,
                "decision_boundary": "research_only; not an order, portfolio weight, backtest feature, or alpha promotion",
                "known_failure_scenario": f"{tech.get('failure_scenario', '')}; {fund.get('failure_scenario', '')}",
            }
        )
    return views, trace


def build_synthesis_formula_spec_rows() -> list[dict[str, Any]]:
    return [
        {
            "rule": "balanced_technical_fundamental",
            "condition": "technical and fundamental are available, not conflicting, and fundamental confidence >= 0.45",
            "formula": "0.55 * technical_score + 0.45 * fundamental_score",
            "confidence_policy": "min(0.82, 0.55 * technical_confidence + 0.45 * fundamental_confidence)",
            "view_policy": "score_to_view with 0.60/0.40 thresholds",
        },
        {
            "rule": "technical_primary_fundamental_low_confidence",
            "condition": "no hard conflict but fundamental confidence < 0.45",
            "formula": "0.72 * technical_score + 0.28 * fundamental_score",
            "confidence_policy": "min(0.55, 0.72 * technical_confidence + 0.28 * fundamental_confidence)",
            "view_policy": "score_to_view with 0.60/0.40 thresholds",
        },
        {
            "rule": "conflict_penalty_balanced",
            "condition": "technical and fundamental views conflict and fundamental confidence >= 0.45",
            "formula": "0.5 + 0.25 * (technical_score - 0.5) + 0.25 * (fundamental_score - 0.5)",
            "confidence_policy": "min(0.55, 0.45 * technical_confidence + 0.45 * fundamental_confidence)",
            "view_policy": "cap final score to [0.41, 0.59] so hard conflicts stay neutral",
        },
        {
            "rule": "conflict_penalty_fundamental_low_confidence",
            "condition": "technical and fundamental views conflict and fundamental confidence < 0.45",
            "formula": "0.5 + 0.35 * (technical_score - 0.5) + 0.10 * (fundamental_score - 0.5)",
            "confidence_policy": "min(0.48, 0.65 * technical_confidence + 0.20 * fundamental_confidence)",
            "view_policy": "cap final score to [0.41, 0.59] so hard conflicts stay neutral",
        },
        {
            "rule": "technical_only_due_missing_fundamental",
            "condition": "technical score available and fundamental score missing",
            "formula": "technical_score",
            "confidence_policy": "min(technical_confidence, 0.42)",
            "view_policy": "technical-only view remains research-only",
        },
        {
            "rule": "fundamental_only_due_missing_technical",
            "condition": "fundamental score available and technical score missing",
            "formula": "fundamental_score",
            "confidence_policy": "0.65 * fundamental_confidence",
            "view_policy": "fundamental-only view remains research-only",
        },
    ]


def build_synthesis_score_reconciliation_rows(synthesized_views: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in synthesized_views:
        rule = row.get("synthesis_rule")
        tech_score = row.get("technical_score")
        fund_score = row.get("fundamental_score")
        expected = None
        if rule == "fundamental_only_due_missing_technical" and fund_score is not None:
            expected = float(fund_score)
        elif rule == "technical_only_due_missing_fundamental" and tech_score is not None:
            expected = float(tech_score)
        elif rule == "conflict_penalty_balanced" and tech_score is not None and fund_score is not None:
            expected = 0.5 + 0.25 * (float(tech_score) - 0.5) + 0.25 * (float(fund_score) - 0.5)
        elif rule == "conflict_penalty_fundamental_low_confidence" and tech_score is not None and fund_score is not None:
            expected = 0.5 + 0.35 * (float(tech_score) - 0.5) + 0.10 * (float(fund_score) - 0.5)
        elif rule == "technical_primary_fundamental_low_confidence" and tech_score is not None and fund_score is not None:
            expected = 0.72 * float(tech_score) + 0.28 * float(fund_score)
        elif rule == "balanced_technical_fundamental" and tech_score is not None and fund_score is not None:
            expected = 0.55 * float(tech_score) + 0.45 * float(fund_score)
        expected_after_caps = expected
        if expected_after_caps is not None and truthy_cell(row.get("hard_conflict")):
            expected_after_caps = min(0.59, max(0.41, float(expected_after_caps)))
        actual = row.get("final_score")
        if expected_after_caps is None or actual is None:
            abs_diff = None
            status = "fail"
        else:
            abs_diff = abs(round(float(expected_after_caps), 6) - float(actual))
            status = "pass" if abs_diff <= 1e-6 else "fail"
        rows.append(
            {
                "object_id": row["object_id"],
                "synthesis_rule": rule,
                "raw_recomputed_score": round(float(expected), 6) if expected is not None else None,
                "reported_raw_synthesis_score": row.get("raw_synthesis_score"),
                "recomputed_final_score": round(float(expected_after_caps), 6) if expected_after_caps is not None else None,
                "reported_final_score": actual,
                "abs_diff": round(float(abs_diff), 9) if abs_diff is not None else None,
                "status": status,
            }
        )
    return rows


def build_synthesis_input_contract_check_rows(
    universe: list[dict[str, Any]],
    technical_rows: list[dict[str, Any]],
    fundamental_rows: list[dict[str, Any]],
    synthesized_views: list[dict[str, Any]],
    decision_trace: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    tech_by_id = {row["object_id"]: row for row in technical_rows}
    fund_by_id = {row["object_id"]: row for row in fundamental_rows}
    view_by_id = {row["object_id"]: row for row in synthesized_views}
    trace_by_id = {row["object_id"]: row for row in decision_trace}
    rows: list[dict[str, Any]] = []
    forbidden_action_terms = ["buy", "sell", "order", "trade", "long", "reduce", "hold"]
    for obj in universe:
        object_id = obj["object_id"]
        view = view_by_id.get(object_id, {})
        checks = {
            "technical_input_present": object_id in tech_by_id,
            "fundamental_input_present": object_id in fund_by_id,
            "synthesized_view_present": object_id in view_by_id,
            "decision_trace_present": object_id in trace_by_id,
            "view_scope_research_only": view.get("view_scope") == "research_only",
            "not_historical_backtest_feature": not truthy_cell(view.get("historical_backtest_allowed")),
            "not_order_instruction": not truthy_cell(view.get("order_instruction_allowed")),
            "not_portfolio_weight": not truthy_cell(view.get("portfolio_weight_allowed")),
            "action_bias_is_research_only": not any(term in str(view.get("action_bias", "")).lower() for term in forbidden_action_terms),
            "hard_conflict_stays_neutral": (not truthy_cell(view.get("hard_conflict"))) or str(view.get("final_view")) == "neutral",
        }
        failed = [name for name, ok in checks.items() if not ok]
        rows.append(
            {
                "object_id": object_id,
                "object_name": obj["object_name"],
                "status": "pass" if not failed else "fail",
                "failed_checks": ";".join(failed),
                "check_count": len(checks),
            }
        )
    return rows


def build_architecture_rows() -> list[dict[str, Any]]:
    return [
        {
            "agent": "chief_quant_orchestrator",
            "role_class": "governance",
            "new_responsibility": "route object-level research requests and separate research views from tradable strategy changes",
            "fixed_input": "task brief, research object, horizon, allowed data sources",
            "required_output": "research plan, acceptance checklist, handoff target",
            "forbidden": "promote views to portfolio rules without validation",
        },
        {
            "agent": "data_steward",
            "role_class": "data",
            "new_responsibility": "maintain PIT data coverage map for prices, valuation, macro, fundamentals, and corporate actions",
            "fixed_input": "source contracts, raw data paths, available_date policy",
            "required_output": "coverage matrix, data gap register, lineage refs",
            "forbidden": "use current snapshots as historical features",
        },
        {
            "agent": "factor_researcher",
            "role_class": "research",
            "new_responsibility": "convert repeated research-assistant evidence into predeclared factor hypotheses only after data coverage is approved",
            "fixed_input": "approved data, object-level views, hypothesis brief",
            "required_output": "factor card, observation diagnostics, candidate or rejection decision",
            "forbidden": "treat one-off assistant views as validated factors",
        },
        {
            "agent": "regime_timing_researcher",
            "role_class": "research",
            "new_responsibility": "map object-level views to regime and timing hypotheses without bypassing holdout validation",
            "fixed_input": "market state data, macro context, object-level view history",
            "required_output": "regime diagnostics, state definitions, timing failure modes",
            "forbidden": "promote timing rules from visual agreement or same-sample fit",
        },
        {
            "agent": "technical_market_analyst",
            "role_class": "research_assistant",
            "new_responsibility": "produce trend, momentum, drawdown, volatility, and liquidity evidence for indexes, sectors, and stocks",
            "fixed_input": "clean OHLCV or close-only PIT series",
            "required_output": "technical signal panel and latest view",
            "forbidden": "use forward returns or optimize thresholds on the evaluation sample",
        },
        {
            "agent": "fundamental_equity_analyst",
            "role_class": "research_assistant",
            "new_responsibility": "produce valuation, dividend, growth, quality, and revision views with confidence tied to PIT coverage",
            "fixed_input": "PIT valuation/fundamental data with available_date",
            "required_output": "fundamental signal panel and data limitations",
            "forbidden": "backfill latest fundamentals into history",
        },
        {
            "agent": "investment_view_synthesizer",
            "role_class": "research_assistant",
            "new_responsibility": "combine technical, fundamental, macro, and risk evidence into one object-level research view",
            "fixed_input": "agent outputs only, no private scratch context",
            "required_output": "final view, confidence, action bias, invalidation, failure scenarios",
            "forbidden": "hide conflicts between agents",
        },
        {
            "agent": "portfolio_risk_engineer",
            "role_class": "portfolio",
            "new_responsibility": "consume assistant views only after they become governed strategy signals; enforce exposure, turnover, drawdown, and cash rules",
            "fixed_input": "validated signals, constraints, cost assumptions",
            "required_output": "target weights, constraint checks, exposure report",
            "forbidden": "turn research-only views into weights",
        },
        {
            "agent": "backtest_validation_auditor",
            "role_class": "validation",
            "new_responsibility": "audit whether assistant-derived hypotheses use point-in-time data, avoid leakage, and survive independent validation",
            "fixed_input": "candidate artifacts, source rows, baseline, validation design",
            "required_output": "leakage checklist, robustness report, validation decision",
            "forbidden": "promote alpha or overwrite research conclusions",
        },
        {
            "agent": "execution_cost_analyst",
            "role_class": "implementation",
            "new_responsibility": "review whether assistant-derived model candidates remain viable after turnover, liquidity, spread, and capacity costs",
            "fixed_input": "trade list, target weights, liquidity data, cost grid",
            "required_output": "cost sensitivity and capacity flags",
            "forbidden": "ignore high turnover because research views look plausible",
        },
        {
            "agent": "research_reporter",
            "role_class": "reporting",
            "new_responsibility": "render structured research outputs into readable reports and HTML dashboards",
            "fixed_input": "synthesized views, traces, limitations",
            "required_output": "markdown report, HTML dashboard, reproducibility manifest",
            "forbidden": "turn low-confidence outputs into high-conviction language",
        },
        {
            "agent": "code_quality_engineer",
            "role_class": "quality",
            "new_responsibility": "verify assistant scripts, manifests, output paths, schemas, and role-roster consistency",
            "fixed_input": "code, configs, output directories, manifests, task board",
            "required_output": "smoke checks, output integrity checks, framework check result",
            "forbidden": "certify research validity or alpha quality",
        },
    ]


def build_architecture_governance_coverage_rows() -> list[dict[str, Any]]:
    checks = [
        ("agent_roster", ROOT / "strategy_lab" / "agent_framework_check.py", "registered 12-agent roster and task-brief owner checks"),
        ("readme_extension", ROOT / "strategy_lab" / "agents" / "README.md", "documents research assistant extension and default work order"),
        ("raci_extension", ROOT / "strategy_lab" / "agents" / "RACI_MATRIX.md", "documents research assistant RACI responsibilities"),
        ("io_contract_extension", ROOT / "strategy_lab" / "agents" / "AGENT_IO_CONTRACT.md", "documents artifacts and research-only boundary"),
        ("workflow_extension", ROOT / "strategy_lab" / "agents" / "AGENT_WORKFLOW.md", "documents research view phase gate"),
        ("technical_agent_spec", ROOT / "strategy_lab" / "agents" / "technical_market_analyst" / "AGENT.md", "fixed inputs, outputs, forbidden actions, gates"),
        ("fundamental_agent_spec", ROOT / "strategy_lab" / "agents" / "fundamental_equity_analyst" / "AGENT.md", "fixed inputs, outputs, forbidden actions, gates"),
        ("synthesizer_agent_spec", ROOT / "strategy_lab" / "agents" / "investment_view_synthesizer" / "AGENT.md", "fixed inputs, outputs, forbidden actions, gates"),
    ]
    return [
        {
            "coverage_item": item,
            "path": rel_path(path),
            "exists": path.exists(),
            "purpose": purpose,
        }
        for item, path, purpose in checks
    ]


def build_object_schema_rows() -> list[dict[str, Any]]:
    return [
        {"field": "object_id", "type": "string", "required": True, "description": "stable security, index, sector, ETF, or portfolio identifier"},
        {"field": "object_name", "type": "string", "required": True, "description": "display name resolved from source when available"},
        {"field": "object_type", "type": "enum", "required": True, "description": "market_index, industry_index, etf, stock, or portfolio"},
        {"field": "sleeve", "type": "string", "required": True, "description": "research grouping such as large_cap, broad_market, or industry sleeve"},
        {"field": "horizon", "type": "enum", "required": True, "description": "intraday, 1w, 1m_6m, 1y_plus"},
        {"field": "asof_date", "type": "date", "required": True, "description": "research date; all data must be available on or before this date"},
        {"field": "price_path", "type": "path", "required": True, "description": "primary price source path"},
        {"field": "price_status", "type": "enum", "required": True, "description": "usable, limited_history, or missing"},
        {"field": "row_count", "type": "integer", "required": True, "description": "number of usable positive close rows"},
        {"field": "start_date", "type": "date", "required": True, "description": "first usable price date"},
        {"field": "end_date", "type": "date", "required": True, "description": "latest usable price date"},
        {"field": "data_source", "type": "string", "required": True, "description": "source adapter or provider label from the primary price file"},
        {"field": "technical_view", "type": "enum", "required": False, "description": "bullish, neutral, bearish, or blocked"},
        {"field": "fundamental_view", "type": "enum", "required": False, "description": "bullish, neutral, bearish, or blocked"},
        {"field": "final_view", "type": "enum", "required": False, "description": "synthesized research view, not a trade order"},
        {"field": "confidence", "type": "float", "required": False, "description": "0 to 1 confidence capped by data coverage and agreement"},
        {"field": "invalidation", "type": "string", "required": False, "description": "conditions that would invalidate the research view"},
    ]


def build_research_object_contract_check_rows(schema_rows: list[dict[str, Any]], universe: list[dict[str, Any]]) -> list[dict[str, Any]]:
    universe_fields = set(universe[0].keys()) if universe else set()
    rows: list[dict[str, Any]] = []
    for field in [row["field"] for row in schema_rows if str(row.get("required", "")).lower() == "true"]:
        present = field in universe_fields
        non_empty = present and all(str(obj.get(field, "")).strip() != "" for obj in universe)
        rows.append(
            {
                "check": f"required_field:{field}",
                "field": field,
                "status": "pass" if present and non_empty else "fail",
                "detail": "field exists and is non-empty for all sample objects" if present and non_empty else "required schema field missing or empty in sample universe",
            }
        )
    for obj in universe:
        path = ROOT / obj["price_path"]
        rows.append(
            {
                "check": f"price_path_exists:{obj['object_id']}",
                "field": "price_path",
                "status": "pass" if path.exists() else "fail",
                "detail": obj["price_path"],
            }
        )
    return rows


def build_technical_formula_spec_rows() -> list[dict[str, Any]]:
    return [
        {
            "component": "ma_stack",
            "weight": 0.38,
            "usage": "directional_score",
            "definition": "average of close>MA20, MA20>MA60, and MA60>MA120",
            "rationale": "captures medium-term trend alignment",
            "failure_mode": "range-bound whipsaw",
        },
        {
            "component": "momentum_score",
            "weight": 0.34,
            "usage": "directional_score",
            "definition": "bounded blend of 20d and 60d returns with symmetric clipping",
            "rationale": "captures recent participation and continuation",
            "failure_mode": "late-cycle acceleration reversal",
        },
        {
            "component": "drawdown_repair_score",
            "weight": 0.18,
            "usage": "directional_score",
            "definition": "1 plus 120d drawdown clipped between -35pct and 0",
            "rationale": "penalizes unresolved deep drawdowns",
            "failure_mode": "bear-market rebound before base repair",
        },
        {
            "component": "liquidity_score",
            "weight": 0.10,
            "usage": "directional_score",
            "definition": "20d amount average divided by 120d amount average, clipped from 0.5 to 1.5 then shifted",
            "rationale": "requires activity confirmation",
            "failure_mode": "volume spike from panic or one-off policy event",
        },
        {
            "component": "volatility_context",
            "weight": 0.0,
            "usage": "confidence_and_risk_overlay",
            "definition": "20d annualized volatility percentile over a 252-day rolling window; elevated/high states cap confidence",
            "rationale": "volatility is not directional by itself, but high-volatility breakouts and breakdowns are less stable",
            "failure_mode": "low volatility can precede expansion and high volatility can mark capitulation",
        },
        {
            "component": "data_quality_status",
            "weight": 0.0,
            "usage": "confidence_overlay",
            "definition": "latest row is classified as full_ohlc, close_only, or unknown; close-only bars reduce confidence",
            "rationale": "close-only data limits validation of intraday range, wick, and true liquidity behavior",
            "failure_mode": "index source metadata can understate usable information in some vendor feeds",
        },
    ]


def build_technical_input_contract_check_rows(universe: list[dict[str, Any]], technical_latest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    universe_ids = {row["object_id"] for row in universe}
    latest_ids = {row["object_id"] for row in technical_latest}
    rows: list[dict[str, Any]] = [
        {
            "check": "object_set_matches_v3_87_universe",
            "status": "pass" if universe_ids == latest_ids else "fail",
            "detail": f"universe={sorted(universe_ids)} latest={sorted(latest_ids)}",
        },
        {
            "check": "latest_row_count_matches_universe",
            "status": "pass" if len(universe) == len(technical_latest) else "fail",
            "detail": f"universe={len(universe)} latest={len(technical_latest)}",
        },
    ]
    by_id = {row["object_id"]: row for row in technical_latest}
    for obj in universe:
        latest = by_id.get(obj["object_id"], {})
        rows.append(
            {
                "check": f"asof_date_aligned:{obj['object_id']}",
                "status": "pass" if latest.get("date") == obj.get("asof_date") else "fail",
                "detail": f"universe_asof={obj.get('asof_date')} latest_date={latest.get('date')}",
            }
        )
        rows.append(
            {
                "check": f"score_available:{obj['object_id']}",
                "status": "pass" if latest.get("technical_score") is not None else "fail",
                "detail": "technical_score exists" if latest.get("technical_score") is not None else "technical_score missing",
            }
        )
    return rows


def build_data_coverage_rows(universe: list[dict[str, Any]], macro_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for obj in universe:
        rows.append(
            {
                "object_id": obj["object_id"],
                "dataset": "price_daily",
                "status": obj["price_status"],
                "path": obj["price_path"],
                "coverage_start": obj["start_date"],
                "coverage_end": obj["end_date"],
                "available_date_field": "date",
                "pit_policy": "daily market data; no forward labels used",
                "research_use_status": "approved_for_latest_and_historical_research" if obj["price_status"] == "usable" else "blocked_or_limited",
                "downstream_allowed": obj["price_status"] == "usable",
                "limitation": "close-only early rows may limit intraday/OHLC research" if obj["object_type"] == "market_index" else "industry index source, not constituent-level PIT data",
                "repair_action": "ingest full OHLCV/vendor adjusted source if strategy validation needs richer fields",
            }
        )
        if obj["object_type"] == "market_index":
            valuation = load_index_valuation(obj["object_id"])
            valuation_parts = [
                ("valuation_pe_history", valuation.get("pe_source", ""), valuation.get("pe_ttm") is not None),
                ("valuation_pb_history", valuation.get("pb_source", ""), valuation.get("pb") is not None),
            ]
            for dataset, source_path, usable in valuation_parts:
                rows.append(
                    {
                        "object_id": obj["object_id"],
                        "dataset": dataset,
                        "status": "historical_available" if usable else "missing",
                        "path": source_path,
                        "coverage_start": "",
                        "coverage_end": obj["end_date"],
                        "available_date_field": "date",
                        "pit_policy": "historical valuation only; latest snapshot is not backfilled",
                        "research_use_status": "approved_for_latest_research" if usable else "blocked_missing_source",
                        "downstream_allowed": usable,
                        "limitation": "valuation date is treated as available on market date; use vendor release timing before backtest promotion",
                        "repair_action": "ingest vendor PIT valuation with explicit available_date for strict historical factor tests",
                    }
                )
        elif obj["object_type"] == "industry_index":
            rows.append(
                {
                    "object_id": obj["object_id"],
                    "dataset": "industry_valuation",
                    "status": "current_snapshot_only",
                    "path": "data_raw/index/akshare_sw_industry/analysis_latest_sw/analysis_latest_combined.csv",
                    "coverage_start": "",
                    "coverage_end": obj["end_date"],
                    "available_date_field": "requested_analysis_date/resolved_analysis_date",
                    "pit_policy": "research-only snapshot until historical available_date series is ingested",
                    "research_use_status": "latest_research_only_not_historical_backtest",
                    "downstream_allowed": False,
                    "limitation": "current snapshot cannot be used as historical feature",
                    "repair_action": "ingest historical industry valuation with explicit available_date",
                }
            )
    rows.append(
        {
            "object_id": "macro",
            "dataset": "macro_pit_panel",
            "status": "usable" if macro_rows else "missing",
            "path": "data_raw/macro/macro_pit_panel.csv",
            "coverage_start": min((row["date"] for row in macro_rows), default=""),
            "coverage_end": max((row["date"] for row in macro_rows), default=""),
            "available_date_field": "available_date",
            "pit_policy": "available_date as-of join required",
            "research_use_status": "approved_for_asof_research",
            "downstream_allowed": bool(macro_rows),
            "limitation": "macro panel is event-release snapshot without vintage revision history for some series",
            "repair_action": "add vintage histories when strict macro backtests require revision-aware data",
        }
    )
    return rows


def render_markdown_report(views: list[dict[str, Any]], gaps: list[dict[str, Any]], macro_rows: list[dict[str, Any]]) -> str:
    hard_conflict_count = sum(truthy_cell(row.get("hard_conflict")) for row in views)
    blocked_trade_count = sum(not truthy_cell(row.get("order_instruction_allowed")) for row in views)
    lines = [
        "# Quant Research Assistant Report",
        "",
        "This report is an object-level research view. It is not a portfolio backtest, order recommendation, portfolio weight, or alpha promotion.",
        "",
        "## Use Boundary",
        "",
        f"- Objects covered: {len(views)}.",
        f"- Hard technical/fundamental conflicts: {hard_conflict_count}.",
        f"- Rows blocked from order use: {blocked_trade_count}.",
        "- Latest synthesized rows are not historical backtest features; rebuild governed inputs by date and available_date before validation.",
        "",
        "## Synthesized Views",
        "",
        "| Object | Type | View | Raw Score | Final Score | Confidence | Conflict | Cap Reason | Usage |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in views:
        lines.append(
            f"| {row['object_id']} {row['object_name']} | {row['object_type']} | {row['final_view']} | {row.get('raw_synthesis_score')} | {row['final_score']} | {row['confidence']} | {row.get('hard_conflict')} | {row.get('confidence_cap_reason')} | {row.get('action_bias_usage')} |"
        )
    lines.extend(
        [
            "",
            "## Macro PIT Snapshot",
            "",
            "| Series | As Of | Date | Available Date | Stale Days | Value | PIT Quality | Usage |",
            "|---|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in macro_rows[:20]:
        lines.append(
            f"| {row['series_id']} | {row.get('asof_date', '')} | {row['date']} | {row['available_date']} | {row.get('stale_days', '')} | {row['value']} | {row['pit_quality']} | {row.get('macro_usage', '')} |"
        )
    lines.extend(["", "## Data Gaps", "", "| Object | Gap | Severity | Impact | Repair |", "|---|---|---:|---|---|"])
    if gaps:
        for gap in gaps:
            lines.append(f"| {gap['object_id']} {gap['object_name']} | {gap['gap_type']} | {gap['severity']} | {gap.get('impact', '')} | {gap['repair']} |")
    else:
        lines.append("| none | none | low | none | none |")
    return "\n".join(lines) + "\n"


def _render_html_dashboard_legacy(views: list[dict[str, Any]], technical: list[dict[str, Any]], fundamental: list[dict[str, Any]], gaps: list[dict[str, Any]]) -> str:
    tech_by_id = {row["object_id"]: row for row in technical}
    fund_by_id = {row["object_id"]: row for row in fundamental}

    def bar(score: Any) -> str:
        if score is None or pd.isna(score):
            return '<span class="muted">n/a</span>'
        width = max(4, min(100, int(float(score) * 100)))
        return f'<span class="bar"><span style="width:{width}%"></span></span><span class="score">{float(score):.2f}</span>'

    cards = []
    for row in views:
        tech = tech_by_id.get(row["object_id"], {})
        fund = fund_by_id.get(row["object_id"], {})
        view_class = html.escape(str(row["final_view"]))
        cards.append(
            f"""
      <article class="item">
        <div class="item-head">
          <div>
            <h2>{html.escape(str(row["object_id"]))} {html.escape(str(row["object_name"]))}</h2>
            <p>{html.escape(str(row["object_type"]))} · {html.escape(str(row["date"]))}</p>
          </div>
          <span class="badge {view_class}">{html.escape(str(row["final_view"]))}</span>
        </div>
        <dl>
          <div><dt>Final</dt><dd>{bar(row.get("final_score"))}</dd></div>
          <div><dt>Technical</dt><dd>{bar(tech.get("technical_score"))}</dd></div>
          <div><dt>Fundamental</dt><dd>{bar(fund.get("fundamental_score"))}</dd></div>
          <div><dt>Confidence</dt><dd>{bar(row.get("confidence"))}</dd></div>
        </dl>
        <p class="bias">{html.escape(str(row["action_bias"]))}</p>
        <p class="note">{html.escape(str(row["synthesis_rule"]))}</p>
      </article>
            """
        )
    gap_rows = "\n".join(
        f"<tr><td>{html.escape(str(g['object_id']))}</td><td>{html.escape(str(g['gap_type']))}</td><td>{html.escape(str(g['severity']))}</td><td>{html.escape(str(g['repair']))}</td></tr>"
        for g in gaps
    ) or '<tr><td colspan="4">No material data gaps.</td></tr>'
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Quant Research Assistant V3.91</title>
  <style>
    :root {{ color-scheme: light; --ink:#16202a; --muted:#5e6b78; --line:#d9e0e7; --bg:#f6f8fa; --panel:#ffffff; --good:#0f7b5f; --bad:#a53b3b; --mid:#5c6570; --accent:#2457a6; }}
    body {{ margin:0; font-family: Arial, Helvetica, sans-serif; background:var(--bg); color:var(--ink); }}
    main {{ max-width:1180px; margin:0 auto; padding:24px; }}
    header {{ display:flex; align-items:flex-end; justify-content:space-between; gap:16px; border-bottom:1px solid var(--line); padding-bottom:16px; }}
    h1 {{ font-size:26px; line-height:1.2; margin:0 0 6px; letter-spacing:0; }}
    p {{ margin:0; color:var(--muted); line-height:1.5; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:14px; margin:20px 0; }}
    .item {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px; }}
    .item-head {{ display:flex; justify-content:space-between; gap:12px; align-items:flex-start; margin-bottom:14px; }}
    h2 {{ font-size:16px; margin:0 0 4px; letter-spacing:0; }}
    .badge {{ display:inline-flex; min-width:72px; justify-content:center; padding:5px 8px; border-radius:6px; color:#fff; font-size:13px; text-transform:capitalize; }}
    .badge.bullish {{ background:var(--good); }} .badge.bearish {{ background:var(--bad); }} .badge.neutral {{ background:var(--mid); }} .badge.blocked {{ background:#7a4a00; }}
    dl {{ display:grid; gap:10px; margin:0; }}
    dl div {{ display:grid; grid-template-columns:88px 1fr; gap:8px; align-items:center; }}
    dt {{ color:var(--muted); font-size:13px; }}
    dd {{ margin:0; display:flex; align-items:center; gap:8px; min-width:0; }}
    .bar {{ height:8px; background:#e8edf2; border-radius:999px; overflow:hidden; flex:1; min-width:80px; }}
    .bar span {{ display:block; height:100%; background:var(--accent); }}
    .score {{ font-variant-numeric:tabular-nums; font-size:13px; color:var(--ink); min-width:34px; }}
    .bias {{ margin-top:12px; color:var(--ink); font-size:13px; }}
    .note {{ margin-top:4px; font-size:12px; }}
    table {{ width:100%; border-collapse:collapse; background:var(--panel); border:1px solid var(--line); border-radius:8px; overflow:hidden; }}
    th,td {{ padding:9px 10px; border-bottom:1px solid var(--line); text-align:left; font-size:13px; vertical-align:top; }}
    th {{ color:var(--muted); font-weight:600; }}
    @media (max-width:640px) {{ main {{ padding:14px; }} header {{ display:block; }} dl div {{ grid-template-columns:76px 1fr; }} }}
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>Quant Research Assistant V3.91</h1>
      <p>Object-level technical and fundamental research view. This is not a backtest or an order system.</p>
    </div>
    <p>{len(views)} objects · {len(gaps)} data gaps</p>
  </header>
  <section class="grid">
    {''.join(cards)}
  </section>
  <section>
    <h2>Data Gaps</h2>
    <table><thead><tr><th>Object</th><th>Gap</th><th>Severity</th><th>Repair</th></tr></thead><tbody>{gap_rows}</tbody></table>
  </section>
</main>
</body>
</html>
"""


def render_html_dashboard(
    views: list[dict[str, Any]],
    technical: list[dict[str, Any]],
    fundamental: list[dict[str, Any]],
    gaps: list[dict[str, Any]],
    macro_rows: list[dict[str, Any]] | None = None,
) -> str:
    tech_by_id = {row["object_id"]: row for row in technical}
    fund_by_id = {row["object_id"]: row for row in fundamental}
    macro_rows = macro_rows or []

    def bar(score: Any) -> str:
        if score is None or pd.isna(score):
            return '<span class="muted">n/a</span>'
        width = max(4, min(100, int(float(score) * 100)))
        return f'<span class="bar"><span style="width:{width}%"></span></span><span class="score">{float(score):.2f}</span>'

    hard_conflict_count = sum(truthy_cell(row.get("hard_conflict")) for row in views)
    neutral_conflict_count = sum(truthy_cell(row.get("hard_conflict")) and str(row.get("final_view")) == "neutral" for row in views)
    blocked_order_count = sum(not truthy_cell(row.get("order_instruction_allowed")) for row in views)

    cards = []
    for row in views:
        tech = tech_by_id.get(row["object_id"], {})
        fund = fund_by_id.get(row["object_id"], {})
        view_class = html.escape(str(row["final_view"]))
        conflict_label = "Hard conflict" if truthy_cell(row.get("hard_conflict")) else "Aligned or mixed"
        use_label = "No order / no backtest" if not truthy_cell(row.get("order_instruction_allowed")) else "Review boundary"
        cards.append(
            f"""
      <article class="item">
        <div class="item-head">
          <div>
            <h2>{html.escape(str(row["object_id"]))} {html.escape(str(row["object_name"]))}</h2>
            <p>{html.escape(str(row["object_type"]))} · {html.escape(str(row["date"]))}</p>
          </div>
          <span class="badge {view_class}">{html.escape(str(row["final_view"]))}</span>
        </div>
        <div class="tags">
          <span>{html.escape(conflict_label)}</span>
          <span>{html.escape(use_label)}</span>
          <span>{html.escape(str(row.get("view_scope", "research_only")))}</span>
        </div>
        <dl>
          <div><dt>Raw</dt><dd>{bar(row.get("raw_synthesis_score"))}</dd></div>
          <div><dt>Final</dt><dd>{bar(row.get("final_score"))}</dd></div>
          <div><dt>Technical</dt><dd>{bar(tech.get("technical_score"))}</dd></div>
          <div><dt>Fundamental</dt><dd>{bar(fund.get("fundamental_score"))}</dd></div>
          <div><dt>Confidence</dt><dd>{bar(row.get("confidence"))}</dd></div>
        </dl>
        <p class="bias">{html.escape(str(row["action_bias"]))}</p>
        <p class="note">{html.escape(str(row["synthesis_rule"]))}</p>
        <p class="reason">{html.escape(str(row.get("confidence_cap_reason", "none")))}</p>
      </article>
            """
        )

    gap_rows = "\n".join(
        f"<tr><td>{html.escape(str(g['object_id']))} {html.escape(str(g.get('object_name', '')))}</td><td>{html.escape(str(g['gap_type']))}</td><td>{html.escape(str(g['severity']))}</td><td>{html.escape(str(g.get('impact', '')))}</td><td>{html.escape(str(g['repair']))}</td></tr>"
        for g in gaps
    ) or '<tr><td colspan="5">No material data gaps.</td></tr>'
    macro_table_rows = "\n".join(
        f"<tr><td>{html.escape(str(row.get('series_id', '')))}</td><td>{html.escape(str(row.get('asof_date', '')))}</td><td>{html.escape(str(row.get('date', '')))}</td><td>{html.escape(str(row.get('available_date', '')))}</td><td>{html.escape(str(row.get('stale_days', '')))}</td><td>{html.escape(str(row.get('value', '')))}</td><td>{html.escape(str(row.get('pit_quality', '')))}</td></tr>"
        for row in macro_rows[:20]
    ) or '<tr><td colspan="7">No macro PIT rows available.</td></tr>'

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Quant Research Assistant V3.91</title>
  <style>
    :root {{ color-scheme: light; --ink:#16202a; --muted:#5e6b78; --line:#d9e0e7; --bg:#f6f8fa; --panel:#ffffff; --soft:#eef3f7; --good:#0f7b5f; --bad:#a53b3b; --mid:#5c6570; --accent:#2457a6; --warn:#8a5a00; }}
    body {{ margin:0; font-family: Arial, Helvetica, sans-serif; background:var(--bg); color:var(--ink); }}
    main {{ max-width:1180px; margin:0 auto; padding:24px; }}
    header {{ display:flex; align-items:flex-end; justify-content:space-between; gap:16px; border-bottom:1px solid var(--line); padding-bottom:16px; }}
    h1 {{ font-size:26px; line-height:1.2; margin:0 0 6px; letter-spacing:0; }}
    p {{ margin:0; color:var(--muted); line-height:1.5; }}
    .boundary {{ margin:16px 0 0; padding:12px 14px; border:1px solid #d8c27b; background:#fff8df; border-radius:8px; color:#3f3100; font-size:13px; line-height:1.5; }}
    .stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:10px; margin:18px 0 4px; }}
    .stat {{ background:var(--soft); border:1px solid var(--line); border-radius:8px; padding:10px 12px; }}
    .stat b {{ display:block; font-size:20px; line-height:1.1; margin-bottom:4px; }}
    .stat span {{ color:var(--muted); font-size:12px; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:14px; margin:20px 0; }}
    .item {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px; }}
    .item-head {{ display:flex; justify-content:space-between; gap:12px; align-items:flex-start; margin-bottom:14px; }}
    h2 {{ font-size:16px; margin:0 0 4px; letter-spacing:0; }}
    .badge {{ display:inline-flex; min-width:72px; justify-content:center; padding:5px 8px; border-radius:6px; color:#fff; font-size:13px; text-transform:capitalize; }}
    .badge.bullish {{ background:var(--good); }} .badge.bearish {{ background:var(--bad); }} .badge.neutral {{ background:var(--mid); }} .badge.blocked {{ background:#7a4a00; }}
    .tags {{ display:flex; flex-wrap:wrap; gap:6px; margin:0 0 12px; }}
    .tags span {{ background:var(--soft); border:1px solid var(--line); border-radius:6px; padding:3px 6px; color:var(--ink); font-size:12px; }}
    dl {{ display:grid; gap:10px; margin:0; }}
    dl div {{ display:grid; grid-template-columns:92px 1fr; gap:8px; align-items:center; }}
    dt {{ color:var(--muted); font-size:13px; }}
    dd {{ margin:0; display:flex; align-items:center; gap:8px; min-width:0; }}
    .bar {{ height:8px; background:#e8edf2; border-radius:999px; overflow:hidden; flex:1; min-width:80px; }}
    .bar span {{ display:block; height:100%; background:var(--accent); }}
    .score {{ font-variant-numeric:tabular-nums; font-size:13px; color:var(--ink); min-width:34px; }}
    .bias {{ margin-top:12px; color:var(--ink); font-size:13px; }}
    .note {{ margin-top:4px; font-size:12px; }}
    .reason {{ margin-top:6px; color:var(--warn); font-size:12px; overflow-wrap:anywhere; }}
    section {{ margin-top:20px; }}
    .table-wrap {{ overflow:auto; border:1px solid var(--line); border-radius:8px; background:var(--panel); }}
    table {{ width:100%; border-collapse:collapse; background:var(--panel); border:1px solid var(--line); border-radius:8px; overflow:hidden; }}
    th,td {{ padding:9px 10px; border-bottom:1px solid var(--line); text-align:left; font-size:13px; vertical-align:top; }}
    th {{ color:var(--muted); font-weight:600; }}
    td {{ overflow-wrap:anywhere; }}
    @media (max-width:640px) {{ main {{ padding:14px; }} header {{ display:block; }} dl div {{ grid-template-columns:82px 1fr; }} .grid {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>Quant Research Assistant V3.91</h1>
      <p>Object-level technical and fundamental research view. This is not a backtest or an order system.</p>
    </div>
    <p>{len(views)} objects · {len(gaps)} data gaps</p>
  </header>
  <div class="boundary">Research-only latest view. Do not treat these rows as orders, portfolio weights, historical backtest features, or alpha promotion evidence. Reconstruct governed inputs with PIT rules before validation.</div>
  <section class="stats" aria-label="report summary">
    <div class="stat"><b>{len(views)}</b><span>Objects</span></div>
    <div class="stat"><b>{hard_conflict_count}</b><span>Hard conflicts</span></div>
    <div class="stat"><b>{neutral_conflict_count}</b><span>Conflicts capped neutral</span></div>
    <div class="stat"><b>{blocked_order_count}</b><span>Rows blocked from order use</span></div>
  </section>
  <section class="grid">
    {''.join(cards)}
  </section>
  <section>
    <h2>Data Gaps</h2>
    <div class="table-wrap"><table><thead><tr><th>Object</th><th>Gap</th><th>Severity</th><th>Impact</th><th>Repair</th></tr></thead><tbody>{gap_rows}</tbody></table></div>
  </section>
  <section>
    <h2>Macro PIT Snapshot</h2>
    <div class="table-wrap"><table><thead><tr><th>Series</th><th>As Of</th><th>Date</th><th>Available</th><th>Stale Days</th><th>Value</th><th>PIT Quality</th></tr></thead><tbody>{macro_table_rows}</tbody></table></div>
  </section>
</main>
</body>
</html>
"""
