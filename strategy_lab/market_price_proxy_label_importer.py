"""Guarded MARKET price-proxy label importer for HIRSSM V3.59.

The importer creates forward labels from the V3.58 long MARKET price-index
proxy. It is deliberately separate from the official total-return importer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_PROXY_COLUMNS = {
    "date",
    "asset_or_index",
    "source_symbol",
    "market_level",
    "available_date",
    "data_source",
    "source_vintage",
    "source_type",
    "return_basis_candidate",
    "price_adjustment_status",
    "official_total_return",
    "proxy_label_generation_allowed",
    "official_label_generation_allowed",
    "model_promotion_allowed",
    "performance_claim_allowed",
}

REQUIRED_SIGNAL_COLUMNS = {
    "signal_id",
    "signal_date",
    "asset",
    "signal_value",
    "signal_direction",
    "available_date",
    "model_promotion_allowed",
    "performance_claim_allowed",
}

LABEL_COLUMNS = [
    "signal_date",
    "asset",
    "horizon",
    "forward_price_index_return",
    "return_basis",
    "label_available_date",
    "price_proxy_source",
    "source_symbol",
    "official_total_return",
    "proxy_label_generation_allowed",
    "official_label_generation_allowed",
    "model_promotion_allowed",
    "performance_claim_allowed",
    "diagnostic_usage",
]

FORBIDDEN_OUTPUT_TERMS = {
    "rank_ic",
    "icir",
    "hit_rate",
    "sharpe",
    "nav",
    "annualized_return",
    "portfolio_return",
    "max_drawdown",
    "deflated_sharpe",
    "pbo",
}


@dataclass(frozen=True)
class PriceProxyLabelConfig:
    proxy_source_path: Path
    signal_panel_path: Path
    v3_58_manifest_path: Path
    output_dir: Path
    catalog_path: Path
    canonical_label_path: Path
    official_total_return_source_path: Path
    horizons: tuple[int, ...]
    source_symbol: str
    min_proxy_rows: int
    min_signal_rows: int
    min_label_rows: int
    min_source_window_coverage_ratio: float
    min_all_signal_coverage_ratio: float


def normalize_date(values: pd.Series) -> pd.Series:
    cleaned = (
        values.astype(str)
        .str.strip()
        .str.replace("-", "", regex=False)
        .str.replace("/", "", regex=False)
        .str.replace(".0", "", regex=False)
    )
    return pd.to_datetime(cleaned, format="%Y%m%d", errors="coerce")


def format_date(values: pd.Series) -> pd.Series:
    return normalize_date(values).dt.strftime("%Y%m%d")


def _bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    return values.astype(str).str.lower().isin({"true", "1", "yes"})


def _status(ok: bool, fail_status: str = "blocked") -> str:
    return "pass" if ok else fail_status


def empty_labels() -> pd.DataFrame:
    return pd.DataFrame(columns=LABEL_COLUMNS)


def validate_proxy_source(proxy: pd.DataFrame, config: PriceProxyLabelConfig) -> pd.DataFrame:
    missing = sorted(REQUIRED_PROXY_COLUMNS.difference(proxy.columns))
    rows: list[dict[str, Any]] = [
        {
            "check": "proxy_required_columns_present",
            "status": _status(not missing),
            "detail": ",".join(missing),
        },
        {
            "check": "proxy_minimum_rows",
            "status": _status(len(proxy) >= config.min_proxy_rows),
            "detail": f"rows={len(proxy)};min={config.min_proxy_rows}",
        },
    ]
    if missing:
        return pd.DataFrame(rows)

    dates = format_date(proxy["date"])
    available = format_date(proxy["available_date"])
    levels = pd.to_numeric(proxy["market_level"], errors="coerce")
    source_symbols = set(proxy["source_symbol"].astype(str).unique())
    basis = set(proxy["return_basis_candidate"].astype(str).unique())
    source_type = set(proxy["source_type"].astype(str).unique())
    official_total_return = _bool_series(proxy["official_total_return"])
    proxy_allowed = _bool_series(proxy["proxy_label_generation_allowed"])
    official_allowed = _bool_series(proxy["official_label_generation_allowed"])
    model_allowed = _bool_series(proxy["model_promotion_allowed"])
    performance_allowed = _bool_series(proxy["performance_claim_allowed"])
    duplicate_dates = int(dates.duplicated().sum())
    lineage = " ".join(proxy["data_source"].astype(str).unique()).lower()
    forbidden_official_terms = any(term in lineage for term in ["total_return_index", "official_total_return"])

    rows.extend(
        [
            {
                "check": "proxy_dates_parseable",
                "status": _status(dates.notna().all()),
                "detail": f"bad_rows={int(dates.isna().sum())}",
            },
            {
                "check": "proxy_available_date_not_before_date",
                "status": _status((available >= dates).all()),
                "detail": f"bad_rows={int((available < dates).sum())}",
            },
            {
                "check": "proxy_market_level_positive_finite",
                "status": _status(np.isfinite(levels).all() and (levels > 0).all()),
                "detail": f"bad_rows={int((~np.isfinite(levels) | (levels <= 0)).sum())}",
            },
            {
                "check": "proxy_no_duplicate_dates",
                "status": _status(duplicate_dates == 0),
                "detail": f"duplicates={duplicate_dates}",
            },
            {
                "check": "proxy_source_symbol_expected",
                "status": _status(source_symbols == {config.source_symbol}),
                "detail": ",".join(sorted(source_symbols)),
            },
            {
                "check": "proxy_basis_is_price_index_return",
                "status": _status(basis == {"price_index_return"}),
                "detail": ",".join(sorted(basis)),
            },
            {
                "check": "proxy_source_type_expected",
                "status": _status(source_type == {"csindex_price_index_proxy"}),
                "detail": ",".join(sorted(source_type)),
            },
            {
                "check": "proxy_not_official_total_return",
                "status": _status(not official_total_return.any() and not official_allowed.any()),
                "detail": f"official_total_return_any={bool(official_total_return.any())};official_allowed_any={bool(official_allowed.any())}",
            },
            {
                "check": "proxy_label_generation_allowed",
                "status": _status(proxy_allowed.all()),
                "detail": f"false_rows={int((~proxy_allowed).sum())}",
            },
            {
                "check": "proxy_model_and_performance_flags_false",
                "status": _status(not model_allowed.any() and not performance_allowed.any()),
                "detail": f"model_allowed={bool(model_allowed.any())};performance_allowed={bool(performance_allowed.any())}",
            },
            {
                "check": "proxy_lineage_not_official_total_return",
                "status": _status(not forbidden_official_terms),
                "detail": "lineage must remain price-index proxy lineage",
            },
        ]
    )
    return pd.DataFrame(rows)


def validate_signal_panel(signals: pd.DataFrame, config: PriceProxyLabelConfig) -> pd.DataFrame:
    missing = sorted(REQUIRED_SIGNAL_COLUMNS.difference(signals.columns))
    rows: list[dict[str, Any]] = [
        {
            "check": "signal_required_columns_present",
            "status": _status(not missing),
            "detail": ",".join(missing),
        },
        {
            "check": "signal_minimum_rows",
            "status": _status(len(signals) >= config.min_signal_rows),
            "detail": f"rows={len(signals)};min={config.min_signal_rows}",
        },
    ]
    if missing:
        return pd.DataFrame(rows)
    signal_dates = format_date(signals["signal_date"])
    available = format_date(signals["available_date"])
    assets = set(signals["asset"].astype(str).unique())
    model_allowed = _bool_series(signals["model_promotion_allowed"])
    performance_allowed = _bool_series(signals["performance_claim_allowed"])
    rows.extend(
        [
            {
                "check": "signal_dates_parseable",
                "status": _status(signal_dates.notna().all()),
                "detail": f"bad_rows={int(signal_dates.isna().sum())}",
            },
            {
                "check": "signal_available_date_not_after_signal_date",
                "status": _status((available <= signal_dates).all()),
                "detail": f"bad_rows={int((available > signal_dates).sum())}",
            },
            {
                "check": "signal_scope_is_market_only",
                "status": _status(assets == {"MARKET"}),
                "detail": ",".join(sorted(assets)),
            },
            {
                "check": "signal_model_and_performance_flags_false",
                "status": _status(not model_allowed.any() and not performance_allowed.any()),
                "detail": f"model_allowed={bool(model_allowed.any())};performance_allowed={bool(performance_allowed.any())}",
            },
        ]
    )
    return pd.DataFrame(rows)


def prepare_proxy(proxy: pd.DataFrame) -> pd.DataFrame:
    out = proxy.copy()
    out["_date"] = format_date(out["date"])
    out["_available_date"] = format_date(out["available_date"])
    out["_level"] = pd.to_numeric(out["market_level"], errors="coerce")
    return out.sort_values("_date").drop_duplicates("_date", keep="last").reset_index(drop=True)


def build_price_proxy_forward_labels(
    signal_panel: pd.DataFrame,
    proxy: pd.DataFrame,
    config: PriceProxyLabelConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if signal_panel.empty or proxy.empty:
        return empty_labels(), pd.DataFrame()

    prepared = prepare_proxy(proxy)
    signals = signal_panel.loc[signal_panel["asset"].astype(str) == "MARKET"].copy()
    unique_signal_dates = sorted(format_date(signals["signal_date"]).dropna().unique())
    source_dates = prepared["_date"].astype(str).tolist()
    source_date_set = set(source_dates)
    source_min = min(source_dates) if source_dates else ""
    source_max = max(source_dates) if source_dates else ""
    source_window_signal_dates = [date for date in unique_signal_dates if source_min <= date <= source_max]
    pos_by_date = {date: idx for idx, date in enumerate(source_dates)}

    rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    source_symbol = str(prepared["source_symbol"].iloc[0])
    lineage = (
        str(prepared["data_source"].iloc[0])
        + "|"
        + str(prepared["source_vintage"].iloc[0])
        + "|"
        + source_symbol
        + "|price_proxy_label_v3_59"
    )
    for horizon in config.horizons:
        matched = 0
        enough_future = 0
        label_available_ok = 0
        for signal_date in unique_signal_dates:
            pos = pos_by_date.get(signal_date)
            if pos is None:
                continue
            matched += 1
            future_pos = pos + int(horizon)
            if future_pos >= len(prepared):
                continue
            enough_future += 1
            current_level = float(prepared.loc[pos, "_level"])
            future_level = float(prepared.loc[future_pos, "_level"])
            label_available_date = str(prepared.loc[future_pos, "_available_date"])
            if label_available_date <= signal_date:
                continue
            label_available_ok += 1
            rows.append(
                {
                    "signal_date": signal_date,
                    "asset": "MARKET",
                    "horizon": int(horizon),
                    "forward_price_index_return": future_level / current_level - 1.0,
                    "return_basis": "price_index_return",
                    "label_available_date": label_available_date,
                    "price_proxy_source": lineage,
                    "source_symbol": source_symbol,
                    "official_total_return": False,
                    "proxy_label_generation_allowed": True,
                    "official_label_generation_allowed": False,
                    "model_promotion_allowed": False,
                    "performance_claim_allowed": False,
                    "diagnostic_usage": "non_official_price_proxy_label_only",
                }
            )
        coverage_rows.append(
            {
                "horizon": int(horizon),
                "all_unique_signal_dates": len(unique_signal_dates),
                "source_window_signal_dates": len(source_window_signal_dates),
                "matched_source_dates": matched,
                "enough_future_dates": enough_future,
                "label_available_date_ok": label_available_ok,
                "coverage_all_signal_dates": label_available_ok / max(len(unique_signal_dates), 1),
                "coverage_source_window_signal_dates": label_available_ok / max(len(source_window_signal_dates), 1),
                "source_date_min": source_min,
                "source_date_max": source_max,
            }
        )
    return pd.DataFrame(rows, columns=LABEL_COLUMNS), pd.DataFrame(coverage_rows)


def build_label_contract(labels: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "column": "signal_date",
                "required": True,
                "rule": "signal date, YYYYMMDD, joined to MARKET signal panel",
            },
            {
                "column": "forward_price_index_return",
                "required": True,
                "rule": "future MARKET price-index proxy level / current level - 1",
            },
            {
                "column": "return_basis",
                "required": True,
                "rule": "must equal price_index_return; never total_return",
            },
            {
                "column": "official_total_return",
                "required": True,
                "rule": "must remain false",
            },
            {
                "column": "diagnostic_usage",
                "required": True,
                "rule": "must state non_official_price_proxy_label_only",
            },
        ]
    )


def build_label_diagnostics(labels: pd.DataFrame, config: PriceProxyLabelConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if labels.empty:
        return pd.DataFrame()
    for horizon, group in labels.groupby("horizon", dropna=False):
        values = pd.to_numeric(group["forward_price_index_return"], errors="coerce").dropna()
        rows.append(
            {
                "horizon": int(horizon),
                "label_rows": int(len(group)),
                "unique_signal_dates": int(group["signal_date"].astype(str).nunique()),
                "finite_rows": int(values.shape[0]),
                "return_min": float(values.min()) if not values.empty else np.nan,
                "return_p01": float(values.quantile(0.01)) if not values.empty else np.nan,
                "return_p99": float(values.quantile(0.99)) if not values.empty else np.nan,
                "return_max": float(values.max()) if not values.empty else np.nan,
                "diagnostic_type": "distribution_sanity_not_performance_validation",
            }
        )
    return pd.DataFrame(rows).sort_values("horizon")


def build_readiness_checks(
    proxy_checks: pd.DataFrame,
    signal_checks: pd.DataFrame,
    labels: pd.DataFrame,
    coverage: pd.DataFrame,
    v3_58_manifest: dict[str, Any],
    config: PriceProxyLabelConfig,
) -> pd.DataFrame:
    min_source_window_coverage = float(coverage["coverage_source_window_signal_dates"].min()) if not coverage.empty else 0.0
    min_all_signal_coverage = float(coverage["coverage_all_signal_dates"].min()) if not coverage.empty else 0.0
    rows = [
        {
            "check": "v3_58_manifest_accepted",
            "status": _status(bool(v3_58_manifest.get("self_check_pass")) and bool(v3_58_manifest.get("metrics", {}).get("long_price_proxy_written"))),
            "detail": f"self_check={v3_58_manifest.get('self_check_pass')};proxy_written={v3_58_manifest.get('metrics', {}).get('long_price_proxy_written')}",
        },
        {
            "check": "proxy_source_checks_passed",
            "status": _status(proxy_checks["status"].eq("pass").all()),
            "detail": ";".join(proxy_checks.loc[proxy_checks["status"] != "pass", "check"].astype(str)),
        },
        {
            "check": "signal_contract_checks_passed",
            "status": _status(signal_checks["status"].eq("pass").all()),
            "detail": ";".join(signal_checks.loc[signal_checks["status"] != "pass", "check"].astype(str)),
        },
        {
            "check": "price_proxy_label_rows_produced",
            "status": _status(len(labels) >= config.min_label_rows),
            "detail": f"label_rows={len(labels)};min={config.min_label_rows}",
        },
        {
            "check": "source_window_coverage_passed",
            "status": _status(min_source_window_coverage >= config.min_source_window_coverage_ratio),
            "detail": f"min_coverage={min_source_window_coverage:.4f};required={config.min_source_window_coverage_ratio:.4f}",
        },
        {
            "check": "all_signal_coverage_documented",
            "status": _status(min_all_signal_coverage >= config.min_all_signal_coverage_ratio, "warn"),
            "detail": f"min_coverage={min_all_signal_coverage:.4f};required={config.min_all_signal_coverage_ratio:.4f};pre_2005_signals_are_unlabelled",
        },
        {
            "check": "performance_validation_allowed_now",
            "status": "blocked",
            "detail": "V3.59 creates non-official price-proxy labels only; no IC, hit-rate, or model promotion",
        },
    ]
    return pd.DataFrame(rows)


def build_no_promotion_guard(labels: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "result_type": "market_price_proxy_forward_labels",
                "produced": len(labels) > 0,
                "blocked": len(labels) == 0,
                "reason": "non-official price-index proxy labels",
            },
            {
                "result_type": "official_market_total_return_labels",
                "produced": False,
                "blocked": True,
                "reason": "V3.59 does not use or create total-return labels",
            },
            {
                "result_type": "state_stratified_signal_validation",
                "produced": False,
                "blocked": True,
                "reason": "validation must be a separate guarded V3.60 task",
            },
            {
                "result_type": "portfolio_backtest_or_model_promotion",
                "produced": False,
                "blocked": True,
                "reason": "labels are not performance evidence",
            },
        ]
    )


def build_acceptance_checks(
    readiness: pd.DataFrame,
    labels: pd.DataFrame,
    guard: pd.DataFrame,
    official_total_return_source_exists: bool,
    output_column_names: list[str],
) -> pd.DataFrame:
    unexpected_blocked = readiness.loc[
        (readiness["status"] == "blocked") & (~readiness["check"].isin(["performance_validation_allowed_now"]))
    ]
    label_basis = set(labels.get("return_basis", pd.Series(dtype=str)).astype(str).unique())
    official_flags = _bool_series(labels.get("official_total_return", pd.Series(False, index=labels.index))) if not labels.empty else pd.Series(dtype=bool)
    forbidden_columns = sorted({term for term in FORBIDDEN_OUTPUT_TERMS if term in " ".join(output_column_names).lower()})
    forbidden_produced = bool(
        guard.loc[
            guard["result_type"].isin(
                ["official_market_total_return_labels", "state_stratified_signal_validation", "portfolio_backtest_or_model_promotion"]
            ),
            "produced",
        ].any()
    )
    return pd.DataFrame(
        [
            {
                "check": "all_non_performance_readiness_checks_passed",
                "status": "pass" if unexpected_blocked.empty else "fail",
                "detail": ";".join(unexpected_blocked["check"].astype(str)),
            },
            {
                "check": "labels_are_price_index_return_only",
                "status": "pass" if label_basis == {"price_index_return"} else "fail",
                "detail": ",".join(sorted(label_basis)),
            },
            {
                "check": "labels_not_official_total_return",
                "status": "pass" if len(labels) > 0 and not official_flags.any() else "fail",
                "detail": f"official_true_rows={int(official_flags.sum()) if not official_flags.empty else 0}",
            },
            {
                "check": "official_total_return_source_not_created",
                "status": "pass" if not official_total_return_source_exists else "fail",
                "detail": "data_raw/market_labels/market_total_return_index.csv",
            },
            {
                "check": "performance_outputs_not_produced",
                "status": "pass" if not forbidden_produced and not forbidden_columns else "fail",
                "detail": ",".join(forbidden_columns),
            },
        ]
    )


def markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 16) -> list[str]:
    if frame.empty:
        return ["_No rows._"]
    safe = frame.loc[:, [col for col in columns if col in frame.columns]].head(max_rows)
    lines = ["| " + " | ".join(safe.columns) + " |", "| " + " | ".join(["---"] * len(safe.columns)) + " |"]
    for row in safe.itertuples(index=False):
        rendered = []
        for value in row:
            if isinstance(value, float):
                rendered.append(f"{value:.4f}")
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return lines


def build_report(
    proxy: pd.DataFrame,
    signal_panel: pd.DataFrame,
    labels: pd.DataFrame,
    coverage: pd.DataFrame,
    diagnostics: pd.DataFrame,
    readiness: pd.DataFrame,
    acceptance: pd.DataFrame,
    config: PriceProxyLabelConfig,
) -> str:
    proxy_dates = format_date(proxy["date"]) if not proxy.empty else pd.Series(dtype=str)
    signal_dates = format_date(signal_panel["signal_date"]) if not signal_panel.empty else pd.Series(dtype=str)
    label_dates = labels["signal_date"].astype(str) if not labels.empty else pd.Series(dtype=str)
    lines = [
        "# V3.59 MARKET Price-Proxy Label Importer",
        "",
        "## Decision",
        "",
        "- V3.59 builds forward labels from the V3.58 `000985.CSI` price-index proxy.",
        "- Labels use `return_basis=price_index_return`; they are not official total-return labels.",
        "- It does not run IC, hit rate, state validation, portfolio NAV, Sharpe, or model promotion.",
        "",
        "## Input And Output Scope",
        "",
        f"- Proxy rows: `{len(proxy)}`",
        f"- Proxy date range: `{proxy_dates.min() if not proxy_dates.empty else ''}` to `{proxy_dates.max() if not proxy_dates.empty else ''}`",
        f"- Signal rows: `{len(signal_panel)}`",
        f"- Signal date range: `{signal_dates.min() if not signal_dates.empty else ''}` to `{signal_dates.max() if not signal_dates.empty else ''}`",
        f"- Label rows: `{len(labels)}`",
        f"- Label date range: `{label_dates.min() if not label_dates.empty else ''}` to `{label_dates.max() if not label_dates.empty else ''}`",
        f"- Canonical label path: `{config.canonical_label_path}`",
        "",
        "## Coverage",
        "",
    ]
    lines.extend(
        markdown_table(
            coverage,
            [
                "horizon",
                "all_unique_signal_dates",
                "source_window_signal_dates",
                "label_available_date_ok",
                "coverage_all_signal_dates",
                "coverage_source_window_signal_dates",
            ],
        )
    )
    lines.extend(["", "## Label Distribution Sanity", ""])
    lines.extend(markdown_table(diagnostics, ["horizon", "label_rows", "unique_signal_dates", "return_min", "return_p99", "return_max"]))
    lines.extend(["", "## Readiness", ""])
    lines.extend(markdown_table(readiness, ["check", "status", "detail"], max_rows=20))
    lines.extend(["", "## Acceptance", ""])
    lines.extend(markdown_table(acceptance, ["check", "status", "detail"], max_rows=20))
    lines.extend(
        [
            "",
            "## Next Use",
            "",
            "- V3.60 may run a guarded state-stratified validation using these proxy labels only.",
            "- Any V3.60 result must be labelled non-official price-index proxy evidence.",
            "- Formal dividend-inclusive or investable total-return validation remains blocked until official total-return data arrives.",
        ]
    )
    return "\n".join(lines)


def build_catalog(config: PriceProxyLabelConfig, labels: pd.DataFrame) -> str:
    label_dates = labels["signal_date"].astype(str) if not labels.empty else pd.Series(dtype=str)
    return "\n".join(
        [
            "# A-share MARKET Price-Proxy Label Importer V3.59",
            "",
            "## Dataset Role",
            "",
            "V3.59 creates MARKET-level forward labels from the V3.58 long price-index proxy.",
            "",
            "## Governance",
            "",
            "- Return basis: `price_index_return` only.",
            "- Official total-return status: false.",
            "- Allowed use: proxy-labelled diagnostics and future guarded state validation.",
            "- Forbidden use: official total-return validation, dividend-inclusive performance claims, portfolio promotion, or default model promotion.",
            "",
            "## Produced Shape",
            "",
            f"- Label rows: `{len(labels)}`",
            f"- Label date range: `{label_dates.min() if not label_dates.empty else ''}` to `{label_dates.max() if not label_dates.empty else ''}`",
            f"- Canonical label file: `{config.canonical_label_path}`",
        ]
    )
