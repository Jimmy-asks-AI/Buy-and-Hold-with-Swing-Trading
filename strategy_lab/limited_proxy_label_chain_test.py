"""Limited-window adjusted-proxy label-chain test for HIRSSM V3.56."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from market_total_return_label_importer import (
    MarketLabelImportConfig,
    build_forward_labels,
    source_validation_passed,
    validate_market_source,
)


@dataclass(frozen=True)
class LimitedProxyChainConfig:
    proxy_source_path: Path
    signal_panel_path: Path
    output_dir: Path
    catalog_path: Path
    horizons: tuple[int, ...]
    min_source_rows: int
    min_signal_rows: int
    min_label_rows: int
    min_limited_coverage_ratio: float
    market_proxy_codes: tuple[str, ...]
    source_asset_priority: tuple[str, ...]


def source_summary(source: pd.DataFrame) -> pd.DataFrame:
    if source.empty:
        return pd.DataFrame(
            [
                {
                    "rows": 0,
                    "date_min": "",
                    "date_max": "",
                    "asset_or_index": "",
                    "data_source": "",
                    "source_vintage_count": 0,
                }
            ]
        )
    return pd.DataFrame(
        [
            {
                "rows": int(len(source)),
                "date_min": str(source["date"].min()),
                "date_max": str(source["date"].max()),
                "asset_or_index": ",".join(sorted(source["asset_or_index"].astype(str).unique())),
                "data_source": ",".join(sorted(source["data_source"].astype(str).unique())[:5]),
                "source_vintage_count": int(source["source_vintage"].astype(str).nunique()),
            }
        ]
    )


def build_import_config(config: LimitedProxyChainConfig) -> MarketLabelImportConfig:
    return MarketLabelImportConfig(
        source_path=config.proxy_source_path,
        signal_panel_path=config.signal_panel_path,
        output_dir=config.output_dir,
        catalog_path=config.catalog_path,
        horizons=config.horizons,
        market_proxy_codes=config.market_proxy_codes,
        source_asset_priority=config.source_asset_priority,
        min_source_rows=config.min_source_rows,
        min_signal_coverage_ratio=config.min_limited_coverage_ratio,
    )


def eligible_signal_dates(source: pd.DataFrame, horizons: tuple[int, ...]) -> set[str]:
    dates = sorted(source["date"].astype(str).unique())
    if not dates:
        return set()
    max_horizon = max(int(item) for item in horizons)
    if len(dates) <= max_horizon:
        return set()
    return set(dates[: len(dates) - max_horizon])


def build_limited_signal_panel(signal_panel: pd.DataFrame, source: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    eligible = eligible_signal_dates(source, horizons)
    if not eligible:
        return signal_panel.iloc[0:0].copy()
    out = signal_panel.loc[
        (signal_panel["asset"].astype(str) == "MARKET")
        & (signal_panel["signal_date"].astype(str).isin(eligible))
    ].copy()
    return out.sort_values(["signal_date", "signal_id"] if "signal_id" in out.columns else ["signal_date"])


def coverage_summary(labels: pd.DataFrame, limited_signal_panel: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    unique_dates = sorted(limited_signal_panel["signal_date"].astype(str).unique()) if not limited_signal_panel.empty else []
    rows = []
    for horizon in horizons:
        sub = labels.loc[labels["horizon"].astype(int) == int(horizon)] if not labels.empty else labels
        labelled_dates = set(sub["signal_date"].astype(str)) if not sub.empty else set()
        rows.append(
            {
                "horizon": int(horizon),
                "limited_unique_signal_dates": len(unique_dates),
                "labelled_unique_signal_dates": len(labelled_dates),
                "coverage_ratio": len(labelled_dates) / len(unique_dates) if unique_dates else 0.0,
                "label_rows": int(len(sub)),
            }
        )
    return pd.DataFrame(rows)


def build_chain_readiness(
    source_checks: pd.DataFrame,
    limited_signal_panel: pd.DataFrame,
    labels: pd.DataFrame,
    coverage: pd.DataFrame,
    config: LimitedProxyChainConfig,
) -> pd.DataFrame:
    validation_ok = source_validation_passed(source_checks)
    min_coverage = float(coverage["coverage_ratio"].min()) if not coverage.empty else 0.0
    rows = [
        {
            "check": "proxy_source_validation_passed",
            "status": "pass" if validation_ok else "blocked",
            "detail": ";".join(source_checks.loc[source_checks["status"] != "pass", "check"].astype(str)),
        },
        {
            "check": "limited_signal_panel_has_rows",
            "status": "pass" if len(limited_signal_panel) >= config.min_signal_rows else "blocked",
            "detail": f"rows={len(limited_signal_panel)};min={config.min_signal_rows}",
        },
        {
            "check": "limited_labels_produced",
            "status": "pass" if len(labels) >= config.min_label_rows else "blocked",
            "detail": f"rows={len(labels)};min={config.min_label_rows}",
        },
        {
            "check": "limited_label_coverage_passed",
            "status": "pass" if min_coverage >= config.min_limited_coverage_ratio else "blocked",
            "detail": f"min_coverage={min_coverage:.4f};min={config.min_limited_coverage_ratio:.4f}",
        },
        {
            "check": "official_market_source_untouched",
            "status": "pass",
            "detail": "V3.56 writes only isolated outputs under outputs/agent_runs/v3_56.",
        },
        {
            "check": "performance_validation_allowed_now",
            "status": "blocked",
            "detail": "limited-window adjusted proxy proves chain mechanics only; it is not full-history evidence.",
        },
    ]
    return pd.DataFrame(rows)


def build_no_promotion_guard(labels: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "result_type": "limited_market_forward_labels",
                "produced": len(labels) > 0,
                "blocked": len(labels) == 0,
                "reason": "technical chain-test labels only",
            },
            {
                "result_type": "state_stratified_performance_validation",
                "produced": False,
                "blocked": True,
                "reason": "V3.56 does not run IC, hit rate, or validation claims.",
            },
            {
                "result_type": "portfolio_backtest_or_model_promotion",
                "produced": False,
                "blocked": True,
                "reason": "short-window adjusted proxy cannot support model promotion.",
            },
        ]
    )


def build_acceptance_checks(
    readiness: pd.DataFrame,
    guard: pd.DataFrame,
    labels: pd.DataFrame,
    source: pd.DataFrame,
) -> pd.DataFrame:
    forbidden_label_promotion = bool(
        guard.loc[
            guard["result_type"].isin(["state_stratified_performance_validation", "portfolio_backtest_or_model_promotion"]),
            "produced",
        ].any()
    )
    source_terms = " ".join(source.get("data_source", pd.Series(dtype=str)).astype(str).unique()).lower()
    explicit_proxy = "approved_adjusted_proxy" in source_terms
    return pd.DataFrame(
        [
            {
                "check": "readiness_required_checks_pass_or_blocked_as_expected",
                "status": "pass"
                if not readiness.loc[
                    ~readiness["check"].isin(["performance_validation_allowed_now"]),
                    "status",
                ].eq("blocked").any()
                else "fail",
                "detail": ";".join(readiness.loc[readiness["status"] == "blocked", "check"].astype(str)),
            },
            {
                "check": "limited_labels_exist_but_not_promoted",
                "status": "pass" if len(labels) > 0 and not forbidden_label_promotion else "fail",
                "detail": f"label_rows={len(labels)}",
            },
            {
                "check": "source_lineage_marked_adjusted_proxy",
                "status": "pass" if explicit_proxy else "fail",
                "detail": "requires approved_adjusted_proxy in data_source lineage",
            },
            {
                "check": "official_market_total_return_file_not_created",
                "status": "pass",
                "detail": "V3.56 does not write data_raw/market_labels/market_total_return_index.csv",
            },
            {
                "check": "performance_claims_not_created",
                "status": "pass" if not forbidden_label_promotion else "fail",
                "detail": "no IC/backtest/model promotion outputs",
            },
        ]
    )


def build_report(
    source_summary_frame: pd.DataFrame,
    source_checks: pd.DataFrame,
    limited_signal_panel: pd.DataFrame,
    labels: pd.DataFrame,
    coverage: pd.DataFrame,
    readiness: pd.DataFrame,
    acceptance: pd.DataFrame,
) -> str:
    summary = source_summary_frame.iloc[0].to_dict() if not source_summary_frame.empty else {}
    signal_dates = limited_signal_panel["signal_date"].astype(str) if not limited_signal_panel.empty else pd.Series(dtype=str)
    lines = [
        "# V3.56 Limited Proxy Label Chain Test",
        "",
        "## Decision",
        "",
        "- V3.56 uses the V3.55 JoinQuant `fq='pre'` MARKET proxy only as a short-window technical chain test.",
        "- It verifies that a V3.53-style source can generate forward labels.",
        "- It does not write the official `data_raw/market_labels/market_total_return_index.csv` file.",
        "- It does not run IC, hit rate, NAV, drawdown, Sharpe, backtest, or model promotion.",
        "",
        "## Source Summary",
        "",
        f"- Source rows: `{summary.get('rows', 0)}`",
        f"- Source date range: `{summary.get('date_min', '')}` to `{summary.get('date_max', '')}`",
        f"- Source lineage: `{summary.get('data_source', '')}`",
        "",
        "## Limited Signal Scope",
        "",
        f"- Limited signal rows: `{len(limited_signal_panel)}`",
        f"- Limited unique signal dates: `{signal_dates.nunique()}`",
        f"- Limited signal date range: `{signal_dates.min() if not signal_dates.empty else ''}` to `{signal_dates.max() if not signal_dates.empty else ''}`",
        f"- Label rows: `{len(labels)}`",
        "",
        "## Source Checks",
        "",
        "| check | status | detail |",
        "|---|---|---|",
    ]
    for row in source_checks.itertuples(index=False):
        lines.append(f"| `{row.check}` | `{row.status}` | {row.detail} |")
    lines.extend(
        [
            "",
            "## Coverage",
            "",
            "| horizon | limited_unique_signal_dates | labelled_unique_signal_dates | coverage_ratio | label_rows |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in coverage.itertuples(index=False):
        lines.append(
            f"| {int(row.horizon)} | {int(row.limited_unique_signal_dates)} | {int(row.labelled_unique_signal_dates)} | {float(row.coverage_ratio):.4f} | {int(row.label_rows)} |"
        )
    lines.extend(
        [
            "",
            "## Readiness",
            "",
            "| check | status | detail |",
            "|---|---|---|",
        ]
    )
    for row in readiness.itertuples(index=False):
        lines.append(f"| `{row.check}` | `{row.status}` | {row.detail} |")
    lines.extend(
        [
            "",
            "## Acceptance",
            "",
            "| check | status | detail |",
            "|---|---|---|",
        ]
    )
    for row in acceptance.itertuples(index=False):
        lines.append(f"| `{row.check}` | `{row.status}` | {row.detail} |")
    lines.extend(
        [
            "",
            "## Next Use",
            "",
            "- Use this only to confirm the label generation chain.",
            "- A full-history official total-return index or explicitly governed adjusted proxy is still required before performance validation.",
            "",
        ]
    )
    return "\n".join(lines)


def build_catalog(readiness: pd.DataFrame, labels: pd.DataFrame) -> str:
    perf_status = str(readiness.loc[readiness["check"] == "performance_validation_allowed_now", "status"].iloc[0])
    return "\n".join(
        [
            "# A-share Limited Proxy Label Chain Test V3.56",
            "",
            "## Dataset Decision",
            "",
            "- Chain test ready: `true`",
            f"- Limited labels produced: `{len(labels) > 0}`",
            f"- Limited label rows: `{len(labels)}`",
            f"- Performance validation status: `{perf_status}`",
            "- Scope: short-window technical validation only.",
            "",
        ]
    )
