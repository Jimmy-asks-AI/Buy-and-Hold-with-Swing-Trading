"""Narrow walk-forward proxy review for HIRSSM V3.64.

V3.64 reviews only V3.63 survivor signal-horizon rows. It uses rolling
five-calendar-year train windows and one-year test windows against non-official
price-index proxy labels. Results are proxy diagnostics only: no portfolio
backtest, NAV, official total-return evidence, or model promotion.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from state_stratified_proxy_validation import FORBIDDEN_PROMOTION_TERMS, _corr, normalize_date


@dataclass(frozen=True)
class NarrowWalkForwardProxyConfig:
    joined_panel_path: Path
    candidate_decision_path: Path
    v3_63_manifest_path: Path
    output_dir: Path
    catalog_path: Path
    train_years: int
    test_years: int
    min_train_observations: int
    min_test_observations: int
    min_proxy_spearman: float
    min_proxy_qspread: float
    min_top_directional_alignment: float
    min_gated_windows: int
    min_oos_pass_rate: float
    min_oos_median_spearman: float
    min_oos_positive_qspread_share: float
    top_quantile: float
    bottom_quantile: float


def _status(ok: bool, fail_status: str = "fail") -> str:
    return "pass" if ok else fail_status


def _bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    return values.astype(str).str.lower().isin({"true", "1", "yes"})


def _bucket_metrics(group: pd.DataFrame, config: NarrowWalkForwardProxyConfig) -> dict[str, Any]:
    clean = group.dropna(subset=["signal_value", "expected_signed_proxy_return"]).copy()
    observations = int(len(clean))
    if observations == 0:
        return {
            "observations": 0,
            "proxy_spearman_corr": np.nan,
            "proxy_pearson_corr": np.nan,
            "proxy_qspread_top_minus_bottom": np.nan,
            "top_bucket_directional_alignment_share": np.nan,
            "directional_alignment_share": np.nan,
            "expected_signed_proxy_mean": np.nan,
            "top_bucket_rows": 0,
            "bottom_bucket_rows": 0,
        }
    top_cut = clean["signal_value"].quantile(config.top_quantile)
    bottom_cut = clean["signal_value"].quantile(config.bottom_quantile)
    top = clean.loc[clean["signal_value"] >= top_cut]
    bottom = clean.loc[clean["signal_value"] <= bottom_cut]
    top_mean = float(top["expected_signed_proxy_return"].mean()) if not top.empty else np.nan
    bottom_mean = float(bottom["expected_signed_proxy_return"].mean()) if not bottom.empty else np.nan
    return {
        "observations": observations,
        "proxy_spearman_corr": _corr(clean["signal_value"], clean["expected_signed_proxy_return"], "spearman"),
        "proxy_pearson_corr": _corr(clean["signal_value"], clean["expected_signed_proxy_return"], "pearson"),
        "proxy_qspread_top_minus_bottom": top_mean - bottom_mean if pd.notna(top_mean) and pd.notna(bottom_mean) else np.nan,
        "top_bucket_directional_alignment_share": float(top["active_directional_alignment"].mean()) if not top.empty else np.nan,
        "directional_alignment_share": float(clean["active_directional_alignment"].mean()),
        "expected_signed_proxy_mean": float(clean["expected_signed_proxy_return"].mean()),
        "top_bucket_rows": int(len(top)),
        "bottom_bucket_rows": int(len(bottom)),
    }


def _gate_metrics(metrics: dict[str, Any], min_observations: int, config: NarrowWalkForwardProxyConfig) -> tuple[bool, str]:
    reasons: list[str] = []
    if int(metrics["observations"]) < min_observations:
        reasons.append("insufficient_observations")
    if pd.isna(metrics["proxy_spearman_corr"]) or float(metrics["proxy_spearman_corr"]) < config.min_proxy_spearman:
        reasons.append("spearman_below_threshold")
    if pd.isna(metrics["proxy_qspread_top_minus_bottom"]) or float(metrics["proxy_qspread_top_minus_bottom"]) <= config.min_proxy_qspread:
        reasons.append("qspread_not_positive")
    if (
        pd.isna(metrics["top_bucket_directional_alignment_share"])
        or float(metrics["top_bucket_directional_alignment_share"]) < config.min_top_directional_alignment
    ):
        reasons.append("top_bucket_alignment_below_threshold")
    return len(reasons) == 0, ";".join(reasons) if reasons else "passed"


def select_survivors(decisions: pd.DataFrame) -> pd.DataFrame:
    required = {"signal_id", "horizon", "artifact_review_status"}
    missing = sorted(required.difference(decisions.columns))
    if missing:
        raise ValueError(f"candidate decision missing columns: {missing}")
    survivors = decisions.loc[
        decisions["artifact_review_status"].astype(str).eq("survives_for_walk_forward_proxy_review"),
        ["signal_id", "horizon", "proxy_evidence_status", "artifact_review_status"],
    ].copy()
    survivors["horizon"] = pd.to_numeric(survivors["horizon"], errors="coerce").astype(int)
    return survivors.sort_values(["signal_id", "horizon"]).reset_index(drop=True)


def build_survivor_panel(joined: pd.DataFrame, survivors: pd.DataFrame) -> pd.DataFrame:
    data = joined.copy()
    data["signal_date"] = normalize_date(data["signal_date"])
    data["source_trade_date"] = normalize_date(data["source_trade_date"])
    data["horizon"] = pd.to_numeric(data["horizon"], errors="coerce").astype(int)
    data["signal_value"] = pd.to_numeric(data["signal_value"], errors="coerce")
    data["expected_signed_proxy_return"] = pd.to_numeric(data["expected_signed_proxy_return"], errors="coerce")
    panel = data.merge(survivors.loc[:, ["signal_id", "horizon"]], on=["signal_id", "horizon"], how="inner")
    panel["signal_year"] = pd.to_datetime(panel["signal_date"], format="%Y%m%d", errors="coerce").dt.year
    panel["default_model_allowed"] = False
    panel["portfolio_backtest_allowed"] = False
    panel["official_total_return_evidence"] = False
    panel["walk_forward_scope"] = "narrow_survivor_price_proxy_review_only"
    return panel.sort_values(["signal_id", "horizon", "signal_date"]).reset_index(drop=True)


def validate_inputs(panel: pd.DataFrame, survivors: pd.DataFrame, v3_63_manifest: dict[str, Any], config: NarrowWalkForwardProxyConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {
            "check": "v3_63_manifest_accepted",
            "status": _status(bool(v3_63_manifest.get("self_check_pass"))),
            "detail": f"self_check={v3_63_manifest.get('self_check_pass')}",
        },
        {
            "check": "survivor_candidates_present",
            "status": _status(not survivors.empty),
            "detail": f"rows={len(survivors)}",
        },
        {
            "check": "survivor_panel_has_rows",
            "status": _status(not panel.empty),
            "detail": f"rows={len(panel)}",
        },
    ]
    if panel.empty:
        return pd.DataFrame(rows)
    source_dates = pd.to_datetime(panel["source_trade_date"], format="%Y%m%d", errors="coerce")
    signal_dates = pd.to_datetime(panel["signal_date"], format="%Y%m%d", errors="coerce")
    model_allowed = _bool_series(panel["model_promotion_allowed_signal"]) if "model_promotion_allowed_signal" in panel.columns else pd.Series(False, index=panel.index)
    performance_allowed = _bool_series(panel["performance_claim_allowed_signal"]) if "performance_claim_allowed_signal" in panel.columns else pd.Series(False, index=panel.index)
    portfolio_allowed = _bool_series(panel["portfolio_backtest_allowed"])
    official = _bool_series(panel["official_total_return_evidence"])
    rows.extend(
        [
            {
                "check": "source_trade_date_before_signal_date",
                "status": _status((source_dates < signal_dates).all()),
                "detail": f"bad_rows={int((source_dates >= signal_dates).sum())}",
            },
            {
                "check": "survivor_panel_large_enough",
                "status": _status(len(panel) >= config.min_test_observations * max(1, len(survivors))),
                "detail": f"rows={len(panel)}",
            },
            {
                "check": "no_model_or_performance_promotion_flags",
                "status": _status(not model_allowed.any() and not performance_allowed.any() and not portfolio_allowed.any() and not official.any()),
                "detail": f"model={bool(model_allowed.any())};performance={bool(performance_allowed.any())};portfolio={bool(portfolio_allowed.any())};official={bool(official.any())}",
            },
        ]
    )
    return pd.DataFrame(rows)


def build_walk_forward_windows(panel: pd.DataFrame, config: NarrowWalkForwardProxyConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if panel.empty:
        return pd.DataFrame(rows)
    min_year = int(panel["signal_year"].min())
    max_year = int(panel["signal_year"].max())
    first_test_year = min_year + config.train_years
    for (signal_id, horizon), group in panel.groupby(["signal_id", "horizon"], dropna=False):
        first = group.iloc[0]
        for test_year in range(first_test_year, max_year + 1, config.test_years):
            train_start = test_year - config.train_years
            train_end = test_year - 1
            test_end = test_year + config.test_years - 1
            train = group.loc[(group["signal_year"] >= train_start) & (group["signal_year"] <= train_end)].copy()
            test = group.loc[(group["signal_year"] >= test_year) & (group["signal_year"] <= test_end)].copy()
            if test.empty:
                continue
            train_metrics = _bucket_metrics(train, config)
            test_metrics = _bucket_metrics(test, config)
            train_gate, train_reason = _gate_metrics(train_metrics, config.min_train_observations, config)
            oos_gate, oos_reason = _gate_metrics(test_metrics, config.min_test_observations, config)
            rows.append(
                {
                    "signal_id": signal_id,
                    "horizon": int(horizon),
                    "signal_family": first.get("signal_family", ""),
                    "source_component_type": first.get("source_component_type", ""),
                    "train_start_year": int(train_start),
                    "train_end_year": int(train_end),
                    "test_start_year": int(test_year),
                    "test_end_year": int(test_end),
                    "train_observations": train_metrics["observations"],
                    "train_proxy_spearman_corr": train_metrics["proxy_spearman_corr"],
                    "train_proxy_qspread_top_minus_bottom": train_metrics["proxy_qspread_top_minus_bottom"],
                    "train_top_bucket_directional_alignment_share": train_metrics["top_bucket_directional_alignment_share"],
                    "train_gate_pass": bool(train_gate),
                    "train_gate_reason": train_reason,
                    "test_observations": test_metrics["observations"],
                    "oos_proxy_spearman_corr": test_metrics["proxy_spearman_corr"],
                    "oos_proxy_qspread_top_minus_bottom": test_metrics["proxy_qspread_top_minus_bottom"],
                    "oos_top_bucket_directional_alignment_share": test_metrics["top_bucket_directional_alignment_share"],
                    "oos_directional_alignment_share": test_metrics["directional_alignment_share"],
                    "oos_expected_signed_proxy_mean": test_metrics["expected_signed_proxy_mean"],
                    "oos_gate_pass": bool(oos_gate),
                    "oos_gate_reason": oos_reason,
                    "train_gate_and_oos_pass": bool(train_gate and oos_gate),
                    "default_model_allowed": False,
                    "portfolio_backtest_allowed": False,
                    "official_total_return_evidence": False,
                }
            )
    return pd.DataFrame(rows).sort_values(["horizon", "signal_id", "test_start_year"]).reset_index(drop=True)


def build_oos_summary(windows: pd.DataFrame, config: NarrowWalkForwardProxyConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if windows.empty:
        return pd.DataFrame(rows)
    for (signal_id, horizon), group in windows.groupby(["signal_id", "horizon"], dropna=False):
        gated = group.loc[group["train_gate_pass"].astype(bool)].copy()
        gated_count = int(len(gated))
        oos_pass_count = int(gated["oos_gate_pass"].astype(bool).sum()) if gated_count else 0
        pass_rate = float(oos_pass_count / gated_count) if gated_count else np.nan
        median_spearman = float(gated["oos_proxy_spearman_corr"].median()) if gated_count else np.nan
        positive_qspread_share = float((gated["oos_proxy_qspread_top_minus_bottom"] > 0).mean()) if gated_count else np.nan
        top_alignment_median = float(gated["oos_top_bucket_directional_alignment_share"].median()) if gated_count else np.nan
        qualifies = (
            gated_count >= config.min_gated_windows
            and pd.notna(pass_rate)
            and pass_rate >= config.min_oos_pass_rate
            and pd.notna(median_spearman)
            and median_spearman >= config.min_oos_median_spearman
            and pd.notna(positive_qspread_share)
            and positive_qspread_share >= config.min_oos_positive_qspread_share
        )
        rows.append(
            {
                "signal_id": signal_id,
                "horizon": int(horizon),
                "total_windows": int(len(group)),
                "train_gated_windows": gated_count,
                "oos_gate_pass_windows": oos_pass_count,
                "oos_gate_pass_rate_on_gated_windows": pass_rate,
                "oos_median_proxy_spearman_corr": median_spearman,
                "oos_positive_qspread_share": positive_qspread_share,
                "oos_median_top_bucket_directional_alignment_share": top_alignment_median,
                "walk_forward_proxy_review_status": "passes_narrow_proxy_walk_forward" if qualifies else "watchlist_or_repair",
                "default_model_allowed": False,
                "portfolio_backtest_allowed": False,
                "official_total_return_evidence": False,
            }
        )
    return pd.DataFrame(rows).sort_values(["walk_forward_proxy_review_status", "horizon", "signal_id"]).reset_index(drop=True)


def build_candidate_decision(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in summary.itertuples(index=False):
        if row.walk_forward_proxy_review_status == "passes_narrow_proxy_walk_forward":
            next_action = "eligible_for_stricter_label_source_review_not_model_promotion"
        else:
            next_action = "keep_observation_or_repair_before_any_escalation"
        rows.append(
            {
                "signal_id": row.signal_id,
                "horizon": int(row.horizon),
                "walk_forward_proxy_review_status": row.walk_forward_proxy_review_status,
                "decision": "proxy_review_candidate_only_no_default_promotion",
                "next_action": next_action,
                "reason": "non-official price-index proxy walk-forward review is not investable evidence",
                "default_model_allowed": False,
                "portfolio_backtest_allowed": False,
                "official_total_return_evidence": False,
            }
        )
    return pd.DataFrame(rows).sort_values(["walk_forward_proxy_review_status", "horizon", "signal_id"]).reset_index(drop=True)


def build_no_promotion_guard(windows: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    passed = int(decisions["walk_forward_proxy_review_status"].astype(str).eq("passes_narrow_proxy_walk_forward").sum()) if not decisions.empty else 0
    return pd.DataFrame(
        [
            {
                "result_type": "narrow_walk_forward_proxy_review",
                "produced": not windows.empty,
                "blocked": windows.empty,
                "reason": "non-official price-index proxy diagnostics",
            },
            {
                "result_type": "stricter_label_source_review_queue",
                "produced": passed > 0,
                "blocked": passed == 0,
                "reason": "candidate queue only; not model promotion",
            },
            {
                "result_type": "official_total_return_validation",
                "produced": False,
                "blocked": True,
                "reason": "V3.64 does not use official total-return labels",
            },
            {
                "result_type": "portfolio_backtest",
                "produced": False,
                "blocked": True,
                "reason": "V3.64 does not create positions, trades, NAV, Sharpe, or drawdown",
            },
            {
                "result_type": "model_promotion",
                "produced": False,
                "blocked": True,
                "reason": "proxy walk-forward diagnostics cannot promote default model",
            },
        ]
    )


def build_acceptance_checks(
    input_checks: pd.DataFrame,
    windows: pd.DataFrame,
    summary: pd.DataFrame,
    decisions: pd.DataFrame,
    guard: pd.DataFrame,
    output_column_names: list[str],
) -> pd.DataFrame:
    forbidden_columns = sorted({term for term in FORBIDDEN_PROMOTION_TERMS if term in " ".join(output_column_names).lower()})
    forbidden_produced = bool(
        guard.loc[
            guard["result_type"].isin(["official_total_return_validation", "portfolio_backtest", "model_promotion"]),
            "produced",
        ].any()
    )
    return pd.DataFrame(
        [
            {
                "check": "input_checks_passed",
                "status": "pass" if input_checks["status"].eq("pass").all() else "fail",
                "detail": ";".join(input_checks.loc[input_checks["status"] != "pass", "check"].astype(str)),
            },
            {
                "check": "walk_forward_windows_produced",
                "status": "pass" if not windows.empty else "fail",
                "detail": f"rows={len(windows)}",
            },
            {
                "check": "summary_and_decisions_produced",
                "status": "pass" if not summary.empty and not decisions.empty else "fail",
                "detail": f"summary={len(summary)};decisions={len(decisions)}",
            },
            {
                "check": "candidate_decisions_do_not_promote",
                "status": "pass" if not decisions["default_model_allowed"].astype(bool).any() else "fail",
                "detail": f"rows={len(decisions)}",
            },
            {
                "check": "portfolio_or_promotion_outputs_not_produced",
                "status": "pass" if not forbidden_produced and not forbidden_columns else "fail",
                "detail": ",".join(forbidden_columns),
            },
        ]
    )


def markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 24) -> list[str]:
    if frame.empty:
        return ["_No rows._"]
    safe = frame.loc[:, [col for col in columns if col in frame.columns]].head(max_rows)
    lines = ["| " + " | ".join(safe.columns) + " |", "| " + " | ".join(["---"] * len(safe.columns)) + " |"]
    for row in safe.itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def build_report(
    survivors: pd.DataFrame,
    windows: pd.DataFrame,
    summary: pd.DataFrame,
    decisions: pd.DataFrame,
    input_checks: pd.DataFrame,
    acceptance: pd.DataFrame,
) -> str:
    passed = int(summary["walk_forward_proxy_review_status"].astype(str).eq("passes_narrow_proxy_walk_forward").sum()) if not summary.empty else 0
    train_gated = int(windows["train_gate_pass"].astype(bool).sum()) if not windows.empty else 0
    oos_pass = int(windows["train_gate_and_oos_pass"].astype(bool).sum()) if not windows.empty else 0
    lines = [
        "# V3.64 Narrow Walk-Forward Proxy Review",
        "",
        "## Decision",
        "",
        "- V3.64 reviews only V3.63 survivor signal-horizon rows.",
        "- Each test window uses the prior five calendar years as training and the next calendar year as OOS proxy review.",
        "- It does not run a portfolio backtest, write NAV, or promote any default model.",
        "",
        "## Scope",
        "",
        f"- V3.63 survivor rows reviewed: `{len(survivors)}`",
        f"- Walk-forward windows: `{len(windows)}`",
        f"- Train-gated windows: `{train_gated}`",
        f"- Train-gated and OOS-passing windows: `{oos_pass}`",
        f"- Signal-horizon rows passing narrow proxy walk-forward: `{passed}`",
        "",
        "## Candidate Decisions",
        "",
    ]
    lines.extend(markdown_table(decisions, ["signal_id", "horizon", "walk_forward_proxy_review_status", "decision", "next_action"], max_rows=30))
    lines.extend(["", "## OOS Summary", ""])
    lines.extend(
        markdown_table(
            summary,
            [
                "signal_id",
                "horizon",
                "total_windows",
                "train_gated_windows",
                "oos_gate_pass_windows",
                "oos_gate_pass_rate_on_gated_windows",
                "oos_median_proxy_spearman_corr",
                "oos_positive_qspread_share",
                "walk_forward_proxy_review_status",
            ],
            max_rows=30,
        )
    )
    lines.extend(["", "## Window Sample", ""])
    lines.extend(
        markdown_table(
            windows,
            [
                "signal_id",
                "horizon",
                "test_start_year",
                "train_gate_pass",
                "oos_gate_pass",
                "oos_proxy_spearman_corr",
                "oos_proxy_qspread_top_minus_bottom",
                "oos_top_bucket_directional_alignment_share",
            ],
            max_rows=30,
        )
    )
    lines.extend(["", "## Input Checks", ""])
    lines.extend(markdown_table(input_checks, ["check", "status", "detail"], max_rows=16))
    lines.extend(["", "## Acceptance", ""])
    lines.extend(markdown_table(acceptance, ["check", "status", "detail"], max_rows=16))
    lines.extend(
        [
            "",
            "## Next Use",
            "",
            "- Passing rows can enter stricter label-source review only; they are not tradable evidence.",
            "- Rows that fail narrow walk-forward should remain observation or be repaired.",
            "- A future investable decision still requires official total-return labels and a separate portfolio harness.",
        ]
    )
    return "\n".join(lines)


def build_catalog(summary: pd.DataFrame, config: NarrowWalkForwardProxyConfig) -> str:
    passed = int(summary["walk_forward_proxy_review_status"].astype(str).eq("passes_narrow_proxy_walk_forward").sum()) if not summary.empty else 0
    return "\n".join(
        [
            "# A-share Narrow Walk-Forward Proxy Review V3.64",
            "",
            "## Dataset Role",
            "",
            "V3.64 reviews V3.63 survivor MARKET signal-horizon rows with rolling train/OOS proxy diagnostics.",
            "",
            "## Governance",
            "",
            "- Return basis: `price_index_return` proxy evidence only.",
            "- Official total-return evidence: false.",
            "- Portfolio backtest: not produced.",
            "- Default model promotion: not allowed.",
            "",
            "## Configuration",
            "",
            f"- Train years: `{config.train_years}`",
            f"- Test years: `{config.test_years}`",
            "",
            "## Produced Shape",
            "",
            f"- Summary rows: `{len(summary)}`",
            f"- Rows passing narrow proxy walk-forward: `{passed}`",
        ]
    )
