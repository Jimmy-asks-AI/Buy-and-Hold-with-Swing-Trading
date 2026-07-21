"""Artifact-risk audit for V3.60 proxy-positive MARKET signals.

V3.61 checks whether V3.60 price-index proxy observations look economically
plausible or are likely caused by timing artifacts, same-day price effects,
regime dependence, or weak negative controls. It does not run a portfolio
backtest and cannot promote any default model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from state_stratified_proxy_validation import FORBIDDEN_PROMOTION_TERMS, _corr, normalize_date


@dataclass(frozen=True)
class ProxyArtifactAuditConfig:
    joined_panel_path: Path
    signal_summary_path: Path
    state_summary_path: Path
    negative_control_path: Path
    market_proxy_source_path: Path
    v3_60_manifest_path: Path
    output_dir: Path
    catalog_path: Path
    horizons: tuple[int, ...]
    positive_status: str
    min_abs_proxy_spearman: float
    min_control_degradation: float
    max_same_day_corr_ratio: float
    max_future_lead_corr_excess: float
    min_primary_state_support: int
    max_bull_state_share: float
    trend_window: int
    bull_return_threshold: float
    bear_return_threshold: float


def _safe_div(numerator: float, denominator: float) -> float:
    if pd.isna(numerator) or pd.isna(denominator) or abs(denominator) < 1e-12:
        return np.nan
    return float(numerator / denominator)


def _status(ok: bool, fail_status: str = "fail") -> str:
    return "pass" if ok else fail_status


def _bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    return values.astype(str).str.lower().isin({"true", "1", "yes"})


def load_market_context(source: pd.DataFrame, config: ProxyArtifactAuditConfig) -> pd.DataFrame:
    data = source.copy()
    data["signal_date"] = normalize_date(data["date"])
    data["market_close"] = pd.to_numeric(data.get("close", data.get("market_level")), errors="coerce")
    data["same_day_price_return"] = pd.to_numeric(data.get("pct_chg"), errors="coerce") / 100.0
    if data["same_day_price_return"].isna().all():
        data["same_day_price_return"] = data["market_close"].pct_change()
    data["prev_day_price_return"] = data["same_day_price_return"].shift(1)
    data["rolling_trend_return"] = data["market_close"] / data["market_close"].shift(config.trend_window) - 1.0
    data["market_trend_state"] = "range_or_transition"
    data.loc[data["rolling_trend_return"] >= config.bull_return_threshold, "market_trend_state"] = "bull_price_proxy_state"
    data.loc[data["rolling_trend_return"] <= config.bear_return_threshold, "market_trend_state"] = "bear_price_proxy_state"
    keep = [
        "signal_date",
        "same_day_price_return",
        "prev_day_price_return",
        "rolling_trend_return",
        "market_trend_state",
    ]
    return data.loc[:, keep].dropna(subset=["signal_date"]).drop_duplicates("signal_date")


def select_proxy_positive_candidates(summary: pd.DataFrame, config: ProxyArtifactAuditConfig) -> pd.DataFrame:
    required = {"signal_id", "horizon", "proxy_evidence_status"}
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(f"signal summary missing columns: {sorted(missing)}")
    candidates = summary.loc[summary["proxy_evidence_status"].astype(str).eq(config.positive_status)].copy()
    candidates["horizon"] = pd.to_numeric(candidates["horizon"], errors="coerce").astype(int)
    return candidates.sort_values(["signal_id", "horizon"]).reset_index(drop=True)


def build_audit_panel(joined: pd.DataFrame, market_context: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    candidate_keys = candidates.loc[:, ["signal_id", "horizon"]].copy()
    candidate_keys["horizon"] = pd.to_numeric(candidate_keys["horizon"], errors="coerce").astype(int)
    panel = joined.copy()
    panel["signal_date"] = normalize_date(panel["signal_date"])
    panel["horizon"] = pd.to_numeric(panel["horizon"], errors="coerce").astype(int)
    panel["signal_value"] = pd.to_numeric(panel["signal_value"], errors="coerce")
    panel["expected_signed_proxy_return"] = pd.to_numeric(panel["expected_signed_proxy_return"], errors="coerce")
    panel = panel.merge(candidate_keys, on=["signal_id", "horizon"], how="inner")
    panel = panel.merge(market_context, on="signal_date", how="left")
    panel = panel.sort_values(["signal_id", "horizon", "signal_date"]).reset_index(drop=True)
    panel["signal_lag1"] = panel.groupby(["signal_id", "horizon"])["signal_value"].shift(1)
    panel["signal_lead1_future_unseen"] = panel.groupby(["signal_id", "horizon"])["signal_value"].shift(-1)
    panel["same_day_return_signed"] = panel["same_day_price_return"] * panel["direction_multiplier"]
    panel["prev_day_return_signed"] = panel["prev_day_price_return"] * panel["direction_multiplier"]
    panel["default_model_allowed"] = False
    panel["portfolio_backtest_allowed"] = False
    panel["official_total_return_evidence"] = False
    panel["audit_scope"] = "artifact_risk_diagnostics_only"
    return panel


def build_temporal_artifact_audit(panel: pd.DataFrame, config: ProxyArtifactAuditConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (signal_id, horizon), group in panel.groupby(["signal_id", "horizon"], dropna=False):
        real_corr = _corr(group["signal_value"], group["expected_signed_proxy_return"], "spearman")
        lag1_corr = _corr(group["signal_lag1"], group["expected_signed_proxy_return"], "spearman")
        lead1_corr = _corr(group["signal_lead1_future_unseen"], group["expected_signed_proxy_return"], "spearman")
        same_day_corr = _corr(group["signal_value"], group["same_day_return_signed"], "spearman")
        prev_day_corr = _corr(group["signal_value"], group["prev_day_return_signed"], "spearman")
        same_day_ratio = _safe_div(abs(same_day_corr), abs(real_corr))
        future_excess = abs(lead1_corr) - abs(real_corr) if pd.notna(lead1_corr) and pd.notna(real_corr) else np.nan
        rows.append(
            {
                "signal_id": signal_id,
                "horizon": int(horizon),
                "observations": int(len(group)),
                "real_proxy_spearman_corr": real_corr,
                "lag1_signal_proxy_spearman_corr": lag1_corr,
                "future_lead1_signal_proxy_spearman_corr": lead1_corr,
                "same_day_price_signed_spearman_corr": same_day_corr,
                "prev_day_price_signed_spearman_corr": prev_day_corr,
                "same_day_to_forward_abs_corr_ratio": same_day_ratio,
                "future_lead_abs_corr_excess": future_excess,
                "same_day_artifact_flag": bool(pd.notna(same_day_ratio) and same_day_ratio > config.max_same_day_corr_ratio),
                "future_signal_artifact_flag": bool(pd.notna(future_excess) and future_excess > config.max_future_lead_corr_excess),
                "default_model_allowed": False,
                "official_total_return_evidence": False,
            }
        )
    return pd.DataFrame(rows).sort_values(["horizon", "signal_id"]).reset_index(drop=True)


def build_negative_control_audit(controls: pd.DataFrame, candidates: pd.DataFrame, config: ProxyArtifactAuditConfig) -> pd.DataFrame:
    keys = candidates.loc[:, ["signal_id", "horizon"]].copy()
    keys["horizon"] = keys["horizon"].astype(int)
    data = controls.copy()
    data["horizon"] = pd.to_numeric(data["horizon"], errors="coerce").astype(int)
    data = data.merge(keys, on=["signal_id", "horizon"], how="inner")
    data["negative_control_degraded"] = pd.to_numeric(data["abs_corr_degradation"], errors="coerce") >= config.min_control_degradation
    data["negative_control_artifact_flag"] = ~data["negative_control_degraded"]
    data["default_model_allowed"] = False
    data["official_total_return_evidence"] = False
    return data.sort_values(["horizon", "signal_id"]).reset_index(drop=True)


def build_state_dependence_audit(state_summary: pd.DataFrame, candidates: pd.DataFrame, config: ProxyArtifactAuditConfig) -> pd.DataFrame:
    keys = candidates.loc[:, ["signal_id", "horizon"]].copy()
    keys["horizon"] = keys["horizon"].astype(int)
    data = state_summary.copy()
    data["horizon"] = pd.to_numeric(data["horizon"], errors="coerce").astype(int)
    data["proxy_spearman_corr"] = pd.to_numeric(data["proxy_spearman_corr"], errors="coerce")
    data["proxy_qspread_top_minus_bottom"] = pd.to_numeric(data["proxy_qspread_top_minus_bottom"], errors="coerce")
    data["top_bucket_directional_alignment_share"] = pd.to_numeric(data["top_bucket_directional_alignment_share"], errors="coerce")
    data["observations"] = pd.to_numeric(data["observations"], errors="coerce")
    data = data.merge(keys, on=["signal_id", "horizon"], how="inner")
    data = data.loc[data["validation_role"].astype(str).eq("primary_proxy_stratum")].copy()
    data["supportive_state"] = (
        (data["proxy_spearman_corr"] >= config.min_abs_proxy_spearman)
        & (data["proxy_qspread_top_minus_bottom"] > 0)
        & (data["top_bucket_directional_alignment_share"] >= 0.52)
    )

    rows: list[dict[str, Any]] = []
    for (signal_id, horizon), group in data.groupby(["signal_id", "horizon"], dropna=False):
        composite = group.loc[group["state_column"].eq("composite_state")].copy()
        bull_mask = composite["state_value"].astype(str).str.contains("risk_on|breadth_recovery|liquidity", case=False, regex=True)
        bull_obs = float(composite.loc[bull_mask & composite["supportive_state"], "observations"].sum())
        support_obs = float(composite.loc[composite["supportive_state"], "observations"].sum())
        bull_share = _safe_div(bull_obs, support_obs)
        column_support = (
            group.groupby("state_column")["supportive_state"].sum().reset_index(name="supportive_primary_state_count")
        )
        rows.append(
            {
                "signal_id": signal_id,
                "horizon": int(horizon),
                "primary_state_rows": int(len(group)),
                "supportive_primary_state_rows": int(group["supportive_state"].sum()),
                "state_columns_with_support": int((column_support["supportive_primary_state_count"] > 0).sum()),
                "composite_supportive_state_rows": int(composite["supportive_state"].sum()) if not composite.empty else 0,
                "composite_bull_support_observation_share": bull_share,
                "state_support_too_sparse_flag": bool(group["supportive_state"].sum() < config.min_primary_state_support),
                "bull_state_proxy_flag": bool(pd.notna(bull_share) and bull_share > config.max_bull_state_share),
                "default_model_allowed": False,
                "official_total_return_evidence": False,
            }
        )
    return pd.DataFrame(rows).sort_values(["horizon", "signal_id"]).reset_index(drop=True)


def build_regime_proxy_audit(panel: pd.DataFrame, candidates: pd.DataFrame, config: ProxyArtifactAuditConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (signal_id, horizon, state), group in panel.groupby(["signal_id", "horizon", "market_trend_state"], dropna=False):
        corr = _corr(group["signal_value"], group["expected_signed_proxy_return"], "spearman")
        qspread = _bucket_spread(group)
        rows.append(
            {
                "signal_id": signal_id,
                "horizon": int(horizon),
                "market_trend_state": str(state),
                "observations": int(len(group)),
                "proxy_spearman_corr": corr,
                "proxy_qspread_top_minus_bottom": qspread,
                "default_model_allowed": False,
                "official_total_return_evidence": False,
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    total = result.groupby(["signal_id", "horizon"])["observations"].transform("sum")
    result["observation_share"] = result["observations"] / total
    return result.sort_values(["horizon", "signal_id", "market_trend_state"]).reset_index(drop=True)


def _bucket_spread(group: pd.DataFrame) -> float:
    clean = group.dropna(subset=["signal_value", "expected_signed_proxy_return"]).copy()
    if len(clean) < 20 or clean["signal_value"].nunique() < 3:
        return np.nan
    top_cut = clean["signal_value"].quantile(0.8)
    bottom_cut = clean["signal_value"].quantile(0.2)
    top = clean.loc[clean["signal_value"] >= top_cut, "expected_signed_proxy_return"]
    bottom = clean.loc[clean["signal_value"] <= bottom_cut, "expected_signed_proxy_return"]
    if top.empty or bottom.empty:
        return np.nan
    return float(top.mean() - bottom.mean())


def build_candidate_artifact_decision(
    temporal: pd.DataFrame,
    controls: pd.DataFrame,
    state_audit: pd.DataFrame,
    config: ProxyArtifactAuditConfig,
) -> pd.DataFrame:
    merged = temporal.merge(
        controls.loc[:, ["signal_id", "horizon", "abs_corr_degradation", "negative_control_degraded", "negative_control_artifact_flag"]],
        on=["signal_id", "horizon"],
        how="left",
    ).merge(
        state_audit.loc[
            :,
            [
                "signal_id",
                "horizon",
                "supportive_primary_state_rows",
                "state_columns_with_support",
                "composite_bull_support_observation_share",
                "state_support_too_sparse_flag",
                "bull_state_proxy_flag",
            ],
        ],
        on=["signal_id", "horizon"],
        how="left",
    )
    rows: list[dict[str, Any]] = []
    for row in merged.itertuples(index=False):
        flags = []
        if bool(row.same_day_artifact_flag):
            flags.append("same_day_price_artifact_risk")
        if bool(row.future_signal_artifact_flag):
            flags.append("future_signal_serial_state_risk")
        if bool(row.negative_control_artifact_flag):
            flags.append("negative_control_not_degraded")
        if bool(row.state_support_too_sparse_flag):
            flags.append("state_support_too_sparse")
        if bool(row.bull_state_proxy_flag):
            flags.append("bull_state_proxy_dependence")
        if not flags:
            review_status = "plausible_for_stricter_forward_validation"
            next_action = "carry_to_walk_forward_proxy_review_only"
        elif len(flags) <= 2 and "negative_control_not_degraded" not in flags:
            review_status = "mixed_evidence_requires_manual_review"
            next_action = "inspect_source_formula_and_state_breakdown"
        else:
            review_status = "artifact_risk_blocks_escalation"
            next_action = "do_not_escalate_until_signal_redefined_or_new_label_source_available"
        rows.append(
            {
                "signal_id": row.signal_id,
                "horizon": int(row.horizon),
                "artifact_review_status": review_status,
                "artifact_flags": ";".join(flags) if flags else "none",
                "next_action": next_action,
                "default_model_allowed": False,
                "portfolio_backtest_allowed": False,
                "official_total_return_evidence": False,
            }
        )
    return pd.DataFrame(rows).sort_values(["horizon", "signal_id"]).reset_index(drop=True)


def build_readiness_checks(
    joined: pd.DataFrame,
    summary: pd.DataFrame,
    controls: pd.DataFrame,
    market_context: pd.DataFrame,
    v3_60_manifest: dict[str, Any],
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        {
            "check": "v3_60_manifest_accepted",
            "status": _status(bool(v3_60_manifest.get("self_check_pass"))),
            "detail": f"self_check={v3_60_manifest.get('self_check_pass')}",
        },
        {
            "check": "joined_panel_present",
            "status": _status(not joined.empty),
            "detail": f"rows={len(joined)}",
        },
        {
            "check": "signal_summary_present",
            "status": _status(not summary.empty),
            "detail": f"rows={len(summary)}",
        },
        {
            "check": "negative_control_present",
            "status": _status(not controls.empty),
            "detail": f"rows={len(controls)}",
        },
        {
            "check": "market_context_present",
            "status": _status(not market_context.empty),
            "detail": f"rows={len(market_context)}",
        },
        {
            "check": "proxy_positive_candidates_present",
            "status": _status(not candidates.empty),
            "detail": f"rows={len(candidates)}",
        },
        {
            "check": "portfolio_or_model_promotion_allowed_now",
            "status": "blocked",
            "detail": "V3.61 is artifact-risk diagnostics only",
        },
    ]
    return pd.DataFrame(rows)


def build_no_promotion_guard(decisions: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "result_type": "proxy_artifact_risk_audit",
                "produced": not decisions.empty,
                "blocked": decisions.empty,
                "reason": "research diagnostics only",
            },
            {
                "result_type": "official_total_return_validation",
                "produced": False,
                "blocked": True,
                "reason": "V3.61 only reuses non-official price-index proxy evidence",
            },
            {
                "result_type": "portfolio_backtest",
                "produced": False,
                "blocked": True,
                "reason": "V3.61 does not create positions, NAV, returns, Sharpe, or drawdown",
            },
            {
                "result_type": "model_promotion",
                "produced": False,
                "blocked": True,
                "reason": "artifact audit cannot promote default model",
            },
        ]
    )


def build_acceptance_checks(
    readiness: pd.DataFrame,
    temporal: pd.DataFrame,
    decisions: pd.DataFrame,
    guard: pd.DataFrame,
    output_column_names: list[str],
) -> pd.DataFrame:
    unexpected = readiness.loc[
        (readiness["status"] != "pass") & (~readiness["check"].isin(["portfolio_or_model_promotion_allowed_now"]))
    ]
    forbidden_columns = sorted({term for term in FORBIDDEN_PROMOTION_TERMS if term in " ".join(output_column_names).lower()})
    produced_forbidden = bool(
        guard.loc[
            guard["result_type"].isin(["official_total_return_validation", "portfolio_backtest", "model_promotion"]),
            "produced",
        ].any()
    )
    return pd.DataFrame(
        [
            {
                "check": "readiness_checks_passed",
                "status": "pass" if unexpected.empty else "fail",
                "detail": ";".join(unexpected["check"].astype(str)),
            },
            {
                "check": "temporal_audit_produced",
                "status": "pass" if not temporal.empty else "fail",
                "detail": f"rows={len(temporal)}",
            },
            {
                "check": "candidate_decisions_do_not_promote",
                "status": "pass" if not decisions.empty and not decisions["default_model_allowed"].astype(bool).any() else "fail",
                "detail": f"rows={len(decisions)}",
            },
            {
                "check": "portfolio_or_promotion_outputs_not_produced",
                "status": "pass" if not produced_forbidden and not forbidden_columns else "fail",
                "detail": ",".join(forbidden_columns),
            },
        ]
    )


def markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 20) -> list[str]:
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
    candidates: pd.DataFrame,
    temporal: pd.DataFrame,
    controls: pd.DataFrame,
    state_audit: pd.DataFrame,
    regime_audit: pd.DataFrame,
    decisions: pd.DataFrame,
    readiness: pd.DataFrame,
    acceptance: pd.DataFrame,
) -> str:
    blocked = int(decisions["artifact_review_status"].astype(str).eq("artifact_risk_blocks_escalation").sum()) if not decisions.empty else 0
    plausible = int(decisions["artifact_review_status"].astype(str).eq("plausible_for_stricter_forward_validation").sum()) if not decisions.empty else 0
    lines = [
        "# V3.61 Proxy Artifact Audit",
        "",
        "## Decision",
        "",
        "- V3.61 audits V3.60 proxy-positive signal-horizon observations for artifact risk.",
        "- It uses non-official `price_index_return` proxy evidence only.",
        "- It does not run a portfolio backtest, write NAV, or promote any default model.",
        "",
        "## Scope",
        "",
        f"- Proxy-positive candidate rows audited: `{len(candidates)}`",
        f"- Plausible for stricter proxy review: `{plausible}`",
        f"- Artifact-risk blocked rows: `{blocked}`",
        "",
        "## Candidate Artifact Decisions",
        "",
    ]
    lines.extend(markdown_table(decisions, ["signal_id", "horizon", "artifact_review_status", "artifact_flags", "next_action"], max_rows=24))
    lines.extend(["", "## Temporal Artifact Audit", ""])
    lines.extend(
        markdown_table(
            temporal,
            [
                "signal_id",
                "horizon",
                "real_proxy_spearman_corr",
                "same_day_to_forward_abs_corr_ratio",
                "future_lead_abs_corr_excess",
                "same_day_artifact_flag",
                "future_signal_artifact_flag",
            ],
            max_rows=24,
        )
    )
    lines.extend(["", "## Negative Control Audit", ""])
    lines.extend(
        markdown_table(
            controls,
            [
                "signal_id",
                "horizon",
                "real_proxy_spearman_corr",
                "lag_broken_control_spearman_corr",
                "abs_corr_degradation",
                "negative_control_degraded",
            ],
            max_rows=24,
        )
    )
    lines.extend(["", "## State Dependence Audit", ""])
    lines.extend(
        markdown_table(
            state_audit,
            [
                "signal_id",
                "horizon",
                "supportive_primary_state_rows",
                "state_columns_with_support",
                "composite_bull_support_observation_share",
                "state_support_too_sparse_flag",
                "bull_state_proxy_flag",
            ],
            max_rows=24,
        )
    )
    lines.extend(["", "## Market Trend Regime Sample", ""])
    lines.extend(markdown_table(regime_audit, ["signal_id", "horizon", "market_trend_state", "observations", "proxy_spearman_corr", "observation_share"], max_rows=24))
    lines.extend(["", "## Readiness", ""])
    lines.extend(markdown_table(readiness, ["check", "status", "detail"], max_rows=16))
    lines.extend(["", "## Acceptance", ""])
    lines.extend(markdown_table(acceptance, ["check", "status", "detail"], max_rows=16))
    lines.extend(
        [
            "",
            "## Next Use",
            "",
            "- Treat `plausible_for_stricter_forward_validation` as a research queue, not a promotion decision.",
            "- Treat `artifact_risk_blocks_escalation` as a block until the signal is redefined or an official label source is available.",
            "- If no signal-horizon row survives, V3.62 should repair signal definitions with stricter lag discipline instead of running an empty walk-forward queue.",
        ]
    )
    return "\n".join(lines)


def build_catalog(decisions: pd.DataFrame, config: ProxyArtifactAuditConfig) -> str:
    blocked = int(decisions["artifact_review_status"].astype(str).eq("artifact_risk_blocks_escalation").sum()) if not decisions.empty else 0
    plausible = int(decisions["artifact_review_status"].astype(str).eq("plausible_for_stricter_forward_validation").sum()) if not decisions.empty else 0
    return "\n".join(
        [
            "# A-share Proxy Artifact Audit V3.61",
            "",
            "## Dataset Role",
            "",
            "V3.61 audits V3.60 proxy-positive MARKET signal observations for likely artifacts.",
            "",
            "## Governance",
            "",
            "- Return basis: `price_index_return` proxy evidence only.",
            "- Official total-return evidence: false.",
            "- Portfolio backtest: not produced.",
            "- Default model promotion: not allowed.",
            "",
            "## Inputs",
            "",
            f"- Joined panel: `{config.joined_panel_path}`",
            f"- Signal summary: `{config.signal_summary_path}`",
            f"- Negative control: `{config.negative_control_path}`",
            f"- Market proxy source: `{config.market_proxy_source_path}`",
            "",
            "## Produced Shape",
            "",
            f"- Candidate decisions: `{len(decisions)}`",
            f"- Plausible for stricter proxy review: `{plausible}`",
            f"- Artifact-risk blocked: `{blocked}`",
        ]
    )
