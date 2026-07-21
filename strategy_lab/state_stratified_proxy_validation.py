"""Guarded state-stratified proxy validation for HIRSSM V3.60.

This module evaluates V3.50 MARKET signals against V3.59 price-index proxy
labels. Results are proxy diagnostics only, not official total-return evidence
and not a portfolio backtest.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STATE_COLUMNS = [
    "composite_state",
    "liquidity_state",
    "breadth_state",
    "activity_state",
    "concentration_state",
    "limit_crowding_state",
]

FORBIDDEN_PROMOTION_TERMS = {
    "nav",
    "sharpe",
    "annualized_return",
    "portfolio_return",
    "max_drawdown",
    "official_total_return_label",
    "default_enabled",
}


@dataclass(frozen=True)
class ProxyValidationConfig:
    signal_panel_path: Path
    label_path: Path
    v3_59_manifest_path: Path
    output_dir: Path
    catalog_path: Path
    horizons: tuple[int, ...]
    state_columns: tuple[str, ...]
    min_joined_rows: int
    min_signal_observations: int
    min_state_observations: int
    top_quantile: float
    bottom_quantile: float
    min_abs_proxy_spearman: float
    min_proxy_qspread: float
    min_top_directional_alignment: float
    negative_control_shift: int


def normalize_date(values: pd.Series) -> pd.Series:
    cleaned = (
        values.astype(str)
        .str.strip()
        .str.replace("-", "", regex=False)
        .str.replace("/", "", regex=False)
        .str.replace(".0", "", regex=False)
    )
    return pd.to_datetime(cleaned, format="%Y%m%d", errors="coerce").dt.strftime("%Y%m%d")


def _bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    return values.astype(str).str.lower().isin({"true", "1", "yes"})


def _status(ok: bool, fail_status: str = "blocked") -> str:
    return "pass" if ok else fail_status


def _corr(x: pd.Series, y: pd.Series, method: str) -> float:
    clean = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(clean) < 3 or clean["x"].nunique() < 2 or clean["y"].nunique() < 2:
        return np.nan
    if method == "spearman":
        return float(clean["x"].rank().corr(clean["y"].rank(), method="pearson"))
    return float(clean["x"].corr(clean["y"], method=method))


def validate_signal_panel(signals: pd.DataFrame, config: ProxyValidationConfig) -> pd.DataFrame:
    required = {
        "signal_id",
        "signal_date",
        "asset",
        "signal_value",
        "signal_direction",
        "available_date",
        "model_promotion_allowed",
        "performance_claim_allowed",
        *config.state_columns,
    }
    missing = sorted(required.difference(signals.columns))
    rows: list[dict[str, Any]] = [
        {
            "check": "signal_required_columns_present",
            "status": _status(not missing),
            "detail": ",".join(missing),
        }
    ]
    if missing:
        return pd.DataFrame(rows)

    signal_dates = normalize_date(signals["signal_date"])
    available = normalize_date(signals["available_date"])
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


def validate_proxy_labels(labels: pd.DataFrame, config: ProxyValidationConfig) -> pd.DataFrame:
    required = {
        "signal_date",
        "asset",
        "horizon",
        "forward_price_index_return",
        "return_basis",
        "label_available_date",
        "official_total_return",
        "proxy_label_generation_allowed",
        "official_label_generation_allowed",
        "model_promotion_allowed",
        "performance_claim_allowed",
        "diagnostic_usage",
    }
    missing = sorted(required.difference(labels.columns))
    rows: list[dict[str, Any]] = [
        {
            "check": "label_required_columns_present",
            "status": _status(not missing),
            "detail": ",".join(missing),
        }
    ]
    if missing:
        return pd.DataFrame(rows)

    signal_dates = normalize_date(labels["signal_date"])
    available = normalize_date(labels["label_available_date"])
    returns = pd.to_numeric(labels["forward_price_index_return"], errors="coerce")
    basis = set(labels["return_basis"].astype(str).unique())
    horizons = set(pd.to_numeric(labels["horizon"], errors="coerce").dropna().astype(int))
    official = _bool_series(labels["official_total_return"])
    proxy_allowed = _bool_series(labels["proxy_label_generation_allowed"])
    official_allowed = _bool_series(labels["official_label_generation_allowed"])
    model_allowed = _bool_series(labels["model_promotion_allowed"])
    performance_allowed = _bool_series(labels["performance_claim_allowed"])
    usage = set(labels["diagnostic_usage"].astype(str).unique())
    rows.extend(
        [
            {
                "check": "label_dates_parseable",
                "status": _status(signal_dates.notna().all()),
                "detail": f"bad_rows={int(signal_dates.isna().sum())}",
            },
            {
                "check": "label_available_date_after_signal_date",
                "status": _status((available > signal_dates).all()),
                "detail": f"bad_rows={int((available <= signal_dates).sum())}",
            },
            {
                "check": "label_return_basis_is_price_proxy",
                "status": _status(basis == {"price_index_return"}),
                "detail": ",".join(sorted(basis)),
            },
            {
                "check": "label_horizons_match_config",
                "status": _status(horizons == set(config.horizons)),
                "detail": f"actual={sorted(horizons)};expected={sorted(config.horizons)}",
            },
            {
                "check": "label_returns_finite",
                "status": _status(returns.notna().all() and np.isfinite(returns).all()),
                "detail": f"bad_rows={int((returns.isna() | ~np.isfinite(returns)).sum())}",
            },
            {
                "check": "label_not_official_total_return",
                "status": _status(not official.any() and not official_allowed.any()),
                "detail": f"official_any={bool(official.any())};official_allowed_any={bool(official_allowed.any())}",
            },
            {
                "check": "label_proxy_allowed_only",
                "status": _status(proxy_allowed.all() and not model_allowed.any() and not performance_allowed.any()),
                "detail": f"proxy_false_rows={int((~proxy_allowed).sum())};model_any={bool(model_allowed.any())};performance_any={bool(performance_allowed.any())}",
            },
            {
                "check": "label_usage_non_official",
                "status": _status(usage == {"non_official_price_proxy_label_only"}),
                "detail": ",".join(sorted(usage)),
            },
        ]
    )
    return pd.DataFrame(rows)


def build_joined_panel(signals: pd.DataFrame, labels: pd.DataFrame, config: ProxyValidationConfig) -> pd.DataFrame:
    signal_cols = [
        "signal_id",
        "signal_date",
        "asset",
        "signal_value",
        "signal_direction",
        "available_date",
        "candidate_status",
        "signal_family",
        "formula_name",
        *config.state_columns,
        "data_quality_state",
        "model_promotion_allowed",
        "performance_claim_allowed",
    ]
    keep_cols = [col for col in signal_cols if col in signals.columns]
    sig = signals.loc[signals["asset"].astype(str) == "MARKET", keep_cols].copy()
    sig["signal_date"] = normalize_date(sig["signal_date"])
    sig["signal_value"] = pd.to_numeric(sig["signal_value"], errors="coerce")
    lab = labels.copy()
    lab["signal_date"] = normalize_date(lab["signal_date"])
    lab["horizon"] = pd.to_numeric(lab["horizon"], errors="coerce").astype("Int64")
    lab["forward_price_index_return"] = pd.to_numeric(lab["forward_price_index_return"], errors="coerce")
    joined = sig.merge(lab, on=["signal_date", "asset"], how="inner", suffixes=("_signal", "_label"))
    direction = joined["signal_direction"].astype(str).str.lower().map({"positive": 1.0, "negative": -1.0}).fillna(0.0)
    joined["direction_multiplier"] = direction
    joined["expected_signed_proxy_return"] = joined["forward_price_index_return"] * joined["direction_multiplier"]
    joined["active_directional_alignment"] = joined["expected_signed_proxy_return"] > 0
    joined["validation_scope"] = "state_stratified_price_proxy_only"
    joined["default_model_allowed"] = False
    joined["portfolio_backtest_allowed"] = False
    joined["official_total_return_evidence"] = False
    return joined.sort_values(["signal_id", "horizon", "signal_date"]).reset_index(drop=True)


def _bucket_means(group: pd.DataFrame, config: ProxyValidationConfig) -> dict[str, float]:
    clean = group.dropna(subset=["signal_value", "expected_signed_proxy_return"]).copy()
    if clean.empty or clean["signal_value"].nunique() < 2:
        return {
            "bottom_bucket_expected_signed_return_mean": np.nan,
            "top_bucket_expected_signed_return_mean": np.nan,
            "proxy_qspread_top_minus_bottom": np.nan,
            "top_bucket_directional_alignment_share": np.nan,
            "top_bucket_rows": 0,
            "bottom_bucket_rows": 0,
        }
    lo = clean["signal_value"].quantile(config.bottom_quantile)
    hi = clean["signal_value"].quantile(config.top_quantile)
    bottom = clean.loc[clean["signal_value"] <= lo]
    top = clean.loc[clean["signal_value"] >= hi]
    bottom_mean = float(bottom["expected_signed_proxy_return"].mean()) if not bottom.empty else np.nan
    top_mean = float(top["expected_signed_proxy_return"].mean()) if not top.empty else np.nan
    return {
        "bottom_bucket_expected_signed_return_mean": bottom_mean,
        "top_bucket_expected_signed_return_mean": top_mean,
        "proxy_qspread_top_minus_bottom": top_mean - bottom_mean if pd.notna(top_mean) and pd.notna(bottom_mean) else np.nan,
        "top_bucket_directional_alignment_share": float(top["active_directional_alignment"].mean()) if not top.empty else np.nan,
        "top_bucket_rows": int(len(top)),
        "bottom_bucket_rows": int(len(bottom)),
    }


def _summarize_group(group: pd.DataFrame, config: ProxyValidationConfig) -> dict[str, Any]:
    clean = group.dropna(subset=["signal_value", "expected_signed_proxy_return"]).copy()
    bucket = _bucket_means(clean, config)
    return {
        "observations": int(len(clean)),
        "unique_signal_dates": int(clean["signal_date"].astype(str).nunique()) if not clean.empty else 0,
        "proxy_pearson_corr": _corr(clean["signal_value"], clean["expected_signed_proxy_return"], "pearson"),
        "proxy_spearman_corr": _corr(clean["signal_value"], clean["expected_signed_proxy_return"], "spearman"),
        "expected_signed_proxy_return_mean": float(clean["expected_signed_proxy_return"].mean()) if not clean.empty else np.nan,
        "directional_alignment_share": float(clean["active_directional_alignment"].mean()) if not clean.empty else np.nan,
        **bucket,
    }


def build_signal_validation_summary(joined: pd.DataFrame, config: ProxyValidationConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (signal_id, horizon), group in joined.groupby(["signal_id", "horizon"], dropna=False):
        summary = _summarize_group(group, config)
        enough = summary["observations"] >= config.min_signal_observations
        spearman = summary["proxy_spearman_corr"]
        qspread = summary["proxy_qspread_top_minus_bottom"]
        alignment = summary["top_bucket_directional_alignment_share"]
        proxy_positive = (
            enough
            and pd.notna(spearman)
            and abs(float(spearman)) >= config.min_abs_proxy_spearman
            and pd.notna(qspread)
            and float(qspread) >= config.min_proxy_qspread
            and pd.notna(alignment)
            and float(alignment) >= config.min_top_directional_alignment
        )
        rows.append(
            {
                "signal_id": signal_id,
                "horizon": int(horizon),
                "signal_direction": ",".join(sorted(group["signal_direction"].astype(str).unique())),
                "signal_family": ",".join(sorted(group.get("signal_family", pd.Series(dtype=str)).astype(str).unique())[:3]),
                **summary,
                "proxy_evidence_status": "proxy_positive_observation" if proxy_positive else "proxy_observation_only",
                "default_model_allowed": False,
                "official_total_return_evidence": False,
            }
        )
    return pd.DataFrame(rows).sort_values(["horizon", "proxy_evidence_status", "proxy_spearman_corr"], ascending=[True, True, False])


def build_state_coverage(joined: pd.DataFrame, config: ProxyValidationConfig) -> pd.DataFrame:
    base = joined.drop_duplicates(["signal_date", *config.state_columns]).copy()
    rows: list[dict[str, Any]] = []
    for column in config.state_columns:
        counts = base.groupby(column, dropna=False)["signal_date"].nunique().reset_index(name="unique_signal_dates")
        total = max(base["signal_date"].nunique(), 1)
        for row in counts.itertuples(index=False):
            value = str(getattr(row, column))
            observations = int(row.unique_signal_dates)
            rows.append(
                {
                    "state_column": column,
                    "state_value": value,
                    "unique_signal_dates": observations,
                    "share_of_joined_dates": observations / total,
                    "validation_role": "primary_proxy_stratum" if observations >= config.min_state_observations else "monitor_only_low_sample",
                }
            )
    return pd.DataFrame(rows).sort_values(["state_column", "unique_signal_dates"], ascending=[True, False])


def build_state_stratified_summary(joined: pd.DataFrame, config: ProxyValidationConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in config.state_columns:
        for (signal_id, horizon, state_value), group in joined.groupby(["signal_id", "horizon", column], dropna=False):
            summary = _summarize_group(group, config)
            observations = summary["observations"]
            rows.append(
                {
                    "state_column": column,
                    "state_value": str(state_value),
                    "signal_id": signal_id,
                    "horizon": int(horizon),
                    **summary,
                    "validation_role": "primary_proxy_stratum" if observations >= config.min_state_observations else "monitor_only_low_sample",
                    "default_model_allowed": False,
                    "official_total_return_evidence": False,
                }
            )
    return pd.DataFrame(rows).sort_values(["horizon", "state_column", "observations"], ascending=[True, True, False])


def build_negative_control_summary(joined: pd.DataFrame, config: ProxyValidationConfig) -> pd.DataFrame:
    control = joined.sort_values(["horizon", "signal_date", "signal_id"]).copy()
    control["control_expected_signed_proxy_return"] = control.groupby("horizon")["expected_signed_proxy_return"].shift(config.negative_control_shift)
    rows: list[dict[str, Any]] = []
    for (signal_id, horizon), group in control.groupby(["signal_id", "horizon"], dropna=False):
        real = _summarize_group(group, config)
        control_clean = group.dropna(subset=["signal_value", "control_expected_signed_proxy_return"]).copy()
        control_corr = _corr(control_clean["signal_value"], control_clean["control_expected_signed_proxy_return"], "spearman")
        rows.append(
            {
                "signal_id": signal_id,
                "horizon": int(horizon),
                "real_proxy_spearman_corr": real["proxy_spearman_corr"],
                "lag_broken_control_spearman_corr": control_corr,
                "abs_corr_degradation": abs(real["proxy_spearman_corr"]) - abs(control_corr)
                if pd.notna(real["proxy_spearman_corr"]) and pd.notna(control_corr)
                else np.nan,
                "control_shift_rows": config.negative_control_shift,
                "control_role": "diagnostic_only_not_gate",
            }
        )
    return pd.DataFrame(rows).sort_values(["horizon", "abs_corr_degradation"], ascending=[True, False])


def build_candidate_gate_decision(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in summary.itertuples(index=False):
        rows.append(
            {
                "signal_id": row.signal_id,
                "horizon": int(row.horizon),
                "proxy_evidence_status": row.proxy_evidence_status,
                "decision": "observe_only_no_default_promotion",
                "reason": "price-index proxy evidence is not official total-return evidence",
                "default_model_allowed": False,
                "portfolio_backtest_allowed": False,
                "official_total_return_evidence": False,
            }
        )
    return pd.DataFrame(rows)


def build_readiness_checks(
    signal_checks: pd.DataFrame,
    label_checks: pd.DataFrame,
    joined: pd.DataFrame,
    summary: pd.DataFrame,
    v3_59_manifest: dict[str, Any],
    config: ProxyValidationConfig,
) -> pd.DataFrame:
    positive_count = int((summary["proxy_evidence_status"] == "proxy_positive_observation").sum()) if not summary.empty else 0
    rows = [
        {
            "check": "v3_59_manifest_accepted",
            "status": _status(bool(v3_59_manifest.get("self_check_pass")) and bool(v3_59_manifest.get("metrics", {}).get("price_proxy_labels_written"))),
            "detail": f"self_check={v3_59_manifest.get('self_check_pass')};labels_written={v3_59_manifest.get('metrics', {}).get('price_proxy_labels_written')}",
        },
        {
            "check": "signal_contract_checks_passed",
            "status": _status(signal_checks["status"].eq("pass").all()),
            "detail": ";".join(signal_checks.loc[signal_checks["status"] != "pass", "check"].astype(str)),
        },
        {
            "check": "price_proxy_label_checks_passed",
            "status": _status(label_checks["status"].eq("pass").all()),
            "detail": ";".join(label_checks.loc[label_checks["status"] != "pass", "check"].astype(str)),
        },
        {
            "check": "joined_proxy_validation_panel_has_rows",
            "status": _status(len(joined) >= config.min_joined_rows),
            "detail": f"rows={len(joined)};min={config.min_joined_rows}",
        },
        {
            "check": "proxy_signal_summary_produced",
            "status": _status(not summary.empty),
            "detail": f"rows={len(summary)};proxy_positive_observations={positive_count}",
        },
        {
            "check": "official_total_return_validation_allowed_now",
            "status": "blocked",
            "detail": "labels are price_index_return only",
        },
        {
            "check": "portfolio_or_model_promotion_allowed_now",
            "status": "blocked",
            "detail": "V3.60 emits proxy diagnostics only",
        },
    ]
    return pd.DataFrame(rows)


def build_no_promotion_guard(summary: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "result_type": "state_stratified_price_proxy_validation",
                "produced": not summary.empty,
                "blocked": summary.empty,
                "reason": "non-official price-index proxy signal diagnostics",
            },
            {
                "result_type": "official_total_return_validation",
                "produced": False,
                "blocked": True,
                "reason": "V3.60 does not use official total-return labels",
            },
            {
                "result_type": "portfolio_backtest",
                "produced": False,
                "blocked": True,
                "reason": "V3.60 is not a portfolio harness",
            },
            {
                "result_type": "model_promotion",
                "produced": False,
                "blocked": True,
                "reason": "proxy evidence cannot promote default model",
            },
        ]
    )


def build_acceptance_checks(
    readiness: pd.DataFrame,
    summary: pd.DataFrame,
    gates: pd.DataFrame,
    guard: pd.DataFrame,
    official_source_exists: bool,
    output_column_names: list[str],
) -> pd.DataFrame:
    unexpected_blocked = readiness.loc[
        (readiness["status"] == "blocked")
        & (~readiness["check"].isin(["official_total_return_validation_allowed_now", "portfolio_or_model_promotion_allowed_now"]))
    ]
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
                "check": "all_proxy_readiness_checks_passed",
                "status": "pass" if unexpected_blocked.empty else "fail",
                "detail": ";".join(unexpected_blocked["check"].astype(str)),
            },
            {
                "check": "summary_is_proxy_only",
                "status": "pass" if not summary.empty and not summary["official_total_return_evidence"].astype(bool).any() else "fail",
                "detail": f"summary_rows={len(summary)}",
            },
            {
                "check": "candidate_gates_do_not_promote_default_model",
                "status": "pass" if not gates.empty and not gates["default_model_allowed"].astype(bool).any() else "fail",
                "detail": f"gate_rows={len(gates)}",
            },
            {
                "check": "official_total_return_source_not_created",
                "status": "pass" if not official_source_exists else "fail",
                "detail": "data_raw/market_labels/market_total_return_index.csv",
            },
            {
                "check": "portfolio_or_promotion_outputs_not_produced",
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
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def build_report(
    joined: pd.DataFrame,
    summary: pd.DataFrame,
    state_coverage: pd.DataFrame,
    state_summary: pd.DataFrame,
    controls: pd.DataFrame,
    readiness: pd.DataFrame,
    acceptance: pd.DataFrame,
) -> str:
    positive = int((summary["proxy_evidence_status"] == "proxy_positive_observation").sum()) if not summary.empty else 0
    lines = [
        "# V3.60 Guarded State-Stratified Proxy Validation",
        "",
        "## Decision",
        "",
        "- V3.60 evaluates V3.50 MARKET signals against V3.59 `price_index_return` proxy labels.",
        "- Results are non-official price-index proxy diagnostics only.",
        "- It does not run a portfolio backtest, write NAV, or promote any default model.",
        "",
        "## Scope",
        "",
        f"- Joined proxy validation rows: `{len(joined)}`",
        f"- Signal-horizon summary rows: `{len(summary)}`",
        f"- Proxy-positive observation rows: `{positive}`",
        "",
        "## Signal-Horizon Summary",
        "",
    ]
    lines.extend(
        markdown_table(
            summary,
            [
                "signal_id",
                "horizon",
                "observations",
                "proxy_spearman_corr",
                "proxy_qspread_top_minus_bottom",
                "top_bucket_directional_alignment_share",
                "proxy_evidence_status",
            ],
            max_rows=20,
        )
    )
    lines.extend(["", "## State Coverage", ""])
    lines.extend(markdown_table(state_coverage, ["state_column", "state_value", "unique_signal_dates", "share_of_joined_dates", "validation_role"], max_rows=24))
    lines.extend(["", "## State-Stratified Sample", ""])
    lines.extend(
        markdown_table(
            state_summary,
            [
                "state_column",
                "state_value",
                "signal_id",
                "horizon",
                "observations",
                "proxy_spearman_corr",
                "proxy_qspread_top_minus_bottom",
                "validation_role",
            ],
            max_rows=20,
        )
    )
    lines.extend(["", "## Negative Control", ""])
    lines.extend(markdown_table(controls, ["signal_id", "horizon", "real_proxy_spearman_corr", "lag_broken_control_spearman_corr", "abs_corr_degradation"], max_rows=20))
    lines.extend(["", "## Readiness", ""])
    lines.extend(markdown_table(readiness, ["check", "status", "detail"], max_rows=16))
    lines.extend(["", "## Acceptance", ""])
    lines.extend(markdown_table(acceptance, ["check", "status", "detail"], max_rows=16))
    lines.extend(
        [
            "",
            "## Next Use",
            "",
            "- Use V3.60 only to prioritize further research questions.",
            "- Do not promote a signal without official total-return labels, walk-forward validation, and a separate portfolio harness.",
            "- V3.61 should inspect the strongest proxy observations and decide whether they are economically plausible or artifacts.",
        ]
    )
    return "\n".join(lines)


def build_catalog(labels_path: Path, summary: pd.DataFrame) -> str:
    positive = int((summary["proxy_evidence_status"] == "proxy_positive_observation").sum()) if not summary.empty else 0
    return "\n".join(
        [
            "# A-share State-Stratified Proxy Validation V3.60",
            "",
            "## Dataset Role",
            "",
            "V3.60 evaluates V3.50 MARKET signals against V3.59 non-official price-index proxy labels.",
            "",
            "## Governance",
            "",
            "- Return basis: `price_index_return` only.",
            "- Official total-return evidence: false.",
            "- Portfolio backtest: not produced.",
            "- Default model promotion: not allowed.",
            "",
            "## Inputs",
            "",
            f"- Proxy labels: `{labels_path}`",
            "",
            "## Produced Shape",
            "",
            f"- Signal-horizon summary rows: `{len(summary)}`",
            f"- Proxy-positive observation rows: `{positive}`",
        ]
    )
