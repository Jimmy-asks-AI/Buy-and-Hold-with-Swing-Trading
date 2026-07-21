"""JoinQuant MARKET proxy probe for HIRSSM V3.55.

This module verifies whether JQData can supply a market proxy series. It does
not treat ordinary index price data as a total-return label source.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_PROXY_OUTPUT_COLUMNS = [
    "date",
    "asset_or_index",
    "total_return_index_or_adjusted_close",
    "available_date",
    "data_source",
    "source_vintage",
]

FORBIDDEN_PROXY_TERMS = {
    "raw_close",
    "price_only",
    "unadjusted",
    "benchmark_nav",
    "strategy_nav",
    "backtest",
}


@dataclass(frozen=True)
class JoinQuantProbeConfig:
    output_dir: Path
    catalog_path: Path
    target_source_path: Path
    target_security: str
    target_asset_or_index: str
    start_date: str
    end_date: str
    fields: tuple[str, ...]
    fq: str
    execute_probe: bool
    allow_adjusted_proxy_write: bool
    approved_proxy_basis: str
    allow_overwrite_target: bool


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_credentials(root: Path) -> dict[str, Any]:
    path = root / "configs" / "data_credentials.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def _truthy_secret(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    upper = text.upper()
    if "PASTE_" in upper or upper in {"NONE", "NULL", "TODO"}:
        return False
    return True


def credential_pair(root: Path) -> tuple[str | None, str | None]:
    credentials = read_credentials(root)
    jq_cfg = credentials.get("joinquant", {}) if isinstance(credentials, dict) else {}
    username = os.getenv("JQDATA_USERNAME") or os.getenv("JOINQUANT_USERNAME") or os.getenv("JQ_USER") or jq_cfg.get("username")
    password = os.getenv("JQDATA_PASSWORD") or os.getenv("JOINQUANT_PASSWORD") or os.getenv("JQ_PASSWORD") or jq_cfg.get("password")
    return username if _truthy_secret(username) else None, password if _truthy_secret(password) else None


def get_price_signature_detail() -> tuple[bool, str]:
    if importlib.util.find_spec("jqdatasdk") is None:
        return False, "jqdatasdk_not_installed"
    try:
        import jqdatasdk as jq  # type: ignore

        sig = inspect.signature(jq.get_price)
        params = set(sig.parameters)
        required = {"security", "start_date", "end_date", "frequency", "fields", "fq"}
        missing = sorted(required.difference(params))
        return not missing, f"signature={sig};missing={missing}"
    except Exception as exc:  # pragma: no cover - defensive SDK introspection.
        return False, f"{type(exc).__name__}: {exc}"


def readiness_frame(root: Path, config: JoinQuantProbeConfig) -> pd.DataFrame:
    username, password = credential_pair(root)
    sig_ok, sig_detail = get_price_signature_detail()
    sdk_ready = importlib.util.find_spec("jqdatasdk") is not None
    rows = [
        {
            "check": "jqdatasdk_installed",
            "ready": sdk_ready,
            "status": "pass" if sdk_ready else "blocked",
            "detail": "python package jqdatasdk",
        },
        {
            "check": "joinquant_username_present",
            "ready": username is not None,
            "status": "pass" if username is not None else "blocked",
            "detail": "env var or configs/data_credentials.json username",
        },
        {
            "check": "joinquant_password_present",
            "ready": password is not None,
            "status": "pass" if password is not None else "blocked",
            "detail": "env var or configs/data_credentials.json password",
        },
        {
            "check": "get_price_signature_supported",
            "ready": sig_ok,
            "status": "pass" if sig_ok else "blocked",
            "detail": sig_detail,
        },
        {
            "check": "target_security_configured",
            "ready": bool(config.target_security),
            "status": "pass" if config.target_security else "blocked",
            "detail": config.target_security,
        },
        {
            "check": "adjusted_proxy_write_policy_approved",
            "ready": config.allow_adjusted_proxy_write and config.approved_proxy_basis == "adjusted_proxy",
            "status": "pass" if config.allow_adjusted_proxy_write and config.approved_proxy_basis == "adjusted_proxy" else "blocked",
            "detail": "write requires allow_adjusted_proxy_write=true and approved_proxy_basis=adjusted_proxy",
        },
    ]
    return pd.DataFrame(rows)


def credentials_ready(readiness: pd.DataFrame) -> bool:
    required = {
        "jqdatasdk_installed",
        "joinquant_username_present",
        "joinquant_password_present",
        "get_price_signature_supported",
        "target_security_configured",
    }
    rows = readiness.loc[readiness["check"].isin(required)]
    return bool(len(rows) == len(required) and rows["ready"].astype(bool).all())


def build_probe_plan(config: JoinQuantProbeConfig, readiness: pd.DataFrame) -> pd.DataFrame:
    creds_ready = credentials_ready(readiness)
    return pd.DataFrame(
        [
            {
                "probe_id": "jq_get_price_market_proxy",
                "provider": "joinquant",
                "endpoint": "jqdatasdk.get_price",
                "target_security": config.target_security,
                "start_date": config.start_date,
                "end_date": config.end_date,
                "fields": ",".join(config.fields),
                "fq": config.fq,
                "execute_probe": config.execute_probe,
                "ready_to_execute": bool(config.execute_probe and creds_ready),
                "expected_output": "daily market proxy price series for capability check",
                "acceptance_boundary": "probe success is not total-return-source acceptance",
            },
            {
                "probe_id": "adjusted_proxy_write_gate",
                "provider": "local",
                "endpoint": "write_market_total_return_index_csv",
                "target_security": config.target_security,
                "start_date": config.start_date,
                "end_date": config.end_date,
                "fields": ",".join(config.fields),
                "fq": config.fq,
                "execute_probe": config.execute_probe,
                "ready_to_execute": bool(
                    config.execute_probe
                    and creds_ready
                    and config.allow_adjusted_proxy_write
                    and config.approved_proxy_basis == "adjusted_proxy"
                ),
                "expected_output": config.target_source_path.as_posix(),
                "acceptance_boundary": "write only after explicit adjusted_proxy approval",
            },
        ]
    )


def normalize_probe_result(data: pd.DataFrame, config: JoinQuantProbeConfig) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame(columns=REQUIRED_PROXY_OUTPUT_COLUMNS)
    work = data.copy()
    if "time" in work.columns:
        date_values = pd.to_datetime(work["time"], errors="coerce")
    else:
        date_values = pd.to_datetime(work.index, errors="coerce")
    if "close" not in work.columns:
        raise ValueError("JoinQuant probe data missing close column")
    close = pd.to_numeric(work["close"], errors="coerce")
    dates = date_values.strftime("%Y%m%d")
    out = pd.DataFrame(
        {
            "date": dates,
            "asset_or_index": config.target_asset_or_index,
            "total_return_index_or_adjusted_close": close,
            "available_date": dates,
            "data_source": f"jqdatasdk.get_price.{config.target_security}.fq_{config.fq}.approved_adjusted_proxy",
            "source_vintage": f"probe_fetched_at_{now_text()}",
        }
    )
    return out.dropna(subset=["date", "total_return_index_or_adjusted_close"]).sort_values("date")


def execute_joinquant_probe(root: Path, config: JoinQuantProbeConfig, readiness: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not config.execute_probe:
        return (
            pd.DataFrame(
                [
                    {
                        "probe_id": "jq_get_price_market_proxy",
                        "status": "not_executed",
                        "rows": 0,
                        "detail": "execute_probe=false",
                    }
                ]
            ),
            pd.DataFrame(columns=REQUIRED_PROXY_OUTPUT_COLUMNS),
        )
    if not credentials_ready(readiness):
        blocked = ",".join(readiness.loc[~readiness["ready"].astype(bool), "check"].astype(str))
        return (
            pd.DataFrame(
                [
                    {
                        "probe_id": "jq_get_price_market_proxy",
                        "status": "blocked",
                        "rows": 0,
                        "detail": blocked,
                    }
                ]
            ),
            pd.DataFrame(columns=REQUIRED_PROXY_OUTPUT_COLUMNS),
        )

    username, password = credential_pair(root)
    import jqdatasdk as jq  # type: ignore

    jq.auth(username, password)
    raw = jq.get_price(
        config.target_security,
        start_date=config.start_date,
        end_date=config.end_date,
        frequency="daily",
        fields=list(config.fields),
        fq=config.fq,
    )
    if not isinstance(raw, pd.DataFrame):
        raise TypeError(f"unexpected get_price result type: {type(raw).__name__}")
    normalized = normalize_probe_result(raw, config)
    probe_result = pd.DataFrame(
        [
            {
                "probe_id": "jq_get_price_market_proxy",
                "status": "ok",
                "rows": int(len(raw)),
                "detail": f"normalized_rows={len(normalized)}",
            }
        ]
    )
    return probe_result, normalized


def write_policy_frame(config: JoinQuantProbeConfig, normalized: pd.DataFrame) -> pd.DataFrame:
    source_text = ""
    if not normalized.empty:
        source_text = " ".join(normalized["data_source"].astype(str).head(5)).lower()
    forbidden_hit = any(term in source_text for term in FORBIDDEN_PROXY_TERMS)
    approved = bool(config.allow_adjusted_proxy_write and config.approved_proxy_basis == "adjusted_proxy" and not forbidden_hit)
    return pd.DataFrame(
        [
            {
                "policy": "write_market_total_return_index_csv",
                "approved": approved,
                "target_path": config.target_source_path.as_posix(),
                "reason": "explicit adjusted_proxy approval" if approved else "write blocked unless adjusted proxy is explicitly approved",
                "rows_available": int(len(normalized)),
            },
            {
                "policy": "treat_probe_as_certified_total_return",
                "approved": False,
                "target_path": config.target_source_path.as_posix(),
                "reason": "JoinQuant get_price probe is not official total-return certification",
                "rows_available": int(len(normalized)),
            },
        ]
    )


def maybe_write_proxy_source(config: JoinQuantProbeConfig, normalized: pd.DataFrame, write_policy: pd.DataFrame) -> pd.DataFrame:
    approved = bool(write_policy.loc[write_policy["policy"] == "write_market_total_return_index_csv", "approved"].iloc[0])
    if not approved:
        return pd.DataFrame(
            [
                {
                    "action": "write_target_source",
                    "status": "blocked",
                    "rows": 0,
                    "detail": "write policy not approved",
                }
            ]
        )
    if normalized.empty:
        return pd.DataFrame(
            [
                {
                    "action": "write_target_source",
                    "status": "blocked",
                    "rows": 0,
                    "detail": "no normalized probe rows",
                }
            ]
        )
    if config.target_source_path.exists() and not config.allow_overwrite_target:
        return pd.DataFrame(
            [
                {
                    "action": "write_target_source",
                    "status": "blocked",
                    "rows": 0,
                    "detail": "target exists and allow_overwrite_target=false",
                }
            ]
        )
    config.target_source_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(config.target_source_path, index=False, encoding="utf-8-sig")
    return pd.DataFrame(
        [
            {
                "action": "write_target_source",
                "status": "written",
                "rows": int(len(normalized)),
                "detail": config.target_source_path.as_posix(),
            }
        ]
    )


def build_readiness_checks(
    readiness: pd.DataFrame,
    probe_result: pd.DataFrame,
    write_policy: pd.DataFrame,
    write_result: pd.DataFrame,
    config: JoinQuantProbeConfig,
) -> pd.DataFrame:
    creds_ready = credentials_ready(readiness)
    probe_ok = bool((probe_result["status"] == "ok").any()) if not probe_result.empty else False
    write_approved = bool(write_policy.loc[write_policy["policy"] == "write_market_total_return_index_csv", "approved"].iloc[0])
    target_written = bool((write_result["status"] == "written").any()) if not write_result.empty else False
    return pd.DataFrame(
        [
            {
                "check": "joinquant_credentials_ready",
                "status": "pass" if creds_ready else "blocked",
                "detail": ",".join(readiness.loc[~readiness["ready"].astype(bool), "check"].astype(str)),
            },
            {
                "check": "probe_execution_allowed",
                "status": "pass" if config.execute_probe and creds_ready else "blocked",
                "detail": f"execute_probe={config.execute_probe}",
            },
            {
                "check": "get_price_probe_succeeded",
                "status": "pass" if probe_ok else "blocked",
                "detail": ";".join(probe_result["detail"].astype(str)) if not probe_result.empty else "",
            },
            {
                "check": "adjusted_proxy_write_approved",
                "status": "pass" if write_approved else "blocked",
                "detail": "requires allow_adjusted_proxy_write=true and approved_proxy_basis=adjusted_proxy",
            },
            {
                "check": "target_source_written",
                "status": "pass" if target_written else "blocked",
                "detail": ";".join(write_result["detail"].astype(str)) if not write_result.empty else "",
            },
            {
                "check": "performance_validation_allowed_now",
                "status": "blocked",
                "detail": "V3.55 is probe/write gate only; V3.53 must validate source and V3.49 must validate signals.",
            },
        ]
    )


def build_no_promotion_guard(write_result: pd.DataFrame) -> pd.DataFrame:
    target_written = bool((write_result["status"] == "written").any()) if not write_result.empty else False
    return pd.DataFrame(
        [
            {
                "result_type": "joinquant_market_proxy_probe",
                "produced": False,
                "blocked": True,
                "reason": "probe result is capability evidence, not a performance result",
            },
            {
                "result_type": "market_total_return_source_csv",
                "produced": target_written,
                "blocked": not target_written,
                "reason": "written only if explicit adjusted proxy write gate passed",
            },
            {
                "result_type": "labels_or_state_validation_or_backtest",
                "produced": False,
                "blocked": True,
                "reason": "V3.55 does not run labels, validation, NAV, Sharpe, drawdown, or model promotion",
            },
        ]
    )


def build_acceptance_checks(
    readiness: pd.DataFrame,
    probe_plan: pd.DataFrame,
    write_policy: pd.DataFrame,
    guard: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "check": "credentials_not_printed",
                "status": "pass",
                "detail": "readiness outputs booleans and generic details only",
            },
            {
                "check": "probe_plan_has_acceptance_boundary",
                "status": "pass" if probe_plan["acceptance_boundary"].astype(str).str.len().gt(0).all() else "fail",
                "detail": f"probe_rows={len(probe_plan)}",
            },
            {
                "check": "ordinary_get_price_not_certified_total_return",
                "status": "pass"
                if not bool(write_policy.loc[write_policy["policy"] == "treat_probe_as_certified_total_return", "approved"].iloc[0])
                else "fail",
                "detail": "certified total return remains false",
            },
            {
                "check": "labels_and_backtest_not_run",
                "status": "pass"
                if not bool(guard.loc[guard["result_type"] == "labels_or_state_validation_or_backtest", "produced"].iloc[0])
                else "fail",
                "detail": "probe only",
            },
            {
                "check": "missing_credentials_block_probe",
                "status": "pass"
                if credentials_ready(readiness) or not bool(probe_plan["ready_to_execute"].any())
                else "fail",
                "detail": "ready_to_execute requires credentials",
            },
        ]
    )


def credential_instructions() -> str:
    return "\n".join(
        [
            "# JoinQuant Credential Instructions V3.55",
            "",
            "To run the JoinQuant probe, provide credentials locally by either:",
            "",
            "```powershell",
            "$env:JQDATA_USERNAME = '<your_joinquant_username>'",
            "$env:JQDATA_PASSWORD = '<your_joinquant_password>'",
            "```",
            "",
            "or fill the local-only section in `configs/data_credentials.json`:",
            "",
            "```json",
            "{\"joinquant\":{\"username\":\"...\",\"password\":\"...\"}}",
            "```",
            "",
            "Then run V3.55 with `--execute-probe`.",
            "",
            "Writing `data_raw/market_labels/market_total_return_index.csv` also requires config approval:",
            "",
            "- `allow_adjusted_proxy_write`: `true`",
            "- `approved_proxy_basis`: `adjusted_proxy`",
            "",
            "This remains an adjusted proxy, not an official total-return index.",
            "",
        ]
    )


def build_report(
    config: JoinQuantProbeConfig,
    readiness: pd.DataFrame,
    probe_plan: pd.DataFrame,
    probe_result: pd.DataFrame,
    write_policy: pd.DataFrame,
    write_result: pd.DataFrame,
    checks: pd.DataFrame,
    acceptance: pd.DataFrame,
) -> str:
    lines = [
        "# V3.55 JoinQuant Market Proxy Probe",
        "",
        "## Decision",
        "",
        "- V3.55 adds a JoinQuant `get_price` capability probe for a MARKET proxy series.",
        "- Default behavior does not execute network calls and does not write V3.53 input.",
        "- Ordinary JoinQuant index price data is not promoted as certified total-return data.",
        "- No labels, IC, returns, NAV, drawdown, Sharpe, portfolio backtest, or model promotion is produced.",
        "",
        "## Config",
        "",
        f"- Target security: `{config.target_security}`",
        f"- Date range: `{config.start_date}` to `{config.end_date}`",
        f"- FQ mode: `{config.fq}`",
        f"- Execute probe: `{config.execute_probe}`",
        f"- Allow adjusted proxy write: `{config.allow_adjusted_proxy_write}`",
        f"- Approved proxy basis: `{config.approved_proxy_basis}`",
        "",
        "## Readiness",
        "",
        "| check | status | ready | detail |",
        "|---|---|---|---|",
    ]
    for row in readiness.itertuples(index=False):
        lines.append(f"| `{row.check}` | `{row.status}` | `{row.ready}` | {row.detail} |")
    lines.extend(
        [
            "",
            "## Probe Plan",
            "",
            "| probe_id | endpoint | ready_to_execute | acceptance_boundary |",
            "|---|---|---|---|",
        ]
    )
    for row in probe_plan.itertuples(index=False):
        lines.append(f"| `{row.probe_id}` | `{row.endpoint}` | `{row.ready_to_execute}` | {row.acceptance_boundary} |")
    lines.extend(
        [
            "",
            "## Probe Result",
            "",
            "| probe_id | status | rows | detail |",
            "|---|---|---:|---|",
        ]
    )
    for row in probe_result.itertuples(index=False):
        lines.append(f"| `{row.probe_id}` | `{row.status}` | {int(row.rows)} | {row.detail} |")
    lines.extend(
        [
            "",
            "## Write Policy",
            "",
            "| policy | approved | rows_available | reason |",
            "|---|---|---:|---|",
        ]
    )
    for row in write_policy.itertuples(index=False):
        lines.append(f"| `{row.policy}` | `{row.approved}` | {int(row.rows_available)} | {row.reason} |")
    lines.extend(
        [
            "",
            "## Write Result",
            "",
            "| action | status | rows | detail |",
            "|---|---|---:|---|",
        ]
    )
    for row in write_result.itertuples(index=False):
        lines.append(f"| `{row.action}` | `{row.status}` | {int(row.rows)} | {row.detail} |")
    lines.extend(
        [
            "",
            "## Checks",
            "",
            "| check | status | detail |",
            "|---|---|---|",
        ]
    )
    for row in checks.itertuples(index=False):
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
            "- Add JoinQuant credentials locally, rerun V3.55 with `--execute-probe`, and inspect probe rows.",
            "- Only after explicit adjusted-proxy approval should V3.55 write the V3.53 source CSV.",
            "- Rerun V3.53 after a source CSV exists; then V3.49 can validate signals.",
            "",
        ]
    )
    return "\n".join(lines)


def build_catalog(checks: pd.DataFrame, write_result: pd.DataFrame) -> str:
    target_written = bool((write_result["status"] == "written").any()) if not write_result.empty else False
    perf_status = str(checks.loc[checks["check"] == "performance_validation_allowed_now", "status"].iloc[0])
    return "\n".join(
        [
            "# A-share JoinQuant Market Proxy Probe V3.55",
            "",
            "## Dataset Decision",
            "",
            "- JoinQuant probe harness ready: `true`",
            f"- Target source written: `{target_written}`",
            f"- Performance validation status: `{perf_status}`",
            "- Ordinary `get_price` output is not an official total-return source.",
            "",
        ]
    )
