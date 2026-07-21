#!/usr/bin/env python
"""Generate concise model run reports from quant model system outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _table_block(table: pd.DataFrame, max_rows: int = 20) -> list[str]:
    if table.empty:
        return ["empty", ""]
    return ["```text", table.head(max_rows).to_string(index=False), "```", ""]


def _status_counts(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty or "status" not in table.columns:
        return pd.DataFrame()
    return table["status"].value_counts(dropna=False).rename_axis("status").reset_index(name="count")


def _load_paper_state(run_dir: Path) -> dict[str, object]:
    path = run_dir / "paper_tracking" / "paper_state.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def generate_model_run_report(run_dir: str | Path, output_path: str | Path | None = None) -> Path:
    run = Path(run_dir)
    output = Path(output_path) if output_path else run / "MODEL_RUN_REPORT.md"
    panel_validation = _read_csv(run / "panel_validation.csv")
    data_quality_gates = _read_csv(run / "data_quality" / "gates.csv")
    data_quality_summary = _read_csv(run / "data_quality" / "summary.csv")
    registry_summary = _read_csv(run / "registry_summary.csv")
    meta = _read_csv(run / "walk_forward" / "meta.csv")
    performance = _read_csv(run / "walk_forward" / "walk_forward_performance.csv")
    split_summary = _read_csv(run / "walk_forward" / "split_summary.csv")
    selected = _read_csv(run / "walk_forward" / "selected.csv")
    ledger_row = _read_csv(run / "ledger_row.csv")
    checklist = _read_csv(run / "paper_tracking" / "paper_monitoring_checklist.csv")
    drift_summary = _read_csv(run / "paper_tracking" / "paper_drift_summary.csv")
    paper_state = _load_paper_state(run)

    fail_count = int((panel_validation.get("status", pd.Series(dtype=str)) == "fail").sum()) if not panel_validation.empty else 0
    warn_count = int((panel_validation.get("status", pd.Series(dtype=str)) == "warn").sum()) if not panel_validation.empty else 0
    dq_fail_count = int((data_quality_gates.get("status", pd.Series(dtype=str)) == "fail").sum()) if not data_quality_gates.empty else 0
    dq_warn_count = int((data_quality_gates.get("status", pd.Series(dtype=str)) == "warn").sum()) if not data_quality_gates.empty else 0
    decision = str(ledger_row["decision"].iloc[-1]) if not ledger_row.empty and "decision" in ledger_row.columns else "unknown"
    live_allowed = bool(paper_state.get("live_trading_allowed", False))
    selected_factors = int(selected["factor"].nunique()) if not selected.empty and "factor" in selected.columns else int(selected.shape[0])
    selected_families = int(selected["family"].nunique()) if not selected.empty and "family" in selected.columns else 0

    lines = [
        "# Model Run Report",
        "",
        f"Run directory: `{run}`",
        "",
        "## Decision",
        "",
        f"- Ledger decision: `{decision}`",
        f"- Panel validation fails: `{fail_count}`",
        f"- Panel validation warnings: `{warn_count}`",
        f"- Data-quality gate fails: `{dq_fail_count}`",
        f"- Data-quality gate warnings: `{dq_warn_count}`",
        f"- Selected factors: `{selected_factors}`",
        f"- Selected families: `{selected_families}`",
        f"- Paper status: `{paper_state.get('status', 'missing')}`",
        f"- Live trading allowed: `{str(live_allowed).lower()}`",
        "",
        "Interpretation: `promote_to_paper` only permits paper tracking. It is not live-trading approval.",
        "",
        "## Panel Validation Status",
        "",
    ]
    lines.extend(_table_block(_status_counts(panel_validation)))
    lines.extend(["## Data Quality Summary", ""])
    lines.extend(_table_block(data_quality_summary))
    lines.extend(["## Data Quality Gates", ""])
    lines.extend(_table_block(data_quality_gates, max_rows=50))
    lines.extend(["## Walk-forward Meta", ""])
    lines.extend(_table_block(meta))
    lines.extend(["## Walk-forward Performance", ""])
    lines.extend(_table_block(performance))
    lines.extend(["## Split Summary", ""])
    lines.extend(_table_block(split_summary))
    lines.extend(["## Registry Summary", ""])
    lines.extend(_table_block(registry_summary))
    lines.extend(["## Ledger Row", ""])
    lines.extend(_table_block(ledger_row))
    lines.extend(["## Paper State", ""])
    if paper_state:
        lines.extend(["```json", json.dumps(paper_state, ensure_ascii=False, indent=2), "```", ""])
    else:
        lines.extend(["missing", ""])
    lines.extend(["## Paper Drift Summary", ""])
    lines.extend(_table_block(drift_summary))
    lines.extend(["## Monitoring Checklist", ""])
    lines.extend(_table_block(checklist, max_rows=50))
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    path = generate_model_run_report(args.run_dir, args.output)
    print(f"saved={path.resolve()}")


if __name__ == "__main__":
    main()
