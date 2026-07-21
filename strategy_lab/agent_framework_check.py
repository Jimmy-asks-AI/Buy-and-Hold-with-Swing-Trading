from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

if __package__:
    from .model_run_manifest import SCHEMA_VERSION, validate_model_run_manifest
else:
    from model_run_manifest import SCHEMA_VERSION, validate_model_run_manifest


ROOT = Path(__file__).resolve().parents[1]
AGENTS_DIR = ROOT / "strategy_lab" / "agents"
REPORTS_DIR = ROOT / "reports"
AGENT_RUNS_DIR = ROOT / "outputs" / "agent_runs"
LEGACY_DEBT_ALLOWLIST_PATH = AGENTS_DIR / "_templates" / "legacy_governance_debt_allowlist.json"

AGENTS = [
    "chief_quant_orchestrator",
    "data_steward",
    "factor_researcher",
    "regime_timing_researcher",
    "technical_market_analyst",
    "fundamental_equity_analyst",
    "investment_view_synthesizer",
    "portfolio_risk_engineer",
    "backtest_validation_auditor",
    "execution_cost_analyst",
    "research_reporter",
    "code_quality_engineer",
]

REQUIRED_SECTIONS = [
    "## Role",
    "## Scope",
    "## Context Policy",
    "## Fixed Inputs",
    "## Required Outputs",
    "## Interface Contract",
    "## Forbidden",
    "## Acceptance Criteria",
    "## Quality Gates",
    "## Failure Conditions",
    "## Handoff Format",
]

REQUIRED_FILES = [
    AGENTS_DIR / "README.md",
    AGENTS_DIR / "AGENT_WORKFLOW.md",
    AGENTS_DIR / "RACI_MATRIX.md",
    AGENTS_DIR / "AGENT_IO_CONTRACT.md",
    AGENTS_DIR / "task_briefs" / "README.md",
    AGENTS_DIR / "_templates" / "AGENT_SPEC_TEMPLATE.md",
    AGENTS_DIR / "_templates" / "RESEARCH_TASK_TEMPLATE.md",
    AGENTS_DIR / "_templates" / "agent_run_manifest.schema.json",
    AGENTS_DIR / "_templates" / "model_run_manifest.schema.json",
    AGENTS_DIR / "_templates" / "task_brief.schema.json",
    LEGACY_DEBT_ALLOWLIST_PATH,
    AGENTS_DIR / "_templates" / "candidate_registry.schema.json",
    AGENTS_DIR / "_templates" / "split_manifest.schema.json",
    AGENTS_DIR / "_templates" / "candidate_gate_decision.schema.json",
    REPORTS_DIR / "AGENT_TASK_BOARD.md",
    REPORTS_DIR / "MODEL_DECISION_LOG.md",
    REPORTS_DIR / "AGENT_REVIEW_SUMMARY.md",
]


def check_file(path: Path) -> list[str]:
    if not path.exists():
        return [f"missing file: {path.relative_to(ROOT)}"]
    if not path.read_text(encoding="utf-8").strip():
        return [f"empty file: {path.relative_to(ROOT)}"]
    return []


def check_agent(agent: str) -> list[str]:
    path = AGENTS_DIR / agent / "AGENT.md"
    errors = check_file(path)
    if errors:
        return errors
    text = path.read_text(encoding="utf-8")
    missing = [section for section in REQUIRED_SECTIONS if section not in text]
    return [f"{agent}: missing section {section}" for section in missing]


def check_governance_docs() -> list[str]:
    errors: list[str] = []
    readme = AGENTS_DIR / "README.md"
    if readme.exists():
        text = readme.read_text(encoding="utf-8")
        required_phrases = [
            "HIRSSM V3.10 Clean Rank-Vol Core",
            "hirssm_v3_11_nested_candidate_harness.py",
            "AGENT_WORKFLOW.md",
            "RACI_MATRIX.md",
            "AGENT_IO_CONTRACT.md",
        ]
        for phrase in required_phrases:
            if phrase not in text:
                errors.append(f"README missing governance phrase: {phrase}")
    workflow = AGENTS_DIR / "AGENT_WORKFLOW.md"
    if workflow.exists():
        text = workflow.read_text(encoding="utf-8")
        for phrase in ["Phase Gates", "Minimum Promotion Evidence", "Full-sample ranking", "Research Yield Stop-Loss", "Task Brief Discipline"]:
            if phrase not in text:
                errors.append(f"AGENT_WORKFLOW.md missing phrase: {phrase}")
    raci = AGENTS_DIR / "RACI_MATRIX.md"
    if raci.exists() and "Decision Authority" not in raci.read_text(encoding="utf-8"):
        errors.append("RACI_MATRIX.md missing Decision Authority")
    contract = AGENTS_DIR / "AGENT_IO_CONTRACT.md"
    if contract.exists():
        text = contract.read_text(encoding="utf-8")
        for phrase in ["Task Brief Required Fields", "Required Manifest Fields", "Standard Artifact Names", "Promotion Boundary", "Effectiveness Review", "task_status", "model_decision"]:
            if phrase not in text:
                errors.append(f"AGENT_IO_CONTRACT.md missing phrase: {phrase}")
    return errors


def check_manifest_schema(path: Path) -> list[str]:
    errors = check_file(path)
    if errors:
        return errors
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"invalid json schema: {path.relative_to(ROOT)}: {exc}"]
    required = set(data.get("required", []))
    expected = {
        "run_id",
        "task_id",
        "agent",
        "version",
        "baseline",
        "status",
        "started_at",
        "command",
        "config",
        "data_refs",
        "code_refs",
        "output_dir",
        "allowed_inputs",
        "artifacts",
        "outputs",
        "changed_files",
        "metrics",
        "self_check_pass",
        "fail_count",
        "warn_count",
        "limitations",
        "risk_flags",
        "next_decision",
        "handoff_summary",
    }
    missing = expected - required
    return [f"manifest schema missing required field: {field}" for field in sorted(missing)]


def check_json_schema_file(path: Path) -> list[str]:
    errors = check_file(path)
    if errors:
        return errors
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"invalid json schema: {path.relative_to(ROOT)}: {exc}"]
    if "type" not in data:
        return [f"schema missing type: {path.relative_to(ROOT)}"]
    return []


def check_task_briefs() -> list[str]:
    errors: list[str] = []
    briefs_dir = AGENTS_DIR / "task_briefs"
    schema_path = AGENTS_DIR / "_templates" / "task_brief.schema.json"
    if not briefs_dir.exists():
        return [f"missing task_briefs directory: {briefs_dir.relative_to(ROOT)}"]
    if not schema_path.exists():
        return [f"missing task brief schema: {schema_path.relative_to(ROOT)}"]
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"invalid task brief schema json: {schema_path.relative_to(ROOT)}: {exc}"]
    required = set(schema.get("required", []))
    board_rows = task_board_rows_by_id()
    brief_paths = sorted(briefs_dir.glob("*.json"))
    if not brief_paths:
        errors.append(f"no task brief json files found in {briefs_dir.relative_to(ROOT)}")
        return errors
    for path in brief_paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"invalid task brief json: {path.relative_to(ROOT)}: {exc}")
            continue
        missing = required - set(data)
        for field in sorted(missing):
            errors.append(f"{path.relative_to(ROOT)} missing required field: {field}")
        task_id = str(data.get("task_id", ""))
        if task_id and task_id not in board_rows:
            errors.append(f"{path.relative_to(ROOT)} task_id not found in task board: {task_id}")
        assigned_agent = str(data.get("assigned_agent", "")).strip()
        if assigned_agent and assigned_agent not in AGENTS:
            errors.append(f"{path.relative_to(ROOT)} assigned_agent not in registered roster: {assigned_agent}")
        next_handoff = str(data.get("next_handoff", "")).strip()
        strict_handoff_task = strict_handoff_required(task_id)
        if strict_handoff_task and next_handoff and next_handoff.lower() not in {"none", "n/a"} and next_handoff not in AGENTS:
            errors.append(f"{path.relative_to(ROOT)} next_handoff not in registered roster: {next_handoff}")
        for field in ["allowed_inputs", "allowed_writes", "required_outputs", "forbidden", "acceptance_criteria", "failure_conditions", "quality_gates"]:
            value = data.get(field, [])
            if not isinstance(value, list) or not value:
                errors.append(f"{path.relative_to(ROOT)} field must be a non-empty list: {field}")
        baseline = data.get("baseline", {})
        if not isinstance(baseline, dict) or not {"version", "script", "output_dir"}.issubset(set(baseline)):
            errors.append(f"{path.relative_to(ROOT)} baseline missing version/script/output_dir")
    return errors


def read_task_board_rows() -> list[dict[str, str]]:
    path = REPORTS_DIR / "AGENT_TASK_BOARD.md"
    if not path.exists():
        return []
    rows: list[dict[str, str]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    header: list[str] | None = None
    for line in lines:
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and cells[0] == "---":
            continue
        if "Task ID" in cells:
            header = cells
            continue
        if header and len(cells) == len(header):
            rows.append(dict(zip(header, cells)))
    return rows


def split_refs(text: str) -> list[str]:
    if not text or text.lower() in {"none", "n/a"}:
        return []
    return [item.strip().strip("`") for item in text.split(",") if item.strip()]


def task_board_rows_by_id() -> dict[str, dict[str, str]]:
    return {row.get("Task ID", ""): row for row in read_task_board_rows() if row.get("Task ID")}


def strict_handoff_required(task_id: str) -> bool:
    match = re.match(r"(?P<date>\d{8})_v(?P<major>\d+)_(?P<minor>\d+)", task_id)
    if not match:
        return False
    date = match.group("date")
    major = int(match.group("major"))
    minor = int(match.group("minor"))
    return date > "20260602" or (date == "20260602" and (major, minor) >= (3, 86))


def _normalise_rel_text(value: str) -> str:
    return value.replace("\\", "/").strip()


def load_legacy_debt_allowlist() -> tuple[dict, list[str]]:
    if not LEGACY_DEBT_ALLOWLIST_PATH.exists():
        return {"enabled": False}, []
    try:
        data = json.loads(LEGACY_DEBT_ALLOWLIST_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"enabled": False}, [f"invalid legacy debt allowlist json: {LEGACY_DEBT_ALLOWLIST_PATH.relative_to(ROOT)}: {exc}"]
    if not isinstance(data, dict):
        return {"enabled": False}, [f"legacy debt allowlist must be an object: {LEGACY_DEBT_ALLOWLIST_PATH.relative_to(ROOT)}"]
    paths = set(str(path) for path in data.get("legacy_task_brief_paths", []))
    paths.update(str(path) for path in data.get("legacy_agent_run_manifest_paths", []))
    validation_errors: list[str] = []
    for path in sorted(paths):
        if not (ROOT / path).exists():
            validation_errors.append(f"legacy debt allowlist path missing: {path}")
    return data, validation_errors


def legacy_debt_paths(allowlist: dict) -> set[str]:
    if not allowlist.get("enabled", False):
        return set()
    paths = set(_normalise_rel_text(str(path)) for path in allowlist.get("legacy_task_brief_paths", []))
    paths.update(_normalise_rel_text(str(path)) for path in allowlist.get("legacy_agent_run_manifest_paths", []))
    return paths


def error_path_token(error: str) -> str:
    if not error:
        return ""
    token = error.split(" ", 1)[0]
    return _normalise_rel_text(token)


def partition_legacy_debt_errors(errors: list[str], allowlist: dict) -> tuple[list[str], list[str]]:
    allowed_paths = legacy_debt_paths(allowlist)
    if not allowed_paths:
        return errors, []
    active_errors: list[str] = []
    legacy_errors: list[str] = []
    for error in errors:
        if error_path_token(error) in allowed_paths:
            legacy_errors.append(error)
        else:
            active_errors.append(error)
    return active_errors, legacy_errors


def legacy_debt_summary(errors: list[str]) -> list[dict[str, object]]:
    counts: dict[str, int] = {}
    for error in errors:
        path = error_path_token(error)
        counts[path] = counts.get(path, 0) + 1
    return [{"path": path, "error_count": count} for path, count in sorted(counts.items())]


def check_task_board_outputs() -> list[str]:
    errors: list[str] = []
    for row in read_task_board_rows():
        if row.get("Status") != "accepted":
            continue
        task_id = row.get("Task ID", "unknown")
        for ref in split_refs(row.get("Output Refs", "")):
            path = ROOT / ref.rstrip("/")
            if not path.exists():
                errors.append(f"accepted task {task_id} output ref missing: {ref}")
    return errors


def load_csv_records(path: Path) -> list[dict]:
    df = pd.read_csv(path, encoding="utf-8-sig")
    records: list[dict] = []
    for raw in df.to_dict(orient="records"):
        record: dict = {}
        for key, value in raw.items():
            if pd.isna(value):
                record[key] = None
            elif isinstance(value, str):
                low = value.strip().lower()
                if low == "true":
                    record[key] = True
                elif low == "false":
                    record[key] = False
                else:
                    record[key] = value
            elif hasattr(value, "item"):
                record[key] = value.item()
            else:
                record[key] = value
        records.append(record)
    return records


def schema_required_fields(schema: dict) -> set[str]:
    if schema.get("type") == "array":
        return set(schema.get("items", {}).get("required", []))
    return set(schema.get("required", []))


def check_records_against_schema(records: list[dict], schema_path: Path, label: str) -> list[str]:
    errors: list[str] = []
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    required = schema_required_fields(schema)
    if not records:
        errors.append(f"{label} has no records")
        return errors
    for idx, record in enumerate(records):
        missing = required - set(record)
        for field in sorted(missing):
            errors.append(f"{label} row {idx} missing field: {field}")
    return errors


def candidate_output_dirs() -> list[Path]:
    outputs = ROOT / "outputs"
    if not outputs.exists():
        return []
    dirs = []
    for path in sorted(outputs.glob("hirssm_v*_*/")):
        if path.is_dir() and (
            (path / "candidate_registry.csv").exists()
            or (path / "split_manifest.csv").exists()
            or (path / "candidate_gate_decision.csv").exists()
        ):
            dirs.append(path)
    return dirs


def check_structured_outputs() -> list[str]:
    errors: list[str] = []
    for out_dir in candidate_output_dirs():
        checks = [
            ("candidate_registry", out_dir / "candidate_registry.csv", AGENTS_DIR / "_templates" / "candidate_registry.schema.json"),
            ("split_manifest", out_dir / "split_manifest.csv", AGENTS_DIR / "_templates" / "split_manifest.schema.json"),
            ("candidate_gate_decision", out_dir / "candidate_gate_decision.csv", AGENTS_DIR / "_templates" / "candidate_gate_decision.schema.json"),
        ]
        for label, csv_path, schema_path in checks:
            if not csv_path.exists():
                errors.append(f"missing structured output: {csv_path.relative_to(ROOT)}")
                continue
            try:
                records = load_csv_records(csv_path)
            except Exception as exc:
                errors.append(f"cannot parse {csv_path.relative_to(ROOT)}: {exc}")
                continue
            errors.extend(check_records_against_schema(records, schema_path, f"{out_dir.name}/{label}"))
    return errors


def bool_series(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(dtype=bool)
    return series.map(lambda value: str(value).strip().lower() == "true")


def selection_rows_for_gate(selection: pd.DataFrame) -> pd.DataFrame:
    if selection.empty or "selection_status" not in selection.columns:
        return pd.DataFrame()
    statuses = {"selected_by_prior_window", "baseline_fallback_prior_window"}
    return selection[selection["selection_status"].astype(str).isin(statuses)].copy()


def check_candidate_gate_consistency() -> list[str]:
    errors: list[str] = []
    for out_dir in candidate_output_dirs():
        gate_path = out_dir / "candidate_gate_decision.csv"
        selection_path = out_dir / "nested_selection_by_fold.csv"
        if not gate_path.exists() or not selection_path.exists():
            continue
        try:
            gate = pd.read_csv(gate_path, encoding="utf-8-sig")
            selection = pd.read_csv(selection_path, encoding="utf-8-sig")
        except Exception as exc:
            errors.append(f"cannot parse gate consistency inputs in {out_dir.relative_to(ROOT)}: {exc}")
            continue
        if gate.empty or selection.empty:
            continue
        selected = selection_rows_for_gate(selection)
        if selected.empty:
            errors.append(f"{out_dir.relative_to(ROOT)} has no train-sufficient selected_by_prior_window folds")
            continue
        for _, row in gate.iterrows():
            cost = float(row["cost_bps"])
            cost_selection = selected[selected["cost_bps"].astype(float).eq(cost)]
            if cost_selection.empty:
                errors.append(f"{out_dir.relative_to(ROOT)} gate cost {cost:g} has no matching selected folds")
                continue
            actual_rate = float((cost_selection["selected_variant"].astype(str) != "v3_10_clean_rank_vol_core").mean())
            reported_rate = float(row["nonbaseline_selection_rate"])
            if abs(actual_rate - reported_rate) > 1e-9:
                errors.append(
                    f"{out_dir.relative_to(ROOT)} nonbaseline_selection_rate mismatch at {cost:g}bps: reported={reported_rate:.12f}, actual={actual_rate:.12f}"
                )
            if "baseline_fallback_used" in cost_selection.columns:
                fallback_count = int(bool_series(cost_selection["baseline_fallback_used"]).sum())
                if fallback_count > 0 and "baseline_fallback_used" not in selection.columns:
                    errors.append(f"{out_dir.relative_to(ROOT)} fallback count exists but baseline_fallback_used column missing")
    return errors


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def check_forward_label_signal_gates() -> list[str]:
    errors: list[str] = []
    if not AGENT_RUNS_DIR.exists():
        return errors
    for signal_path in AGENT_RUNS_DIR.rglob("signal_validation.csv"):
        try:
            signal = pd.read_csv(signal_path, encoding="utf-8-sig")
        except Exception as exc:
            errors.append(f"cannot parse {signal_path.relative_to(ROOT)}: {exc}")
            continue
        uses_forward_labels = any("forward" in col.lower() or col.lower().endswith("_fwd") for col in signal.columns)
        if not uses_forward_labels:
            continue
        out_dir = signal_path.parent
        holdout_path = out_dir / "signal_gate_holdout_validation.csv"
        if not holdout_path.exists():
            errors.append(f"{signal_path.relative_to(ROOT)} uses forward labels but missing signal_gate_holdout_validation.csv")
            continue
        try:
            holdout = pd.read_csv(holdout_path, encoding="utf-8-sig")
        except Exception as exc:
            errors.append(f"cannot parse {holdout_path.relative_to(ROOT)}: {exc}")
            continue
        if holdout.empty or "split" not in holdout.columns or "eligible_for_implementation" not in holdout.columns:
            errors.append(f"{holdout_path.relative_to(ROOT)} missing split/eligible_for_implementation evidence")
            continue
        holdout_ok = {
            str(row["variant"]): truthy(row["eligible_for_implementation"])
            for _, row in holdout[holdout["split"].astype(str).eq("holdout")].iterrows()
            if "variant" in holdout.columns
        }
        spec_path = out_dir / "implementation_candidate_spec.csv"
        if spec_path.exists():
            try:
                specs = pd.read_csv(spec_path, encoding="utf-8-sig")
            except Exception as exc:
                errors.append(f"cannot parse {spec_path.relative_to(ROOT)}: {exc}")
                continue
            for _, row in specs.iterrows():
                if truthy(row.get("implementation_allowed", False)):
                    variant = str(row.get("variant", ""))
                    if not holdout_ok.get(variant, False):
                        errors.append(f"{spec_path.relative_to(ROOT)} allows implementation without holdout gate pass: {variant}")
        registry_path = out_dir / "candidate_registry.csv"
        if registry_path.exists():
            try:
                registry = pd.read_csv(registry_path, encoding="utf-8-sig")
            except Exception as exc:
                errors.append(f"cannot parse {registry_path.relative_to(ROOT)}: {exc}")
                continue
            for _, row in registry.iterrows():
                if str(row.get("role", "")).strip().lower() == "candidate" and "signal" in str(row.get("selection_source", "")).lower():
                    variant = str(row.get("variant", ""))
                    if not holdout_ok.get(variant, False):
                        errors.append(f"{registry_path.relative_to(ROOT)} marks signal candidate without holdout gate pass: {variant}")
    return errors


def check_run_manifests() -> list[str]:
    if not AGENT_RUNS_DIR.exists():
        return []
    schema_path = AGENTS_DIR / "_templates" / "agent_run_manifest.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    required = set(schema["required"])
    errors: list[str] = []
    task_rows = task_board_rows_by_id()
    for path in AGENT_RUNS_DIR.rglob("agent_run_manifest.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"invalid run manifest json: {path.relative_to(ROOT)}: {exc}")
            continue
        missing = required - set(data)
        for field in sorted(missing):
            errors.append(f"{path.relative_to(ROOT)} missing field: {field}")
        output_dir = ROOT / str(data.get("output_dir", ""))
        if output_dir and not output_dir.exists():
            errors.append(f"{path.relative_to(ROOT)} output_dir missing: {data.get('output_dir')}")
        for field in ["artifacts", "outputs", "changed_files"]:
            value = data.get(field, [])
            if not isinstance(value, list):
                errors.append(f"{path.relative_to(ROOT)} field is not a list: {field}")
                continue
            for item in value:
                item_text = str(item)
                if "*" in item_text or "?" in item_text:
                    matches = list(ROOT.glob(item_text.replace("\\", "/")))
                    if not matches:
                        errors.append(f"{path.relative_to(ROOT)} {field} glob has no matches: {item}")
                    continue
                artifact_path = ROOT / item_text
                if not artifact_path.exists():
                    errors.append(f"{path.relative_to(ROOT)} {field} path missing: {item}")
        task_id = data.get("task_id")
        if task_id:
            task_board_row = task_rows.get(str(task_id))
            if not task_board_row:
                errors.append(f"{path.relative_to(ROOT)} task_id not found in task board: {task_id}")
            elif task_board_row.get("Status") == "backlog":
                errors.append(f"{path.relative_to(ROOT)} has manifest but task board status is backlog: {task_id}")
        manifest_status = str(data.get("status", "")).strip().lower()
        fail_count = int(data.get("fail_count", 0) or 0)
        if manifest_status == "pass" and fail_count > 0:
            errors.append(f"{path.relative_to(ROOT)} status pass but fail_count is {fail_count}")
    return errors


def artifact_paths_from_model_manifest(data: dict) -> list[str]:
    paths: list[str] = []
    for item in data.get("artifacts", []):
        if isinstance(item, dict):
            path = item.get("path")
        else:
            path = item
        if path:
            paths.append(str(path).replace("\\", "/").strip())
    return paths


def check_model_run_manifests() -> list[str]:
    errors: list[str] = []
    candidates = list((ROOT / "outputs").rglob("model_run_manifest.json")) if (ROOT / "outputs").exists() else []
    candidates += list((ROOT / "outputs").rglob("run_manifest.json")) if (ROOT / "outputs").exists() else []
    for path in candidates:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if data.get("schema_version") != SCHEMA_VERSION:
            continue
        findings = validate_model_run_manifest(data)
        for finding in findings:
            if finding["severity"] == "fail":
                errors.append(f"{path.relative_to(ROOT)} model manifest fail: {finding['field']} - {finding['message']}")
        manifest_rel = path.relative_to(ROOT).as_posix()
        for artifact_path in artifact_paths_from_model_manifest(data):
            artifact_rel = artifact_path
            if artifact_rel.startswith("Introduction-to-Quantitative-Finance/"):
                artifact_rel = artifact_rel.split("Introduction-to-Quantitative-Finance/", 1)[1]
            if artifact_rel == manifest_rel or artifact_rel.endswith("/model_run_manifest_check.csv"):
                errors.append(
                    f"{path.relative_to(ROOT)} model manifest self-references generated manifest/check artifact: {artifact_path}"
                )
    return errors


def main() -> int:
    errors: list[str] = []
    legacy_allowlist, legacy_allowlist_errors = load_legacy_debt_allowlist()
    for path in REQUIRED_FILES:
        errors.extend(check_file(path))
    errors.extend(legacy_allowlist_errors)
    for agent in AGENTS:
        errors.extend(check_agent(agent))
    errors.extend(check_governance_docs())
    errors.extend(check_manifest_schema(AGENTS_DIR / "_templates" / "agent_run_manifest.schema.json"))
    for path in [
        AGENTS_DIR / "_templates" / "model_run_manifest.schema.json",
        AGENTS_DIR / "_templates" / "task_brief.schema.json",
        AGENTS_DIR / "_templates" / "candidate_registry.schema.json",
        AGENTS_DIR / "_templates" / "split_manifest.schema.json",
        AGENTS_DIR / "_templates" / "candidate_gate_decision.schema.json",
    ]:
        errors.extend(check_json_schema_file(path))
    errors.extend(check_task_briefs())
    errors.extend(check_task_board_outputs())
    errors.extend(check_structured_outputs())
    errors.extend(check_candidate_gate_consistency())
    errors.extend(check_forward_label_signal_gates())
    errors.extend(check_run_manifests())
    errors.extend(check_model_run_manifests())
    raw_error_count = len(errors)
    active_errors, legacy_errors = partition_legacy_debt_errors(errors, legacy_allowlist)

    result = {
        "agent_count": len(AGENTS),
        "required_file_count": len(REQUIRED_FILES),
        "required_section_count": len(REQUIRED_SECTIONS),
        "run_manifest_count": len(list(AGENT_RUNS_DIR.rglob("agent_run_manifest.json"))) if AGENT_RUNS_DIR.exists() else 0,
        "self_check_pass": not active_errors,
        "errors": active_errors,
        "raw_error_count": raw_error_count,
        "legacy_debt_count": len(legacy_errors),
        "legacy_debt_path_count": len(legacy_debt_summary(legacy_errors)),
        "legacy_debt_allowlist": str(LEGACY_DEBT_ALLOWLIST_PATH.relative_to(ROOT)),
        "legacy_debt_summary": legacy_debt_summary(legacy_errors),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if active_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
