# Research Task Template

## Task ID

`YYYYMMDD_<version>_<agent>_<short_name>`

## Parent ID

Use `none` for root tasks.

## Assigned Agent

Name one agent only.

## Status

Allowed values: `backlog`, `ready`, `in_progress`, `blocked`, `review`, `accepted`, `rejected`, `archived`.

## Objective

One concrete outcome.

## Baseline

- Version:
- Script:
- Output directory:
- Metrics to compare:

## Allowed Inputs

List exact files or directories.

## Allowed Writes

List exact output directory. Use a unique path under `outputs/agent_runs/<version>/<agent>/`.

## Dependencies

List task IDs that must finish first.

## Required Outputs

- `agent_run_manifest.json`
- `agent_report.md`
- role-specific CSV/JSON/MD artifacts
- `changed_files.txt` if edits are made

## Interface Contract

- Standard status: `pass`, `observation`, `fail`, or `blocked`.
- Standard artifact names must follow `AGENT_IO_CONTRACT.md`.
- Any model-producing task must emit `model_run_manifest.json`.
- Full-sample metrics are diagnostic unless this task explicitly says otherwise.

## Forbidden

List task-specific forbidden actions.

## Acceptance Criteria

List measurable completion checks.

## Quality Gates

List gates that must pass before handoff, including data timing, leakage, cost, reproducibility, and output integrity when relevant.

## Handoff

Write the final decision-relevant summary for the orchestrator.

## Next Handoff

Name the next agent or `chief_quant_orchestrator` if the task is ready for decision.
