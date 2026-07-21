# Agent Spec Template

## Role

State the agent's single durable responsibility.

## Scope

Define what the agent owns and what it does not own.

## Context Policy

- Default to isolated context.
- Read only the files listed in the task brief.
- Share only structured artifacts.

## Fixed Inputs

- Agent spec: this file or the role-specific `AGENT.md`.
- Task brief: objective, version, baseline, allowed paths, output directory.
- Required baseline: usually `HIRSSM V3.8` unless the orchestrator states otherwise.
- Allowed artifacts: list exact files or directories.

## Required Outputs

- `agent_run_manifest.json`
- `agent_report.md`
- role-specific tables or metrics
- `changed_files.txt` if code or documents are edited

## Interface Contract

- Consume only paths listed in `allowed_inputs`.
- Write only paths listed in `allowed_writes`.
- Use standard artifact names from `AGENT_IO_CONTRACT.md`.
- Set status to `pass`, `observation`, `fail`, or `blocked`.
- State whether the result is eligible for default promotion within this role's authority.

## Forbidden

- Do not promote experiments into the default model.
- Do not use future data, current index constituents, or current weights in historical tests.
- Do not overwrite another agent's output directory.
- Do not optimize the target metric outside the assigned scope.

## Acceptance Criteria

Define objective pass/fail checks for the role.

## Quality Gates

List reproducibility, leakage, cost, data, or schema gates that must pass before handoff.

## Failure Conditions

Define conditions that require a failed or blocked status.

## Handoff Format

Summarize findings in a structure the orchestrator can merge without reading scratch work.
