# Quant Research Subagent Framework Retrospective

## What Has Been Built

The current system has evolved from a single strategy backtest into a governed quant research workflow.

Completed foundation:

- 9 persistent subagent roles under `strategy_lab/agents/`.
- Isolated-context operating rule: agents receive only assigned specs, task briefs, and explicit artifacts.
- Task board, model decision log, and review summary under `reports/`.
- Strict `agent_run_manifest.json` and `model_run_manifest.json` standards.
- V3.10 clean governance baseline: `HIRSSM V3.10 Clean Rank-Vol Core`.
- V3.11 nested candidate validation harness with candidate registry, split manifest, purge/embargo audit, inner selection, outer OOS, PBO, leakage checklist, and gate decision.

Current accepted baseline:

- Baseline: `HIRSSM V3.10 Clean Rank-Vol Core`.
- Validation path: V3.11 nested candidate harness.
- Current candidate decision: `reject_for_default_observation_only`.

## Main Structural Problems Found

The initial agent framework was useful but still too loose in three areas.

1. Baseline drift:
   The agent README and some role specs still referenced V3.8 even after V3.10 became the clean governance baseline.

2. Missing process map:
   Roles existed, but the handoff path from data approval to factor research, portfolio construction, validation, code quality, reporting, and final decision was not explicit enough.

3. Weak interface contract:
   Agents had required outputs, but there was no shared contract for task fields, status semantics, artifact naming, or promotion boundaries.

4. Role-overlap risk:
   Several roles could accidentally imply promotion authority. Validation can block, portfolio can block invalid weights, code quality can block reproducibility, but only the orchestrator can promote.

5. Self-check coverage:
   `agent_framework_check.py` checked basic files and sections, but not the newer governance files or expanded role-interface requirements.

## Optimizations Implemented

### Governance Files

Added:

- `strategy_lab/agents/AGENT_WORKFLOW.md`
- `strategy_lab/agents/RACI_MATRIX.md`
- `strategy_lab/agents/AGENT_IO_CONTRACT.md`

These define phase gates, role accountability, required task fields, artifact names, status terms, and promotion boundaries.

### Agent Specs

Each `AGENT.md` now has:

- `## Interface Contract`
- `## Quality Gates`

The goal is to make every subagent remember:

- what it may consume;
- what it may write;
- what status terms it can use;
- what it is forbidden to decide;
- which gates must pass before handoff.

### Baseline and Promotion Flow

Updated the framework baseline from V3.8 to:

- `HIRSSM V3.10 Clean Rank-Vol Core`

Updated the validation path to:

- `HIRSSM V3.11 Nested Candidate Harness`

The default flow is now:

1. Orchestrator defines task.
2. Data steward approves data.
3. Factor or regime researcher creates signal evidence.
4. Portfolio risk engineer constructs weights.
5. Execution cost analyst tests costs and capacity.
6. Backtest validation auditor reviews leakage and robustness.
7. Code quality engineer checks scripts, outputs, schemas, and manifests.
8. Research reporter writes final report.
9. Orchestrator records promote, observe, reject, or block.

### Self-Check

`strategy_lab/agent_framework_check.py` now requires:

- 11 governance files;
- 11 required sections in every agent spec;
- governance phrases in README/workflow/contract files;
- strict model manifest validation for model outputs.

Current result:

- `agent_count`: 9
- `required_file_count`: 11
- `required_section_count`: 11
- `run_manifest_count`: 10
- `self_check_pass`: true

## Role Improvements

| Agent | Key Clarification |
| --- | --- |
| `chief_quant_orchestrator` | Only role allowed to promote; owns task board and decision log. |
| `data_steward` | Owns PIT and dataset approval; can mark data `research_only`. |
| `factor_researcher` | Owns factor definitions and evidence, not portfolio or promotion. |
| `regime_timing_researcher` | Owns timing state definitions, not direct target weights or promotion. |
| `portfolio_risk_engineer` | Owns constrained weights and risk controls, not alpha validity. |
| `execution_cost_analyst` | Owns net-of-cost and capacity evidence, not factor definitions. |
| `backtest_validation_auditor` | Can block promotion; cannot tune candidates or promote. |
| `research_reporter` | Owns communication and dashboards; cannot change decisions. |
| `code_quality_engineer` | Owns reproducibility and output integrity; not research validity. |

## Remaining Gaps

- No automated enforcement yet that each new task board row has a matching task brief file.
- No machine-readable schema yet for `candidate_registry.csv`, `split_manifest.csv`, and validation reports.
- No central dashboard showing every agent task status and gate status together.
- No dedicated `risk_committee` role; currently the orchestrator absorbs final risk committee responsibility.
- No live data-refresh monitor; data quality is checked per task.

## Recommended Next Optimizations

1. Add `task_briefs/` with one markdown or JSON brief per task board row.
2. Add schemas for candidate registry, split manifest, leakage checklist, and gate decision.
3. Extend `agent_framework_check.py` to verify required outputs for every accepted task board row.
4. Create an agent dashboard HTML that shows baseline, candidate status, gate status, and blocked reasons.
5. Add a small `risk_committee` review step or checklist before any future default promotion.
