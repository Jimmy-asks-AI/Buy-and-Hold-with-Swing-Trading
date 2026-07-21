# Subagent Framework Effectiveness Review V4

## Critical Judgment

The multi-agent framework has been effective as a governance and falsification system, but not yet effective as an alpha-production system.

It did the important defensive job:

- weak candidates were not promoted;
- forward-label and holdout boundaries became explicit;
- nested/PBO gates stopped unstable variants;
- reproducibility and manifest defects are now caught by framework checks.

But it also shows a structural weakness:

- 6 model-producing harnesses have all been rejected for default promotion;
- 39 accepted tasks produced only governance-safe negative evidence, not a better default model;
- the research loop became too serial, often repairing one branch rather than exploring independent sources;
- `accepted` task status can be misread as `promoted` model status unless reports make the distinction explicit.

## Quantitative Scorecard

- Model harnesses with gate decisions: `6`
- Promoted models: `0`
- Default-rejected models: `6`
- Average 10bps annual delta versus V3.10: `-0.1071%`
- Best 10bps annual delta versus V3.10: `+0.0686%`
- Average 10bps PBO: `0.4484`
- Manifest pass rate: `100%`
- Machine-readable task briefs introduced: `1`

## Structural Defects

1. `accepted_task_vs_model_promotion_confusion`
   - Task artifacts can be accepted while model decisions remain rejected.
   - Reports and manifests must separate `task_status` from `model_decision`.

2. `low_model_yield_after_many_versions`
   - The system is strong at blocking weak candidates but weak at generating robust candidates.
   - Five failed model versions should trigger source discovery or a cross-version review.

3. `validation_agent_overload`
   - The validation auditor has often implemented harnesses and attribution.
   - Portfolio construction hypotheses should be owned by portfolio/risk before validation.

4. `task_brief_not_machine_enforced`
   - The schema existed but no task brief directory existed.
   - Task brief validation is now added for present JSON briefs.

5. `post_hoc_learning_can_turn_into_parameter_search`
   - Failure attribution naturally tempts threshold and amplitude tuning.
   - Attribution findings must become a new predeclared hypothesis brief before implementation.

## Optimizations Implemented

- Added `strategy_lab/subagent_effectiveness_review.py`.
- Added machine-readable `strategy_lab/agents/task_briefs/`.
- Added the current governance task brief.
- Extended `agent_framework_check.py` to validate present task briefs.
- Updated workflow rules for:
  - task/model decision separation;
  - five-version research-yield stop-loss;
  - attribution-to-hypothesis boundaries;
  - task brief discipline.
- Updated RACI to reduce validation-agent role overlap.

## Next Operating Rule

Before another chain of small implementation versions, require a task brief that states whether the next step is:

- source/data discovery;
- independent signal batch research;
- portfolio construction redesign;
- or validation of a predeclared candidate.

If five more model-producing versions still fail to beat V3.10 after costs and PBO, stop implementation work and return to source discovery.
