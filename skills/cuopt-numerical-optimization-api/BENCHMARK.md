# Skill Benchmark: cuopt-numerical-optimization-api

> ❌ **Overall verdict: FAIL — Publication blocked**

The skill should be reviewed before publication. Address the blocking findings below, then rerun Skill Evaluator.

## Evaluation Metadata

- Skill: `cuopt-numerical-optimization-api`
- Validation status: `failed`
- Evaluation date: 2026-11-09
- Evaluator version: `1.6.9`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 9 evaluation tasks (9 positive)
- Dataset digest: `sha256:f385c69ce235035e8e5cbfd91a8d3245b75473f0bf0d694451e988e439f6e9e1` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| cuopt-numerical-optimization-api | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| cuopt-numerical-optimization-api | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| cuopt-numerical-optimization-api | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| cuopt-numerical-optimization-api | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| cuopt-numerical-optimization-api | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |

- Controls for `cuopt-numerical-optimization-api`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

## What This Report Answers

The three-tier evaluation checks whether the skill:

- is safe to use;
- produces correct answers;
- is discovered and activated when needed;
- helps the agent complete the user's goal and expected workflow; and
- avoids wasted skill and tool usage.

## Results at a Glance

| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 94.3% — baseline ran, but no comparable score was available; uplift unavailable | 85.1% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 82.2% → 97.8% (+15.6 points) | 91.1% → 93.3% (+2.2 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 92.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 74.0% → 90.0% (+16.0 points) | 52.4% → 66.1% (+13.7 points) |
| Efficiency | 83.6% — baseline ran, but no comparable score was available; uplift unavailable | 73.4% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 98.7% → 96.8% (-1.9 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 87.6% → 88.8% (+1.2 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,128,193 | 537,122 | +591,071 | +110.04% | skill 9/9; base 9/9 |
| claude-code | numopt-c-eval-001-milp-api-call-sequence | 154,232 | 36,279 | +117,953 | +325.13% | skill 1/1; base 1/1 |
| claude-code | numopt-c-eval-002-parameter-function-wrong-name | 151,443 | 71,272 | +80,171 | +112.49% | skill 1/1; base 1/1 |
| claude-code | numopt-c-eval-003-csr-constraint-matrix | 111,269 | 35,832 | +75,437 | +210.53% | skill 1/1; base 1/1 |
| claude-code | numopt-c-eval-004-qp-restrictions | 111,645 | 143,383 | -31,738 | -22.14% | skill 1/1; base 1/1 |
| claude-code | numopt-cli-eval-001-mps-sections-and-cli-command | 112,355 | 71,591 | +40,764 | +56.94% | skill 1/1; base 1/1 |
| claude-code | numopt-py-eval-001-lp-api-call-sequence | 112,580 | 35,660 | +76,920 | +215.70% | skill 1/1; base 1/1 |
| claude-code | numopt-py-eval-002-status-case-sensitivity | 110,562 | 35,809 | +74,753 | +208.75% | skill 1/1; base 1/1 |
| claude-code | numopt-py-eval-003-integer-vs-continuous-workers | 110,781 | 36,055 | +74,726 | +207.26% | skill 1/1; base 1/1 |
| claude-code | numopt-py-eval-004-qp-maximize-workaround | 153,326 | 71,241 | +82,085 | +115.22% | skill 1/1; base 1/1 |
| codex | All cases | 523,828 | 258,944 | +264,884 | +102.29% | skill 9/9; base 9/9 |
| codex | numopt-c-eval-001-milp-api-call-sequence | 48,461 | 38,433 | +10,028 | +26.09% | skill 1/1; base 1/1 |
| codex | numopt-c-eval-002-parameter-function-wrong-name | 48,551 | 26,134 | +22,417 | +85.78% | skill 1/1; base 1/1 |
| codex | numopt-c-eval-003-csr-constraint-matrix | 48,705 | 17,790 | +30,915 | +173.78% | skill 1/1; base 1/1 |
| codex | numopt-c-eval-004-qp-restrictions | 67,262 | 56,084 | +11,178 | +19.93% | skill 1/1; base 1/1 |
| codex | numopt-cli-eval-001-mps-sections-and-cli-command | 52,215 | 26,279 | +25,936 | +98.69% | skill 1/1; base 1/1 |
| codex | numopt-py-eval-001-lp-api-call-sequence | 48,603 | 26,146 | +22,457 | +85.89% | skill 1/1; base 1/1 |
| codex | numopt-py-eval-002-status-case-sensitivity | 77,074 | 21,610 | +55,464 | +256.66% | skill 1/1; base 1/1 |
| codex | numopt-py-eval-003-integer-vs-continuous-workers | 62,164 | 21,716 | +40,448 | +186.26% | skill 1/1; base 1/1 |
| codex | numopt-py-eval-004-qp-maximize-workaround | 70,793 | 24,752 | +46,041 | +186.01% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 1,652,021 | 796,066 | +855,955 | +107.52% | skill 18/18; base 18/18 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **FAILED** | 11 validator(s); 11 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 2 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 9 task(s) |

## Blocking Findings

- **MEDIUM** BANDIT/B310:blacklist: Audit url open for permitted schemes. Allowing use of file:/ or custom schemes is often unexpected. (CWE-22) (`skills/cuopt-numerical-optimization-api/assets/python/mps_solver/model.py:39`)

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found across SKILL.md and references/qp_examples.md:
  "## QP Rules (all interfaces)" in SKILL.md (lines 51-57)
  vs "## Critical Reminders" in references/qp_examples.md (lines 180-188) (`SKILL.md:51`)
- **MEDIUM** BANDIT/B310:blacklist: Audit url open for permitted schemes. Allowing use of file:/ or custom schemes is often unexpected. (CWE-22) (`skills/cuopt-numerical-optimization-api/assets/python/mps_solver/model.py:39`)
- **MEDIUM** DUPLICATE/duplicate: Duplicate content found across assets/python/portfolio/README.md and assets/python/portfolio/model.py and references/python_api.md and references/qp_examples.md:
  "# Portfolio optimization (QP)" in assets/python/portfolio/README.md (lines 1-7)
  vs "(module docstring)" in assets/python/portfolio/model.py (lines 1-3)
  vs "## QP Example (beta — MINIMIZE only)" in references/python_api.md (lines 51-73)
  vs "## Portfolio Optimization" in references/qp_examples.md (lines 3-19) (`assets/python/portfolio/README.md:1`)
- **MEDIUM** QUALITY/quality_efficiency: Deeply nested references in c_api.md (`skills/cuopt-numerical-optimization-api/SKILL.md`)
- **MEDIUM** SCHEMA/frontmatter_field_placement: Root field 'version' is ignored; use 'metadata.version' (`skills/cuopt-numerical-optimization-api/SKILL.md`)
- 8 additional finding(s) are available in the full evaluation artifacts.

</details>

## Scoring Methodology

<details>
<summary>Show dimension definitions, source signals, and thresholds</summary>

| Dimension | Question | Scored signals |
|---|---|---|
| Security | Is it safe to use? | `security` (100%) |
| Correctness | Is the answer correct? | `accuracy` (100%) |
| Discoverability | Was the right skill loaded when needed? | `skill_execution` (100%) |
| Effectiveness | Did the skill help complete the task? | `goal_accuracy` (50%) + `behavior_check` (50%) |
| Efficiency | Did it avoid wasted tool calls and token usage? | `skill_efficiency` (50%) + `token_efficiency` (50%) |

- Dimension bands: PASS at 50% or above; NEUTRAL from 40% to below 50%; FAIL below 40%.
- Overall Tier 3 lift: PASS at +5 points or more; FAIL at -10 points or less; values between those bands are NEUTRAL.
- Overall verdict: PASS only when every configured dimension passes for at least one supported agent. Lift is reported as diagnostic evidence and does not override this gate.
- The 50% attempt pass threshold is a separate per-task gate; it is not the dimension pass threshold.
- Effectiveness is the equal-weight mean of goal completion (`goal_accuracy`) and expected workflow adherence (`behavior_check`).
- Efficiency is 50% tool-call productivity (the backward-compatible `skill_efficiency` wire id) and 50% `token_efficiency`. Positive-case skill routing is scored under Discoverability, not Efficiency; a negative case without a routing target is N/A. N/A sources are omitted, remaining weights are renormalized, and the dimension is marked partial.

Signals present in this run:

- `security` (Security): unsafe operations, secret leakage, and unauthorized access.
- `skill_execution` (Skill Execution): whether the expected skill was selected, decoys were avoided, and the workflow executed.
- `skill_efficiency` (Tool Productivity): tool-call productivity (legacy wire id; routing is scored under Discoverability).
- `accuracy` (Accuracy): final-answer correctness against the reference answer.
- `goal_accuracy` (Goal Accuracy): whether the user's goal was achieved.
- `behavior_check` (Behavior Check): whether the expected workflow behavior was followed.
- `token_efficiency` (Token Efficiency): actual uncached prompt plus completion usage (50% of Efficiency).

</details>

## Freshness

Regenerate this benchmark when the skill, evaluation dataset, target agent/model, evaluator version, environment, or scoring policy changes.
