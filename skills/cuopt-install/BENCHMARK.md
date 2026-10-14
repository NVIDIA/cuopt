# Skill Benchmark: cuopt-install

> **Overall verdict: NEUTRAL — One or more dimensions remain below PASS**

Live evaluation did not show a material gain or regression. Collect more evidence or improve the skill before making a publication decision.

## Evaluation Metadata

- Skill: `cuopt-install`
- Evaluation date: 2026-10-14
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 7 evaluation tasks (7 positive)
- Dataset digest: `sha256:69515ef8677f7773d6ae7ac097b3a02767971b308d79ba74759f647dd33794c5` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

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
| Overall | 92.4% — baseline ran, but no comparable score was available; uplift unavailable | 87.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 92.9% → 100.0% (+7.1 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 97.1% → 91.4% (-5.7 points) | 91.4% → 88.6% (-2.8 points) |
| Discoverability | 97.9% — baseline ran, but no comparable score was available; uplift unavailable | 93.6% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 77.6% → 83.2% (+5.6 points) | 78.6% → 80.5% (+1.9 points) |
| Efficiency | 89.7% — baseline ran, but no comparable score was available; uplift unavailable | 72.2% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 954,833 | 1,026,680 | -71,847 | -7.00% | skill 7/7; base 7/7 |
| claude-code | inst-eval-001-docker-server | 60,727 | 31,010 | +29,717 | +95.83% | skill 1/1; base 1/1 |
| claude-code | inst-eval-002-cuda-suffix-must-match | 60,594 | 126,543 | -65,949 | -52.12% | skill 1/1; base 1/1 |
| claude-code | inst-eval-003-python-implies-c-not-reverse | 94,907 | 372,535 | -277,628 | -74.52% | skill 1/1; base 1/1 |
| claude-code | inst-eval-004-unsupported-gpu | 60,644 | 28,971 | +31,673 | +109.33% | skill 1/1; base 1/1 |
| claude-code | inst-eval-005-underspecified-install-request | 453,499 | 215,170 | +238,329 | +110.76% | skill 1/1; base 1/1 |
| claude-code | inst-eval-006-no-pip-and-conda-together | 60,595 | 30,150 | +30,445 | +100.98% | skill 1/1; base 1/1 |
| claude-code | inst-eval-007-module-not-found-triage | 163,867 | 222,301 | -58,434 | -26.29% | skill 1/1; base 1/1 |
| codex | All cases | 837,908 | 1,063,079 | -225,171 | -21.18% | skill 7/7; base 7/7 |
| codex | inst-eval-001-docker-server | 39,316 | 25,203 | +14,113 | +56.00% | skill 1/1; base 1/1 |
| codex | inst-eval-002-cuda-suffix-must-match | 55,315 | 47,822 | +7,493 | +15.67% | skill 1/1; base 1/1 |
| codex | inst-eval-003-python-implies-c-not-reverse | 44,517 | 58,137 | -13,620 | -23.43% | skill 1/1; base 1/1 |
| codex | inst-eval-004-unsupported-gpu | 32,838 | 26,336 | +6,502 | +24.69% | skill 1/1; base 1/1 |
| codex | inst-eval-005-underspecified-install-request | 604,264 | 848,412 | -244,148 | -28.78% | skill 1/1; base 1/1 |
| codex | inst-eval-006-no-pip-and-conda-together | 33,084 | 17,958 | +15,126 | +84.23% | skill 1/1; base 1/1 |
| codex | inst-eval-007-module-not-found-triage | 28,574 | 39,211 | -10,637 | -27.13% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 1,792,741 | 2,089,759 | -297,018 | -14.21% | skill 14/14; base 14/14 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 16 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **NEUTRAL** | 2 agent(s); 7 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** SCHEMA/frontmatter_field_placement: Root field 'version' is ignored; use 'metadata.version' (`skills/cuopt-install/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/cuopt-install/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/cuopt-install/SKILL.md`)
- **MEDIUM** SECURITY/Unknown (RP1): MCP Rug Pull: Docker image references without a specific tag (:latest is implicit) or digest (@sha256:...) can be silently replaced by (`SKILL.md:112`)
- **MEDIUM** SECURITY/Unknown (SQP-2): The skill instructs users to launch the cuOpt REST server bound to '0.0.0.0' (all network interfaces) on port 8000 witho (`SKILL.md:119`)
- 11 additional finding(s) are available in the full evaluation artifacts.

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
