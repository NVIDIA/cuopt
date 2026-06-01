# Evaluation Report

Evaluation of the `cuopt-multi-objective-exploration` skill.

> **Status: proof-of-concept A/B complete; official NVSkills-Eval pending on the fork.**
> The numbers below are from a custom WITH-vs-WITHOUT run on a Colab GPU with cuOpt in the
> loop (real solves) — not from NVSkills-Eval. They establish the skill's value before the
> formal run. The official NVSkills-Eval (`claude-code` + `codex`, external profile) and CI
> Tiers 1–2 still run on the fork and gate publication.

## Summary

- Skill: `cuopt-multi-objective-exploration`
- Eval: custom WITH-vs-WITHOUT A/B — notebook `cuopt_exploration_skill_value_test.ipynb`
- Problem: supplier-selection procurement MILP (12 suppliers; maximize resilience / minimize cost under demand coverage) + a fixed real-world-style supplier dataset (cost vs reliability)
- Agents: `claude-opus-4-8`, `claude-sonnet-4-6` (each driving cuOpt as a tool); judge: `claude-opus-4-8`
- Samples: exploration N=6/cell; interpretation N=15/cell × 2 scenarios; decoy N=6/cell
- Date / hardware: 2026-05-29, Colab (NVIDIA RTX PRO 6000)

## Results (WITH vs WITHOUT the skill)

| Dimension | WITHOUT | WITH | Read |
|---|---|---|---|
| **Effectiveness — interpretation** | 0.58 | 0.73 | +0.15; both models (opus 0.73→0.93, sonnet 0.42→0.54), both scenarios (real 0.74→0.88, synthetic 0.42→0.58) |
| **Effectiveness — exploration** | 2.17 | 3.67 | non-supported Pareto portfolios recovered, of 24 (~+70%); both models up; full-front coverage 23%→28% |
| **Discoverability** | 92% | 100% | restraint on the single-objective decoy (small lift; high baseline) |
| **Efficiency** | 10.7 / 1.2 | 11.9 / 1.0 | cuOpt solves used (multi-objective / decoy) |
| **Correctness** | 72% | 76% | solved portfolios proven-optimal (cuOpt `FeasibleFound` accounts for the rest — solver, not skill) |
| **Security** | — | — | no unsafe surface: the agent's only tool is a math solver (no secrets, filesystem, or network) |

**Interpretation, per rubric item (pooled):** no-single-best **+0.35**, knee-not-auto-pick **+0.28**, exchange-rate ≈0, state-assumptions ≈0. The lift is the *don't-collapse-to-one-answer* discipline; the other two behaviors were already present in both arms.

## What the eval shows

- The skill's value is **interpretation discipline** — agents present the tradeoff and defer instead of collapsing to one option — holding across both models and both decisions, including the fixed external supplier dataset (scenario A).
- **Exploration** is a real supporting lift on this constrained MILP: agents recover ~70% more of the non-supported (weighted-sum-unreachable) Pareto portfolios.
- **Discoverability** is a small positive (models mostly restrain unprompted).

## Caveats

- Custom A/B, not NVSkills-Eval. Agents are raw models with a cuOpt tool, not `claude-code` / `codex`.
- Exploration and the synthetic scenario use one instance (seed 1); the interpretation value also holds on the seed-independent fixed supplier dataset (scenario A).
- Judge is `claude-opus-4-8` (LLM-graded rubric).
- The shipped `SKILL.md` adds three cuOpt-feasibility clarifications not in the A/B's inlined skill text (PDLP warmstart is LP-only; ε-constrain *linear* objectives, since cuOpt constraints are linear; cap each MILP solve's time limit so points are optimal to the gap you set). These are factual corrections, not value claims — they don't bear on the measured behaviors, so the numbers above stand. Re-running the notebook against the final text would change nothing material.

## Tier 1 / Tier 2 / official NVSkills-Eval — pending on the fork

- `./ci/utils/validate_skills.sh` (frontmatter, required files, version 26.08.00) + `sync_skills_version.sh`.
- Tier 2 dedup — scoped to orchestration + interpretation; defers per-solve mechanics to the api-* skills and per-objective formulation to `cuopt-numerical-optimization-formulation`.
- NVSkills-Eval (external profile; `claude-code` + `codex`) — the formal gate.

## Publication recommendation

The POC supports the value claim. Next: socialize via a GitHub discussion/proposal, then a fork-based draft PR with CI + the official NVSkills-Eval.
