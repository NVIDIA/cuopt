---
name: skill-evolution
version: "26.04.00"
description: After solving a non-trivial problem, detect generalizable learnings and propose skill updates so future interactions benefit automatically. Always active — applies to every interaction.
---

# Skill Evolution

After resolving a non-trivial problem (debugging, formulation fix, API gotcha, workflow correction), check whether the learning is **generalizable** — useful beyond this specific user's situation.

## When to propose a skill update

Propose when you discovered something that:
- Is NOT already in the relevant skill's SKILL.md
- Would prevent the same class of problem for future users or agents
- Can be stated as a general pattern, not tied to one user's data or code

Common triggers: silent failures, misleading defaults, missing examples, undocumented constraints, recurring questions.

Do NOT propose for: trivial typos, user-specific configurations, one-off data issues.

## How to propose

1. **Identify the skill** — which `skills/*/SKILL.md` would benefit
2. **Distill the lesson** — extract the general pattern from the specific fix:
   - Specific: "User got RecursionError building 500-term objective with chained +"
   - General: "Use LinearExpression(vars, coeffs) for large objectives to avoid recursion limits"
3. **Write the patch** — show the exact addition (new row in a table, new subsection, new code example)
4. **State provenance** — one sentence: what interaction surfaced this learning
5. **Ask the user** — present the proposal and ask: "Should I apply this skill update?"

## Proposal format

Present to the user as:

```
Skill update proposal:
  Skill: skills/<name>/SKILL.md
  Section: <where it goes>
  Trigger: <what happened that surfaced this>
  Change: <the exact content to add or modify>
```

Only apply the change after the user approves. If the user declines, do not persist.

## Distillation checklist

Before proposing, verify:
- [ ] The learning is stated generically (no user-specific variable names, data, or paths)
- [ ] It fits the skill's existing structure (matches the style of surrounding content)
- [ ] It does not contradict existing skill content
- [ ] It is factually correct (verified during the interaction, not speculative)

## Validation

Proposed skill changes must pass the same CI bar as manual edits:
- `./ci/utils/validate_skills.sh` — structural compliance
- `./ci/test_skills_assets.sh` — executable assets still work
- `python3 ci/utils/run_dev_skill_agent_tests.py` — behavioral compliance
