## Description: <br>
Multi-objective exploration — trace and interpret the Pareto frontier across competing objectives by orchestrating repeated single-objective cuOpt solves (weighted-sum and ε-constraint), then read the tradeoffs with discipline. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers use this skill when a problem has two or more competing objectives with no agreed weighting (cost vs. service, return vs. risk, distance vs. vehicles). It turns a sequence of single-objective cuOpt solves into a Pareto frontier and provides the interpretation discipline to read tradeoffs, knee points, and convexity blind spots. <br>

### Deployment Geography for Use: <br>
Global <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [cuOpt User Guide](https://docs.nvidia.com/cuopt/user-guide/latest/introduction.html) <br>
- [cuopt-examples](https://github.com/NVIDIA/cuopt-examples) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Code] <br>
**Output Format:** [Markdown with mathematical formulations and a Pareto frontier (table/plot of non-dominated points)] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- claude-code <br>
- codex <br>



## Evaluation Tasks: <br>
Three tasks (see `evals/evals.json`): two positive — interpretation on a real cost-vs-reliability supplier front, and frontier exploration on a supplier-selection MILP — plus one single-objective decoy (no activation expected). A pre-publication WITH/WITHOUT A/B over these has been run on a Colab GPU with real cuOpt solves (see `BENCHMARK.md`). The official NVSkills-Eval (external profile, `claude-code` + `codex`) has not been run from this environment (cuOpt is Linux + NVIDIA-GPU only) and is maintainer-triggered via `/nvskills-ci` on a non-fork `NVIDIA/cuopt` branch, as for the sibling skills (the NVSkills CI doesn't support fork PRs). <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks whether skill-assisted execution avoids unsafe behavior such as secret leakage, destructive commands, or unauthorized access. <br>
- Correctness: Checks whether the agent follows the expected workflow and produces the correct final output. <br>
- Discoverability: Checks whether the agent loads the skill when relevant and avoids using it when irrelevant. <br>
- Effectiveness: Checks whether the agent performs measurably better with the skill than without it. <br>
- Efficiency: Checks whether the agent uses fewer tokens and avoids redundant work. <br>

Underlying evaluation signals used in this run: <br>
- `skill_execution`: Verifies that the agent loaded the expected skill and workflow. <br>
- `skill_efficiency`: Checks routing quality, decoy avoidance, and redundant tool usage. <br>
- `accuracy`: Grades final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Checks whether the overall user task completed successfully. <br>
- `behavior_check`: Verifies expected behavior steps, including safety expectations. <br>
- `token_efficiency`: Compares token usage with and without the skill. <br>



## Evaluation Results: <br>
A pre-publication WITH/WITHOUT A/B (Colab GPU, real cuOpt) supports the value claim — interpretation discipline +0.15 across both models and both decisions, with frontier exploration as a supporting lift; full numbers and caveats in `BENCHMARK.md`. The official NVSkills-Eval table below (`claude-code` / `codex`, external profile) is PENDING — it's maintainer-run (`/nvskills-ci`, non-fork); the values are placeholders until then. <br>

| Dimension | Num | `claude-code` | `codex` |
|---|---:|---:|---:|
| Security | — | PENDING POC | PENDING POC |
| Correctness | — | PENDING POC | PENDING POC |
| Discoverability | — | PENDING POC | PENDING POC |
| Effectiveness | — | PENDING POC | PENDING POC |
| Efficiency | — | PENDING POC | PENDING POC |

## Skill Version(s): <br>
26.08.00 (source: frontmatter, git tag) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
