## Description: <br>
cuOpt REST server — start server, endpoints, Python/curl client examples. Use when the user is deploying or calling the REST API. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers deploying the NVIDIA cuOpt REST server or building client applications that submit optimization problems (routing, LP, MILP) to it via HTTP. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [No] <br>
**Credential Type(s):** [None] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [cuOpt User Guide](https://docs.nvidia.com/cuopt/user-guide/latest/introduction.html) <br>
- [cuopt-examples (GitHub)](https://github.com/NVIDIA/cuopt-examples) <br>
- [nvidia/cuopt (Docker Hub)](https://hub.docker.com/r/nvidia/cuopt) <br>


## Skill Output: <br>
**Output Type(s):** [API Calls, Shell commands, Code] <br>
**Output Format:** [Markdown with inline code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
8 evaluation tasks (8 positive), 1 attempt per task, in isolated k8s-sandbox pods. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill is safe to use (unsafe operations, secret leakage, unauthorized access). <br>
- Correctness: Whether the answer produced is correct against the reference answer. <br>
- Discoverability: Whether the right skill was loaded and executed when needed. <br>
- Effectiveness: Whether the skill helped complete the user’s goal and expected workflow. <br>
- Efficiency: Whether wasted tool or skill usage was avoided. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was found and executed. <br>
- `skill_efficiency`: Routing quality, workspace-aware skill reads, and productive tool use. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user’s goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill) | Codex (Baseline → Skill) |
|---|---:|---:|
| Overall | 55% → 99% (+44 pts) | 58% → 95% (+37 pts) |
| Security | 100% → 100% (±0 pts) | 100% → 100% (±0 pts) |
| Correctness | 82% → 100% (+18 pts) | 78% → 98% (+20 pts) |
| Discoverability | 28% → 100% (+72 pts) | 42% → 92% (+50 pts) |
| Effectiveness | 59% → 98% (+39 pts) | 51% → 89% (+38 pts) |
| Efficiency | 6% → 96% (+90 pts) | 17% → 96% (+79 pts) |

## Skill Version(s): <br>
26.10.00 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
