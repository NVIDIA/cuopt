## Description: <br>
After solving a non-trivial problem, this skill reads the in-conversation history to detect generalizable learnings and drafts a proposal that modifies a single existing SKILL.md file. The proposal is shown to the user and is not applied without explicit approval. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner: NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
At the end of any interaction the agent checks whether one of four deterministic post-correction triggers occurred in the session: the user explicitly corrected the agent's output, the agent's initial approach failed and required a retry, the agent observed an API behavior not described in any existing SKILL.md, or the agent applied a workaround not described in any existing SKILL.md. If a trigger occurred, the agent reads the in-conversation history, distills a generalizable learning, and produces a proposal in a fixed four-field format (Target, Trigger, Scored, Diff) that names exactly one existing SKILL.md file for modification. The proposal is presented to the user; no SKILL.md is modified without the user's explicit approval. <br>

### Deployment Geography for Use: <br>
Global <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [cuOpt User Guide](https://docs.nvidia.com/cuopt/user-guide/latest/introduction.html) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Code] <br>
**Output Format:** [Markdown with inline code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Skill Version(s): <br>
26.08.00 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
