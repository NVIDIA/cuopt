# cuoptopt-agent

Autonomous optimization agent for NVIDIA cuOpt. Given a natural-language improvement
query, the agent:

1. Loads relevant domain skills (GPU architecture, CUDA, algorithms)
2. Searches Google Scholar and arxiv for applicable techniques
3. Asks an LLM to generate targeted code diffs
4. Runs the cuOpt benchmark suite before and after
5. Auto-rejects speed regressions; asks for human review on quality changes
6. On human approval, creates a branch and opens a GitHub PR

Supports **Claude** (Anthropic), **GPT-4o** (OpenAI), and **NVIDIA NIM** as LLM backends.

---

## Quick Start

### 1. Install

```bash
pip install -e ".[dev]"
```

### 2. Set API keys

```bash
export ANTHROPIC_API_KEY="sk-ant-..."        # for --model claude
export OPENAI_API_KEY="sk-..."               # for --model gpt
export NVIDIA_API_KEY="nvapi-..."            # for --model nvidia
export GITHUB_TOKEN="ghp_..."               # for PR creation
```

### 3. Run

```bash
# From the cuoptopt-agent directory
cuoptopt-agent "Improve presolver for dense LP systems on L40" --model claude

# Skip literature search for faster offline iteration
cuoptopt-agent "Optimize SpMV kernel memory coalescing" --model nvidia --skip-research

# Verbose debug output
cuoptopt-agent "Speed up branch-and-bound on H100" --model gpt --verbose
```

### From Cursor (no terminal needed)

`Ctrl+Shift+P` → **Tasks: Run Task** → choose one of:
- **cuoptopt-agent: Run (interactive)** — prompts for query and model
- **cuoptopt-agent: Run (skip research, fast)** — offline mode
- **cuoptopt-agent: Run (verbose)** — with debug logging
- **cuoptopt-agent: Install dependencies** — installs the package

---

## CLI Reference

```
cuoptopt-agent [OPTIONS] QUERY

Arguments:
  QUERY    Optimization goal in natural language

Options:
  -m, --model    claude | gpt | nvidia  [default: claude]
  --max-iter     Max implement-test cycles  [default: 5]
  --config-dir   Path to config/  [default: cuoptopt-agent/config]
  --repo-root    Path to cuOpt repo root  [default: auto-detected]
  --skip-research  Skip Scholar/arxiv search
  -v, --verbose  Debug logging
```

---

## Configuration

### `config/models.yaml`

Defines LLM backends. API keys are read from environment variables — never hardcode them.

```yaml
claude:
  provider: anthropic
  model: claude-opus-4-5
  api_key_env: ANTHROPIC_API_KEY

nvidia:
  provider: nvidia_nim
  base_url: https://integrate.api.nvidia.com/v1
  model: nvidia/llama-3.1-nemotron-70b-instruct
  api_key_env: NVIDIA_API_KEY
```

To use a different NVIDIA NIM model, change the `model` field. See available models at
[build.nvidia.com](https://build.nvidia.com/explore/discover).

### `config/thresholds.yaml`

Controls regression detection:

| Key | Default | Meaning |
|-----|---------|---------|
| `speed_tolerance_pct` | 5.0 | Auto-reject if > 5% slower |
| `quality_tolerance_pct` | 1.0 | Ask human if > 1% worse objective |
| `max_reassess_iterations` | 5 | Give up after N LLM loops |
| `benchmark_warmup_runs` | 2 | GPU warm-up solves (discarded) |
| `benchmark_timed_runs` | 5 | Timed solves (median used) |

---

## Domain Skills

Skills live in `cuoptopt-agent/skills/`. The agent auto-selects the most relevant ones
via TF-IDF keyword matching against each skill's `description` field.

| Skill | Keywords |
|-------|---------|
| `cuda-optimization` | CUDA kernel, warp, shared memory, coalescing |
| `nvidia-l40-architecture` | L40, Ada Lovelace, GDDR6, cc 8.9 |
| `nvidia-a100-architecture` | A100, Ampere, HBM2e, NVLink |
| `nvidia-h100-architecture` | H100, H200, Hopper, TMA, wgmma |
| `nvidia-blackwell-architecture` | Blackwell, B200, tcgen05, TMEM, FP4 |
| `presolvers` | presolver, bound tightening, probing, redundancy |
| `mip-algorithms` | branch-and-bound, cutting planes, MILP, Gomory |
| `gpu-profiling` | Nsight, ncu, roofline, bottleneck |
| `benchmarking-methodology` | timing, warmup, statistical, regression |
| `research-literature` | Scholar, arxiv, paper search |

---

## Workflow Diagram

```
Query
  │
  ├─ Skill Loader ──────────────── top-6 relevant skills
  ├─ Research Agent ────────────── Google Scholar + arxiv papers
  │
  └─ Loop (up to max-iter):
       │
       ├─ Implementation Agent ── LLM → unified diffs → patch
       ├─ Benchmark Runner ────── datasets/ before+after
       │
       ├─ Speed regression? ──── YES → auto-reject → loop
       ├─ Quality regression? ── YES → ask human → loop or continue
       └─ Ask final approval ─── YES → branch + commit + push + PR
                               └── NO → loop
```

---

## Environment Variables

| Variable | Required for |
|----------|-------------|
| `ANTHROPIC_API_KEY` | `--model claude` |
| `OPENAI_API_KEY` | `--model gpt` |
| `NVIDIA_API_KEY` | `--model nvidia` |
| `GITHUB_TOKEN` | PR creation (scope: `repo`) |

---

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check cuoptopt_agent/
```

---

## Adding New Skills

1. Create `cuoptopt-agent/skills/<skill-name>/SKILL.md`
2. Add YAML front-matter with `name`, `version`, `description`
3. The skill is auto-discovered — no registration needed
4. The `description` field drives TF-IDF matching; make it keyword-rich

Example:
```markdown
---
name: routing-heuristics
version: "26.04.00"
description: VRP routing heuristics — LKH, SISR, ALNS, large neighborhood search, GPU parallelization.
---

# Routing Heuristics
...
```
