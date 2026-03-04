---
name: lp-milp-formulation
version: "26.04.00"
description: LP/MILP concepts and going from problem text to formulation. What LP/MILP are, required formulation questions, typical modeling elements, and how to parse problem statements (parameters, constraints, decisions, objective).
---

# LP/MILP Formulation

Concepts and workflow for going from a problem description to a clear formulation. No API code here.

## What is LP / MILP

- **LP**: Linear objective, linear constraints, continuous variables.
- **MILP**: Same plus some integer or binary variables (e.g. scheduling, facility location, selection).

## Required questions (problem formulation)

Ask these if not already clear:

1. **Decision variables** — What are they? Bounds?
2. **Objective** — Minimize or maximize? Linear expression in the variables?
3. **Constraints** — Linear inequalities/equalities? Names and meaning?
4. **Variable types** — All continuous (LP) or some integer/binary (MILP)?

## Typical modeling elements

- **Continuous variables** — production amounts, flow, etc.
- **Binary variables** — open/close, yes/no (e.g. facility open, item selected).
- **Linking constraints** — e.g. production only if facility open (Big-M or indicator).
- **Resource constraints** — linear cap on usage (materials, time, capacity).

---

## Problem statement parsing

When the user gives **problem text**, classify every sentence and then summarize before formulating.

**Classify every sentence** as **parameter/given**, **constraint**, **decision**, or **objective**. Watch for **implicit constraints** (e.g. committed vs optional phrasing) and **implicit objectives** (e.g. "determine the plan" + costs → minimize total cost).

**Ambiguity:** If anything is still ambiguous, ask the user or solve all plausible interpretations and report all outcomes; do not assume a single interpretation.

### 🔒 MANDATORY: When in Doubt — Ask

- If there is **any doubt** about whether a constraint or value should be included, **ask the user** and state the possible interpretations.

### 🔒 MANDATORY: Complete-Path Runs — Try All Variants

- When the user asks to **run the complete path** (e.g. end-to-end, full pipeline), run all plausible variants and **report all outcomes** so the user can choose; do not assume a single interpretation.

### Three labels

| Label | Meaning | Examples (sentence type) |
|-------|--------|---------------------------|
| **Parameter / given** | Fixed data, inputs, facts. Not chosen by the model. | "Demand is 100 units." "There are 3 factories." "Costs are $5 per unit." |
| **Constraint** | Something that must hold. May be explicit or **implicit** from phrasing. | "Capacity is 200." "All demand must be met." "At least 2 shifts must be staffed." |
| **Decision** | Something we choose or optimize. | "How much to produce." "Which facilities to open." "How many workers to hire." |
| **Objective** | What to minimize or maximize. May be **explicit** ("minimize cost") or **implicit** ("determine the plan" with costs given). | "Minimize total cost." "Determine the production plan" (with costs) → minimize total cost. |

### Implicit constraints: committed vs optional phrasing

**Committed/fixed phrasing** → treat as **parameter** or **implicit constraint** (everything mentioned is given or must happen). Not a decision.

| Phrasing | Interpretation | Why |
|----------|-----------------|-----|
| "Plans to produce X products" | **Constraint**: all X must be produced. | Commitment; production level is fixed. |
| "Operates 3 factories" | **Parameter**: all 3 are open. Not a location-selection problem. | Current state is fixed. |
| "Employs N workers" | **Parameter**: all N are employed. Not a hiring decision. | Workforce size is given. |
| "Has a capacity of C" | **Parameter** (C) + **constraint**: usage ≤ C. | Capacity is fixed. |
| "Must meet all demand" | **Constraint**: demand satisfaction. | Explicit requirement. |

**Optional/decision phrasing** → treat as **decision**.

| Phrasing | Interpretation | Why |
|----------|-----------------|-----|
| "May produce up to …" | **Decision**: how much to produce. | Optional level. |
| "Can choose to open" (factories, sites) | **Decision**: which to open. | Selection is decided. |
| "Considers hiring" | **Decision**: how many to hire. | Hiring is under consideration. |
| "Decides how much to order" | **Decision**: order quantities. | Explicit decision. |
| "Wants to minimize/maximize …" | **Objective** (drives decisions). | Goal; decisions are the levers. |

### Implicit objectives — do not miss

**If the problem asks to "determine the plan" (or similar) but does not state "minimize" or "maximize" explicitly, the objective is often implicit.** You **MUST** identify it and state it before formulating; do not build a model with no objective.

| Phrasing / context | Likely implicit objective | Why |
|-------------------|---------------------------|-----|
| "Determine the production plan" + costs given (per unit, per hour, etc.) | **Minimize total cost** (production + inspection/sales + overtime, etc.) | Plan is chosen; costs are specified → natural goal is to minimize total cost. |
| "Determine the plan" + costs and revenues given | **Maximize profit** (revenue − cost) | Both sides of the ledger → optimize profit. |
| "Try to determine the monthly production plan" + workshop hour costs, inspection/sales costs | **Minimize total cost** | All cost components are given; no revenue to maximize → minimize total cost. |

**Rule:** When the problem gives cost (or cost and revenue) data and asks to "determine", "find", or "establish" the plan, **always state the objective explicitly** (e.g. "I'm treating the objective as minimize total cost, since only costs are given."). If both cost and revenue are present, state whether you use "minimize cost" or "maximize profit". Ask the user if unclear.

### Parsing workflow

1. **Split** the problem text into sentences or logical clauses.
2. **Label** each: parameter/given | constraint | decision | **objective** (if stated).
3. **Identify the objective (explicit or implicit):** If the problem says "minimize/maximize X", that's the objective. If it only says "determine the plan" (or "find", "establish") but gives costs (and possibly revenues), the objective is **implicit** — state it (e.g. minimize total cost, or maximize profit) and confirm with the user if ambiguous.
4. **Flag implicit constraints**: For each sentence, ask — "Does this state a fixed fact or a requirement (→ parameter/constraint), or something we choose (→ decision)?"
5. **Resolve ambiguity** by checking verbs and modals:
   - "is", "has", "operates", "employs", "plans to" (fixed/committed) → parameter or implicit constraint.
   - "may", "can choose", "considers", "decides", "wants to" (optional) → decision or objective.
6. **🔒 MANDATORY — If anything is still ambiguous** (e.g. a value or constraint could be read two ways): ask the user which interpretation is correct, or solve all plausible interpretations and report all outcomes. Do not assume a single interpretation.
7. **Summarize** for the user: list parameters, constraints (explicit + flagged implicit), decisions, and **objective (explicit or inferred)** before writing the math formulation.

### Parsing checklist

- [ ] Every sentence has a label (parameter | constraint | decision | objective if stated).
- [ ] **Objective is identified:** Explicit ("minimize/maximize X") or implicit ("determine the plan" + costs → minimize total cost; + revenues → maximize profit). Never formulate without stating the objective.
- [ ] Committed phrasing ("plans to", "operates", "employs") → not decisions.
- [ ] Optional phrasing ("may", "can choose", "considers") → decisions.
- [ ] Implicit constraints from committed phrasing are written out (e.g. "all X must be produced").
- [ ] **🔒 MANDATORY — Ambiguity:** Any phrase that could be read two ways → I asked the user or I will solve all interpretations and report all outcomes (no silent single interpretation).
- [ ] Summary is produced before formulating (parameters, constraints, decisions, **objective**).

### Example

**Text:** "The company operates 3 factories and plans to produce 500 units. It may use overtime at extra cost. Minimize total cost."

| Sentence / phrase | Label | Note |
|-------------------|-------|------|
| "Operates 3 factories" | Parameter | All 3 open; not facility selection. |
| "Plans to produce 500 units" | Constraint (implicit) | All 500 must be produced. |
| "May use overtime at extra cost" | Decision | How much overtime is a decision. |
| "Minimize total cost" | Objective | Drives decisions. |

Result: Parameters = 3 factories, 500 units target. Constraints = produce exactly 500 (implicit from "plans to produce"). Decisions = production allocation across factories, overtime amounts. Objective = minimize cost.

**Implicit-objective example:** A problem that asks to "determine the production plan" (or similar) and gives cost components (e.g. workshop, inspection, sales) but does not state "minimize" or "maximize" → **Objective is implicit: minimize total cost**. Always state it explicitly: "The objective is to minimize total cost."
