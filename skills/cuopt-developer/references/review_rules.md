# Review Rules — Pre-PR Self-Review and CodeRabbit

One rule set for two consumers: the author (human or agent) before `git commit`,
and CodeRabbit during review.

- **Author / agent (self-review loop):** before every `git commit`, stage your changes and
  check `git diff --staged` and the commit message against this list, fix each violation,
  stage the fix, and repeat until nothing is found.
- **CodeRabbit:** raise a violation as a short nit that names the rule. Do not raise style
  that is not listed here, or anything `clang-format` or `ruff` already enforces.
- **Reviewer:** name the rule instead of re-explaining. If a rule is wrong or missing, change this
  file in the same PR.

If neighbouring code in the same file does the opposite of a rule, ask which is intended.

## Whole-PR review

Before the line-by-line rules, look at the PR as a whole.

- **Other parts affected.** Which other code, bindings, the server, docs, CI, packaging and
  versions does this change touch or depend on? Does each need a matching change?
- **One coherent change.** Read the full diff as a reviewer would. Does everything in it
  belong, and do the code, comments and description agree?
- **Reuse first.** Is there an existing utility or pattern for this? See Reuse and placement.
- **No secrets or private details.** No keys, tokens, passwords, signed URLs, internal
  hostnames, internal paths or personal data in code, tests, docs or the commit message.
- **Docs and tests cover it.** Do the docs, examples and tests cover the new and the changed
  behaviour, including the other parts affected? See the Tests and Docs sections.

## Comments

- **Describe the code as it is.** No history ("used to be", "previously", "fixes the
  bug where"), no failed attempts, no numbers from one benchmark instance, no line numbers,
  commit hashes or PR numbers. See [conventions.md](conventions.md#comments).
- **Comment only what the code cannot say.** Add a comment for a non-obvious reason, an
  invariant or a unit. Do not restate what the next line does, narrate the steps, or add a
  header comment to trivial code. Keep it to one or two lines, and shorten a longer one as
  far as it will go. Do not address comments to an agent. Define jargon or use the plain
  name.
- **Non-trivial math is explained.** Say what the inputs are and what is computed.

## Naming and terminology

- **Use the project vocabulary; do not coin new terms.** Use the existing words for a
  concept everywhere (code, logs, docs, APIs). Where the codebase has no term, use the name
  from the literature (for example `min_activity`, `dot2`), not an invented one. Current
  choices:

  | Use | Not |
  |---|---|
  | user problem / presolved problem | solver space, barrier space, internal objective |
  | barrier | IPM |
  | problem | model |
  | multi-GPU | distributed |
  | MIP | MILP |
  | `mathematical_optimization` | `linear_programming` |

- **Spell names out and make them accurate.** No cryptic abbreviations in identifiers,
  and a name must describe what the thing is.
- **Be consistent within an API.** Pick one casing and one word per concept (for example
  "setting" or "parameter") and use it in every identifier, doc and example.

## Code

- **Follow the codebase's type conventions.** Use the project's template types (`i_t`,
  `f_t`) rather than bare `int`/`double`, and match the neighbouring files' `const` and
  instantiation patterns.
- **No noise casts or typed literals.** Drop a `static_cast` that does not change the value
  or guard against overflow. Write `0.0` or `0`, not `f_t{0}` or `static_cast<f_t>(0)`; the
  literal converts on its own. See [conventions.md](conventions.md).
- **Explicit ownership.** `std::unique_ptr` by default, `std::shared_ptr` only with a
  reason, `std::span` over pointer plus size, no raw owning pointers.
- **Plain control flow.** Avoid lambdas; use one only for sorting. No scope guards or hidden
  cleanup without a comment. A condition used in two places is one named function.
- **Follow the C++ style rules in [conventions.md](conventions.md#c-implementation-style)**
  (lambdas, local structs, work-estimate gating, `long double`).
- **Sparse-matrix index names.** `i` is a row, `j` is a column, and `p` or `q` indexes the
  nonzero arrays. Never use `j` for a row or `i` for a column. If those are taken, use `k`,
  `h`, `r`, `s`.
- **Argument order.** In a new function, input-only parameters come before output parameters.
  Follow the nearby code, for example `lp_problem_t` first and `solver_settings_t` second.
- **Name your constants.** Use an `enum class` for modes, and named constants or
  settings for limits and tolerances, not literals. Where `-Wswitch` is an error, leave
  `default:` out of a `switch` over an enum so a new value fails to compile.
- **Pick the right check.** An invariant the code itself must maintain is an assert;
  validation of caller input is `CUOPT_EXPECTS`.
- **One threading and RNG model.** Use OpenMP rather than `std::thread`, and the
  project's RNG (see the RNG policy in `SKILL.md`) rather than `<random>` engines.
- **Numerics are careful.** Use the existing compensated routines for sensitive sums,
  round in the safe direction when converting, take tolerances from settings, and think about
  overflow.

## Reuse and placement

- **Search before writing.** Look for an existing helper (utilities, linear algebra,
  presolve, copy and profiling helpers) before adding one. Reuse or extend it, or say why not.
- **Code lives with its domain.** Keep layers separate: the gRPC layer is a plain
  mapper with no solver logic, and CPU-only code does not own device buffers.
- **One representation.** Do not add a new type that holds the same fields as an
  existing one.

## Public API

- **Regular parameter first.** A new option is a regular `CUOPT_*` parameter that the
  bindings pick up from the C API, not a special-case setter or enum in each language.
  Add the gRPC field registry entry too (see [contributing.md](contributing.md)).
- **Hyperparameters stay internal.** Do not expose or document them for users.
- **The C API is the source.** Other bindings derive constants and statuses from it
  rather than copying them by hand or adding accessors it does not have.

## Tests

- **A test must be able to fail.** If the feature were reverted, would it still pass?
  A bug fix needs its reproducer.
- **Test the contract.** Check documented behaviour, not internals (no log-capture
  assertions). Do not test undocumented unsupported cases; documented errors are part of the
  contract. Use `TEST_P` (C++) or `pytest.mark.parametrize` (Python) rather than a loop over
  parameters.
- **Test in the right layer.** `cpp/tests` owns correctness, determinism, algorithms and each
  C API function. Python and any other API built on the core only test the API: functions
  and options are called, results convert, bad input raises the right error. Put a solver
  bug's reproducer in `cpp/tests`.
- **Prefer extending an existing test.** If a test already covers the case or can be
  modified to cover it, change it instead of adding a new one.
- **A skipped test links an open issue and is re-enabled, not removed.**
- **Keep tests small and cheap.** Toy instances and tight time limits; justify anything
  large or slow. Skip tests of trivial plumbing.
- **New paths get the coverage their siblings have.** A new public API function or option is
  called by at least one test.

## Docs, logs and PRs

- **Plain, complete sentences.** Capitalize headings and bullets, spell project terms
  consistently ("Quickstart", "MIP"), and do not describe earlier versions, internals or other APIs.
  Document deprecated APIs only in migration guides such as `docs/cuopt/source/transition.rst`.
- **Generate, do not hand-copy.** Settings, constants and defaults come from the source
  of truth.
- **Useful log lines.** Name the setting to change, and match the format of neighbouring
  lines.
- **Minimal diff.** Revert unrelated changes, split version bumps and dependency
  upgrades, and do not commit build output.
- **Follow-ups are issues.** Link a tracking issue instead of leaving a `TODO` that only
  describes the work.
- **Description and release notes are for readers.** Fill in the description, and write
  release notes for users rather than copying PR titles.
