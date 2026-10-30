# Review Rules — Pre-PR Self-Review and CodeRabbit

One rule set for two consumers: the author (human or agent) before `git push`, and CodeRabbit
during review.

- **Author / agent:** go through this list against `git diff origin/<target-branch>...HEAD` before pushing.
- **CodeRabbit:** raise a violation as a short nit that names the rule. Do not raise style
  that is not listed here, or anything `clang-format` or `ruff` already enforces.
- **Reviewer:** name the rule instead of re-explaining. If a rule is wrong or missing, change this
  file in the same PR.

If neighbouring code in the same file does the opposite of a rule, ask which is intended.

## Comments

- **Describe the code as it is.** No history ("used to be", "previously", "fixes the
  bug where"), no failed attempts, no numbers from one benchmark instance, no line numbers,
  commit hashes or PR numbers. See [conventions.md](conventions.md#comments).
- **Short and plain.** Say what or why in a line or two. Delete comments that restate
  the code or address an agent. Define jargon or use the plain name.
- **Non-trivial math is explained.** Say what the inputs are and what is computed.

## Naming and terminology

- **Use the project vocabulary; do not coin new terms.** Use the existing words for a
  concept everywhere (code, logs, docs, APIs). Current choices:

  | Use | Not |
  |---|---|
  | user problem / presolved problem | solver space, barrier space, internal objective |
  | barrier | IPM |
  | problem | model |
  | multi-GPU | distributed |
  | MIP | MILP |
  | `mathematical_optimization` (new names) | `linear_programming` |

- **Spell names out and make them accurate.** No cryptic abbreviations in identifiers,
  and a name must describe what the thing is.
- **Be consistent within an API.** Pick one casing and one word per concept (for example
  "setting" or "parameter") and use it in every identifier, doc and example.

## Code

- **Follow the codebase's type conventions.** Use the project's template types (`i_t`,
  `f_t`) rather than bare `int`/`double`, drop casts that change nothing, and match the
  neighbouring files' `const` and instantiation patterns.
- **Explicit ownership.** `std::unique_ptr` by default, `std::shared_ptr` only with a
  reason, `std::span` over pointer plus size, no raw owning pointers.
- **Plain control flow.** Short lambdas only, no scope guards or hidden cleanup without a
  comment. A condition used in two places is one named function.
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
- **Keep tests small and cheap.** Toy instances and tight time limits; justify anything
  large or slow. Skip tests of trivial plumbing.
- **New paths get the coverage their siblings have.**

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
