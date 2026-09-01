# gRPC Wire Fields and the Codegen Registry

Read this before touching anything that crosses the cuOpt gRPC wire: problem
fields, solution fields, or solver settings.

## The one rule

**Never hand-edit files under `cpp/src/grpc/codegen/generated/`.** They are
generated from `cpp/src/grpc/codegen/field_registry.yaml` by
`cpp/src/grpc/codegen/generate_conversions.py`, and CI will reject a mismatch.

Every field that crosses the wire is declared once in the registry. One entry
drives ~28 generated artifacts: the `.proto`, the C++ to/from-proto converters,
the chunked upload/download paths, and the size estimator.

## Workflow

```bash
# 1. edit cpp/src/grpc/codegen/field_registry.yaml
# 2. regenerate
./build.sh codegen
# 3. commit BOTH the registry and cpp/src/grpc/codegen/generated/
# 4. optional local pre-check of what CI runs
bash ci/verify_grpc_codegen.sh
```

**Codegen is not part of the build.** `build.sh` guards it behind an explicit
`codegen` argument, and CMake only *consumes* the checked-in `generated/`
directory. Building cuOpt will never regenerate for you, and it will never warn
you that the registry has drifted — `ci/verify_grpc_codegen.sh` is what catches
that, in CI, after you push.

`./build.sh codegen` needs only `pyyaml` — no GPU, no compile. It is cheap to
run and cheap to re-run; there is no reason to skip it.

## Registry sections

| Section | Generates |
|---------|-----------|
| `enums` | proto enums + C++ converters |
| `optimization_problem` | `OptimizationProblem` message, problem converters |
| `pdlp_settings` / `mip_settings` | `PDLPSolverSettings` / `MIPSolverSettings` |
| `lp_solution` / `mip_solution` | solution messages and converters |
| `chunked_result_header` | `ChunkedResultHeader` for the chunked download path |

## Attributes worth knowing

Full reference: `cpp/src/grpc/codegen/FIELD_REGISTRY_REFERENCE.md`. The ones
that bite:

- **`optional`** — emits proto3 presence tracking. Required whenever the C++
  default differs from the proto3 zero value. Without it, a client that *omits*
  the field silently overwrites the solver default with `0` / `false` / the
  first enum value. `bool foo{true}` and enums whose C++ default is not the
  first declared value both need this.
- **`sentinel`** — maps a C++ sentinel (e.g. `numeric_limits<i_t>::max()`) to a
  reserved wire value (e.g. `-1`). Composes with `optional`: the sentinel covers
  the explicitly-sent case, `optional` covers the omitted case. `iteration_limit`
  and `node_limit` need both.
- **`description`** / **`default`** — documentation for a settings field,
  emitted into `cuopt_mcp_schema.json`. `description` becomes the JSON Schema
  `description`, which is what an MCP client shows a model deciding whether to
  set the field, so write it for someone choosing a value rather than for
  someone reading the struct. `default` is a free-text string describing the
  C++ member initializer (`"1e-4"`, `"-1 (automatic)"`); the generator neither
  derives nor validates it, so take it from the C++ struct — `docs/` is known
  to disagree in several places.
- **`param_name`** — the `CUOPT_*` string parameter a client passes to
  `set_parameter`, when it differs from the field name. The field name is the
  *proto* name, and MIP diverges heavily: `relative_mip_gap` is
  `mip_relative_gap`, `mir_cuts` is `mip_mixed_integer_rounding_cuts`, `seed`
  is `random_seed`. **44 of 85 settings fields need this.** Omitting it on a
  diverging field produces a setting that fails at solve time with "Invalid
  parameter". Set it to `null` for a field with no `CUOPT_*` constant at all,
  which drops it from the MCP schema.
  `python/cuopt_mcp/tests/test_parameter_names.py` asserts every advertised
  name exists in `constants.h`.

## Field numbers are permanent

`field_num` and `array_id` are wire identifiers. Never renumber or reuse a
retired number — older clients still send data on it. The registry records
retired numbers in comments (see the note on 37/38 in `mip_settings`); follow
that convention when removing a field.

## Adding a settings field: checklist

1. Add the C++ member with its initializer.
2. Add the registry entry with the next free `field_num`.
3. Add `optional:` if the C++ default is not the proto3 zero value.
4. Add `description:` and `default:` — match the C++ initializer, not the docs
   (see below).
5. `./build.sh codegen`, commit registry + `generated/`.
6. Document the user-facing parameter in `docs/cuopt/source/convex-settings.rst`
   (LP) or `mip-settings.rst` (MIP), and add the constant to
   `cpp/include/cuopt/mathematical_optimization/constants.h`.

## Documented defaults drift from the code

`docs/cuopt/source/*-settings.rst` is the prose source for what a setting means,
but its stated defaults have been observed to disagree with the C++ member
initializers. **The C++ initializer is ground truth.** When writing `default:`,
read the struct — `pdlp/solver_settings.hpp`, `mip/solver_settings.hpp`,
`mip/heuristics_hyper_params.hpp` — and treat a docs mismatch as a docs bug to
report separately, not as something to propagate into the registry.

## The registry is not a complete mirror of the settings structs

Some settings exist in C++ and in the user docs but have no registry entry,
which means they cannot be set over gRPC at all. Before assuming a parameter is
remotely settable, grep `field_registry.yaml` for it. Adding a missing one is a
wire change (new `field_num`), not a documentation change — scope it as its own
PR.
