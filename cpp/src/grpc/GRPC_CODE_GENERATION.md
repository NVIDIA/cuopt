# gRPC Code Generation — Architecture

The code generator reads
[`codegen/field_registry.yaml`](codegen/field_registry.yaml) and produces
`cuopt_remote_data.proto` plus a family of C++ `.inc` files that are
`#include`d directly into mapper source files. This eliminates the need to
hand-write repetitive conversion code or `.proto` definitions — adding or
removing gRPC support for a field is a YAML change.

This doc covers the architecture: what the generator emits, how it's wired
into the build, and how to run it. For the field-level reference (every
attribute, what it does, and step-by-step walkthroughs), see
[`codegen/FIELD_REGISTRY_REFERENCE.md`](codegen/FIELD_REGISTRY_REFERENCE.md).
For the wire-level chunked transfer protocol, see
[`GRPC_INTERFACE.md`](GRPC_INTERFACE.md).

## Quick start

```bash
# Regenerate after editing field_registry.yaml
python cpp/src/grpc/codegen/generate_conversions.py

# Or with explicit paths:
python cpp/src/grpc/codegen/generate_conversions.py \
    --registry cpp/src/grpc/codegen/field_registry.yaml \
    --output-dir cpp/src/grpc/codegen/generated
```

The core generator runs with no external dependencies beyond PyYAML (ships
with conda). The `--auto-number` and `--strip` options additionally require
`ruamel.yaml` (listed in the project's development dependencies).

`./build.sh codegen` runs the generator and stages the output for commit.
After editing `field_registry.yaml`, run codegen and commit the regenerated
files alongside your changes.

## File layout

The codegen lives under `cpp/src/grpc/codegen/`:

- `field_registry.yaml` — source of truth for all fields.
- `generate_conversions.py` — generator script.
- `generated/` — output (`cuopt_remote_data.proto` plus `generated_*.inc`
  files), committed to the repo so builds work without re-running the
  generator.
- `FIELD_REGISTRY_REFERENCE.md` — field-level reference and walkthroughs.

The exact set of generated files is whatever `generate_conversions.py`
currently emits for the registry; run `ls cpp/src/grpc/codegen/generated/` to
see the present inventory rather than relying on a hand-maintained list.
`ci/verify_grpc_codegen.sh` re-runs the generator into a temp directory and
diffs against the committed output, so the directory cannot drift silently.

CMake adds `cpp/src/grpc/codegen/generated` to the include path for both the
`cuopt` and `cuopt_grpc_server` targets, so the bare
`#include "generated_*.inc"` directives resolve without any copy step.

---

## What gets generated

### `cuopt_remote_data.proto`

A complete proto file containing every data message and enum derived from the
registry — `OptimizationProblem`, its `repeated_messages` entries (e.g.
`QuadraticConstraint`), the settings messages, the solution messages,
`ChunkedResultHeader`, plus all referenced enums. See
`generated/cuopt_remote_data.proto` for the current message/enum surface
(regenerated whenever `field_registry.yaml` changes; do not edit by hand).

The hand-maintained `cuopt_remote.proto` and `cuopt_remote_service.proto`
import this generated file to avoid duplicating definitions.

### `.inc` file families

The generator emits one or more `.inc` per concern. Rather than enumerate
every file here (it drifts), look at the `_gen_*` and `generate_*` functions
in `generate_conversions.py` — each one corresponds to one generated `.inc`,
and the docstrings explain what it produces. At a high level the families
are:

- **Enum converters** — per-domain C++ `to_proto_*` / `from_proto_*` switch
  functions, grouped by each enum's `domain:` tag (one `.inc` per domain:
  problem / settings / solution).
- **Settings** — to-proto and from-proto converters for each settings
  message.
- **Solution** — unary and chunked conversion bodies for each solution
  type, including header population and per-array collection.
- **Problem** — unary and chunked conversion bodies for `OptimizationProblem`,
  plus the supporting wire-format helpers used by the chunked path.

---

## How `.inc` files are consumed

The `.inc` files are `#include`d directly inside C++ function bodies in the
gRPC mapper translation units. To list the current consumers, run:

```bash
grep -rln '#include "generated_' cpp/src/grpc/
```

Each include sits inside a function body — the `.inc` provides the body, the
surrounding `.cpp` / `.hpp` provides the signature and any helper lambdas
that the generated code references.

CMake adds `cpp/src/grpc/codegen/generated` to the include path for the
`cuopt` and `cuopt_grpc_server` targets, so the bare
`#include "generated_*.inc"` directives resolve without any copy step.

---

## Related documentation

- [`codegen/FIELD_REGISTRY_REFERENCE.md`](codegen/FIELD_REGISTRY_REFERENCE.md)
  — every YAML attribute, defaults, and walkthroughs for adding scalars /
  arrays / enums / setter groups / repeated messages.
- [`GRPC_INTERFACE.md`](GRPC_INTERFACE.md) — chunked transfer protocol,
  message size limits, error handling.
