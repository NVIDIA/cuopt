# Code Generation for gRPC Proto Definitions and C++ Conversion Code

The code generator reads `field_registry.yaml` and produces `cuopt_remote_data.proto`
and C++ `.inc` files that are `#include`d directly into mapper source files. This
eliminates the need to hand-write repetitive conversion code or `.proto` definitions
— adding or removing a field is a one-line YAML change.

## Quick Start

```bash
# Regenerate after editing field_registry.yaml
python cpp/src/grpc/codegen/generate_conversions.py

# Or with explicit paths:
python cpp/src/grpc/codegen/generate_conversions.py \
    --registry cpp/src/grpc/codegen/field_registry.yaml \
    --output-dir cpp/src/grpc/codegen/generated
```

The generator runs in ~100ms with no external dependencies beyond PyYAML (ships
with conda). The `--auto-number` and `--strip` options additionally require
`ruamel.yaml` (listed in the project's development dependencies). Run it
explicitly via `./build.sh codegen` after editing `field_registry.yaml`, then
commit the regenerated files.

## File Layout

```text
cpp/src/grpc/codegen/
├── field_registry.yaml          # Source of truth for all fields
├── generate_conversions.py      # Generator script
└── generated/                   # Output (committed, regenerated on build)
    ├── cuopt_remote_data.proto
    ├── generated_result_enums.proto.inc
    │
    │   # Enum converters (one per domain)
    ├── generated_enum_converters_problem.inc
    ├── generated_enum_converters_settings.inc
    ├── generated_enum_converters_solution.inc
    │
    │   # Settings conversions
    ├── generated_pdlp_settings_to_proto.inc
    ├── generated_proto_to_pdlp_settings.inc
    ├── generated_mip_settings_to_proto.inc
    ├── generated_proto_to_mip_settings.inc
    │
    │   # LP solution conversions
    ├── generated_lp_solution_to_proto.inc
    ├── generated_proto_to_lp_solution.inc
    ├── generated_lp_chunked_header.inc
    ├── generated_collect_lp_arrays.inc
    ├── generated_chunked_to_lp_solution.inc
    ├── generated_estimate_lp_size.inc
    │
    │   # MIP solution conversions
    ├── generated_mip_solution_to_proto.inc
    ├── generated_proto_to_mip_solution.inc
    ├── generated_mip_chunked_header.inc
    ├── generated_collect_mip_arrays.inc
    ├── generated_chunked_to_mip_solution.inc
    ├── generated_estimate_mip_size.inc
    │
    │   # Problem conversions
    ├── generated_problem_to_proto.inc
    ├── generated_proto_to_problem.inc
    ├── generated_estimate_problem_size.inc
    ├── generated_populate_chunked_header_lp.inc
    ├── generated_populate_chunked_header_mip.inc
    ├── generated_chunked_header_to_problem.inc
    ├── generated_chunked_arrays_to_problem.inc
    ├── generated_build_array_chunks.inc
    └── generated_array_field_element_size.inc
```

The generated `.inc` files are committed to the repo so that builds work without
running the generator. Run `./build.sh codegen` to re-generate them after
editing `field_registry.yaml`; CI verifies they stay in sync. CMake adds
`cpp/src/grpc/codegen/generated` to the include path for both targets, so the `.inc`
files are found at compile time with no copy step.

---

## Registry Structure Overview

`field_registry.yaml` has these top-level sections:

| Section | Purpose |
|---|---|
| `enums` | Shared enum definitions (C++ ↔ proto converters) |
| `lp_solution` | LP solution scalar/array fields and constructor args |
| `mip_solution` | MIP solution scalar/array fields and constructor args |
| `pdlp_settings` | PDLP solver settings field mappings |
| `mip_settings` | MIP solver settings field mappings |
| `optimization_problem` | Problem input scalar/array fields and setter groups |

---

## Convention-Over-Configuration Defaults

The registry uses bare field names wherever possible. Defaults:

| Context | Default type | Default getter | Default setter |
|---|---|---|---|
| Solution scalar | `double` | `get_<name>()` | *(via constructor)* |
| Solution array | `double` | `get_<name>_host()` | *(via constructor)* |
| Problem scalar | `double` | `get_<name>()` | `set_<name>()` |
| Problem array | `repeated double` | `get_<name>_host()` | `set_<name>()` |
| Settings field | `double` | struct member access | struct member assignment |

Proto field names always match the registry field name.

**Enum conventions** (derived from the YAML key unless overridden):

| Property | Convention |
|---|---|
| `cpp_type` | `<key>_t` (e.g. `pdlp_termination_status` → `pdlp_termination_status_t`) |
| `proto_type` | PascalCase from key, with known acronyms uppercased: PDLP, MIP, LP, QP, VRP, PDP, TSP (e.g. `pdlp_termination_status` → `PDLPTerminationStatus`) |
| `default` | First value in the `values` list (the proto3 zero-value) |
| `values` numbering | 0, 1, 2, ... (bare names auto-number; `{Name: N}` resets counter to N) |
| converter fns | `to_proto_<key>()` / `from_proto_<key>()` |

---

## Enums

Each entry under `enums:` defines a C++ ↔ proto enum mapping. The generator
produces `to_proto_<key>()` and `from_proto_<key>()` switch functions, split
into per-domain `.inc` files (`generated_enum_converters_problem.inc`, etc.).

Most properties are derived by convention (see above). Only `domain` and
`values` are required for the common case. Values auto-number from 0 when
written as bare names:

```yaml
enums:
  # Minimal — bare names auto-number 0, 1, 2, ...
  # proto_type, cpp_type, and default are all derived.
  pdlp_termination_status:
    domain: solution         # groups into generated_enum_converters_solution.inc
    proto_prefix: PDLP       # proto value = PDLP_UPPER_SNAKE(CppName)
    values:
    - NoTermination          # = 0
    - NumericalError         # = 1
    - Optimal                # = 2
    - PrimalInfeasible       # = 3

  # Override default when it's not the first value
  pdlp_solver_mode:
    domain: settings
    default: Stable3
    values:
    - Stable1
    - Stable2
    - Methodical1
    - Fast1
    - Stable3

  # Override cpp_type when it doesn't follow <key>_t
  lp_method:
    domain: settings
    cpp_type: method_t
    values:
    - Concurrent
    - PDLP
    - DualSimplex
    - Barrier

  # Explicit values reset the counter (C-style enum semantics):
  # example_with_gaps:
  #   domain: solution
  #   values:
  #   - OK                   # = 0
  #   - Warning: 10          # explicit → resets counter
  #   - Error                # = 11 (continues from 10+1)
  #   - Fatal: 20            # explicit → resets counter
  #   - Panic                # = 21
```

### Enum properties

| Property | Required | Default | Description |
|---|---|---|---|
| `domain` | yes | — | One of `problem`, `settings`, `solution`. Controls which `.inc` file the converters go into |
| `proto_type` | no | PascalCase from key | Protobuf enum type name. Derived via acronym-aware PascalCase (e.g. `pdlp_termination_status` → `PDLPTerminationStatus`). Override when the derived name doesn't match |
| `proto_prefix` | no | *(none)* | Prefix for proto value names. With prefix `PDLP`, C++ `Optimal` becomes proto `PDLP_OPTIMAL` |
| `cpp_type` | no | `<key>_t` | C++ enum type (e.g. `pdlp_termination_status_t`). Override with `cpp_type: method_t` |
| `default` | no | first value | C++ enum value to return for unrecognized proto values. Defaults to the first entry in `values` (the proto3 zero-value). Override when the default differs (e.g. `pdlp_solver_mode` defaults to `Stable3`) |
| `values` | yes | — | List of enum entries. Bare names auto-number from 0. Use `{Name: N}` to override; subsequent bare names continue from N+1 (C-style enum semantics) |

---

## LP / MIP Solution Sections

Each solution section (`lp_solution`, `mip_solution`) generates six `.inc` files
covering unary proto conversion, chunked streaming, size estimation, and array
collection.

```yaml
lp_solution:
  cpp_type: "cpu_lp_solution_t<i_t, f_t>"

  scalars: [...]
  arrays: [...]
  constructor_args: { scalars: [...] }
  warm_start: { ... }        # LP only
```

### Top-level properties

| Property | Description |
|---|---|
| `cpp_type` | Fully-qualified C++ template type for the solution constructor |

The generator derives `ChunkedResultHeader.problem_category` (LP or MIP)
automatically from the section name (`lp_solution` vs `mip_solution`).

### Scalars

Each entry describes one scalar field on both `ChunkedResultHeader` (for
chunked streaming) and the unary solution proto.

```yaml
scalars:
  # Minimal form — double type, getter = get_gap()
  - gap:
      field_num: 1006

  # Enum type
  - lp_termination_status:
      field_num: 1000
      type: pdlp_termination_status    # references an enum key
      getter: get_termination_status()

  # Proto-only (set on header but NOT a constructor arg)
  - error_message:
      field_num: 1001
      type: string
      getter: "get_error_status().what()"
      proto_only: true
```

| Property | Default | Description |
|---|---|---|
| `field_num` | *(required)* | Proto field number in `ChunkedResultHeader`. LP: 1000–1999, MIP: 2000–2999, warm start: 3000–3999 |
| `type` | `double` | One of: `double`, `int32`, `bool`, `string`, or an enum key name |
| `getter` | `get_<name>()` | C++ expression to read from the solution object |
| `proto_only` | `false` | If true, set on the proto header but not passed to the C++ constructor |

#### Type-specific behavior

| `type` | To-proto cast | From-proto cast |
|---|---|---|
| `double` | `static_cast<double>(...)` | `static_cast<f_t>(...)` |
| `int32` | `static_cast<int32_t>(...)` | `static_cast<i_t>(...)` |
| `bool` | *(none)* | *(none)* |
| `string` | *(none)* | *(none)* |
| enum key | `to_proto_<key>(...)` | `from_proto_<key>(...)` |

### Arrays

Each entry describes a solution array, identified by a `ResultFieldId` enum value.

```yaml
arrays:
  - primal_solution:
      field_num: 1
      array_id: 0

  - mip_solution:
      array_id: 12
      field_num: 1
      getter: get_solution_host()   # override the default
```

| Property | Default | Description |
|---|---|---|
| `field_num` | *(required)* | Proto field number in the per-solution unary message |
| `array_id` | *(required)* | Numeric value for the `ResultFieldId` enum (global across LP + MIP) |
| `getter` | `get_<name>_host()` | C++ getter expression on the solution object |

### Constructor Args

Controls the positional argument order when reconstructing a C++ solution
object from proto data:

```yaml
constructor_args:
  scalars:
  - lp_termination_status
  - primal_objective
  - dual_objective
  # ... order must match the C++ constructor
```

Arrays are always passed first (in YAML declaration order) via `std::move`.
Then the scalars listed here. If a `warm_start` section exists and warm start
data is present, `std::move(ws)` is appended as the final argument.

### Warm Start (LP only)

Describes a conditional sub-object for PDLP warm start data:

```yaml
warm_start:
  presence_check: has_warm_start_data()          # predicate on the solution object
  getter: get_cpu_pdlp_warm_start_data()         # accessor for the WS struct

  scalars:
  - initial_primal_weight_:
      field_num: 3000

  arrays:
  - current_primal_solution_:
      field_num: 1
      array_id: 3
```

| Property | Description |
|---|---|
| `presence_check` | C++ predicate expression to test if warm start data is present on the solution object |
| `getter` | C++ expression to access the warm start struct |

Warm start field names match the C++ struct member names directly (e.g.
`initial_primal_weight_` maps to `ws.initial_primal_weight_`). The `member`
attribute is only needed if the proto field name cannot match the C++ name
due to ambiguity.

Warm start detection during chunked deserialization is auto-derived: warm
start data is reconstructed whenever any warm-start array is present in the
chunked payload. The generator emits `arrays.count(...) != 0` checks OR'd
across every warm-start array — so a payload whose first warm-start array
happens to be empty but others carry data is still picked up.

---

## Settings Sections

Each settings section (`pdlp_settings`, `mip_settings`) generates two `.inc`
files: `generated_{label}_settings_to_proto.inc` and
`generated_proto_to_{label}_settings.inc`.

```yaml
pdlp_settings:
  cpp_type: "pdlp_solver_settings_t<i_t, f_t>"
  proto_type: "cuopt::remote::PDLPSolverSettings"

  fields:
    # Nested sub-struct — generates settings.tolerances.<field>
  - tolerances:
    - absolute_gap_tolerance:
        field_num: 1
    - relative_gap_tolerance:
        field_num: 2

    # Top-level fields
  - time_limit:
      field_num: 9
  - iteration_limit:
      field_num: 10
      type: int64
      sentinel:
        to_proto: "std::numeric_limits<i_t>::max()"
        proto_value: -1
        from_proto_guard: ">= 0"
        from_proto_cast: "i_t"
```

Settings fields support **nesting**: a list-valued entry (like `tolerances`
above) represents a sub-struct. The generator automatically prefixes C++ member
access with the sub-struct path (e.g. `settings.tolerances.absolute_gap_tolerance`).

### Settings field properties

| Property | Default | Description |
|---|---|---|
| `field_num` | *(required)* | Proto field number in the settings message |
| `type` | `double` | One of: `double`, `int32`, `int64`, `bool`, `string`, or an enum key |
| `cpp_member` | `<field_name>` (auto-prefixed by nesting path) | Explicit path to the C++ struct member |
| `to_proto_cast` | *(none)* | Explicit cast for C++ → proto (e.g. `int32_t`) |
| `from_proto_cast` | *(none)* | Explicit cast for proto → C++ (e.g. `presolver_t`) |
| `sentinel` | *(none)* | Special handling for sentinel values (see below) |

### Sentinel values

Some fields map a C++ default (like `max()`) to a proto sentinel (like `-1`):

```yaml
- iteration_limit:
    type: int64
    sentinel:
      to_proto: "std::numeric_limits<i_t>::max()"   # if C++ value == this...
      proto_value: -1                                 # ...emit this in proto
      from_proto_guard: ">= 0"                        # only assign if proto value matches guard
      from_proto_cast: "i_t"                           # cast applied when assigning
```

---

## Optimization Problem Section

The `optimization_problem` section generates the most files — problem
serialization/deserialization for both unary and chunked gRPC paths.

```yaml
optimization_problem:
  cpp_type: "cpu_optimization_problem_t<i_t, f_t>"
  proto_message: OptimizationProblem

  scalars: [...]
  arrays: [...]
  setter_groups: { ... }
```

### Scalars

```yaml
scalars:
- problem_name:
    field_num: 1
    type: string
- maximize:
    field_num: 3
    type: bool
    getter: get_sense()
- problem_category:
    field_num: 6
    type: problem_category    # enum key reference
```

| Property | Default | Description |
|---|---|---|
| `field_num` | *(required)* | Proto field number in `OptimizationProblem` |
| `type` | `double` | One of: `double`, `int32`, `bool`, `string`, or an enum key |
| `getter` | `get_<name>()` | C++ getter on `cpu_optimization_problem_t` |
| `setter` | `set_<name>()` | C++ setter (can be overridden via `setter_getter_root`) |
| `setter_getter_root` | `<name>` | Base name for default getter/setter derivation |

### Arrays

```yaml
arrays:
- variable_names:
    array_id: 0
    field_num: 7
    type: repeated string

- A_values:
    array_id: 2
    field_num: 9
    setter_getter_root: constraint_matrix_values
    setter_group: csr_constraint_matrix

- variable_types:
    array_id: 12
    field_num: 19
    type: repeated variable_type    # repeated enum

- row_types:
    array_id: 11
    field_num: 18
    type: bytes
    conditional: true
```

| Property | Default | Description |
|---|---|---|
| `field_num` | *(required)* | Proto field number in `OptimizationProblem` |
| `array_id` | *(required)* | Numeric value for the `ArrayFieldId` enum |
| `type` | `repeated double` | One of: `repeated double`, `repeated int32`, `repeated string`, `bytes`, or `repeated <enum_key>` |
| `getter` | `get_<root>_host()` | C++ getter (strings use `get_<root>()` without `_host`) |
| `setter` | `set_<root>()` | C++ setter |
| `setter_getter_root` | `<name>` | Base name for getter/setter derivation when different from field name |
| `setter_group` | *(none)* | Name of a multi-argument setter group (see below) |
| `conditional` | `false` | If true, serialization is guarded by an emptiness check |
| `skip_conversion` | `false` | If true, the field appears in the proto but is excluded from conversion code |

### Setter Groups

Some C++ setters take multiple arrays at once (e.g. CSR matrix = values +
indices + offsets). Setter groups handle this:

```yaml
setter_groups:
  csr_constraint_matrix:
    setter: set_csr_constraint_matrix
    fields: [A_values, A_indices, A_offsets]

  quadratic_objective:
    setter: set_quadratic_objective_matrix
    fields: [Q_values, Q_indices, Q_offsets]
```

| Property | Description |
|---|---|
| `setter` | C++ setter function name (called with all field arrays as arguments) |
| `fields` | Ordered list of array field names that are passed to the setter |

Arrays that belong to a setter group are excluded from normal per-field
deserialization and handled as a batch instead.

During deserialization, the generator automatically guards setter group calls
on the structural sentinel field — preferring any field whose name ends with
`_offsets` (so CSR-style groups still trigger when nnz=0, because `*_offsets`
is non-empty even when `*_values`/`*_indices` are empty) and falling back to
the first field otherwise. The generated guard looks like
`if (pb_problem.a_offsets_size() > 0)`. For groups that contain
`_values`/`_indices` pairs, the generator additionally emits a size-mismatch
check that throws `std::invalid_argument` if the companion arrays disagree —
no explicit condition attribute is needed in the registry.

---

## Field Number Allocation

Field numbers are required for proto compatibility. They can be assigned
manually or auto-assigned.

### Manual assignment

Specify `field_num` and `array_id` on each field entry. This is the default
workflow.

### Auto-assignment

Run with `--auto-number` to fill in any missing `field_num` or `array_id`
values. This requires `ruamel.yaml` (preserves YAML comments and formatting):

```bash
python cpp/src/grpc/codegen/generate_conversions.py --auto-number
```

### Stripping field numbers

Run with `--strip` to remove all `field_num` and `array_id` values from
`field_registry.yaml`. This is useful for reviewing pure field definitions
without numbering clutter, or for forcing a full re-assignment via
`--auto-number`:

```bash
python cpp/src/grpc/codegen/generate_conversions.py --strip
python cpp/src/grpc/codegen/generate_conversions.py --auto-number
```

If all field numbers have been stripped, running the generator without
`--auto-number` will produce an error.

The numbering ranges:

| Scope | Range |
|---|---|
| `optimization_problem` field_num | 1+ (contiguous, shared across scalars and arrays) |
| `optimization_problem` array_id | 0+ (separate namespace) |
| LP solution scalars (ChunkedResultHeader) | 1000–1999 |
| MIP solution scalars (ChunkedResultHeader) | 2000–2999 |
| Warm start scalars (ChunkedResultHeader) | 3000–3999 |
| Solution array array_id | 0+ (global pool shared across LP, MIP, and warm start) |
| Solution array field_num | 1+ per message (no cap) |
| Settings field_num | 1+ per message (no cap) |

---

## What Gets Generated

### `cuopt_remote_data.proto`

A complete proto file with all data messages and enums derived from the
registry: `OptimizationProblem`, `PDLPSolverSettings`, `MIPSolverSettings`,
`PDLPWarmStartData`, `LPSolution`, `MIPSolution`, `ChunkedResultHeader`,
`ResultArrayDescriptor`, and the enums they reference (`PDLPTerminationStatus`,
`MIPTerminationStatus`, `PDLPSolverMode`, `LPMethod`, `VariableType`,
`ProblemCategory`, `ResultFieldId`, `ArrayFieldId`).

The hand-maintained `cuopt_remote.proto` and `cuopt_remote_service.proto` can
import this generated file to avoid duplicating definitions.

### Enum converter `.inc` files

Per-domain C++ switch functions, split by the `domain` tag on each enum:

- `generated_enum_converters_problem.inc` — enums with `domain: problem`
- `generated_enum_converters_settings.inc` — enums with `domain: settings`
- `generated_enum_converters_solution.inc` — enums with `domain: solution`

### Settings `.inc` files

- `generated_pdlp_settings_to_proto.inc` / `generated_proto_to_pdlp_settings.inc`
- `generated_mip_settings_to_proto.inc` / `generated_proto_to_mip_settings.inc`

### Solution `.inc` files (6 per solution type)

For each of LP and MIP:

| File | Function body it provides |
|---|---|
| `generated_{lp,mip}_solution_to_proto.inc` | Unary C++ solution → proto |
| `generated_proto_to_{lp,mip}_solution.inc` | Unary proto → C++ solution |
| `generated_{lp,mip}_chunked_header.inc` | Populate `ChunkedResultHeader` |
| `generated_collect_{lp,mip}_arrays.inc` | Collect solution arrays as byte maps |
| `generated_chunked_to_{lp,mip}_solution.inc` | Reassemble C++ solution from chunked data |
| `generated_estimate_{lp,mip}_size.inc` | Estimate serialized proto size |

### Problem `.inc` files

| File | Function body it provides |
|---|---|
| `generated_problem_to_proto.inc` | C++ problem → unary proto |
| `generated_proto_to_problem.inc` | Unary proto → C++ problem |
| `generated_estimate_problem_size.inc` | Estimate serialized problem size |
| `generated_populate_chunked_header_{lp,mip}.inc` | Populate chunked problem header |
| `generated_chunked_header_to_problem.inc` | Set problem scalars from chunked header |
| `generated_chunked_arrays_to_problem.inc` | Set problem arrays from chunked byte maps |
| `generated_build_array_chunks.inc` | Build `SendArrayChunkRequest` list for upload |
| `generated_array_field_element_size.inc` | Switch body for per-field element byte size |

---

## How `.inc` Files Are Consumed

The `.inc` files are `#include`d directly inside C++ function bodies in:

- `cpp/src/grpc/grpc_settings_mapper.cpp`
- `cpp/src/grpc/grpc_solution_mapper.cpp`
- `cpp/src/grpc/grpc_problem_mapper.cpp`
- `cpp/src/grpc/server/grpc_field_element_size.hpp`

CMake adds `cpp/src/grpc/codegen/generated` to the include path for the `cuopt` and
`cuopt_grpc_server` targets, so the bare `#include "generated_*.inc"` directives
resolve without any copy step.

---

## Adding a New Field — Walkthroughs

### Add `dual_bound` (double) to MIP solution

1. Add to `mip_solution.scalars`:
   ```yaml
   - dual_bound:
       field_num: 2012
   ```

2. Add to `mip_solution.constructor_args.scalars` in the correct position:
   ```yaml
   constructor_args:
     scalars:
     - mip_termination_status
     - mip_objective
     # ...
     - dual_bound            # ← new, position must match C++ constructor
   ```

3. Add the constructor parameter to `cpu_mip_solution_t`.

4. Regenerate:
   ```bash
   python cpp/src/grpc/codegen/generate_conversions.py
   ```

5. Build and test.

The proto field in `ChunkedResultHeader` and the solution message are generated
automatically — no manual `.proto` edits needed.

### Add `detect_infeasibility_v2` (bool) to PDLP settings

1. Add to `pdlp_settings.fields`:
   ```yaml
   - detect_infeasibility_v2:
       field_num: 31
       type: bool
   ```

2. Add the C++ struct member to `pdlp_solver_settings_t`:
   ```cpp
   bool detect_infeasibility_v2{false};
   ```

3. Regenerate and build. You never touch `grpc_settings_mapper.cpp` or any
   `.proto` file — the proto field in `PDLPSolverSettings` is generated
   automatically.

### Add a new array to the optimization problem

1. Add to `optimization_problem.arrays`:
   ```yaml
   - my_new_array:
       array_id: 18
       field_num: 25
       type: repeated double
   ```

2. Add getter/setter to `cpu_optimization_problem_t`.

3. Regenerate and build.

The `ArrayFieldId` enum entry and the `OptimizationProblem` proto field are
generated automatically.

### Add a tolerance to MIP settings (nested sub-struct)

If the C++ member is nested under `tolerances.`, just add it inside the
`tolerances` list:

```yaml
- tolerances:
  - relative_mip_gap:
      field_num: 2
  - my_new_tolerance:       # ← new
      field_num: 14
```

The generator will access it as `settings.tolerances.my_new_tolerance`.

---

## Related Documentation

- `GRPC_INTERFACE.md` — Chunked transfer protocol, message size limits, error handling.
- `GRPC_SERVER_ARCHITECTURE.md` — Server process model, IPC, threads, job lifecycle.
- `GRPC_QUICK_START.md` — Starting the server and solving remotely from Python, CLI, or C.
