# Field Registry Reference

Practical, field-level reference for editing
[`field_registry.yaml`](field_registry.yaml). Documents every attribute you can
set on a registry entry and walks through the common edits (add a scalar, add an
array, etc.).

For architecture (what the codegen is, how `.inc` files are wired into builds,
where files live), see
[`../GRPC_CODE_GENERATION.md`](../GRPC_CODE_GENERATION.md). For the wire-level
chunked transfer protocol, see [`../GRPC_INTERFACE.md`](../GRPC_INTERFACE.md).

## Contents

- [1. Mental model](#1-mental-model)
- [2. Common attributes (cover once)](#2-common-attributes-cover-once)
  - [2.1 Scalar fields](#21-scalar-fields)
  - [2.2 Array fields](#22-array-fields)
  - [2.3 Enum definitions](#23-enum-definitions)
  - [2.4 The `optional:` flag](#24-the-optional-flag)
  - [2.5 The `sentinel:` block](#25-the-sentinel-block)
  - [2.6 Type vocabulary](#26-type-vocabulary)
  - [2.7 Auto-derived names](#27-auto-derived-names)
- [3. Section-specific extras](#3-section-specific-extras)
  - [3.0 Keys shared by all sections](#30-keys-shared-by-all-sections)
  - [3.1 Settings sections](#31-settings-sections-pdlp_settings-mip_settings)
  - [3.2 Solution sections](#32-solution-sections-lp_solution-mip_solution)
  - [3.3 Optimization problem section](#33-optimization-problem-section-optimization_problem)
- [4. Field number allocation](#4-field-number-allocation)
- [5. How-to walkthroughs](#5-how-to-walkthroughs)
  - [5.1 Add a scalar (any section)](#51-add-a-scalar-any-section)
  - [5.2 Add an array (problem or solution)](#52-add-an-array-problem-or-solution)
  - [5.3 Add a new enum / a value to an existing enum](#53-add-a-new-enum--a-value-to-an-existing-enum)
  - [5.4 Add a setter group](#54-add-a-setter-group-csr-style-multi-array-setter)
  - [5.5 Add a `repeated_messages` entry](#55-add-a-repeated_messages-entry-nested-message-type)
  - [5.6 Add a warm-start field (LP only)](#56-add-a-warm-start-field-lp-only)
  - [5.7 Make a setting `optional`](#57-make-a-setting-optional)
  - [5.8 Map a sentinel value (`max()` ↔ `-1`)](#58-map-a-sentinel-value-max--1)
- [6. V2 cutover checklist](#6-v2-cutover-checklist)
  - [Optional V2 add-ons (items 5–8)](#optional-v2-add-ons-items-58)
- [7. Post-cutover investigations](#7-post-cutover-investigations)

---

## 1. Mental model

The registry is a single YAML file with a small set of top-level sections, each
describing one C++ ↔ proto mapping:

| Section | Describes |
|---|---|
| `enums:` | C++ ↔ proto enum mappings (shared by all other sections). |
| `optimization_problem:` | Problem data (the `OptimizationProblem` proto). |
| `pdlp_settings:`, `mip_settings:` | Solver settings. |
| `lp_solution:`, `mip_solution:` | Solver result data. |
| `chunked_result_header:` | Submessages embedded on `ChunkedResultHeader` (mostly the array descriptors). |

Each section contains zero or more of: `scalars:`, `arrays:`, `fields:` (nested
settings), `repeated_messages:`, `setter_groups:`, `constructor_args:`,
`embeds:`, `warm_start:`.

When the generator runs, it produces:

- One `cuopt_remote_data.proto` (the wire schema for data messages).
- A family of `generated_*.inc` files (C++ function bodies) in `generated/`.
- Auto-derived `ArrayFieldId` / `ResultFieldId` enums for chunk routing.

```bash
python cpp/src/grpc/codegen/generate_conversions.py
bash ci/verify_grpc_codegen.sh           # sanity check: detect drift
```

The committed `generated/` directory is the source of truth that builds compile
against; CI fails the build if it doesn't match what the generator produces.

---

## 2. Common attributes (cover once)

Every element type in the registry follows the same conventions where possible.
This section documents each element type once; section-specific extras (§3)
only describe what diverges.

### 2.1 Scalar fields

A scalar appears in one of these places:

- `optimization_problem.scalars:`
- `pdlp_settings.fields:`, `mip_settings.fields:` (also nested sub-structs)
- `lp_solution.scalars:`, `mip_solution.scalars:`
- `lp_solution.warm_start.scalars:` (LP only)
- `<section>.repeated_messages[*].scalars:`

YAML shape:

```yaml
scalars:
- field_name:
    field_num: 7
    type: double      # default
    # optional attributes:
    getter: "get_field_name()"
    setter: "set_field_name"
    sentinel: { ... }
    optional: true
    proto_only: true
    member: foo.bar
    from_proto_cast: char
```

| Attribute | Required | Default | Effect |
|---|---|---|---|
| `field_num` | yes | — | Wire tag in the containing proto message. See §4 for ranges and auto-numbering. |
| `type` | no | `double` | One of `double`, `int32`, `int64`, `bool`, `string`, or the key of an entry under `enums:`. |
| `getter` | no | varies (see §2.7) | C++ expression to read the value from the C++ object. Accepts either a bare method name (`get_X`) or a complete expression (`get_X()`, `get_error_status().what()`); bare identifiers get `()` appended automatically. |
| `setter` | no | varies | C++ setter name (problem only; settings use struct-member assignment, solutions use constructor). Bare method name (`set_X`); a trailing `()` if present is stripped automatically, because the generator owns the argument list. |
| `optional` | no | `false` | Adds `optional` to the proto and guards from-proto reads with `has_X()`. See §2.4 for full propagation rules. |
| `sentinel` | no | — | Maps a C++ sentinel value (e.g. `max()`) to a reserved wire value (e.g. `-1`). See §2.5. |
| `proto_only` | no | `false` | Set on the proto but excluded from the C++ constructor call. Used for transport-only fields (e.g. `error_message` on solutions). |
| `member` | no | `<name>`, with nesting prefix auto-prepended for settings sub-structs (`tolerances.<name>`, `heuristic_params.<name>`, …) | C++ struct member name. Set explicitly only when the C++ name diverges from the wire field name — in practice, warm-start fields whose C++ members carry a trailing underscore (`initial_primal_weight_`). |
| `from_proto_cast` | no | — | Explicit C++ cast when reading from proto. Most commonly used to wrap a wire `int32` back into an enum class (e.g. `presolver_t`, `method_t`) or a narrow type (e.g. `char` for `constraint_row_type`). When set, the matching to-proto cast back to the wire's C++ type (e.g. `int32_t` for wire `int32`) is derived automatically — no `to_proto_cast:` needed. |
| `to_proto_cast` | no | derived from `type:` when `from_proto_cast` is set; otherwise no cast | Escape hatch: explicit C++ cast when writing to proto (e.g. `int32_t`). Rarely needed — set only when the auto-derivation above is wrong. |

### 2.2 Array fields

An array appears in:

- `optimization_problem.arrays:`
- `lp_solution.arrays:`, `mip_solution.arrays:`
- `lp_solution.warm_start.arrays:`
- `<section>.repeated_messages[*].arrays:`

YAML shape:

```yaml
arrays:
- array_name:
    field_num: 9
    array_id: 2
    type: repeated double    # default
    # optional attributes:
    setter_getter_root: constraint_matrix_values
    setter_group: csr_constraint_matrix
    conditional: true
    skip_conversion: true
    getter: "..."
    setter: "..."
```

| Attribute | Required | Default | Effect |
|---|---|---|---|
| `field_num` | yes | — | Wire tag in the containing message. |
| `array_id` | yes | — | Numeric value for the auto-derived enum: `ArrayFieldId::FIELD_<UPPER_SNAKE(name)>` for problem arrays, `ResultFieldId::RESULT_<UPPER_SNAKE(name)>` for solution arrays (warm-start arrays get the `result_id_prefix`). Container-relative for `repeated_messages` arrays (see §3.3). |
| `type` | no | `repeated double` | `repeated double`, `repeated int32`, `repeated string`, `bytes`, or `repeated <enum_key>`. |
| `setter_getter_root` | no | `<array_name>` | Base name used to derive the getter/setter (e.g. `setter_getter_root: constraint_matrix_values` → getter `get_constraint_matrix_values_host`, setter `set_constraint_matrix_values`). |
| `setter_group` | no | — | Name of a `setter_groups:` entry. Arrays in the same group are passed together to a multi-argument setter. See §3.3. |
| `conditional` | no | `false` | Guards the from-proto setter call on `_size() > 0`. Use for arrays whose absence means "don't touch the existing value." |
| `skip_conversion` | no | `false` | Field appears in the proto but is excluded from conversion code. |
| `getter` | no | derived (below) | Override the default getter expression. |
| `setter` | no | derived | Override the default setter name. |
| `member` | no | `<name>` | Warm-start arrays only: C++ struct member name. Set explicitly when the C++ member carries a trailing underscore (`current_primal_solution_`). |

#### Default getter / setter derivation

| Context | `type` | Default getter | Default setter |
|---|---|---|---|
| Problem array | `repeated string` | `get_<root>()` | `set_<root>` |
| Problem array | anything else | `get_<root>_host()` | `set_<root>` |
| Solution array | `repeated string` | `get_<root>()` | *(via constructor)* |
| Solution array | anything else | `get_<root>_host()` | *(via constructor)* |
| Warm-start array | — | direct member access (`ws.<member>`) | direct member assignment |

`<root>` is `setter_getter_root` if set, else `<array_name>`. The override
exists for both problem and solution arrays so the wire field name and the
C++ accessor root can diverge without writing an explicit `getter:` / `setter:`
expression. Example on the solution side: `mip_solution` (wire field) ↔
`get_solution_host()` (C++) is expressed as `setter_getter_root: solution`.

### 2.3 Enum definitions

Top-level `enums:` section. Each entry maps a C++ enum to a proto enum.

```yaml
enums:
  pdlp_termination_status:
    domain: solution
    proto_prefix: PDLP
    values:
    - NoTermination
    - NumericalError
    - Optimal
    - PrimalInfeasible

  pdlp_solver_mode:
    domain: settings
    default: Stable3       # override default when not the first value
    values:
    - Stable1
    - Stable2
    - Methodical1
    - Fast1
    - Stable3

  lp_method:
    domain: settings
    cpp_type: method_t     # override when not <key>_t
    values:
    - Concurrent
    - PDLP
    - DualSimplex
    - Barrier
```

| Attribute | Required | Default | Effect |
|---|---|---|---|
| `domain` | yes | — | One of `problem`, `settings`, `solution`. Determines which `.inc` file the converters land in (`generated_enum_converters_<domain>.inc`). |
| `values` | yes | — | List of C++ enum value names. Bare names auto-number from 0; `{Name: N}` resets the counter to N (C-style enum semantics). |
| `proto_type` | no | PascalCase from key (with acronym handling) | Override the proto enum type name. `pdlp_termination_status` → `PDLPTerminationStatus` by default; acronyms PDLP, MIP, LP, QP, VRP, PDP, TSP are uppercased. |
| `proto_prefix` | no | — | Prefix applied to value names: with `PDLP`, `Optimal` becomes `PDLP_OPTIMAL`. Without a prefix, value names use the bare CppName. |
| `cpp_type` | no | `<key>_t` | Override the C++ type name. |
| `default` | no | first value | C++ value to return for unrecognized proto values. Override when the C++ default isn't the first one. |
| `aliases` | no | — | (auto-derived enums only — `result_field_id`) Map of extra value names to numeric IDs for backward compatibility. |

Auto-derived from the key:

- C++ converter functions: `to_proto_<key>()` and `from_proto_<key>()`.
- Proto value names: `<proto_prefix>_<UPPER_SNAKE(CppName)>` (or just `<CppName>` if no prefix).

Enums can be referenced from any scalar/array via `type: <enum_key>`, e.g.
`type: pdlp_termination_status` or `type: repeated variable_type`.

### 2.4 The `optional:` flag

Adds the proto3 `optional` keyword to the field declaration. The wire format is
unchanged (no extra bytes; `optional` only enables presence tracking).

When the field is *present* on the wire, behavior is identical to a bare field.
When the field is *absent*, the from-proto code's behavior depends on the
section:

- **Settings / problem-unary**: the C++ field's in-class default is preserved
  (the setter / assignment is skipped). This is the case the flag is designed
  for: without `optional:`, an external client that omits the field silently
  overwrites the C++ default with the proto3 zero.
- **Problem-chunked**: currently *not honored* — the chunked path reads from
  the hand-written `ChunkedProblemHeader` message which does not declare
  `optional` on its fields. Tracked for unification (see §6).
- **Solution**: cosmetic. Solutions are constructor-built each call; there's no
  pre-existing default to preserve. The `optional` keyword adds `has_X()` to
  the proto for client-side presence detection, but the from-proto path still
  passes the proto3 zero to the constructor when the field is absent.

**Rule of thumb**: for new scalars with non-zero C++ defaults in *settings* or
*problem*, add `optional: true` unless you're sure all clients always set the
field.

From-proto codegen for the settings/problem-unary case looks like:

```cpp
if (pb.has_X()) {
  // (optional sentinel guard runs here)
  target = pb.X();
}
```

`optional:` composes with `sentinel:` (§2.5) — the optional guard wraps the
sentinel value-guard.

### 2.5 The `sentinel:` block

Maps a C++ sentinel value to a reserved wire value. Use for fields like
`iteration_limit` where the C++ default is `std::numeric_limits<i_t>::max()`
("no limit") and we want to encode it as `-1` on the wire.

Two forms — a named shorthand for the common case, and a verbose dict for
anything that doesn't fit a convention.

#### Named conventions (preferred when applicable)

```yaml
- iteration_limit:
    field_num: 10
    type: int64
    sentinel: max_as_negative_1
    optional: true
```

| Convention | Pattern | C++ side |
|---|---|---|
| `max_as_negative_1` | C++ `std::numeric_limits<T>::max()` ↔ wire `-1`, guard `>= 0` | `T` is auto-derived from the field's `type`: `i_t` for `int32`/`int64`, `f_t` for `double`. |

Use a named convention when it fits; use the verbose form below for anything
else. New conventions can be added to `SENTINEL_CONVENTIONS` in
`generate_conversions.py`.

#### Verbose form

```yaml
- some_field:
    field_num: 10
    type: int64
    sentinel:
      to_proto: "std::numeric_limits<i_t>::max()"
      proto_value: -1
      from_proto_guard: ">= 0"
      from_proto_cast: "i_t"
```

| Attribute | Effect |
|---|---|
| `to_proto` | C++ expression for the sentinel value. If `field == to_proto`, the wire field is set to `proto_value`. |
| `proto_value` | Wire value representing the sentinel (typically `-1` for `int32`/`int64` fields). |
| `from_proto_guard` | C++ comparison expression. If `pb.X() <guard>` evaluates true, the value is used; otherwise the C++ default is preserved (or, for solutions, the `to_proto` value is round-tripped). |
| `from_proto_cast` | Optional cast applied when reading. Falls back to the field's `from_proto_cast`. |

The to-proto direction is identical across all sections. From-proto direction:

- **Settings / problem**: when the guard fails, the assignment is skipped
  (preserve C++ default).
- **Solution**: because solutions are built via constructor and there are no
  defaults to fall back on, when the guard fails the local takes the
  `to_proto` C++ value (round-trips the sentinel back through the wire).

`-1` is used on the wire (rather than the literal C++ `max()`) because it's
stable across languages and integer widths.

### 2.6 Type vocabulary

| YAML `type:` | Proto type | Typical C++ type |
|---|---|---|
| `double` *(default for scalars)* | `double` | `double` / `f_t` |
| `int32` | `int32` | `int32_t` / `i_t` |
| `int64` | `int64` | `int64_t` / `i_t` |
| `bool` | `bool` | `bool` |
| `string` | `string` | `std::string` |
| `<enum_key>` | proto enum type | C++ enum type |
| `repeated double` *(default for arrays)* | `repeated double` | `std::vector<double>` / host buffer |
| `repeated int32` | `repeated int32` | `std::vector<i_t>` |
| `repeated string` | `repeated string` | `std::vector<std::string>` |
| `repeated <enum_key>` | `repeated <proto_enum>` | `std::vector<<cpp_enum>>` |
| `bytes` | `bytes` | byte buffer |

### 2.7 Auto-derived names

The generator derives many names from the YAML key. You almost never need to
specify these explicitly.

| Auto-derived from | Result |
|---|---|
| Problem / solution scalar `name` | Default getter `get_<name>()` |
| Problem array `name` | Default getter `get_<root>_host()` (numeric/enum/bytes) or `get_<root>()` (`repeated string`) |
| Problem array `name` | `ArrayFieldId::FIELD_<UPPER_SNAKE(name)>` enum value |
| Solution array `name` | Default getter `get_<root>_host()` (numeric) or `get_<root>()` (`repeated string`) |
| Solution array `name` | `ResultFieldId::RESULT_<UPPER_SNAKE(name)>` (warm-start arrays apply `result_id_prefix`) |
| Settings field `name` | `member` = `<name>`, with the YAML sub-struct path prepended for nested fields (`tolerances.<name>`, `heuristic_params.<name>`). Override only when the C++ member name diverges from the wire field name. |
| Enum `key` | C++ type `<key>_t`, proto type acronym-aware PascalCase, converter fns `to_proto_<key>` / `from_proto_<key>` |
| Enum `value` name (CppName) | Proto value `<proto_prefix>_<UPPER_SNAKE(CppName)>` |
| Section name (`lp_solution` / `mip_solution`) | `ChunkedResultHeader.problem_category` value (LP or MIP) |

---

## 3. Section-specific extras

For shared semantics (`scalars:`, `arrays:`, `optional:`, `sentinel:`, enums),
see §2.

### 3.0 Keys shared by all sections

These top-level keys apply across `pdlp_settings`, `mip_settings`,
`lp_solution`, `mip_solution`, and `optimization_problem`. Each subsection
below covers only what's *unique* to that section.

| Key | Where it applies | Description |
|---|---|---|
| `cpp_type` | All sections | Fully-qualified C++ type the section maps to (e.g. `pdlp_solver_settings_t<i_t, f_t>`, `cpu_lp_solution_t<i_t, f_t>`, `cpu_optimization_problem_t<i_t, f_t>`). |
| `scalars:` / `arrays:` | Solutions, optimization_problem (and `repeated_messages` entries) | Lists of scalar / array fields. Standard semantics from §2. Settings sections use `fields:` instead (see §3.1). |
| `embeds:` | Any section that needs a submessage field | Optional list of submessage fields embedded on the section's proto (e.g. `PDLPWarmStartData warm_start_data = N;`). Each entry pins its own `field_num` so the auto-numberer keeps it out of the scalar / array pool. Used today by `pdlp_settings`, `lp_solution`, and `chunked_result_header`. |

Section-level proto message names (`PDLPSolverSettings`, `MIPSolverSettings`,
`OptimizationProblem`, `LPSolution`, `MIPSolution`) are hardcoded in the
generator — each section maps to one fixed proto message, so there's no
override knob.

### 3.1 Settings sections (`pdlp_settings`, `mip_settings`)

Unique to settings:

| Key | Description |
|---|---|
| `fields:` | List of settings fields. Supports nested sub-structs (see below). Used instead of `scalars:` because settings need to express the C++ struct's nesting hierarchy. |

#### Nested sub-structs

A list-valued entry under `fields:` represents a C++ sub-struct. The generator
automatically prefixes `member` with the nesting path.

```yaml
fields:
- tolerances:          # nested sub-struct
  - absolute_gap_tolerance:
      field_num: 1
  - relative_gap_tolerance:
      field_num: 2
- time_limit:          # top-level field
    field_num: 9
```

In the generated code, `tolerances.absolute_gap_tolerance` is accessed as
`settings.tolerances.absolute_gap_tolerance` automatically — the parser walks
the YAML nesting and synthesizes `member = tolerances.absolute_gap_tolerance`
during field parse. You only set `member` explicitly when the C++ member name
diverges from the wire field name (in the current registry, this happens only
for warm-start fields whose C++ members carry a trailing underscore).

### 3.2 Solution sections (`lp_solution`, `mip_solution`)

Unique to solutions:

| Key | Description |
|---|---|
| `constructor_args:` | Positional order of scalars passed to the C++ constructor. |
| `warm_start:` | (LP only) Conditional sub-object for PDLP warm-start data. |

#### `constructor_args:`

Controls only the **scalar** argument ordering. The full constructor call
the generator produces is:

```cpp
return CppType(
    std::move(array_1), std::move(array_2), ...,    // every array, YAML order
    scalar_1, scalar_2, ...,                         // scalars from constructor_args.scalars
    std::move(ws)                                    // appended iff warm_start is declared & present
);
```

- **Arrays.** Every entry in the section's `arrays:` list is passed, in YAML
  declaration order, wrapped in `std::move`. There is no per-array opt-out
  and no separate list — `constructor_args` doesn't mention them at all. If
  you don't want an array to be a constructor argument, don't put it in the
  `arrays:` list.
- **Scalars.** Only the scalars listed (by name) under
  `constructor_args.scalars` are passed, in the order listed. The order must
  match the C++ constructor signature. Scalars present in the section's
  `scalars:` list but absent from `constructor_args.scalars` are still
  serialized on the wire — they just don't reach the constructor.
- **`proto_only: true` scalars.** Excluded from the constructor automatically
  even if listed; in practice you just omit them from `constructor_args.scalars`.
- **Warm-start.** If `warm_start:` is declared on the section and the
  incoming payload actually carries warm-start data, `std::move(ws)` is
  appended as the final argument.

```yaml
constructor_args:
  scalars:
  - lp_termination_status
  - primal_objective
  - dual_objective
  # ... order must match the C++ constructor
```

#### `warm_start:` (LP only)

```yaml
warm_start:
  presence_check: has_warm_start_data()
  getter: get_cpu_pdlp_warm_start_data()
  chunked_header_prefix: ws_     # optional, defaults to ""
  result_id_prefix: WS           # optional
  scalars:
    - ...
  arrays:
    - ...
```

| Attribute | Description |
|---|---|
| `presence_check` | C++ predicate to test if warm-start data exists on the solution object. |
| `getter` | C++ expression that returns the warm-start struct from the solution object. |
| `chunked_header_prefix` | Prefix applied to scalar field names on `ChunkedResultHeader` (avoids tag-name collisions with the main solution scalars). |
| `result_id_prefix` | Prefix applied to the auto-derived `ResultFieldId` enum value for warm-start arrays. |
| `scalars:` / `arrays:` | Standard (see §2). Use `member:` when the C++ struct member name differs from the field name (typical: trailing underscore on private-ish members). |

Warm-start presence during chunked deserialization is auto-detected: the
generator emits `arrays.count(RESULT_WS_*) != 0` OR'd across every warm-start
array, so the data is reconstructed if any warm-start array shows up.

### 3.3 Optimization problem section (`optimization_problem`)

Unique to the optimization-problem section:

| Key | Description |
|---|---|
| `setter_groups:` | Multi-array CSR-style setters (see below). |
| `repeated_messages:` | Nested repeated message types (see below). |

#### Setter groups (CSR-style multi-array setters)

Some C++ setters take multiple arrays at once (e.g. CSR constraint matrix =
values + indices + offsets). A setter group declares the C++ setter and the
participating arrays:

```yaml
setter_groups:
  csr_constraint_matrix:
    setter: set_csr_constraint_matrix
    fields: [A_values, A_indices, A_offsets]
```

Each participating array carries a matching `setter_group: csr_constraint_matrix`
on its definition.

The generator:

- Excludes group fields from normal per-field deserialization.
- Emits a guarded batch call: `if (pb_problem.<sentinel>_size() > 0) cpu_problem.<setter>(values..., sizes...)`.
- Picks the **structural sentinel** as the field whose name ends in
  `_offsets` (so zero-nnz matrices still trigger the setter, since the
  offsets array is non-empty even when values/indices are empty); falls back
  to the first field if no `_offsets` member exists.
- Auto-emits a `(*_values, *_indices)` size-mismatch check that throws
  `std::invalid_argument` on mismatch — no explicit condition attribute
  needed in the registry.

#### `repeated_messages:` (nested message types)

A `repeated_messages:` entry declares a `repeated <MessageType> <name> = <num>;`
field where each entry has its own scalars and arrays. Per-entry arrays are
unbounded in size, so they ride the chunked-array protocol with a
container-relative `array_id`.

Use this when the underlying C++ data is naturally a list of independent
records that *cannot* be folded into a single shared structure on the wire.
Linear constraints, by contrast, share one CSR triple
(`A_values`/`A_indices`/`A_offsets`) for the entire constraint matrix and
ride the standard top-level array path; quadratic constraints can't fold the
same way because each row carries its own independent `Q` matrix with no
shared sparsity pattern, so each row becomes one entry in
`quadratic_constraints` with its own per-row arrays nested inside. That
nesting is also why the chunked path for these arrays uses the three-tuple
`(container_field_num, container_index, array_id)` instead of the flat
`array_id` used for top-level arrays — `container_index` disambiguates which
repeated entry a chunk belongs to.

```yaml
repeated_messages:
- quadratic_constraints:
    field_num: 25                                  # tag in OptimizationProblem
    message_type: QuadraticConstraint
    cpp_inner_type: "typename cpu_optimization_problem_t<i_t, f_t>::quadratic_constraint_t"
    presence_check: has_quadratic_constraints()
    getter: get_quadratic_constraints()
    setter: set_quadratic_constraints
    scalars:
    - constraint_row_index:
        field_num: 1                               # local tag within QuadraticConstraint
        type: int32
        to_proto_cast: int32_t
    # ...
    arrays:
    - linear_values:
        field_num: 5
        array_id: 0                                # container-relative
    - linear_indices:
        field_num: 6
        array_id: 1
        type: repeated int32
    # ...
    companion_pairs:
    - [linear_values, linear_indices]
    - [rows, cols]
    - [cols, vals]
```

| Attribute | Required | Description |
|---|---|---|
| `field_num` | yes | Wire tag in the parent message. |
| `message_type` | yes | Name of the generated nested proto message. |
| `cpp_inner_type` | for unary decode | Fully-qualified C++ type for one entry. |
| `presence_check` | for unary encode | C++ predicate to check if the repeated field is populated. |
| `getter` | for unary encode | C++ getter for the repeated container. |
| `setter` | for unary decode | C++ setter that takes the assembled vector of entries. |
| `scalars:` / `arrays:` | as applicable | Per-entry fields (same semantics as top-level scalars/arrays). |
| `companion_pairs:` | optional | List of `[a, b]` pairs whose `.size()` must match on decode (throws `std::invalid_argument` otherwise). |

Per-entry `field_num` values are local to the nested proto message — independent
of the parent's tag pool.

Per-entry `array_id` values are **container-relative** — small dense ints
starting from 0 within each container, independent of the top-level
`ArrayFieldId` namespace. Chunks route by
`(container_field_num, container_index, array_id)`.

---

## 4. Field number allocation

### 4.1 Manual vs auto

By default, you specify `field_num` and `array_id` on each entry. Auto-numbering
is opt-in:

```bash
# Fill in any missing field_num / array_id values
python cpp/src/grpc/codegen/generate_conversions.py --auto-number

# Strip all field_num / array_id values (e.g. to reassign from scratch)
python cpp/src/grpc/codegen/generate_conversions.py --strip
```

`--auto-number` and `--strip` require `ruamel.yaml` (preserves YAML formatting
and comments). Without `--auto-number`, a fully-stripped registry errors out
and a partially-stripped one silently emits broken code for the unnumbered
entries — so always re-run `--auto-number` after a `--strip`.

### 4.2 Tag ranges

| Scope | Range |
|---|---|
| `optimization_problem` `field_num` | 1+ (shared pool across scalars and arrays) |
| `optimization_problem` `array_id` | 0+ (separate pool) |
| LP solution scalars (`ChunkedResultHeader`) | 1000–1999 |
| MIP solution scalars (`ChunkedResultHeader`) | 2000–2999 |
| Warm-start scalars (`ChunkedResultHeader`) | 3000–3999 |
| Solution array `array_id` | 0+ (global pool across LP, MIP, warm-start) |
| Solution array `field_num` (per unary solution message) | 1+ per message (no cap) |
| Settings `field_num` (per settings message) | 1+ per message (no cap) |
| `repeated_messages` per-entry `field_num` | 1+ within the nested message |
| `repeated_messages` per-entry `array_id` | 0+ within the container (container-relative) |

### 4.3 Validation

The generator runs a uniqueness check per scope and fails with a precise error
on collision. CI runs `verify_grpc_codegen.sh` to ensure the committed
`generated/` directory matches what the generator produces — drift fails the
build.

---

## 5. How-to walkthroughs

Each walkthrough follows the same outline:

1. Edit `field_registry.yaml`.
2. Add the matching C++ member / getter / setter.
3. Regenerate and verify:

   ```bash
   python cpp/src/grpc/codegen/generate_conversions.py
   bash ci/verify_grpc_codegen.sh
   ```

4. Build and run the gRPC tests.

### 5.1 Add a scalar (any section)

Same recipe for problem, settings, and solution scalars. The tag-range and
getter/setter conventions differ by section (see §2.1, §3, §4.2).

1. Add the entry to the appropriate `scalars:` list (or `fields:` for settings):

   ```yaml
   - my_field:
       field_num: <N>      # consult §4.2 for the range
       type: double        # or int32 / bool / string / <enum_key>
       optional: true      # almost always, if the C++ default isn't 0
   ```

2. Add the corresponding C++ member to the target object:
   - **Settings**: add `bool my_field{<default>};` (or the appropriate type)
     to the settings struct.
   - **Problem**: add `set_my_field` / `get_my_field` to
     `cpu_optimization_problem_t`.
   - **Solution**: add `my_field` as a constructor parameter to
     `cpu_lp_solution_t` / `cpu_mip_solution_t`, and add it to
     `constructor_args.scalars` in the registry in the correct position.

3. Regenerate, build, run tests.

For solutions, the new scalar appears on both the unary
(`LPSolution`/`MIPSolution`) and chunked (`ChunkedResultHeader`) wire formats
automatically.

### 5.2 Add an array (problem or solution)

1. Add the entry to the appropriate `arrays:` list:

   ```yaml
   - my_array:
       field_num: <N>
       array_id: <M>
       type: repeated double    # or repeated int32 / repeated string / bytes / repeated <enum>
   ```

2. Add the matching C++ getter/setter:
   - **Problem**: `set_my_array` and `get_my_array_host()` (or `get_my_array()`
     for `repeated string`).
   - **Solution**: `get_my_array_host()` (or `get_my_array()` for strings);
     the array is `std::move`d into the constructor automatically — no entry
     in `constructor_args` needed.

3. Regenerate, build, run tests.

The `ArrayFieldId::FIELD_MY_ARRAY` / `ResultFieldId::RESULT_MY_ARRAY` enum value
is auto-generated.

### 5.3 Add a new enum / a value to an existing enum

For a brand new enum:

1. Add to the top-level `enums:` section:

   ```yaml
   my_status:
     domain: solution
     values:
     - OK
     - Failed
   ```

2. Add the matching C++ `enum class my_status_t { OK, Failed };`.
3. Regenerate.

For a new value on an existing enum:

1. Append the value to the `values:` list (preserving order — proto enums must
   be append-only for wire compatibility):

   ```yaml
   my_status:
     domain: solution
     values:
     - OK
     - Failed
     - Cancelled    # newly appended
   ```

2. Add the matching C++ enumerator.
3. Regenerate.

### 5.4 Add a setter group (CSR-style multi-array setter)

Use when the C++ setter takes multiple arrays at once.

1. Tag each member array under `arrays:` with `setter_group:` and the
   appropriate `setter_getter_root:` so the auto-derived getter name matches
   the C++ accessor:

   ```yaml
   arrays:
   - A_values:
       field_num: 9
       array_id: 2
       setter_getter_root: constraint_matrix_values
       setter_group: csr_constraint_matrix
   - A_indices:
       field_num: 10
       array_id: 3
       type: repeated int32
       setter_getter_root: constraint_matrix_indices
       setter_group: csr_constraint_matrix
   - A_offsets:
       field_num: 11
       array_id: 4
       type: repeated int32
       setter_getter_root: constraint_matrix_offsets
       setter_group: csr_constraint_matrix
   ```

2. Add the matching `setter_groups:` entry naming the C++ batch setter and
   listing the member arrays in argument order:

   ```yaml
   setter_groups:
     csr_constraint_matrix:
       setter: set_csr_constraint_matrix
       fields: [A_values, A_indices, A_offsets]
   ```

3. Regenerate.

The generator picks the structural sentinel (`*_offsets` preferred), emits the
batch setter call, and auto-emits a `(*_values, *_indices)` size-mismatch check
if those names exist.

### 5.5 Add a `repeated_messages` entry (nested message type)

Use when the C++ side has a `std::vector<inner_t>` of nested structures (e.g.
quadratic constraints).

1. Add a `repeated_messages:` entry with the required keys (`field_num`,
   `message_type`, `cpp_inner_type`, `presence_check`, `getter`, `setter`) plus
   the per-entry `scalars:` and `arrays:`. See §3.3.

   ```yaml
   repeated_messages:
   - quadratic_constraints:
       field_num: 25
       message_type: QuadraticConstraint
       cpp_inner_type: "typename cpu_optimization_problem_t<i_t, f_t>::quadratic_constraint_t"
       presence_check: has_quadratic_constraints()
       getter: get_quadratic_constraints()
       setter: set_quadratic_constraints
       scalars:
       - constraint_row_index:
           field_num: 1
           type: int32
       - rhs_value:
           field_num: 4
       arrays:
       - linear_values:
           field_num: 5
           array_id: 0
       - linear_indices:
           field_num: 6
           array_id: 1
           type: repeated int32
   ```

2. Add `companion_pairs:` for any `_values`/`_indices` pairs that should have
   matching lengths.
3. Add the C++ inner type (e.g.
   `cpu_optimization_problem_t<i_t, f_t>::quadratic_constraint_t`) and the
   setter / getter / presence-check methods on the problem type.
4. Regenerate.

The chunked transport infrastructure is generic — per-entry arrays travel via
container-keyed `ArrayChunk` messages with no transport changes needed.

### 5.6 Add a warm-start field (LP only)

1. Add the scalar/array under `lp_solution.warm_start.scalars:` or `.arrays:`.

   - For warm-start scalars, allocate `field_num` in the 3000–3999 range.
   - Warm-start arrays use the global solution `array_id` pool.
   - Set `member:` if the C++ struct member name differs from the field name
     (typical: trailing underscore on private-ish members).

   ```yaml
   warm_start:
     presence_check: has_warm_start_data()
     getter: get_cpu_pdlp_warm_start_data()
     scalars:
     - initial_primal_weight:
         field_num: 3000
         member: initial_primal_weight_
     arrays:
     - current_primal_solution:
         field_num: 1          # warm-start arrays are scoped to the warm-start sub-message
         array_id: 3           # shared solution array_id pool
         member: current_primal_solution_
   ```

2. Add the matching member to `cpu_pdlp_warm_start_data_t<i_t, f_t>`.
3. Regenerate.

### 5.7 Make a setting `optional`

When a settings field has a non-zero C++ default, mark it `optional: true` to
prevent proto3 zero from silently overwriting it:

```yaml
- detect_infeasibility:
    field_num: 14
    type: bool
    optional: true
```

The wire encoding is unchanged (no extra bytes), so this is backward-compatible
with existing clients.

### 5.8 Map a sentinel value (`max()` ↔ `-1`)

For the common case (`max()` on an integer or float field ↔ `-1` on the wire),
use the named convention:

```yaml
- iteration_limit:
    field_num: 10
    type: int64
    sentinel: max_as_negative_1
    optional: true   # compose with sentinel when the C++ default is the sentinel
```

The C++ type (`i_t` for ints, `f_t` for floats) is auto-derived from `type:`.

For sentinel patterns that don't fit a named convention, use the verbose form
documented in §2.5.

`sentinel:` composes with `optional:` — both presence mechanisms can be set on
the same field. `optional:` handles the "omitted on the wire" case (preserve
the C++ default); `sentinel:` handles the "explicit reserved value on the wire"
case (e.g. `-1` decodes to `max()`).

---

## 6. V2 cutover checklist

The following changes are deferred until immediately after the next release,
then bundled into a single `V2` cutover so external clients have one migration
to do rather than several. Release notes will spell out the symbol changes.

Items committed to land together:

1. **Chunked header unification** — retire `ChunkedProblemHeader` /
   `ChunkedResultHeader`; chunked path reuses unary message shapes.
2. **Drop `ResultFieldId` aliases** — remove the 8 backward-compat
   `RESULT_WS_*` abbreviated names.
3. **Promote weakly-typed enum settings to proper proto enums** — `presolver_t`
   and `pdlp_precision_t` move from raw `int32` + `from_proto_cast` to
   first-class proto enums registered in `enums:`.
4. **Eliminate `constructor_args` via solution setters** — give solution
   classes setters so `__to_proto` and `__from_proto` use the symmetric
   getter/setter codegen path the rest of the registry uses.

Additional candidates (potential fixes to bundle with V2 if approved):

5. **Drop trailing underscores from warm-start C++ members** — eliminates 17
   `member:` overrides.
6. **Standardize solution getter naming and enable auto-derivation** —
   eliminates 14 of 16 `getter:` overrides; depends on item #4.
7. **Add cached `error_message` member to solution classes** — eliminates the
   remaining 2 `getter:` overrides plus both `proto_only:` flags.
8. **Auto-derive `presence_check:` from section name** — eliminates both
   `presence_check:` overrides.

The remainder of §6 describes each item in turn.

### Chunked header unification (both directions)

The chunked path today carries its own header messages on both directions:

- **Upload**: a hand-written `ChunkedProblemHeader` (in
  `cuopt_remote_service.proto`) that mirrors a subset of `OptimizationProblem`'s
  fields with divergent tag numbers. This is why `optional:` doesn't fully
  propagate to the chunked problem path today (§2.4).
- **Download**: a generated `ChunkedResultHeader` that re-declares every
  `LPSolution` / `MIPSolution` / warm-start scalar (with a `ws_` prefix on the
  warm-start side to avoid name collisions) plus a `repeated
  ResultArrayDescriptor` listing the arrays that are about to stream.

The plan after the current release is to retire both headers and run the
chunked path on the same proto shapes as the unary RPCs:

- The upload path will reuse `SubmitJobRequest` (carrying `OptimizationProblem`
  with numeric arrays left empty), with array bulk continuing to arrive via
  `SendArrayChunk`.
- The download path will return `LPSolution` / `MIPSolution` with the
  `repeated` array fields left empty, with array bulk continuing to arrive via
  `SendResultChunk`. The message type itself becomes the LP/MIP discriminator;
  warm-start scalars fold into the existing `PDLPWarmStartData` embed; the
  `ResultArrayDescriptor` list moves onto the unary solution message as a
  field that's empty in the unary case.

New `V2` RPC variants will carry the unified envelope; the legacy chunked RPCs
will remain (deprecated) for a transition period. Net effect: both the chunked
upload and chunked download become specialized uses of the unary RPC shapes —
"same message, arrays empty, follow with chunk stream" — and the
`_gen_chunked_header` / `_gen_chunked_to_solution` codegen plus the
1000/2000/3000 solution-scalar tag ranges go away.

No registry changes are anticipated as part of that unification.

### Drop `ResultFieldId` backward-compatibility aliases

`enums.result_field_id.aliases` currently declares 8 abbreviated `RESULT_WS_*`
names (`RESULT_WS_CURRENT_PRIMAL`, `RESULT_WS_CURRENT_DUAL`, …) that
piggyback on the canonical numeric IDs via `option allow_alias = true`. They
exist solely to keep symbol-level compatibility with the abbreviated names
shipped before warm-start fields were renamed to their canonical full forms
(`current_primal_solution`, etc.). Wire compatibility is already preserved
without them — the numeric IDs are identical.

These aliases will be dropped at the V2 cutover. The `aliases:` block and the
`allow_alias` machinery in the generator that supports it both go away. Net
effect: less registry surface for developers to reason about, one fewer
"why is this here?" question on first read of the YAML.

### Promote weakly-typed enum settings to proper proto enums

A handful of enum-valued settings round-trip today as raw `int32` on the wire
with a `from_proto_cast:` naming the C++ enum (§3.1):

- `pdlp_settings.presolver` (`from_proto_cast: presolver_t`)
- `pdlp_settings.pdlp_precision` (`from_proto_cast: pdlp_precision_t`)
- `mip_settings.presolver` (`from_proto_cast: presolver_t`)

This works but is "pattern B" — the wire is structurally a number rather than
the enum, and clients see a bare `int32` with no symbolic value names. The
matching `to_proto_cast` is auto-derived from `from_proto_cast` (§2.1), so the
generated code is fine, but the wire schema is weaker than it needs to be.

At the V2 cutover these will move to "pattern A": each enum gets registered in
the top-level `enums:` block (with the canonical proto3 zero-value handling
described in §2.3) and the field becomes `type: presolver_t` / `type:
pdlp_precision_t`, dropping the `from_proto_cast:` line entirely. Wire-level
the field type changes from `int32` to the proto enum — a breaking change for
clients, hence the V2 batching.

Net effect: every enum setting in the registry follows the same pattern; the
generator's two-enum-pattern split (and its scan for `from_proto_cast` as the
fallback signal) gets simpler; clients see symbolic enum values on the wire.

### Eliminate `constructor_args` via solution setters

The solution sections (`lp_solution`, `mip_solution`) use `constructor_args:`
because the C++ solution classes are built via constructor and don't expose
setters for scalars (§3.2). This forces the generator to use a one-off
`__from_proto` code path: collect every field into locals, then call the
constructor; the rest of the registry uses the symmetric
"getter/setter per field" path with assignment-style emission.

At the V2 cutover the C++ solution classes will gain setters for every scalar.
Once that lands, `constructor_args:` can be removed from the registry and the
generator can use its standard per-field emission for solutions too.

Net effect: one fewer special case in the codegen; the asymmetry between
`scalars:` (always passed) and `constructor_args:` (only the listed ones) goes
away; solutions look like every other section in the YAML.

### Optional V2 add-ons (items 5–8)

These items are smaller in scope than the committed four and are mostly C++
naming/structural cleanups rather than wire-level changes. They pair well
with the V2 batching (single migration story, single set of release notes)
but could also ship independently. They're documented here as candidates so
they can be decided on alongside the V2 plan.

#### 5. Drop trailing underscores from warm-start C++ members

Every scalar and array under `lp_solution.warm_start` currently carries a
`member:` override because the corresponding C++ struct members in
`cpu_pdlp_warm_start_data_t<i_t, f_t>` have trailing underscores
(`initial_primal_weight_`, `current_primal_solution_`, …). Renaming those
members to drop the trailing `_` lets all 17 `member:` overrides go away —
the field name and C++ member name match exactly, so the auto-derived default
(§2.7) applies.

Pure C++ cosmetic change with zero wire impact. Largest single registry-surface
reduction available.

#### 6. Standardize solution getter naming and enable auto-derivation

Solution sections (`lp_solution`, `mip_solution`) currently require an
explicit `getter:` on every scalar (and most arrays) because the generator
doesn't auto-derive getters for solutions. With item #4 (solution setters)
landing, the symmetric setter side becomes auto-derivable too, and the
generator can enable `get_<field>()` auto-derivation for solutions.

Two follow-on edits make all but two of the 16 `getter:` overrides go away:

- C++: rename `cpu_lp_solution_t::solved_by()` → `get_solved_by()` to match
  the auto-derived form.
- Registry: promote `mip_solution.arrays.mip_solution`'s `getter:
  get_solution_host()` to `setter_getter_root: solution`.

The two remaining `getter:` overrides are the `get_error_status().what()`
expressions on `error_message` / `mip_error_message`; item #7 handles those.

#### 7. Add cached `error_message` member to solution classes

`error_message` on each solution is `proto_only: true` and reads via
`getter: "get_error_status().what()"` — it's a wire-only projection of an
internal status object, not a real C++ member. Adding a `std::string
error_message` member to `cpu_lp_solution_t` and `cpu_mip_solution_t`
(populated once at construction time from `get_error_status().what()`) lets
the registry drop the two `getter:` overrides *and* both `proto_only:` flags —
four registry sites collapse for one C++ structural change.

#### 8. Auto-derive `presence_check:` from section name

The two `presence_check:` overrides (`has_quadratic_constraints()` on
`optimization_problem.repeated_messages` and `has_warm_start_data()` on
`lp_solution.warm_start`) both follow a `has_<X>()` convention. Teaching the
generator to default to `has_<section_name>()` when `presence_check:` is
omitted gives `has_quadratic_constraints()` for free. The warm-start one
needs one C++ rename (`has_warm_start_data()` → `has_warm_start()`) so its
auto-derived form matches the section name.

Both overrides go away.

---

## 7. Post-cutover investigations

These items are not committed to the V2 cutover. They depend on V2's
groundwork (the chunked-header unification, the enum promotions, the
`constructor_args` removal) and don't have a strong wire-compatibility case
for batching with V2 itself, so they're worth investigating *after* V2 lands
rather than blocking on them now.

### Cross-domain enum support (promote `method_t` to Pattern A)

After V2 item #3 lands (`presolver_t` and `pdlp_precision_t` move from raw
`int32` + `from_proto_cast:` to proto enums registered in `enums:`), one
Pattern B holdout remains: `lp_solution.solved_by` (`from_proto_cast:
method_t`). The corresponding `lp_method` enum is already registered in
`enums:`, but with `domain: settings`, so its proto-enum converter functions
are emitted into the settings-domain generated file. The solution-domain code
that reads `solved_by` doesn't include that file today, so the field has to
fall back to the raw-`int32` + manual-cast pattern.

A clean fix is a generator-architecture change, not a wire change:

- **Option a**: emit cross-domain enum converters into a shared header that
  every generated file can include.
- **Option b**: let an enum field reference an enum from another domain
  explicitly and emit a forwarding converter in the consuming domain.

Either way, external clients see no wire change (the field is already an int
on the wire, and promoting it to a proto enum is a no-op at the byte level for
varint-encoded values). That's why this doesn't need to batch with V2.

Net effect: kills the last `from_proto_cast` override in the registry; closes
the Pattern A vs Pattern B split entirely so every enum-valued field uses the
same mechanism, regardless of which domain it lives in.
