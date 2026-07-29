# Coding Conventions, Error Handling, and Memory Management

Read this for cuOpt code style: naming, file extensions, include order, error handling, memory management, and test impact.

## Comments

**Never embed volatile details in code comments.** Anything that changes independently of the code will silently become wrong and mislead future readers. Volatile details include:

- Line numbers (`line 869`, `see line 42`)
- Commit hashes or PR numbers (`fixed in abc1234`, `from PR #1234`)
- Timestamps or version strings
- External URLs that may rot

Use stable identifiers instead — member names, function names, type names, section headings, or file paths. These refactor together with the code; volatile references do not.

```cpp
// ✅ GOOD — stable name
window_count)),  // NOSONAR(S836): window_count is declared before window_state_ in this struct

// ❌ BAD — line number goes stale on the next nearby edit
window_count)),  // NOSONAR: window_count declared before window_state_ (line 869 vs 891)
```

This applies to all comment types: inline comments, block comments, suppression directives (`// NOSONAR`, `// NOLINT`, `# noqa`, `# type: ignore`), and doc comments.

## C++ Naming

| Element | Convention | Example |
|---------|------------|---------|
| Variables | `snake_case` | `num_locations` |
| Functions | `snake_case` | `solve_problem()` |
| Classes | `snake_case` | `data_model` |
| Test cases | `PascalCase` | `SolverTest` |
| Device data | `d_` prefix | `d_locations_` |
| Host data | `h_` prefix | `h_data_` |
| Template params | `_t` suffix | `value_t` |
| Private members | `_` suffix | `n_locations_` |

## File Extensions

| Extension | Usage |
|-----------|-------|
| `.hpp` | C++ headers |
| `.cpp` | C++ source |
| `.cu` | CUDA source (nvcc required) |
| `.cuh` | CUDA headers with device code |

## Include Order

1. Local headers
2. RAPIDS headers
3. Related libraries
4. Dependencies
5. STL

## C++ Implementation Style

- Prefer direct loops or named helpers for performance-critical traversal logic. Reserve lambdas for
  short predicates and callbacks; large local lambdas obscure control flow and can lead to repeated
  scans.

### Suppression comments (`// NOSONAR`, `// NOLINT`, etc.)

When suppressing a static-analysis or linter warning at the call site, include the rule ID and explain *why* the flagged code is safe — not just silence the tool:

```cpp
// ✅ GOOD — rule ID + reason in terms of stable names
window_count)),  // NOSONAR(S836): window_count is declared before window_state_ in this struct

// ❌ BAD — no rule ID, reason uses a volatile line number
window_count)),  // NOSONAR: window_count declared before window_state_ (line 869 vs 891)
```

## Python Style

- Follow PEP 8
- Use type hints
- Tests use pytest

## Error Handling

### Runtime Assertions

```cpp
CUOPT_EXPECTS(condition, "Error message");
CUOPT_FAIL("Unreachable code reached");
```

### Assert-only variables

A variable used only inside `cuopt_assert` (or any assertion that compiles out in
release builds) triggers an unused-variable warning when asserts are disabled.
Mark it `[[maybe_unused]]` at the declaration — do **not** suppress the warning
with `static_cast<void>(var);` (or `(void)var;`) statements after the asserts.

```cpp
// ❌ WRONG — trailing void-casts to silence the warning
const f_t lower_bound = lower_bounds[var_idx];
const f_t upper_bound = upper_bounds[var_idx];
cuopt_assert(lower_bound >= -bound_tol, "...");
cuopt_assert(upper_bound <= 1 + bound_tol, "...");
static_cast<void>(lower_bound);
static_cast<void>(upper_bound);

// ✅ CORRECT — annotate at the declaration
[[maybe_unused]] const f_t lower_bound = lower_bounds[var_idx];
[[maybe_unused]] const f_t upper_bound = upper_bounds[var_idx];
cuopt_assert(lower_bound >= -bound_tol, "...");
cuopt_assert(upper_bound <= 1 + bound_tol, "...");
```

### Container indexing — no gratuitous `static_cast<size_t>`

Index with the bare signed type (`i_t`, `int`, loop counters); don't wrap
subscripts in `static_cast<size_t>(...)`. The build uses `-Werror` but not
`-Wsign-conversion`/`-Wconversion`, and there's no `.clang-tidy`, so `v[i]` emits
no warning — the cast is pure noise and inconsistent with the rest of `cpp/src`.

```cpp
perm[static_cast<size_t>(cursor[r])] = static_cast<i_t>(k);  // ❌ noise
perm[cursor[r]] = k;                                         // ✅
```

Cast only when it changes the value or guards real overflow — e.g. sizing from a
signed subtraction (`std::vector<i_t> v(static_cast<size_t>(hi - lo) + 2, 0)`), or
the narrowing `size_t`→`i_t` in `static_cast<i_t>(x.size())` (established style;
keep it)

### CUDA Error Checking

```cpp
RAFT_CUDA_TRY(cudaMemcpy(...));
```

## Memory Management

```cpp
// ❌ WRONG
int* data = new int[100];

// ✅ CORRECT - use RMM
rmm::device_uvector<int> data(100, stream);
```

- All operations should accept `cuda_stream_view`
- Views (`*_view` suffix) are non-owning

Read existing code in `cpp/src/` for real examples of RMM allocation, stream-ordering, RAFT utilities, and kernel launch patterns.

## Test Impact Check

**Before any behavioral change, ask:**

1. What scenarios must be covered?
2. What's the expected behavior contract?
3. Where should tests live?
   - C++ gtests: `cpp/tests/`
   - Python pytest: `python/.../tests/`

**Add at least one regression test for new behavior.**
