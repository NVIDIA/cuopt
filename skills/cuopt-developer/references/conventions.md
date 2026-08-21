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

- **Never write large lambdas inside a function body.** If a lambda is more than a short
  predicate/comparator (roughly more than ~3–5 lines, or it has nested lambdas, local state,
  or non-trivial control flow), extract it as a named free function, file-local helper in an
  anonymous namespace, or private method. Large local lambdas obscure control flow, hide reuse,
  and make work/time accounting harder to reason about.
- Prefer direct loops or named helpers for performance-critical traversal logic. Reserve
  in-function lambdas for short predicates and callbacks only.
- Keep work-estimate and time-limit checks at phase or outer-loop boundaries. Accumulate the work
  performed by cheap inner loops and charge it once when the phase completes; do not gate every
  inner iteration. Do not separately test a sticky limit or raw estimate immediately before or
  after `add_work_estimate` when that call already provides the gate for the same work.
- Charge the operation that is actually performed. For vector copies and sparse traversals, base
  work on the number of visited or copied entries rather than only the number of containers.
  Avoid charging the same traversal in both its caller and callee.

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

### Integer widths — prefer fixed-width types

Prefer `<cstdint>` fixed-width types (`int32_t`, `int64_t`, `uint32_t`, …) over
plain `int` / `long` / `long long` when the value range or ABI width matters
(counts that can exceed 32 bits, device grid math, work estimates, file offsets).

Avoid multi-word functional casts such as `long long(x)` in `.cu`/`.cuh` — they
confuse CUDA-aware tooling (`type name is not allowed`). Use a C-style cast to a
fixed-width type instead: `(int64_t)x`.

```cpp
const long long total = long long(num) * long long(patterns);  // ❌
const int64_t total = (int64_t)num * (int64_t)patterns;          // ✅
```

Keep `i_t` / `f_t` for problem-index and numeric template parameters; use
`int64_t` (etc.) for host-side wide counters outside that abstraction.

### CUDA Error Checking

```cpp
RAFT_CUDA_TRY(cudaMemcpy(...));
```

### Prefer modern CCCL utilities in device code

When writing or editing CUDA kernels, prefer CCCL / libcu++ helpers over hand-rolled
bit math, reductions, or integer tricks. They encode the PTX-friendly form and avoid
boilerplate that compilers often fail to recover from runtime values.

Examples (CUDA 13 / CCCL 3.x era — headers already used elsewhere in `cpp/src`):

| Need | Prefer | Instead of |
|------|--------|------------|
| Extract a bitfield / decode packed indices | `cuda::bitfield_extract` (`<cuda/bit>`) | `%` / `/` by a runtime `1 << k` (nvcc usually will not strength-reduce those to mask/shift) |
| Build a contiguous bit mask | `cuda::bitmask` | Hand-written `((1u << w) - 1u) << start` |
| Test / round to power of two | `cuda::is_power_of_two`, `next_power_of_two`, `prev_power_of_two` (`<cuda/cmath>`), or `cuda::std::has_single_bit` / `bit_ceil` / `bit_floor` (`<cuda/std/bit>`) | Ad-hoc `(x & (x - 1)) == 0` / manual ceil loops |
| Divide/mod by a value that is constant for a launch (or across many ops) but not a compile-time constant | `cuda::fast_mod_div` (`<cuda/cmath>`) — construct on the host (or once), pass into the kernel, use `/` `%` / `cuda::div` | Hot-path `idiv` / handwritten libdivide magic |
| Warp/block algorithms | CUB / CCCL / RAFT primitives already used in-tree | Homegrown shared-memory reductions when an existing primitive fits |

Docs: [CCCL bit extensions](https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/bit.html),
[pow2 helpers](https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/math/pow2.html),
[`cuda::fast_mod_div`](https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/math/fast_mod_div.html).
Check signatures in the installed headers rather than guessing — APIs evolve with CCCL.

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

When a new MIP test loads a MIPLIB instance (e.g. via `make_path_absolute("mip/<name>.mps")`),
that instance must appear in `datasets/mip/download_miplib_test_dataset.sh`'s `INSTANCES`
list. CI and local setups only fetch that allowlist — an unlisted name fails at parse time
with a missing-file error even though the test itself is correct. Add the basename there as part of the same change that introduces the test.

Calling cuOpt MIP internals that use OpenMP taskloops (notably `diversity_manager::run_presolve`
→ `compute_probing_cache`) from a plain gtest must open an OMP team first, the same way
`solve_mip` does (`#pragma omp parallel num_threads(...)` + `#pragma omp masked`, with
`omp_set_max_active_levels(2)` if needed). Probing sizes its pool as
`omp_get_num_threads() - 1`; outside a parallel region that is 0 and probing becomes a silent
no-op (Papilo size unchanged, test finishes in a few hundred ms).
