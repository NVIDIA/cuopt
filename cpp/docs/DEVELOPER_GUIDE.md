# cuOpt C++ Developer Guide

This document serves as a guide for contributors to cuOpt C++ code. Developers should also refer
to these additional files for further documentation of cuOpt best practices.

* [Documentation Guide](TODO) for guidelines on documenting cuOpt code.
* [Testing Guide](TODO) for guidelines on writing unit tests.
* [Benchmarking Guide](TODO) for guidelines on writing unit benchmarks.

# Overview

cuOpt includes a C++ library that provides GPU-accelerated logistics.

## Lexicon

This section defines terminology used within cuOpt


# Directory Structure and File Naming

External/public cuOpt APIs are grouped into an appropriately titled
header file  in `cuopt/cpp/include/cuopt`. For example, `cuopt/cpp/include/cuopt/routing/data_model_view.hpp`
contains the definition of the routing data model object. Note the  `.hpp`
file extension used to indicate a C++ header file.

Header files should use the `#pragma once` include guard.

## File extensions

- `.hpp` : C++ header files
- `.cpp` : C++ source files
- `.cu`  : CUDA C++ source files
- `.cuh` : Headers containing CUDA device code

Header files and source files should use `.hpp` and `.cpp` extensions unless they must
be compiled by nvcc.  `.cu` and `.cuh` files are more expensive to compile, so we want
to minimize the use of these files to only when necessary.  A good indicator of the need
to use a `.cu` or `.cuh` file is the inclusion of `__device__` and other
symbols that are only recognized by `nvcc`. Another indicator is Thrust
algorithm APIs with a device execution policy (always `rmm::exec_policy` in cuOpt).

## Code and Documentation Style and Formatting

cuOpt code uses [snake_case](https://en.wikipedia.org/wiki/Snake_case) for all names except in a
few cases: unit tests and test case names may use Pascal case, aka
[UpperCamelCase](https://en.wikipedia.org/wiki/Camel_case). We do not use
[Hungarian notation](https://en.wikipedia.org/wiki/Hungarian_notation), except for the following examples:
 * device data variables should be prefaced by d_ if it makes the intent clearer
 * host data variables should be prefaced by h_ if it makes the intent clearer
 * template parameters defining a type should be suffixed with _t
 * private member variables are typically suffixed with an underscore

```c++
template <typename i_t>
class locations_t
{
  ...
 private:
  i_t n_locations__{};
  i_t *d_locations_{};
}
```

C++ formatting is enforced using `clang-format`. You should configure `clang-format` on your
machine to use the `cuopt/cpp/.clang-format` configuration file, and run `clang-format` on all
changed code before committing it. The easiest way to do this is to configure your editor to
"format on save".

Aspects of code style not discussed in this document and not automatically enforceable are typically
caught during code review, or not enforced.

### C++ Guidelines

In general, we recommend following
[C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines). We also
recommend watching Sean Parent's [C++ Seasoning talk](https://www.youtube.com/watch?v=W2tWOdzgXHA),
and we try to follow his rules: "No raw loops. No raw pointers. No raw synchronization primitives."

 * Prefer algorithms from STL and Thrust to raw loops.
 * Prefer cuopt and RMM to raw pointers and raw memory allocation.


### Includes

The following guidelines apply to organizing `#include` lines.

 * Group includes by library (e.g. cuOpt, RMM, Thrust, STL). `clang-format` will respect the
   groupings and sort the individual includes within a group lexicographically.
 * Separate groups by a blank line.
 * Order the groups from "nearest" to "farthest". In other words, local includes, then includes
   from other RAPIDS libraries, then includes from related libraries, like `<thrust/...>`, then
   includes from dependencies installed with cuOpt, and then standard headers (for example `<string>`,
   `<iostream>`).
 * Use <> instead of "" unless the header is in the same directory as the source file.
 * Tools like `clangd` often auto-insert includes when they can, but they usually get the grouping
   and brackets wrong.
 * Always check that includes are only necessary for the file in which they are included.
   Try to avoid excessive including especially in header files. Double check this when you remove
   code.

# cuOpt Data Structures

Application data in cuOpt is contained in 3 objects for routing : data model, solver, and assignment.
There are a variety of other data structures you will use when developing cuOpt code.

## Views and Ownership

Resource ownership is an essential concept in cuOpt. In short, an "owning" object owns a
resource (such as device memory). It acquires that resource during construction and releases the
resource in destruction ([RAII](https://en.cppreference.com/w/cpp/language/raii)). A "non-owning"
object does not own resources. Any class in cuOpt with the `*_view` suffix is non-owning.

## `rmm::device_memory_resource`<a name="memory_resource"></a>

cuOpt allocates all device memory via RMM memory resources (MR). See the
[RMM documentation](https://github.com/rapidsai/rmm/blob/main/README.md) for details.

## Streams

CUDA streams are not yet exposed in external cuOpt APIs.

We are currently investigating the best technique for exposing this.

### Memory Management

cuOpt code generally eschews raw pointers and direct memory allocation. Use RMM classes built to
use `device_memory_resource`(*)s for device memory allocation with automated lifetime management.


#### `rmm::device_uvector<T>`

Similar to a `rmm::device_vector`, allocates a contiguous set of elements in device memory but with
key differences:
- As an optimization, elements are uninitialized and no synchronization occurs at construction.
This limits the types `T` to trivially copyable types.
- All operations are stream ordered (i.e., they accept a `cuda_stream_view` specifying the stream
on which the operation is performed).

## Namespaces

### External
All public cuOpt APIs should be placed in the `cuopt` namespace. Example:
```c++
namespace cuopt{
   void public_function(...);
} // namespace cuopt
```

### Internal

Many functions are not meant for public use, so place them in either the `detail` or an *anonymous*
namespace, depending on the situation.

#### `detail` namespace

Functions or objects that will be used across *multiple* translation units (i.e., source files),
should be exposed in an internal header file and placed in the `detail` namespace. Example:

```c++
// some_utilities.hpp
namespace cuopt{
namespace detail{
void reusable_helper_function(...);
} // namespace detail
} // namespace cuopt
```

#### Anonymous namespace

Functions or objects that will only be used in a *single* translation unit should be defined in an
*anonymous* namespace in the source file where it is used. Example:

```c++
// some_file.cpp
namespace{
void isolated_helper_function(...);
} // anonymous namespace
```

[**Anonymous namespaces should *never* be used in a header file.**](https://wiki.sei.cmu.edu/confluence/display/cplusplus/DCL59-CPP.+Do+not+define+an+unnamed+namespace+in+a+header+file)

# Error Handling

cuOpt follows conventions (and provides utilities) enforcing compile-time and run-time
conditions and detecting and handling CUDA errors. Communication of errors is always via C++
exceptions.

## Runtime Conditions

Use the `CUOPT_EXPECTS` macro to enforce runtime conditions necessary for correct execution.

Example usage:
```c++
CUOPT_EXPECTS(lhs.type() == rhs.type(), "Column type mismatch");
```

The first argument is the conditional expression expected to resolve to  `true`  under normal
conditions. If the conditional evaluates to  `false`, then an error has occurred and an instance of  `cuopt::logic_error` is thrown. The second argument to  `CUOPT_EXPECTS` is a short description of the
error that has occurred and is used for the exception's `what()` message.

There are times where a particular code path, if reached, should indicate an error no matter what.
For example, often the `default` case of a `switch` statement represents an invalid alternative.
Use the `CUOPT_FAIL` macro for such errors. This is effectively the same as calling
`CUOPT_EXPECTS(false, reason)`.

Example:
```c++
CUOPT_FAIL("This code path should not be reached.");
```

### CUDA Error Checking

Use the `RAFT_CUDA_TRY` macro to check for the successful completion of CUDA runtime API functions. This
macro throws a `cuopt::cuda_error` exception if the CUDA API return value is not `cudaSuccess`. The
thrown exception includes a description of the CUDA error code in it's  `what()`  message.

Example:

```c++
RAFT_CUDA_TRY( cudaMemcpy(&dst, &src, num_bytes) );
```

## Compile-Time Conditions

Use `static_assert` to enforce compile-time conditions. For example,

```c++
template <typename T>
void trivial_types_only(T t){
   static_assert(std::is_trivial<T>::value, "This function requires a trivial type.");
...
}
```
