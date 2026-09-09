# NVIDIA cuOpt Library Architecture

<!--
  SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

> **Audience:** cuOpt contributors, especially when reviewing or writing a change that moves code
> between translation units, libraries, or packages.
>
> For the runtime design of the gRPC server (processes, shared memory, pipes) see the
> [gRPC server behavior](../cuopt-grpc/grpc-server-architecture.md) page, and
> `cpp/docs/grpc-server-architecture.md` in the repository for the full C++-level detail.
> This document is about structure: what runs where, what depends on what, and which build
> artifact holds which code.

## Why This Document Exists

Most cuOpt changes stay inside one library and this layering never comes up. A recurring class of
change does not: moving a symbol between a `.cu` and a `.cpp`, adding a source to a different
CMake variable, or splitting a class so part of it can build without CUDA. Those changes are easy
to get wrong in ways that compile cleanly and fail later — at link time in a different build
configuration, or at load time in a downstream package.

The layers below are the ones worth checking against when a change crosses them.

## How the Pieces Fit at Run Time

The two entry points that matter for deployment are the HTTP service and the gRPC server. Both run
where the GPU is; both have a client that does not.

```text
  CLIENT MACHINE                 │  GPU HOST
  no GPU, no CUDA runtime        │
                                 │
  ┌───────────────────────────┐  │  ┌──────────────────────┐
  │ cuopt_sh_client           │──┼─►│ cuopt_server         │
  │ python/cuopt_self_hosted  │  │  │ FastAPI + uvicorn    │
  │ deps: requests, msgpack   │  │  └──────────┬───────────┘
  └───────────────────────────┘  │             │ imports
           HTTP  /cuopt/...      │             ▼
                                 │  ┌──────────────────────┐
                                 │  │ cuopt                │
                                 │  │ python/cuopt         │──┐
                                 │  │ Cython modules       │  │
                                 │  │ deps: cudf, cupy,    │  │ links
                                 │  │ pylibraft, rmm       │  │
                                 │  └──────────────────────┘  │
                                 │                            │
  ┌───────────────────────────┐  │  ┌──────────────────────┐  │
  │ any gRPC client           │──┼─►│ cuopt_grpc_server    │  │
  │ CUOPT_REMOTE_HOST / _PORT │  │  │ forks one worker     │  │
  └───────────────────────────┘  │  │ process per solve    │  │
           gRPC                  │  └──────────┬───────────┘  │
                                 │             │ links        │
                                 │             ▼              │
                                 │  ┌──────────────────────┐  │
                                 │  │ libcuopt.so          │◄─┘
                                 │  │ cuopt::cuopt         │
                                 │  │                      │
                                 │  │ also linked by       │
                                 │  │ cuopt_cli            │
                                 │  └──────────┬───────────┘
                                 │             │ CUDA / rmm / raft
                                 │             ▼
                                 │            GPU
```

`cuopt_sh_client` is the existing proof that a CUDA-free client is useful: it depends on nothing
but `requests` and `msgpack`, and reaches the solver over HTTP. The gRPC path has no equivalent
today — its client is `cuopt.grpc.client`, a Cython module inside the `cuopt` package, so using it
pulls the whole CUDA stack. Closing that gap is what `cuopt_client` and #1872 are for.

`cuopt_cli` is a fourth entry point: a local executable linking `libcuopt` directly, no service
involved.

## What Depends on What

Edges, with the mechanism, since "depends" means four different things here:

| Consumer | Provider | Mechanism |
|---|---|---|
| `cuopt_sh_client` | `cuopt_server` | HTTP, across hosts |
| `cuopt.grpc.client` | `cuopt_grpc_server` | gRPC, across hosts |
| `cuopt_server` | `cuopt` | Python import |
| `cuopt` (wheel) | `libcuopt` (wheel), cudf, cupy, pylibraft, rmm | Python dependency |
| `cuopt` extension modules | `cuopt::cuopt` | C++ link |
| `cuopt_cli`, `cuopt_grpc_server` | `cuopt::cuopt` | C++ link |
| `libcuopt` (wheel) | CUDA toolkit, librmm, cudss, nccl | Python dependency |

The distinction matters: a Python-level dependency is what `pip install` resolves, and it applies
regardless of what the extension module links. Relinking a module against a CUDA-free library does
not remove `cudf` from the wheel's dependency list.

## Build Artifacts

```text
┌───────────────────────────────────────────────────────────────────────────┐
│ Packages                                                                  │
│                                                                           │
│   conda:  libcuopt        libcuopt-tests                                  │
│   wheels: libcuopt        cuopt        cuopt_server      cuopt_sh_client  │
└───────────────────────────────────────────────────────────────────────────┘
                                    ▲
                                    │ ships
┌───────────────────────────────────────────────────────────────────────────┐
│ C++ artifacts                                                             │
│                                                                           │
│   cuopt_objs  ─────────►  libcuopt.so        (cuopt::cuopt)               │
│   (OBJECT lib)            hidden visibility; only CUOPT_EXPORT is public  │
│        │                                                                  │
│        └────────────────►  cuopt_static      internal test binaries       │
│                                                                           │
│   cuopt_cli              executables, RPATH $ORIGIN/../lib                │
│   cuopt_grpc_server                                                       │
└───────────────────────────────────────────────────────────────────────────┘
                                    ▲
                                    │ compiles
┌───────────────────────────────────────────────────────────────────────────┐
│ Sources, grouped by the CMake variable that collects them                 │
│                                                                           │
│   CUOPT_SRC_FILES          everything in libcuopt                         │
│   MPS_FAST_SRC_FILES       fast MPS parser                                │
│                                                                           │
│   cpp/src/{pdlp,mip_heuristics,branch_and_bound,barrier,dual_simplex,     │
│            cuts,linear_algebra,routing,distance,io,grpc,                  │
│            math_optimization,utilities}                                   │
└───────────────────────────────────────────────────────────────────────────┘
```

## Things That Are Easy to Get Wrong

### Explicit Instantiation Is the Public Surface

Template members defined out-of-line in a `.cpp`/`.cu` exist only if that translation unit
explicitly instantiates them. Removing or mis-guarding an instantiation does not fail the build of
that file — it fails whenever something else references the symbol, which may be a different build
configuration or a downstream consumer.

The guards are not interchangeable:

| Macro | Value | Meaning |
|---|---|---|
| `MIP_INSTANTIATE_FLOAT` | `CUOPT_INSTANTIATE_FLOAT`, currently `0` | float MIP/settings members |
| `MIP_INSTANTIATE_DOUBLE` | `CUOPT_INSTANTIATE_DOUBLE`, currently `1` | double members |
| `PDLP_INSTANTIATE_FLOAT` | hardcoded `1` | float PDLP members |

A function guarded on `PDLP_INSTANTIATE_FLOAT` that calls a member guarded on
`MIP_INSTANTIATE_FLOAT` will not link, because the first is always on and the second is always off.
Guard a symbol the same way as the symbols it calls.

A translation unit whose instantiations are all guarded off compiles to zero symbols, silently.
`nm --defined-only` on the object file is the quick check.

### Comparing Exported Symbols Against Main

The reliable way to catch an accidental ABI change is to diff the exported set:

```bash
nm -D --defined-only libcuopt.so | c++filt | sort -u > /tmp/branch.txt
# same for a build of main, then:
comm -23 /tmp/main.txt /tmp/branch.txt   # symbols this branch dropped
```

`ci/check_symbols.sh` is a related but different check: it asserts that internal namespaces
(`cuopt::*::detail`, `rmm::`, `thrust::`, …) are *not* exported. It does not detect a symbol that
went missing.

### `ldd -r` Does Not Prove There Is No Cycle

`ldd -r` resolves symbols across everything already loaded, so a mutual dependency between two
cuOpt libraries looks fine. A cycle surfaces only when the call happens, as an
`undefined symbol` at run time. Check the intended direction of the dependency, not just the
resolved set.

### Visibility Differs by Target on Purpose

`cuopt_objs` is built with hidden visibility, so `libcuopt.so` exports only what is marked
`CUOPT_EXPORT` — a curated API. A library carved out of the *internals* rather than designed as an
export surface cannot use that setting, because the code left behind depends on hundreds of its
symbols. Where the two settings differ, the difference is deliberate and commented at the target.

## Direction

Three in-flight changes reshape the C++ and package layers. They are listed here so the diagram
above does not read as settled.

- **#1804** adds `cuopt_client` / `libcuopt_client.so`: a CPU-only library holding the host-side
  problem representation, the gRPC wire protocol, and the LP/MIP gRPC client, with no CUDA in
  `NEEDED`. `libcuopt` links it `PUBLIC`. The routing gRPC arm stays in `libcuopt`, because its
  mappers call routing accessors that live in CUDA translation units.
- **#1622** splits `libcuopt` into component libraries along solver lines.
- **#1872** covers the packaging half of #1804. `libcuopt_client.so` currently ships inside the
  `libcuopt` package, which depends on CUDA, and every Python extension module links
  `cuopt::cuopt`, so a GPU-free client install is not yet possible.

## Where to Look

| Question | File |
|---|---|
| Which sources go in which library | `cpp/src/*/CMakeLists.txt` |
| Target definitions, visibility, RPATH, install/export | `cpp/CMakeLists.txt` |
| Instantiation guards | `cpp/src/mip_heuristics/mip_constants.hpp` |
| Conda outputs and run dependencies | `conda/recipes/libcuopt/recipe.yaml` |
| Wheel contents and vendoring exclusions | `ci/build_wheel_*.sh` |
| Python-level dependencies | `dependencies.yaml`, `python/*/pyproject.toml` |
| gRPC server runtime design | `cpp/docs/grpc-server-architecture.md` |
| C++ conventions and code layout | `cpp/docs/DEVELOPER_GUIDE.md` |
