# libcuopt-client

Host-side cuOpt client: problem representation, parsers and the gRPC client. Contains no CUDA kernels.

This package ships one component of cuOpt's C++ library. Install [`libcuopt`](https://pypi.org/project/libcuopt/) instead to get all of them.

## Building against this package with CMake

`find_package(cuopt)` provides `cuopt::client`. The wheels are split across separate
site-packages directories, so pass every wheel root and every directory holding a `*-config.cmake`:

```bash
SP=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
ROOTS=$(find "$SP" -mindepth 3 -maxdepth 3 -path '*/lib64/cmake' -printf '%h\n' | xargs -n1 dirname | sort -u | tr '\n' ';')
CFGS=$(find "$SP" \( -name '*-config.cmake' -o -name '*Config.cmake' \) -printf '%h\n' | sort -u | tr '\n' ';')
cmake -S . -B build -DCMAKE_PREFIX_PATH="$ROOTS$CFGS"
```

The client needs no CUDA Toolkit, `rmm` or `raft`, except that `cuopt/error.hpp` includes
`raft/core/error.hpp`. The problem parsers and solver headers ship with `libcuopt-mathopt`.
