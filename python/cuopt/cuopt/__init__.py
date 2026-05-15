# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

try:
    from cuopt._build_variant import CUOPT_PYTHON_BUILD_COMPONENT as _CUOPT_SLICE
except ModuleNotFoundError:
    _CUOPT_SLICE = "FULL"

# Native loader: try split wheels first, then full libcuopt
if _CUOPT_SLICE == "LP":
    for _native in ("libcuopt_lp", "libcuopt"):
        try:
            _m = __import__(_native, fromlist=["load_library"])
        except ModuleNotFoundError:
            continue
        _m.load_library()
        del _m
        break
    del _native
elif _CUOPT_SLICE == "ROUTING":
    for _native in ("libcuopt_routing", "libcuopt"):
        try:
            _m = __import__(_native, fromlist=["load_library"])
        except ModuleNotFoundError:
            continue
        _m.load_library()
        del _m
        break
    del _native
else:
    for _native_pkg in ("libcuopt", "libcuopt_lp", "libcuopt_routing"):
        try:
            _m = __import__(_native_pkg, fromlist=["load_library"])
        except ModuleNotFoundError:
            continue
        _m.load_library()
        del _m
        break
    del _native_pkg

from cuopt._version import __git_commit__, __version__, __version_major_minor__

if _CUOPT_SLICE == "ROUTING":
    _submodules = ["routing", "distance_engine"]
elif _CUOPT_SLICE == "LP":
    _submodules = ["linear_programming"]
else:
    _submodules = ["linear_programming", "routing", "distance_engine"]


def __getattr__(name):
    """Lazy import submodules to support CPU-only hosts with remote solve."""
    if name in _submodules:
        import importlib
        return importlib.import_module(f"cuopt.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(dict.fromkeys(__all__ + _submodules))


__all__ = ["__git_commit__", "__version__", "__version_major_minor__"]
