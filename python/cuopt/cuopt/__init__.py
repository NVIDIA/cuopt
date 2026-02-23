# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

try:
    import libcuopt
except ModuleNotFoundError:
    pass
else:
    libcuopt.load_library()
    del libcuopt

from cuopt._version import __git_commit__, __version__, __version_major_minor__

# Lazy imports for linear_programming, routing, and distance_engine modules
# This allows cuopt to be imported on CPU-only hosts when remote solve is configured
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
