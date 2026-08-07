# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import ctypes
import os

# Loading with RTLD_LOCAL adds the library itself to the loader's
# loaded library cache without loading any symbols into the global
# namespace. This allows libraries that express a dependency on
# this library to be loaded later and successfully satisfy this dependency
# without polluting the global symbol table with symbols from
# libcudf that could conflict with symbols from other DSOs.
PREFERRED_LOAD_FLAG = ctypes.RTLD_LOCAL


def _load_system_installation(soname: str):
    """Try to dlopen() the library indicated by ``soname``
    Raises ``OSError`` if library cannot be loaded.
    """
    return ctypes.CDLL(soname, PREFERRED_LOAD_FLAG)


def _load_wheel_installation(soname: str):
    """Try to dlopen() the library indicated by ``soname``

    Returns ``None`` if the library cannot be loaded.
    """
    if os.path.isfile(
        lib := os.path.join(os.path.dirname(__file__), "lib64", soname)
    ):
        return ctypes.CDLL(lib, PREFERRED_LOAD_FLAG)
    return None


def load_library():
    """Dynamically load libcuopt.so and its dependencies"""
    try:
        # librmm and libraft must be loaded before libcuopt
        # because libcuopt references them.
        import libraft
        import librmm
        import rapids_logger

        rapids_logger.load_library()
        librmm.load_library()
        libraft.load_library()
    except ModuleNotFoundError:
        pass

    prefer_system_installation = (
        os.getenv("RAPIDS_LIBCUOPT_PREFER_SYSTEM_LIBRARY", "false").lower()
        != "false"
    )

    # cuOpt ships as component libraries. libcuopt.so is a linker script,
    # not an ELF object, so it cannot be dlopen()ed -- load the components
    # instead. Each pulls its own dependencies in through DT_NEEDED, so this
    # order only needs to be valid, not exhaustive. routing and grpc are
    # optional (SKIP_ROUTING_BUILD, SKIP_GRPC_BUILD) and may be absent.
    components = [
        ("libcuopt_base.so", True),
        ("libcuopt_routing.so", False),
        ("libcuopt_mathematical_optimization.so", True),
        ("libcuopt_grpc.so", False),
    ]
    loaded = []
    for soname, required in components:
        lib = _load_component(soname, prefer_system_installation, required)
        if lib is not None:
            loaded.append(lib)
    return loaded


def _load_component(
    soname: str, prefer_system_installation: bool, required: bool
):
    """Load one cuOpt component.

    Returns the handle, or ``None`` if it could not be loaded. Failing to
    load an optional component is silent; failing a required one warns.
    """
    libcuopt_lib = None
    if prefer_system_installation:
        # Prefer a system library if one is present to
        # avoid clobbering symbols that other packages might expect, but if no
        # other library is present use the one in the wheel.
        try:
            libcuopt_lib = _load_system_installation(soname)
        except OSError:
            libcuopt_lib = _load_wheel_installation(soname)
    else:
        # Prefer the libraries bundled in this package. If they aren't found
        # (which might be the case in builds where the library
        # was prebuilt before packaging the wheel), look for a
        # system installation.
        try:
            libcuopt_lib = _load_wheel_installation(soname)
            if libcuopt_lib is None:
                libcuopt_lib = _load_system_installation(soname)
        except OSError as e:
            # If none of the searches above succeed, just silently return None
            # and rely on other mechanisms (like RPATHs on other DSOs) to
            # help the loader find the library.
            if not required:
                return None

            import warnings

            warnings.warn(
                f"Failed to load libcuopt library: {soname}. "
                f"Error: {str(e)}. "
                "Falling back to relying on system loader. "
                "cuOpt functionality may be unavailable. "
                f"This might lead to a generic error such as "
                f"'{soname} missing' if the library cannot be found.",
                RuntimeWarning,
            )
    # The caller almost never needs to do anything with this library, but no
    # harm in offering the option since this object at least provides a handle
    # to inspect where libcuopt was loaded from.
    return libcuopt_lib
