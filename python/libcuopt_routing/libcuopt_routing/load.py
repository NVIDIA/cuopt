# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Loader for the cuOpt VRP / TSP / PDP routing component.

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


def load_library() -> list:
    """Dynamically load libcuopt_routing.so and its dependencies. It loads libcuopt_client first, which
    this library has a DT_NEEDED on.

    Returns a single-element list holding the ``ctypes.CDLL`` handle, or an
    empty list when the library could not be loaded; callers rarely need the
    handle itself. A missing library warns rather than raising, so that the
    system loader still gets a chance to resolve it.
    """
    try:
        # librmm and rapids_logger must be loaded before libcuopt_routing.so,
        # which references them.
        import librmm
        import rapids_logger

        rapids_logger.load_library()
        librmm.load_library()
    except ModuleNotFoundError:
        pass

    # This component has a DT_NEEDED on the client, so the client package must
    # be loaded first now that the two live in separate wheels.
    import libcuopt_client

    libcuopt_client.load_library()

    prefer_system_installation = (
        os.getenv("RAPIDS_LIBCUOPT_PREFER_SYSTEM_LIBRARY", "false").lower()
        != "false"
    )

    lib = _load_component(
        "libcuopt_routing.so", prefer_system_installation, True
    )
    return [lib] if lib is not None else []


def _load_component(
    soname: str, prefer_system_installation: bool, required: bool
):
    """Load one cuOpt component.

    Returns the handle, or ``None`` if it could not be loaded. Failing to
    load an optional component is silent; failing a required one warns.
    """
    libcuopt_lib = None
    try:
        if prefer_system_installation:
            # Prefer a system library if one is present to
            # avoid clobbering symbols that other packages might expect, but if no
            # other library is present use the one in the wheel.
            try:
                libcuopt_lib = _load_system_installation(soname)
            except OSError:
                # The fallback needs to sit inside the outer handler too. An
                # optional component that is present but fails to load would
                # otherwise raise out of here and stop the remaining components
                # from being loaded at all.
                libcuopt_lib = _load_wheel_installation(soname)
        else:
            # Prefer the libraries bundled in this package. If they aren't found
            # (which might be the case in builds where the library
            # was prebuilt before packaging the wheel), look for a
            # system installation.
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
