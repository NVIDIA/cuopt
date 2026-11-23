# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


def load_library():
    """Dynamically load libcuopt.so and its dependencies.

    libcuopt is now a thin metapackage: it carries no engine library itself
    (libcuopt.so is a linker script, not an ELF object, so it cannot be
    dlopen()ed anyway). The actual libraries ship in the libcuopt-client,
    libcuopt-mathopt and libcuopt-routing wheels, so loading delegates to
    their own load_library(), which already handles the client-before-engine
    DT_NEEDED ordering and the system/wheel installation fallback.
    """
    loaded = []

    # mathopt is required; routing is optional (SKIP_ROUTING_BUILD) and may
    # not be installed at all.
    import libcuopt_mathopt

    loaded.extend(libcuopt_mathopt.load_library())

    try:
        import libcuopt_routing
    except ModuleNotFoundError:
        pass
    else:
        loaded.extend(libcuopt_routing.load_library())

    return loaded
