# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cuopt.utilities.exception_handler import (
    InputRuntimeError,
    InputValidationError,
    OutOfMemoryError,
    catch_cuopt_exception,
)

# Lazy imports for CUDA-dependent modules to support CPU-only hosts
# These will be imported when first accessed


def __getattr__(name):
    """Lazy import CUDA-dependent utilities."""
    if name == "type_cast":
        from cuopt.utilities.type_casting import type_cast
        return type_cast
    elif name == "series_from_buf":
        from cuopt.utilities.utils import series_from_buf
        return series_from_buf
    elif name == "check_solution":
        from cuopt.utilities.utils import check_solution
        return check_solution
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
