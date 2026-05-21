# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from cuopt.linear_programming.data_model import DataModel
from cuopt.linear_programming.io import parser_wrapper
from cuopt.linear_programming.io.utilities import (
    catch_io_exception,
)


@catch_io_exception
def ParseProblem(file_path: str, fixed_mps_format: bool = False) -> DataModel:
    """Read an optimization problem from a file, dispatching on extension.

    Dispatches to the MPS/QPS or LP parser based on the filename suffix
    (case-insensitive), matching the C++ ``parse_problem`` entry point:

    - ``.mps``, ``.mps.gz``, ``.mps.bz2``, ``.qps``, ``.qps.gz``, ``.qps.bz2``
      → MPS/QPS reader
    - ``.lp``, ``.lp.gz``, ``.lp.bz2`` → LP reader

    Parameters
    ----------
    file_path : str
        Path to an MPS, QPS, or LP file (optionally ``.gz`` / ``.bz2``
        compressed).
    fixed_mps_format : bool
        If the MPS/QPS reader should parse as fixed MPS format. Ignored for
        LP inputs. False by default.

    Returns
    -------
    data_model : DataModel
        A fully formed LP/MILP/QP problem.

    Raises
    ------
    InputValidationError, InputRuntimeError, OutOfMemoryError
        Parser errors from the underlying C++ readers (via
        :func:`catch_io_exception`).
    RuntimeError
        If the file extension is not one of the supported suffixes (raised by
        the C++ ``parse_problem`` dispatch).
    """
    return parser_wrapper.ParseProblem(file_path, fixed_mps_format)