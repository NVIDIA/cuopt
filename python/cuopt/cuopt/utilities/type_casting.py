# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings

import numpy as np

import cudf


def type_cast(cudf_obj, np_type, name):
    if isinstance(cudf_obj, cudf.Series):
        cudf_type = cudf_obj.dtype
    elif isinstance(cudf_obj, cudf.DataFrame):
        if all([np.issubdtype(dtype, np.number) for dtype in cudf_obj.dtypes]):
            cudf_type = cudf_obj.dtypes[0]
        else:
            msg = "All columns in " + name + " should be numeric"
            raise Exception(msg)
    if (
        (
            np.issubdtype(np_type, np.floating)
            and (not np.issubdtype(cudf_type, np.floating))
        )
        or (
            np.issubdtype(np_type, np.integer)
            and (not np.issubdtype(cudf_type, np.integer))
        )
        or (
            np.issubdtype(np_type, np.bool_)
            and (not np.issubdtype(cudf_type, np.bool_))
        )
        or (
            np.issubdtype(np_type, np.int8)
            and (not np.issubdtype(cudf_type, np.int8))
        )
    ):
        msg = (
            "Casting "
            + name
            + " from "
            + str(cudf_type)
            + " to "
            + str(np.dtype(np_type))
        )
        warnings.warn(msg)
    cudf_obj = cudf_obj.astype(np_type)
    return cudf_obj
