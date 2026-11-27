# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping

import numpy as np


def validate_distance_matrix(distance_matrix, comparison_matrix=None):
    """Validate distance matrices and their correspondence to cost matrices."""
    if not isinstance(distance_matrix, Mapping) or not distance_matrix:
        return (False, "Distance matrix cannot be null or empty")
    shape = None
    for vehicle_type, matrix in distance_matrix.items():
        if (
            not isinstance(vehicle_type, (int, np.integer))
            or not 0 <= vehicle_type <= 255
        ):
            return (
                False,
                "Matrix vehicle types must be integers within [0, 255]",
            )
        try:
            values = np.asarray(matrix)
        except (ValueError, TypeError):
            return (
                False,
                "Distance matrix must be a rectangular numeric array",
            )
        if (
            values.ndim != 2
            or not values.size
            or values.shape[0] != values.shape[1]
        ):
            return (False, "Distance matrix must be a square matrix")
        if values.dtype.kind not in "iuf":
            return (False, "Distance matrix must contain real numeric values")
        if np.ma.isMaskedArray(matrix) and np.ma.is_masked(matrix):
            return (False, "Distance matrix cannot have NULL values")
        if values.min() < 0:
            return (False, "All values in distance matrix must be >= 0")
        if not np.isfinite(values).all():
            return (False, "All values in distance matrix must be finite")
        if values.max() > np.finfo(np.float32).max:
            return (
                False,
                "All values in distance matrix must be representable as float32",
            )
        if shape is None:
            shape = values.shape
        elif shape != values.shape:
            return (
                False,
                "Distance matrices for all vehicle types must be the same shape",
            )
        if comparison_matrix is not None and (
            vehicle_type not in comparison_matrix
            or values.shape != comparison_matrix[vehicle_type].shape
        ):
            return (
                False,
                "Distance matrix shape must match the cost matrix shape",
            )
    return (True, "Valid Distance Matrix")
