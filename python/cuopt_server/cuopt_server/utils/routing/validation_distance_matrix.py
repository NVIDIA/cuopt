# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np


def _has_distance_tiers(vehicle_distance_tiers):
    if vehicle_distance_tiers is None:
        return False
    if len(vehicle_distance_tiers) == 0:
        return False
    return all(
        tiers is not None and len(tiers) > 0
        for tiers in vehicle_distance_tiers
    )


def validate_distance_matrix(
    distance_matrix,
    vehicle_distance_tiers=None,
    require_distance_tiers=True,
    comparison_matrix=None,
):
    if distance_matrix is None or len(distance_matrix) == 0:
        return (False, "Distance matrix cannot be null or empty")

    if require_distance_tiers and not _has_distance_tiers(
        vehicle_distance_tiers
    ):
        return (
            False,
            "vehicle_distance_tiers must be set when distance matrix data is provided",
        )

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
        if matrix is None or len(matrix) == 0:
            return (False, "Distance matrix cannot be null or empty")

        row_lengths = [len(row) for row in matrix]
        if not len(set(row_lengths)) == 1:
            return (
                False,
                "All rows in the distance matrix must be of the same length",
            )

        if len(matrix) != len(matrix[0]):
            return (False, "Distance matrix must be a square matrix")

        np_distance_matrix = np.array(matrix)
        if np_distance_matrix.min() < 0:
            return (False, "All values in distance matrix must be >= 0")

        if not np.isfinite(np_distance_matrix).all():
            return (False, "All values in distance matrix must be finite")
        if np_distance_matrix.max() > np.finfo(np.float32).max:
            return (
                False,
                "All values in distance matrix must be representable as float32",
            )

        if shape is None:
            shape = np_distance_matrix.shape
        elif shape != np_distance_matrix.shape:
            return (
                False,
                "Distance matrices for all vehicle types must be the same shape",
            )

        if comparison_matrix is not None and (
            vehicle_type not in comparison_matrix
            or np_distance_matrix.shape
            != comparison_matrix[vehicle_type].shape
        ):
            return (
                False,
                "Distance matrix shape must match the cost matrix shape",
            )

    return (True, "Valid Distance Matrix")
