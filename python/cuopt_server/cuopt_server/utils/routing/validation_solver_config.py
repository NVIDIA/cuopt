# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


def validate_solver_config(
    time_limit,
    objectives,
    config_file,
    verbose_mode,
    error_logging,
    updating=False,
    comparison_time_limit=None,
):
    if updating and comparison_time_limit is None:
        return (
            False,
            "No solver config to update. The set_solver_config endpoint must be used before update is available",  # noqa
        )

    if (time_limit is not None) and (time_limit <= 0):
        return (False, "SolverSettings time limit must be greater than 0")

    if config_file is not None and len(config_file) == 0:
        return (
            False,
            "File path to save configuration should be valid and not empty",
        )

    return (True, "Valid SolverSettings Config")
