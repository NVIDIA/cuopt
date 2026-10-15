# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import time

import numpy

from cuopt_server.utils.linear_programming.data_definition import LPData


def transform_lp_data(data):
    np = numpy
    tmap = {
        "csr_constraint_matrix": {
            "offsets": (True, np.int32),
            "indices": (True, np.int32),
            "values": (True, np.float64),
        },
        "constraint_bounds": {
            "bounds": (True, np.float64),
            "upper_bounds": (True, np.float64),
            "lower_bounds": (True, np.float64),
            "types": (True, "U1"),
        },
        "initial_solution": {
            "primal": (True, np.float64),
            "dual": (True, np.float64),
        },
        "objective_data": {
            "coefficients": (True, np.float64),
        },
        "variable_bounds": {
            "upper_bounds": (True, np.float64),
            "lower_bounds": (True, np.float64),
        },
        "variable_types": (True, "U1"),
    }

    def modify(value, key, dtype=None, indent=""):
        if isinstance(value, list):
            if "inf" in value or "ninf" in value:
                value = [
                    np.inf if x == "inf" else -np.inf if x == "ninf" else x
                    for x in value
                ]
            if dtype is None:
                return np.array(value)
            return np.array(value, dtype)
        return value

    def apply(data, tmap, indent=""):
        for key, value in data.items():
            try:
                if isinstance(value, dict) and key in tmap:
                    apply(value, tmap[key], indent + " ")
                elif key in tmap and tmap[key][0]:
                    data[key] = modify(value, key, tmap[key][1], indent)
            except Exception as e:
                logging.debug(e)
                logging.debug(
                    f"{indent}exception key is {key} value is {value}"
                )
                raise

    def apply_LPData(data):
        data.csr_constraint_matrix.indices = modify(
            data.csr_constraint_matrix.indices,
            "csr_constraint_matrix.indices",
            np.int32,
        )
        data.csr_constraint_matrix.offsets = modify(
            data.csr_constraint_matrix.offsets,
            "csr_constraint_matrix.offsets",
            np.int32,
        )
        data.csr_constraint_matrix.values = modify(
            data.csr_constraint_matrix.values,
            "csr_constraint_matrix.values",
            np.float64,
        )

        data.constraint_bounds.bounds = modify(
            data.constraint_bounds.bounds,
            "constraint_bounds.bounds",
            np.float64,
        )
        data.constraint_bounds.upper_bounds = modify(
            data.constraint_bounds.upper_bounds,
            "constraint_bounds.upper_bounds",
            np.float64,
        )
        data.constraint_bounds.lower_bounds = modify(
            data.constraint_bounds.lower_bounds,
            "constraint_bounds.lower_bounds",
            np.float64,
        )
        data.constraint_bounds.types = modify(
            data.constraint_bounds.types,
            "constraint_bounds.types",
        )

        data.initial_solution.primal = modify(
            data.initial_solution.primal,
            "initial_solution.primal",
            np.float64,
        )

        data.initial_solution.dual = modify(
            data.initial_solution.dual, "initial_solution.dual", np.float64
        )

        data.objective_data.coefficients = modify(
            data.objective_data.coefficients,
            "objective_data.coefficients",
            np.float64,
        )

        data.variable_bounds.upper_bounds = modify(
            data.variable_bounds.upper_bounds,
            "variable_bounds.upper_bounds",
            np.float64,
        )
        data.variable_bounds.lower_bounds = modify(
            data.variable_bounds.lower_bounds,
            "variable_bounds.lower_bounds",
            np.float64,
        )
        data.variable_types = modify(
            data.variable_types,
            "variable_types",
        )

    then = time.time()
    if isinstance(data, LPData):
        apply_LPData(data)
    else:
        apply(data, tmap)
    logging.info(f"transform time {time.time() - then}")
