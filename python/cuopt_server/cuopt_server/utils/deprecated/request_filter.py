# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import sys
from functools import partial

import numpy as np

from cuopt_server.utils.logutil import message

tier = None


# Standard solve time for VRP
def std_solver_time_calc(num_tasks):
    return 10 + num_tasks / 6


def update_solver_time(feature, field, request, field_path):
    num_tasks = len(request["task_data"].task_locations)
    std_solver_time = std_solver_time_calc(num_tasks)
    max_time = int(os.environ.get("CUOPT_MAX_SOLVE_TIME_LIMIT", 3600))
    if "time_limit" in feature and feature["time_limit"] is not None:
        solver_time = min(max_time, feature["time_limit"])
        if feature["time_limit"] != solver_time:
            logging.debug(f"Solver time modified to {solver_time}")
        else:
            logging.debug(f"Using specified solver time {solver_time}")
    else:
        solver_time = min(max_time, std_solver_time)
        logging.debug(
            f"Solver time limit not specified, setting to {solver_time}"
        )
    request["solver_config"].time_limit = solver_time
    return True, ""


def request_required(feature, field, request, field_path):
    if field not in feature:
        msg = field_path + " is required"
        return False, msg
    return True, ""


def request_forbid(feature, field, request, field_path):
    if field in feature and feature[field] is not None:
        msg = "'" + field_path + "' is not permitted"
        return False, msg
    return True, ""


def request_forbid_value(forbidden_value, feature, field, request, field_path):
    if field in feature and feature[field] == forbidden_value:
        msg = field_path + " cannot be set to " + str(forbidden_value)
        return False, msg
    return True, ""


def request_length(required_length, feature, field, request, field_path):
    if field in feature:
        value = feature[field]
        if value is not None and len(value) != required_length:
            msg = field_path + " length should be " + str(required_length)
            return False, msg
    return True, ""


# Expects a dictionary of matrices encoded as lists of lists
def request_symmetric(feature, field, request, field_path):
    if field in feature and isinstance(feature[field], dict):
        for key, values in feature[field].items():
            m = np.array(values)
            if not np.array_equal(m, m.T):
                msg = field_path + "[%s] must be symmetric" % key
                return False, msg
    return True, ""


def request_max_len(max, feature, field, request, field_path):
    if field in feature and len(feature[field]) > max:
        msg = f"length of {field_path} must be less than or equal to {max}"
        return False, msg
    return True, ""


feature_list = {
    "managed_default": {
        "solver_config": {
            "time_limit": update_solver_time,
        },
        "task_data": {
            "task_locations": partial(
                request_max_len,
                int(os.environ.get("CUOPT_TASK_LIMIT", 20000)),
            )
        },
    }
}


def set_tier(t):
    global tier
    tier = t
    if tier not in feature_list:
        raise Exception("tier value '%s' is not supported" % tier)
    logging.debug(message("Tier is %s" % tier))


def get_tier():
    return tier


def tier_list():
    return list(feature_list.keys())


def get_features():
    features = feature_list[tier]
    if callable(features):
        features = features()
    return features


def _get_feature_path(path, field):
    if path == "":
        return field
    return path + "." + field


def check_restrictions(feature_map, feature, request, feature_path=""):
    # "feature_map" is a dictionary with parallel structure
    # to a cuopt request

    # "feature" is a dictionary and on the first call it should be
    # equivalent to "request" (which never changes during recursion)
    status, msg = True, ""
    for field, filter_value in feature_map.items():
        field_path = _get_feature_path(feature_path, field)
        if isinstance(filter_value, str):
            filter_value = getattr(sys.modules[__name__], filter_value)
        if callable(filter_value):
            logging.debug(
                message("Calling %s for %s" % (filter_value, field_path))
            )
            status, msg = filter_value(feature, field, request, field_path)
            if not status:
                logging.error(message(msg))
        elif (
            isinstance(filter_value, dict)
            and field in feature
            and feature[field] is not None
        ):
            status, msg = check_restrictions(
                filter_value,
                dict(feature[field]),
                request,
                field_path,
            )
        if not status:
            break
    return status, msg
