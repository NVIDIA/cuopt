# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Wrap a solver payload in the legacy HTTP envelope used by
# GET /cuopt/solution and validation_only responses.


def make_response(
    response, warnings=None, notes=None, reqId="", total_solve_time=0
):
    r = {"response": response}
    if total_solve_time:
        r["response"]["total_solve_time"] = total_solve_time
    if reqId:
        r["reqId"] = reqId
    if warnings:
        r["warnings"] = warnings
    if notes:
        r["notes"] = notes
    return r
