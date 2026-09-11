# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Wrap a solver payload in the legacy HTTP envelope used by
# GET /cuopt/solution and validation_only responses.

from typing import Any


def make_response(
    response: dict[str, Any],
    warnings: list[str] | None = None,
    notes: list[str] | None = None,
    reqId: str = "",
    total_solve_time: float = 0,
) -> dict[str, Any]:
    """Build the legacy HTTP response envelope.

    Parameters
    ----------
    response
        Solver payload placed under ``response``.
    warnings, notes
        Optional messages placed at the envelope's top level.
    reqId
        Optional request identifier.
    total_solve_time
        Optional solve duration added to the solver payload.

    Returns
    -------
    dict
        The legacy response envelope.
    """
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
