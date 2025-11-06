# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect
import os
from contextvars import ContextVar

cuopt_paths = []

ncaid: ContextVar[str] = ContextVar("ncaid", default="")
requestid: ContextVar[str] = ContextVar("requestid", default="")
solverid: ContextVar[str] = ContextVar("solverid", default="")

# inspect.stack() is the call stack
# element [1] is the calling frame
# element [1][1] is the path containing the caller
# element [1][3] is the calling function


def set_ncaid(id):
    ncaid.set(id)


def get_ncaid():
    return ncaid.get()


def set_requestid(id):
    requestid.set(id)


def get_requestid():
    return requestid.get()


def set_solverid(id):
    solverid.set(id)


def get_solverid():
    return solverid.get()


def message_init():
    # Initialize cuopt_paths to the last directory in the
    # path of the caller, and the directory two levels up
    # from this file (../utils/logutil.py) since we use conda
    # Meant to be called from a file at the root of the source tree
    global cuopt_paths
    fp = inspect.stack()[1][1]
    dir = os.path.dirname(fp)
    cuopt_paths.append(dir[dir.rindex("/") + 1 :] + "/")

    dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cuopt_paths.append(dir[dir.rindex("/") + 1 :] + "/")


def message(msg):
    fp = inspect.stack()[1][1]
    for p in cuopt_paths:
        if p in fp:
            # only keep the portion of the path starting at cuopt path
            fp = fp[fp.rindex(p) :]
            break
    return ("%s:%s " % (fp, inspect.stack()[1][3])) + msg
