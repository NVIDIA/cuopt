# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import traceback

from fastapi.responses import JSONResponse

# This function logs an abbreviated callstack and the
# exception message in a single tuple which is easy to find
# and understand in aggregated logs.


# For any file referenced below "site-packages", the path
# will be clipped to the portion after "site-packages"
def log_traceback(e, msg):
    # shorten messages by clipping path
    # Must use this form of format_exception to be compatible
    # with Python 3.10
    s = traceback.format_exception(type(e), e, e.__traceback__)
    for i, m in enumerate(s):
        if "site-packages" in m:
            p = m.index("site-packages")
            s[i] = m[p + len("site-packages") + 1 :]

    # Break up the messages so they are not clipped in aggregate
    counter = 0
    size = 0
    start = 0
    for i, m in enumerate(s):
        size += len(m)
        if size > 1024:
            logging.error(
                {
                    "cuopt_trace": s[start : i + 1],
                    "counter": counter,
                    "msg": msg,
                }
            )
            start = i + 1
            size = 0
            counter += 1
    if size > 0:
        logging.error(
            {"cuopt_trace": s[start : i + 1], "counter": counter, "msg": msg}
        )


def log_cuopt_exception(b, c):
    logging.error({"cuopt_exception": b["error"], "status_code": c})


def http_exception_handler(exc):
    log_traceback(exc, exc.detail)
    b, c = {"error": str(exc.detail)}, exc.status_code
    log_cuopt_exception(b, c)
    return JSONResponse(b, c)


def validation_exception_handler(exc):
    b, c = {"error": str(exc.errors())}, 422
    log_cuopt_exception(b, c)
    return JSONResponse(b, c)


def exception_handler(exc):
    msg = ":".join(str(exc).split(":")[1:]).split("\n")[0]
    if len(msg) == 0:
        msg = str(exc)
    b, c = (
        {
            "error": "cuOpt unhandled exception, "
            "please include this message in any error report: %s" % (msg)
        },
        500,
    )
    log_traceback(exc, msg)
    log_cuopt_exception(b, c)
    return JSONResponse(b, c)
