# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os

from cuopt_server.utils.logutil import message

datadir = ""
resultdir = ("", 250, 644)


def set_data_dir(dir):
    global datadir
    datadir = dir
    logging.info(message(f"Data directory is {dir}"))
    if datadir:
        if not os.path.isdir(datadir):
            raise ValueError(f"Data directory '{datadir}' does not exist!")
        elif not os.access(datadir, os.R_OK):
            raise ValueError(
                f"Data directory '{dir}' "
                "is not readable by cuopt user "
                f"{os.getuid()}:{os.getgid()}, "
                "check permissions on the directory."
            )


def get_data_dir():
    return datadir


def set_result_dir(dir, maxresult, mode):
    global resultdir
    try:
        dir.lower()
        int(maxresult)
        m = int(mode, 8)
    except (AttributeError, ValueError, TypeError) as e:
        raise ValueError(
            "Bad values passed to set_result_dir() "
            f"{dir}, {maxresult}, {mode}: {str(e)}"
        )
    resultdir = (dir, maxresult, m)
    logging.info(
        message(
            f"Result directory is '{dir}', "
            f"maxresult = {maxresult}, mode = {mode}"
        )
    )
    if dir:
        if not os.path.isdir(dir):
            raise ValueError(f"Result directory '{dir}' does not exist!")
        elif not os.access(dir, os.W_OK):
            raise ValueError(
                f"Result directory '{dir}' "
                "is not writable by cuopt user "
                f"{os.getuid()}:{os.getgid()}, "
                "check permissions on the directory."
            )


def get_result_dir():
    return resultdir
