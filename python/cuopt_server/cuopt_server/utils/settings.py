# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
