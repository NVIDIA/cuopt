# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os

from cuopt_server._version import __version__


def check_client_version(client_vers):
    logging.debug(f"client_vers is {client_vers} in check")
    if os.environ.get("CUOPT_CHECK_CLIENT", True) in ["True", True]:
        major, minor, *_ = __version__.split(".")
        if client_vers == "custom":
            return []
        cv = (client_vers or "").split(".")
        if len(cv) < 2:
            logging.warning("Client version missing or bad format")
            return [
                "Client version missing or not the current format. "
                f"Please upgrade your cuOpt client to '{major}.{minor}', "
                "or set the client version to 'custom' "
                "if this is a custom client."
            ]
        cmajor, cminor = cv[:2]
        if (cmajor, cminor) != (major, minor):
            logging.warning(f"Client version {cmajor}.{cminor} does not match")
            return [
                f"Client version is '{cmajor}.{cminor}' but server "
                f"version is '{major}.{minor}'. Please use a matching client."
            ]
    return []
