# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import threading
from contextlib import contextmanager

from cuopt_server.utils.logutil import message

health_status = "RUNNING"

sema = None


def health_init():
    global sema
    sema = threading.Semaphore(1)


@contextmanager
def sema_acquire():
    res = sema.acquire()
    try:
        yield res
    finally:
        if res:
            sema.release()


def _is_server_broken():
    return health_status != "RUNNING"


def set_unhealthy(msg):
    global health_status
    with sema_acquire():
        logging.error(message("ERROR : '%s' " % msg))
        health_status = "BROKEN"


def health():
    """
    Checks health of the server, and return status and message.
    """
    with sema_acquire():
        status = 1 if _is_server_broken() else 0
        msg = "Server is " + health_status
        return status, msg
