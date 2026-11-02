# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading


class FakeSema:
    _value = -1

    def __enter__(self, *args, **kwargs):
        pass

    def __exit__(self, *args, **kwargs):
        pass

    def acquire(self, *args, **kwargs):
        return True

    def release(self, *args, **kwargs):
        pass


semas = {}


def make_sema(name, count):
    global semas
    if count > 0:
        sema = threading.Semaphore(count)
    else:
        sema = FakeSema()
    if name in semas:
        raise Exception(f"Cannot remake named semaphore {name}")
    semas[name] = sema


def get_sema(name):
    return semas[name]
