# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

try:
    import libcuopt
except ModuleNotFoundError:
    pass
else:
    libcuopt.load_library()
    del libcuopt

from cuopt import linear_programming, routing
from cuopt._version import __git_commit__, __version__, __version_major_minor__
