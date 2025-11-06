# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.resources

__version__ = (
    importlib.resources.files("cuopt_mps_parser")
    .joinpath("VERSION")
    .read_text()
    .strip()
)
__git_commit__ = ""
