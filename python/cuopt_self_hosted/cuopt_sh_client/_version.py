# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.resources

ver = (
    importlib.resources.files("cuopt_sh_client")
    .joinpath("VERSION")
    .read_text()
    .strip()
)

__version__ = ver

__git_commit__ = ""
