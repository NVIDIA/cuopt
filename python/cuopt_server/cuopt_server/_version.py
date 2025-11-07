# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.resources

__version__ = (
    importlib.resources.files("cuopt_server")
    .joinpath("VERSION")
    .read_text()
    .strip()
)
__git_commit__ = ""

__version_major_minor__ = ".".join(__version__.split(".")[:2])
