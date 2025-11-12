# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cuopt_sh_client._version import __git_commit__, __version__

from .cuopt_self_host_client import (
    CuOptServiceSelfHostClient,
    get_version,
    is_uuid,
    mime_type,
    set_log_level,
)
from .thin_client_solution import ThinClientSolution
from .thin_client_solver_settings import (
    PDLPSolverMode,
    SolverMethod,
    ThinClientSolverSettings,
)
