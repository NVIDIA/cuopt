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

from cuopt_sh_client._version import __git_commit__, __version__

from .cuopt_self_host_client import (
    CuOptServiceSelfHostClient,
    get_version,
    is_uuid,
    mime_type,
    set_log_level,
)
from .thin_client_solver_settings import (
    PDLPSolverMode,
    SolverMethod,
    ThinClientSolverSettings,
)
