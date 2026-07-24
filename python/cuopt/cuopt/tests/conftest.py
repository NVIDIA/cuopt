# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

# pytest_plugins must live in the tests-root conftest (see pytest 8+ deprecation).
# Add fixtures/ to sys.path so the plugin module is importable without putting the
# source cuopt package tree on sys.path (which would shadow the installed wheel).
sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))
pytest_plugins = ["grpc_server_fixtures"]
