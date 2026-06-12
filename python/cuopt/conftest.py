# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# pytest_plugins must live in a top-level conftest (see pytest 8+ deprecation).
pytest_plugins = ["cuopt.grpc_server_fixtures"]
