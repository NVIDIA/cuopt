# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# pytest_plugins must live in a top-level conftest (see pytest 8+ deprecation).
pytest_plugins = ["cuopt.tests.fixtures.grpc_server_fixtures"]
