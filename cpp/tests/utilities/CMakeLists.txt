# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

# Add CLI end-to-end test — shared: runs for any component
ConfigureTest(CLI_TEST shared test_cli.cpp)
