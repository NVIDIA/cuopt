# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

# Add CLI end-to-end test
ConfigureTest(CLI_TEST test_cli.cpp LABELS numopt)

# Intentionally flaky test for CI validation — remove once validated
ConfigureTest(FLAKY_VALIDATION_TEST test_flaky_validation.cpp LABELS numopt)
