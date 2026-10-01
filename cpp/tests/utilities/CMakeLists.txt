# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

# Add CLI end-to-end test
ConfigureTest(CLI_TEST test_cli.cpp LABELS numopt)

# Add console-log-callback unit tests
ConfigureTest(CONSOLE_LOG_CALLBACK_TEST test_console_log_callback.cpp LABELS numopt)

# Logger boundary: per-library instances, shared log file, truncate semantics
ConfigureTest(LOGGER_TEST test_logger.cpp LABELS numopt)
