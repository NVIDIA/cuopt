# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

# Add CLI end-to-end test
ConfigureTest(CLI_TEST test_cli.cpp)

# Opt-in SIGSEGV for validating core dump collection (CUOPT_TEST_COREDUMP=1)
ConfigureTest(COREDUMP_SANITY_TEST coredump_sanity_test.cpp)
