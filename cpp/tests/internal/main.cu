/* clang-format off */
/* SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
/* SPDX-License-Identifier: Apache-2.0
/* clang-format on */

// Single main() entry point for combined internal test binaries.
// Individual test files use CUOPT_TEST_PROGRAM_MAIN() which is suppressed
// (via CUOPT_DISABLE_TEST_MAIN) when they are compiled into a combined binary.
#include <utilities/base_fixture.hpp>

// Intentionally NOT suppressed here — this file owns the main() for the binary.
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  auto const cmd_opts = parse_test_options(argc, argv);
  auto const rmm_mode = cmd_opts["rmm_mode"].as<std::string>();
  auto resource       = cuopt::test::create_memory_resource(rmm_mode);
  rmm::mr::set_current_device_resource(resource);
  return RUN_ALL_TESTS();
}
