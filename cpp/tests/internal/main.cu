/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <utilities/base_fixture.hpp>

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  auto const cmd_opts = parse_test_options(argc, argv);
  auto const rmm_mode = cmd_opts["rmm_mode"].as<std::string>();
  auto resource       = cuopt::test::create_memory_resource(rmm_mode);
  rmm::mr::set_current_device_resource(resource);
  return RUN_ALL_TESTS();
}
