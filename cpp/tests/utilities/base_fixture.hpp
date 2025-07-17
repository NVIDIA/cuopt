/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <utilities/error.hpp>
#include <utilities/macros.cuh>
#include "cxxopts.hpp"

#include <gtest/gtest.h>

#include <rmm/mr/device/binning_memory_resource.hpp>
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/owning_wrapper.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

namespace cuopt {
namespace test {

/// MR factory functions
inline auto make_cuda() { return std::make_shared<rmm::mr::cuda_memory_resource>(); }

inline auto make_async() { return std::make_shared<rmm::mr::cuda_async_memory_resource>(); }

inline auto make_managed() { return std::make_shared<rmm::mr::managed_memory_resource>(); }

inline auto make_pool()
{
  // 1GB of initial pool size
  const size_t initial_pool_size = 1024 * 1024 * 1024;
  return rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(make_async(),
                                                                     initial_pool_size);
}

inline auto make_binning()
{
  auto pool = make_pool();
  // Add a fixed_size_memory_resource for bins of size 256, 512, 1024, 2048 and
  // 4096KiB Larger allocations will use the pool resource
  auto mr = rmm::mr::make_owning_wrapper<rmm::mr::binning_memory_resource>(pool, 18, 22);
  return mr;
}

/**
 * @brief Creates a memory resource for the unit test environment given the name
 * of the allocation mode.
 *
 * The returned resource instance must be kept alive for the duration of the
 * tests. Attaching the resource to a TestEnvironment causes issues since the
 * environment objects are not destroyed until after the runtime is shutdown.
 *
 * @throw cuopt::logic_error if the `allocation_mode` is unsupported.
 *
 * @param allocation_mode String identifies which resource type.
 *        Accepted types are "pool", "cuda", and "managed" only.
 * @return Memory resource instance
 */
inline std::shared_ptr<rmm::mr::device_memory_resource> create_memory_resource(
  std::string const& allocation_mode)
{
  if (allocation_mode == "binning") return make_binning();
  if (allocation_mode == "cuda") return make_cuda();
  if (allocation_mode == "pool") return make_pool();
  if (allocation_mode == "managed") return make_managed();
  cuopt_assert(false, "Invalid RMM allocation mode");

  // control will never reach this point
  return make_managed();
}

}  // namespace test
}  // namespace cuopt

/**
 * @brief Parses the cuOpt test command line options.
 *
 * Currently only supports 'rmm_mode' string paramater, which set the rmm
 * allocation mode. The default value of the parameter is 'pool'.
 *
 * @return Parsing results in the form of cxxopts::ParseResult
 */
inline auto parse_test_options(int argc, char** argv)
{
  try {
    cxxopts::Options options(argv[0], " - cuOpt tests command line options");
    options.allow_unrecognised_options().add_options()(
      "rmm_mode", "RMM allocation mode", cxxopts::value<std::string>()->default_value("pool"));

    return options.parse(argc, argv);
  } catch (const std::exception& e) {
    cuopt_assert(false, "Error parsing command line options");
  }

  // control will never reach this point
  cxxopts::Options options(argv[0], " - cuOpt tests command line options");
  return options.parse(argc, argv);
}

/**
 * @brief Macro that defines main function for gtest programs that use rmm
 *
 * Should be included in every test program that uses rmm allocators since it
 * maintains the lifespan of the rmm default memory resource. This `main`
 * function is a wrapper around the google test generated `main`, maintaining
 * the original functionality. In addition, this custom `main` function parses
 * the command line to customize test behavior, like the allocation mode used
 * for creating the default memory resource.
 */
#define CUOPT_TEST_PROGRAM_MAIN()                                        \
  int main(int argc, char** argv)                                        \
  {                                                                      \
    ::testing::InitGoogleTest(&argc, argv);                              \
    auto const cmd_opts = parse_test_options(argc, argv);                \
    auto const rmm_mode = cmd_opts["rmm_mode"].as<std::string>();        \
    auto resource       = cuopt::test::create_memory_resource(rmm_mode); \
    rmm::mr::set_current_device_resource(resource.get());                \
    return RUN_ALL_TESTS();                                              \
  }
