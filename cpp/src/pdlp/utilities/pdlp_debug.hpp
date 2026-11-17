/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

// PDLP_DEBUG_MODE is toggled in pdlp_constants.hpp; it must be seen before the #ifdef below
// regardless of include order in the including file.
#include <pdlp/pdlp_constants.hpp>

#ifdef PDLP_DEBUG_MODE

#include <utilities/copy_helpers.hpp>

#include <raft/util/cudart_utils.hpp>

#include <cstdio>
#include <iostream>
#include <string_view>
#include <type_traits>

// Must be defined before the print overloads below, which use PDLP_DEBUG_SYNC.
#define PDLP_DEBUG_SYNC()          RAFT_CUDA_TRY(cudaDeviceSynchronize())
#define PDLP_DEBUG_LOG(...)        ::cuopt::print(__VA_ARGS__)
#define PDLP_DEBUG_DEVICE_LOG(...) printf(__VA_ARGS__)

namespace cuopt {

// Message only
inline void print(std::string_view const message)
{
  PDLP_DEBUG_SYNC();
  std::cout << message << std::endl;
}

// Scalar (the container overloads live in utilities/copy_helpers.hpp)
template <typename T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
void print(std::string_view const name, T const value)
{
  PDLP_DEBUG_SYNC();
  std::cout << name << "=" << value << std::endl;
}

}  // namespace cuopt

#else

// No-op macros when PDLP_DEBUG_MODE is not defined, so they don't generate code in release builds.
#define PDLP_DEBUG_LOG(...)        ((void)0)
#define PDLP_DEBUG_SYNC()          ((void)0)
#define PDLP_DEBUG_DEVICE_LOG(...) ((void)0)

#endif
