/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/linear_programming/io/parser.hpp>

#include <string_view>

namespace cuopt::test {

inline cuopt::linear_programming::io::mps_data_model_t<int, double> parse_inline_lp(
  std::string_view lp_text)
{
  return cuopt::linear_programming::io::read_lp_from_string<int, double>(lp_text);
}

}  // namespace cuopt::test
