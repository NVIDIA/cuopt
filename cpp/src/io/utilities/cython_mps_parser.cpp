/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/linear_programming/io/parser.hpp>
#include <cuopt/linear_programming/io/utilities/cython_mps_parser.hpp>

namespace cuopt {
namespace cython {

std::unique_ptr<cuopt::linear_programming::io::mps_data_model_t<int, double>> call_parse_mps(
  const std::string& mps_file_path, bool fixed_mps_format)
{
  return std::make_unique<cuopt::linear_programming::io::mps_data_model_t<int, double>>(std::move(
    cuopt::linear_programming::io::parse_mps<int, double>(mps_file_path, fixed_mps_format)));
}

}  // namespace cython
}  // namespace cuopt
