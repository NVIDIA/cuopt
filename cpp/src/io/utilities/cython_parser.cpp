/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/mathematical_optimization/io/parser.hpp>
#include <cuopt/mathematical_optimization/io/utilities/cython_parser.hpp>

namespace cuopt {
namespace cython {

std::unique_ptr<cuopt::mathematical_optimization::io::mps_data_model_t<int, double>> call_read(
  const std::string& file_path, bool fixed_mps_format)
{
  return std::make_unique<cuopt::mathematical_optimization::io::mps_data_model_t<int, double>>(
    std::move(
      cuopt::mathematical_optimization::io::read<int, double>(file_path, fixed_mps_format)));
}

std::unique_ptr<cuopt::mathematical_optimization::io::mps_data_model_t<int, double>> call_parse_mps(
  const std::string& mps_file_path, bool fixed_mps_format)
{
  namespace io = cuopt::mathematical_optimization::io;
  return std::make_unique<io::mps_data_model_t<int, double>>(
    fixed_mps_format ? io::read_mps<int, double>(mps_file_path, true)
                     : io::read_mps_parallel<int, double>(mps_file_path));
}

}  // namespace cython
}  // namespace cuopt
