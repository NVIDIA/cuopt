/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/linear_programming/io/parser.hpp>

#include "fast_parser.hpp"

#include <utilities/logger.hpp>

#include <cstdint>

namespace cuopt::linear_programming::io {

template <typename i_t, typename f_t>
mps_data_model_t<i_t, f_t> read_mps_fast_experimental(const std::string& mps_file_path)
{
  CUOPT_LOG_INFO("Using experimental fast MPS parser for '%s'", mps_file_path.c_str());
  return mps_fast::parse_mps_fast_file<i_t, f_t>(mps_file_path);
}

template mps_data_model_t<int, float> read_mps_fast_experimental(const std::string& mps_file_path);
template mps_data_model_t<int, double> read_mps_fast_experimental(const std::string& mps_file_path);
template mps_data_model_t<int64_t, float> read_mps_fast_experimental(
  const std::string& mps_file_path);
template mps_data_model_t<int64_t, double> read_mps_fast_experimental(
  const std::string& mps_file_path);

}  // namespace cuopt::linear_programming::io
