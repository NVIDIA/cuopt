/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/export.hpp>
#include <cuopt/mathematical_optimization/io/parser.hpp>

#include <mps_parser_internal.hpp>
#include <parallel_parser/fast_parser.hpp>

#include <utilities/logger.hpp>

#include <cstdint>

namespace cuopt::mathematical_optimization::io {

template <typename i_t, typename f_t>
mps_data_model_t<i_t, f_t> read_mps(const std::string& mps_file, bool fixed_mps_format)
{
  mps_data_model_t<i_t, f_t> problem;
  mps_parser_t<i_t, f_t> parser(problem, mps_file, fixed_mps_format);
  return problem;
}

template <typename i_t, typename f_t>
mps_data_model_t<i_t, f_t> read_mps_from_string(std::string_view mps_contents,
                                                bool fixed_mps_format)
{
  mps_data_model_t<i_t, f_t> problem;
  mps_parser_t<i_t, f_t> parser(problem, mps_contents, fixed_mps_format);
  return problem;
}

template CUOPT_EXPORT mps_data_model_t<int, float> read_mps(const std::string& mps_file,
                                                            bool fixed_mps_format);
template CUOPT_EXPORT mps_data_model_t<int, double> read_mps(const std::string& mps_file,
                                                             bool fixed_mps_format);
template CUOPT_EXPORT mps_data_model_t<int64_t, double> read_mps(const std::string& mps_file,
                                                                 bool fixed_mps_format);
template CUOPT_EXPORT mps_data_model_t<int, float> read_mps_from_string(
  std::string_view mps_contents, bool fixed_mps_format);
template CUOPT_EXPORT mps_data_model_t<int, double> read_mps_from_string(
  std::string_view mps_contents, bool fixed_mps_format);

template <typename i_t, typename f_t>
mps_data_model_t<i_t, f_t> read_mps_parallel(const std::string& mps_file_path)
{
  CUOPT_LOG_DEBUG("Using parallel MPS parser for '%s'", mps_file_path.c_str());
  return detail::parse_mps_fast_file<i_t, f_t>(mps_file_path);
}

template CUOPT_EXPORT mps_data_model_t<int, double> read_mps_parallel(
  const std::string& mps_file_path);
template CUOPT_EXPORT mps_data_model_t<int64_t, double> read_mps_parallel(
  const std::string& mps_file_path);

}  // namespace cuopt::mathematical_optimization::io
