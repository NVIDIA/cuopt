/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <mps_parser/parser.hpp>

#include <mps_parser.hpp>

namespace cuopt::mps_parser {

template <typename i_t, typename f_t>
mps_data_model_t<i_t, f_t> parse_mps(const std::string& mps_file, bool fixed_mps_format)
{
  mps_data_model_t<i_t, f_t> problem;
  mps_parser_t<i_t, f_t> parser(problem, mps_file, fixed_mps_format);
  return problem;
}

template mps_data_model_t<int, float> parse_mps(const std::string& mps_file, bool fixed_mps_format);
template mps_data_model_t<int, double> parse_mps(const std::string& mps_file,
                                                 bool fixed_mps_format);

}  // namespace cuopt::mps_parser
