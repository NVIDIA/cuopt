/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <mps_parser/writer.hpp>

#include <mps_parser/mps_writer.hpp>

namespace cuopt::mps_parser {

template <typename i_t, typename f_t>
void write_mps(const data_model_view_t<i_t, f_t>& problem, const std::string& mps_file_path)
{
  mps_writer_t<i_t, f_t> writer(problem);
  writer.write(mps_file_path);
}

template void write_mps<int, float>(const data_model_view_t<int, float>& problem,
                                    const std::string& mps_file_path);
template void write_mps<int, double>(const data_model_view_t<int, double>& problem,
                                     const std::string& mps_file_path);

}  // namespace cuopt::mps_parser
