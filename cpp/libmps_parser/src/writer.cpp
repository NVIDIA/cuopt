/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights
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
