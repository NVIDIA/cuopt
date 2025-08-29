/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <mps_parser/data_model_view.hpp>

#include <stdarg.h>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cuopt::mps_parser {

/**
 * @brief Main writer class for MPS files
 *
 * @tparam f_t  data type of the weights and variables
 * @tparam i_t  data type of the indices
 */
template <typename i_t, typename f_t>
class mps_writer_t {
 public:
  /**
   * @brief Ctor. Takes a data model view as input and writes it out as a MPS formatted file
   *
   * @param[in] problem Data model view to write
   * @param[in] file Path to the MPS file to write
   */
  mps_writer_t(const data_model_view_t<i_t, f_t>& problem);

  /**
   * @brief Writes the problem to an MPS formatted file
   *
   * @param[in] mps_file_path Path to the MPS file to write
   */
  void write(const std::string& mps_file_path);

 private:
  const data_model_view_t<i_t, f_t>& problem_;
};  // class mps_writer_t

}  // namespace cuopt::mps_parser
