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

#pragma once

#include <mps_parser/mps_data_model.hpp>

namespace cuopt::mps_parser {

/**
 * @brief Reads the equation from the input text file which is MPS or QPS formatted
 *
 * Read this link http://lpsolve.sourceforge.net/5.5/mps-format.htm for more
 * details on both free and fixed MPS format.
 *
 * This function supports both standard MPS files (for linear programming) and
 * QPS files (for quadratic programming). QPS files are MPS files with additional
 * sections:
 * - QUADOBJ: Defines quadratic terms in the objective function
 *
 * @param[in] mps_file_path Path to MPS or QPS formatted file.
 * @param[in] fixed_mps_format If MPS/QPS file should be parsed as fixed format, false by default
 * @return mps_data_model_t A fully formed LP/QP problem which represents the given MPS/QPS file
 */
template <typename i_t, typename f_t>
mps_data_model_t<i_t, f_t> parse_mps(const std::string& mps_file_path,
                                     bool fixed_mps_format = false);

}  // namespace cuopt::mps_parser
