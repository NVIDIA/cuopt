/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <dual_simplex/solution.hpp>
#include <dual_simplex/sparse_matrix.hpp>
#include <dual_simplex/types.hpp>

#include <raft/core/handle.hpp>

#include <string>

namespace cuopt::linear_programming::dual_simplex {

enum class variable_type_t : int8_t {
  CONTINUOUS = 0,
  BINARY     = 1,
  INTEGER    = 2,
};

template <typename i_t, typename f_t>
struct user_problem_t {
  user_problem_t(raft::handle_t const* handle_ptr_)
    : handle_ptr(handle_ptr_), A(1, 1, 1), obj_constant(0.0), obj_scale(1.0)
  {
  }
  raft::handle_t const* handle_ptr;
  i_t num_rows;
  i_t num_cols;
  std::vector<f_t> objective;
  csc_matrix_t<i_t, f_t> A;
  std::vector<f_t> rhs;
  std::vector<char> row_sense;
  std::vector<f_t> lower;
  std::vector<f_t> upper;
  std::vector<i_t> range_rows;
  std::vector<f_t> range_value;
  i_t num_range_rows;
  std::string problem_name;
  std::vector<std::string> row_names;
  std::vector<std::string> col_names;
  f_t obj_constant;
  f_t obj_scale;  // 1.0 for min, -1.0 for max
  std::vector<variable_type_t> var_types;
};

}  // namespace cuopt::linear_programming::dual_simplex
