/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <cuopt/linear_programming/mip/solver_settings.hpp>
#include <cuopt/linear_programming/mip/solver_solution.hpp>
#include <cuopt/linear_programming/optimization_problem.hpp>
#include <cuopt/linear_programming/pdlp/solver_settings.hpp>
#include <cuopt/linear_programming/pdlp/solver_solution.hpp>
#include <cuopt/linear_programming/solver_settings.hpp>
#include <cuopt/linear_programming/utilities/internals.hpp>
#include <mps_parser/mps_data_model.hpp>
#include <vector>

namespace cuopt::linear_programming {

/**
 * @brief Linear programming solve function.
 * @note Both primal and dual solutions are zero-initialized. For custom initialization, see
 * op_problem.initial_primal/dual_solution
 *
 * @tparam i_t Data type of indexes
 * @tparam f_t Data type of the variables and their weights in the equations
 *
 * @param[in] op_problem  An optimization_problem_t<i_t, f_t> object with a
 * representation of a linear program
 * @param[in] settings  A pdlp_solver_settings_t<i_t, f_t> object with the settings for the PDLP
 * solver.
 * @param[in] problem_checking  If true, the problem is checked for consistency.
 * @param[in] use_pdlp_solver_modes  If true, the PDLP hyperparameters coming from the
 * pdlp_solver_mode are used (instead of the ones comming from a potential hyper-params file).
 * @return optimization_problem_solution_t<i_t, f_t> owning container for the solver solution
 */
template <typename i_t, typename f_t>
optimization_problem_solution_t<i_t, f_t> solve_lp(
  optimization_problem_t<i_t, f_t>& op_problem,
  pdlp_solver_settings_t<i_t, f_t> const& settings = pdlp_solver_settings_t<i_t, f_t>{},
  bool problem_checking                            = true,
  bool use_pdlp_solver_mode                        = true,
  bool is_batch_mode                               = false);

/**
 * @brief Linear programming solve function.
 * @note Both primal and dual solutions are zero-initialized. For custom initialization, see
 * op_problem.initial_primal/dual_solution
 *
 * @tparam i_t Data type of indexes
 * @tparam f_t Data type of the variables and their weights in the equations
 *
 * @param[in] handle_ptr  A raft::handle_t object with its corresponding CUDA stream.
 * @param[in] mps_data_model  An optimization_problem_t<i_t, f_t> object with a
 * representation of a linear program
 * @param[in] settings  A pdlp_solver_settings_t<i_t, f_t> object with the settings for the PDLP
 * solver.
 * @param[in] problem_checking  If true, the problem is checked for consistency.
 * @param[in] use_pdlp_solver_modes  If true, the PDLP hyperparameters coming from the
 * pdlp_solver_mode are used (instead of the ones comming from a potential hyper-params file).
 * @return optimization_problem_solution_t<i_t, f_t> owning container for the solver solution
 */
template <typename i_t, typename f_t>
optimization_problem_solution_t<i_t, f_t> solve_lp(
  raft::handle_t const* handle_ptr,
  const cuopt::mps_parser::mps_data_model_t<i_t, f_t>& mps_data_model,
  pdlp_solver_settings_t<i_t, f_t> const& settings = pdlp_solver_settings_t<i_t, f_t>{},
  bool problem_checking                            = true,
  bool use_pdlp_solver_mode                        = true);

/**
 * @brief Mixed integer programming solve function.
 *
 * @tparam i_t Data type of indexes
 * @tparam f_t Data type of the variables and their weights in the equations
 *
 * @param[in] op_problem  An optimization_problem_t<i_t, f_t> object with a
 * representation of a linear program
 * @return optimization_problem_solution_t<i_t, f_t> owning container for the solver solution
 */
template <typename i_t, typename f_t>
mip_solution_t<i_t, f_t> solve_mip(
  optimization_problem_t<i_t, f_t>& op_problem,
  mip_solver_settings_t<i_t, f_t> const& settings = mip_solver_settings_t<i_t, f_t>{});

/**
 * @brief Mixed integer programming solve function.
 *
 * @tparam i_t Data type of indexes
 * @tparam f_t Data type of the variables and their weights in the equations
 *
 * @param[in] mps_data_model  An optimization_problem_t<i_t, f_t> object with a
 * representation of a linear program
 * @return optimization_problem_solution_t<i_t, f_t> owning container for the solver solution
 */
template <typename i_t, typename f_t>
mip_solution_t<i_t, f_t> solve_mip(
  raft::handle_t const* handle_ptr,
  const cuopt::mps_parser::mps_data_model_t<i_t, f_t>& mps_data_model,
  mip_solver_settings_t<i_t, f_t> const& settings = mip_solver_settings_t<i_t, f_t>{});

template <typename i_t, typename f_t>
optimization_problem_t<i_t, f_t> mps_data_model_to_optimization_problem(
  raft::handle_t const* handle_ptr,
  const cuopt::mps_parser::mps_data_model_t<i_t, f_t>& data_model);

}  // namespace cuopt::linear_programming
