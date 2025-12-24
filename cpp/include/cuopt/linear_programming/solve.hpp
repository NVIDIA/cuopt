/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/linear_programming/data_model_view.hpp>
#include <cuopt/linear_programming/mip/solver_settings.hpp>
#include <cuopt/linear_programming/mip/solver_solution.hpp>
#include <cuopt/linear_programming/optimization_problem.hpp>
#include <cuopt/linear_programming/pdlp/solver_settings.hpp>
#include <cuopt/linear_programming/pdlp/solver_solution.hpp>
#include <cuopt/linear_programming/solver_settings.hpp>
#include <cuopt/linear_programming/utilities/internals.hpp>
#include <cuopt/linear_programming/utilities/remote_solve.hpp>
#include <mps_parser/mps_data_model.hpp>
#include <string>
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

/**
 * @brief Convert a data_model_view_t to an optimization_problem_t.
 *
 * This function copies data from the view (which points to GPU memory)
 * into an owning optimization_problem_t.
 *
 * @tparam i_t Data type of indexes
 * @tparam f_t Data type of the variables and their weights in the equations
 *
 * @param[in] handle_ptr  A raft::handle_t object with its corresponding CUDA stream.
 * @param[in] view  A data_model_view_t<i_t, f_t> object with spans pointing to GPU memory
 * @return optimization_problem_t<i_t, f_t> owning container for the problem
 */
template <typename i_t, typename f_t>
optimization_problem_t<i_t, f_t> data_model_view_to_optimization_problem(
  raft::handle_t const* handle_ptr, const data_model_view_t<i_t, f_t>& view);

/**
 * @brief Linear programming solve function using data_model_view_t.
 *
 * This overload accepts a non-owning data_model_view_t which can point to either
 * GPU memory (for local solves) or CPU memory (for remote solves).
 * The solve path is automatically determined by checking the CUOPT_REMOTE_HOST
 * and CUOPT_REMOTE_PORT environment variables.
 *
 * @note Both primal and dual solutions are zero-initialized.
 *
 * @tparam i_t Data type of indexes
 * @tparam f_t Data type of the variables and their weights in the equations
 *
 * @param[in] handle_ptr  A raft::handle_t object with its corresponding CUDA stream.
 * @param[in] view  A data_model_view_t<i_t, f_t> with spans pointing to problem data
 * @param[in] settings  A pdlp_solver_settings_t<i_t, f_t> object with the settings for the PDLP
 * solver.
 * @param[in] problem_checking  If true, the problem is checked for consistency.
 * @param[in] use_pdlp_solver_mode  If true, the PDLP hyperparameters coming from the
 * pdlp_solver_mode are used.
 * @return optimization_problem_solution_t<i_t, f_t> owning container for the solver solution
 */
template <typename i_t, typename f_t>
optimization_problem_solution_t<i_t, f_t> solve_lp(
  raft::handle_t const* handle_ptr,
  const data_model_view_t<i_t, f_t>& view,
  pdlp_solver_settings_t<i_t, f_t> const& settings = pdlp_solver_settings_t<i_t, f_t>{},
  bool problem_checking                            = true,
  bool use_pdlp_solver_mode                        = true);

/**
 * @brief Mixed integer programming solve function using data_model_view_t.
 *
 * This overload accepts a non-owning data_model_view_t which can point to either
 * GPU memory (for local solves) or CPU memory (for remote solves).
 * The solve path is automatically determined by checking the CUOPT_REMOTE_HOST
 * and CUOPT_REMOTE_PORT environment variables.
 *
 * @tparam i_t Data type of indexes
 * @tparam f_t Data type of the variables and their weights in the equations
 *
 * @param[in] handle_ptr  A raft::handle_t object with its corresponding CUDA stream.
 * @param[in] view  A data_model_view_t<i_t, f_t> with spans pointing to problem data
 * @param[in] settings  A mip_solver_settings_t<i_t, f_t> object with the settings for the MIP
 * solver.
 * @return mip_solution_t<i_t, f_t> owning container for the solver solution
 */
template <typename i_t, typename f_t>
mip_solution_t<i_t, f_t> solve_mip(
  raft::handle_t const* handle_ptr,
  const data_model_view_t<i_t, f_t>& view,
  mip_solver_settings_t<i_t, f_t> const& settings = mip_solver_settings_t<i_t, f_t>{});

}  // namespace cuopt::linear_programming
