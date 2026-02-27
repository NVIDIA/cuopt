/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/linear_programming/optimization_problem.hpp>
#include <cuopt/linear_programming/optimization_problem_interface.hpp>
#include <cuopt/linear_programming/optimization_problem_solution_interface.hpp>
#include <cuopt/linear_programming/solver_settings.hpp>
#include <cuopt/linear_programming/utilities/cython_types.hpp>

#include <memory>
#include <mps_parser/data_model_view.hpp>
#include <raft/core/handle.hpp>
#include <string>
#include <utility>
#include <vector>

namespace cuopt {
namespace cython {

// Type definitions moved to cython_types.hpp to avoid circular dependencies
// The types linear_programming_ret_t and mip_ret_t are defined in cython_types.hpp.
// Each holds a std::variant internally to support both GPU and CPU solution data.

struct solver_ret_t {
  linear_programming::problem_category_t problem_type;
  linear_programming_ret_t lp_ret;
  mip_ret_t mip_ret;
};

// Wrapper functions to expose the API to Cython.
//
// Ownership convention:
//   call_solve_lp / call_solve_mip  -- return raw pointers; caller does NOT own them.
//     The returned pointers are backed by objects inside the solver_ret_t returned by call_solve.
//   call_solve / call_batch_solve   -- return unique_ptr<solver_ret_t>; caller owns the result.
//     The solver_ret_t holds the solution objects and must outlive any raw pointers obtained above.

linear_programming::lp_solution_interface_t<int, double>* call_solve_lp(
  linear_programming::optimization_problem_interface_t<int, double>* problem_interface,
  linear_programming::pdlp_solver_settings_t<int, double>& solver_settings,
  bool is_batch_mode = false);

// Call solve_mip and return solution interface pointer
linear_programming::mip_solution_interface_t<int, double>* call_solve_mip(
  linear_programming::optimization_problem_interface_t<int, double>* problem_interface,
  linear_programming::mip_solver_settings_t<int, double>& solver_settings);

// Main solve entry point from Python
std::unique_ptr<solver_ret_t> call_solve(cuopt::mps_parser::data_model_view_t<int, double>*,
                                         linear_programming::solver_settings_t<int, double>*,
                                         unsigned int flags = cudaStreamNonBlocking,
                                         bool is_batch_mode = false);

std::pair<std::vector<std::unique_ptr<solver_ret_t>>, double> solve_batch_remote(
  std::vector<cuopt::mps_parser::data_model_view_t<int, double>*>,
  linear_programming::solver_settings_t<int, double>*);

std::pair<std::vector<std::unique_ptr<solver_ret_t>>, double> call_batch_solve(
  std::vector<cuopt::mps_parser::data_model_view_t<int, double>*>,
  linear_programming::solver_settings_t<int, double>*);

}  // namespace cython
}  // namespace cuopt
