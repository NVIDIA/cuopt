/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <dual_simplex/simplex_solver_settings.hpp>
#include <dual_simplex/solution.hpp>
#include <dual_simplex/sparse_matrix.hpp>
#include <dual_simplex/types.hpp>
#include <dual_simplex/user_problem.hpp>

#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t, typename f_t>
struct lp_problem_t {
  lp_problem_t(raft::handle_t const* handle_ptr_, i_t m, i_t n, i_t nz)
    : handle_ptr(handle_ptr_),
      num_rows(m),
      num_cols(n),
      objective(n),
      Q(0, 0, 0),
      A(m, n, nz),
      rhs(m),
      lower(n),
      upper(n),
      obj_constant(0.0)
  {
  }
  raft::handle_t const* handle_ptr;
  i_t num_rows;
  i_t num_cols;
  std::vector<f_t> objective;
  csr_matrix_t<i_t, f_t> Q;
  csc_matrix_t<i_t, f_t> A;
  std::vector<f_t> rhs;
  std::vector<f_t> lower;
  std::vector<f_t> upper;
  f_t obj_constant;
  f_t obj_scale;  // 1.0 for min, -1.0 for max
  bool objective_is_integral{false};

  void write_problem(const std::string& path) const
  {
    FILE* fid = fopen(path.c_str(), "w");
    if (fid) {
      fwrite(&num_rows, sizeof(i_t), 1, fid);
      fwrite(&num_cols, sizeof(i_t), 1, fid);
      fwrite(&obj_constant, sizeof(f_t), 1, fid);
      fwrite(&obj_scale, sizeof(f_t), 1, fid);
      i_t is_integral = objective_is_integral ? 1 : 0;
      fwrite(&is_integral, sizeof(i_t), 1, fid);
      fwrite(objective.data(), sizeof(f_t), num_cols, fid);
      fwrite(rhs.data(), sizeof(f_t), num_rows, fid);
      fwrite(lower.data(), sizeof(f_t), num_cols, fid);
      fwrite(upper.data(), sizeof(f_t), num_cols, fid);
      fwrite(A.col_start.data(), sizeof(i_t), A.col_start.size(), fid);
      fwrite(A.i.data(), sizeof(i_t), A.i.size(), fid);
      fwrite(A.x.data(), sizeof(f_t), A.x.size(), fid);
      fclose(fid);
    }
  }

  void read_problem(const std::string& path)
  {
    FILE* fid = fopen(path.c_str(), "r");
    if (fid) {
      fread(&num_rows, sizeof(i_t), 1, fid);
      fread(&num_cols, sizeof(i_t), 1, fid);
      fread(&obj_constant, sizeof(f_t), 1, fid);
      fread(&obj_scale, sizeof(f_t), 1, fid);
      i_t is_integral;
      fread(&is_integral, sizeof(i_t), 1, fid);
      objective_is_integral = is_integral == 1;
      objective.resize(num_cols);
      fread(objective.data(), sizeof(f_t), num_cols, fid);
      rhs.resize(num_rows);
      fread(rhs.data(), sizeof(f_t), num_rows, fid);
      lower.resize(num_cols);
      fread(lower.data(), sizeof(f_t), num_cols, fid);
      upper.resize(num_cols);
      fread(upper.data(), sizeof(f_t), num_cols, fid);
      A.n = num_cols;
      A.m = num_rows;
      A.col_start.resize(num_cols + 1);
      fread(A.col_start.data(), sizeof(i_t), num_cols + 1, fid);
      A.i.resize(A.col_start[num_cols]);
      fread(A.i.data(), sizeof(i_t), A.i.size(), fid);
      A.x.resize(A.i.size());
      fread(A.x.data(), sizeof(f_t), A.x.size(), fid);
      fclose(fid);
    }
  }

  void write_mps(const std::string& path) const
  {
    std::ofstream mps_file(path);
    if (!mps_file.is_open()) {
      printf("Failed to open file %s\n", path.c_str());
      return;
    }
    mps_file << std::setprecision(std::numeric_limits<f_t>::max_digits10);
    mps_file << "NAME " << "cuopt_lp_problem_t" << "\n";
    mps_file << "ROWS\n";
    mps_file << " N  OBJ\n";
    for (i_t i = 0; i < num_rows; i++) {
      mps_file << " E  R" << i << "\n";
    }
    mps_file << "COLUMNS\n";
    for (i_t j = 0; j < num_cols; j++) {
      const i_t col_start = A.col_start[j];
      const i_t col_end   = A.col_start[j + 1];
      mps_file << "    " << "C" << j << " OBJ " << objective[j] << "\n";
      for (i_t k = col_start; k < col_end; k++) {
        const i_t i          = A.i[k];
        const f_t x          = A.x[k];
        std::string col_name = "C" + std::to_string(j);
        std::string row_name = "R" + std::to_string(i);
        mps_file << "    " << col_name << " " << row_name << " " << x << "\n";
      }
    }
    mps_file << "RHS\n";
    for (i_t i = 0; i < num_rows; i++) {
      mps_file << "    RHS1      R" << i << " " << rhs[i] << "\n";
    }

    mps_file << "BOUNDS\n";
    for (i_t j = 0; j < num_cols; j++) {
      const f_t lb         = lower[j];
      const f_t ub         = upper[j];
      std::string col_name = "C" + std::to_string(j);
      if (lb == -std::numeric_limits<f_t>::infinity() &&
          ub == std::numeric_limits<f_t>::infinity()) {
        mps_file << " FR BOUND1    " << col_name << "\n";
      } else {
        if (lb == -std::numeric_limits<f_t>::infinity()) {
          mps_file << " MI BOUND1    " << col_name << "\n";
        } else {
          mps_file << " LO BOUND1    " << col_name << " " << lb << "\n";
        }
        if (ub != std::numeric_limits<f_t>::infinity()) {
          mps_file << " UP BOUND1    " << col_name << " " << ub << "\n";
        }
      }
    }
    mps_file << "ENDATA\n";
    mps_file.close();
  }
};

template <typename i_t, typename f_t>
struct folding_info_t {
  folding_info_t()
    : D(0, 0, 0),
      C_s(0, 0, 0),
      D_s(0, 0, 0),
      c_tilde(0),
      A_tilde(0, 0, 0),
      num_upper_bounds(0),
      previous_free_variable_pairs({}),
      is_folded(false)
  {
  }
  csc_matrix_t<i_t, f_t> D;
  csc_matrix_t<i_t, f_t> C_s;
  csc_matrix_t<i_t, f_t> D_s;
  std::vector<f_t> c_tilde;
  csc_matrix_t<i_t, f_t> A_tilde;
  i_t num_upper_bounds;
  std::vector<i_t> previous_free_variable_pairs;
  bool is_folded;
};

template <typename i_t, typename f_t>
struct presolve_info_t {
  // indices of variables in the original problem that remain in the presolved problem
  std::vector<i_t> remaining_variables;
  // indicies of variables in the original problem that have been removed in the presolved problem
  std::vector<i_t> removed_variables;
  // values of the removed variables
  std::vector<f_t> removed_values;
  // values of the removed reduced costs
  std::vector<f_t> removed_reduced_costs;
  // Free variable pairs
  std::vector<i_t> free_variable_pairs;
  // Removed lower bounds
  std::vector<f_t> removed_lower_bounds;
  // indices of the constraints in the original problem that remain in the presolved problem
  std::vector<i_t> remaining_constraints;
  // indices of the constraints in the original problem that have been removed in the presolved
  // problem
  std::vector<i_t> removed_constraints;

  folding_info_t<i_t, f_t> folding_info;

  // Variables that were negated to handle -inf < x_j <= u_j
  std::vector<i_t> negated_variables;
};

template <typename i_t, typename f_t>
struct dualize_info_t {
  dualize_info_t()
    : solving_dual(false),
      primal_problem(nullptr, 0, 0, 0),
      zl_start(0),
      zu_start(0),
      equality_rows({}),
      vars_with_upper_bounds({})
  {
  }
  bool solving_dual;
  lp_problem_t<i_t, f_t> primal_problem;
  i_t zl_start;
  i_t zu_start;
  std::vector<i_t> equality_rows;
  std::vector<i_t> vars_with_upper_bounds;
};

template <typename i_t, typename f_t>
void convert_user_problem(const user_problem_t<i_t, f_t>& user_problem,
                          const simplex_solver_settings_t<i_t, f_t>& settings,
                          lp_problem_t<i_t, f_t>& problem,
                          std::vector<i_t>& new_slacks,
                          dualize_info_t<i_t, f_t>& dualize_info);

template <typename i_t, typename f_t>
void convert_user_problem_with_guess(const user_problem_t<i_t, f_t>& user_problem,
                                     const std::vector<f_t>& guess,
                                     lp_problem_t<i_t, f_t>& problem,
                                     std::vector<f_t>& converted_guess);

template <typename i_t, typename f_t>
void convert_user_lp_with_guess(const user_problem_t<i_t, f_t>& user_problem,
                                const lp_solution_t<i_t, f_t>& initial_solution,
                                const std::vector<f_t>& initial_slack,
                                lp_problem_t<i_t, f_t>& lp,
                                lp_solution_t<i_t, f_t>& converted_solution);

template <typename i_t, typename f_t>
i_t presolve(const lp_problem_t<i_t, f_t>& original,
             const simplex_solver_settings_t<i_t, f_t>& settings,
             lp_problem_t<i_t, f_t>& presolved,
             presolve_info_t<i_t, f_t>& presolve_info);

template <typename i_t, typename f_t>
void crush_primal_solution(const user_problem_t<i_t, f_t>& user_problem,
                           const lp_problem_t<i_t, f_t>& problem,
                           const std::vector<f_t>& user_solution,
                           const std::vector<i_t>& new_slacks,
                           std::vector<f_t>& solution);

template <typename i_t, typename f_t>
void crush_primal_solution_with_slack(const user_problem_t<i_t, f_t>& user_problem,
                                      const lp_problem_t<i_t, f_t>& problem,
                                      const std::vector<f_t>& user_solution,
                                      const std::vector<f_t>& user_slack,
                                      const std::vector<i_t>& new_slacks,
                                      std::vector<f_t>& solution);

template <typename i_t, typename f_t>
f_t crush_dual_solution(const user_problem_t<i_t, f_t>& user_problem,
                        const lp_problem_t<i_t, f_t>& problem,
                        const std::vector<i_t>& new_slacks,
                        const std::vector<f_t>& user_y,
                        const std::vector<f_t>& user_z,
                        std::vector<f_t>& y,
                        std::vector<f_t>& z);

template <typename i_t, typename f_t>
void uncrush_primal_solution(const user_problem_t<i_t, f_t>& user_problem,
                             const lp_problem_t<i_t, f_t>& problem,
                             const std::vector<f_t>& solution,
                             std::vector<f_t>& user_solution);

template <typename i_t, typename f_t>
void uncrush_dual_solution(const user_problem_t<i_t, f_t>& user_problem,
                           const lp_problem_t<i_t, f_t>& problem,
                           const std::vector<f_t>& y,
                           const std::vector<f_t>& z,
                           std::vector<f_t>& user_y,
                           std::vector<f_t>& user_z);

template <typename i_t, typename f_t>
void uncrush_solution(const presolve_info_t<i_t, f_t>& presolve_info,
                      const simplex_solver_settings_t<i_t, f_t>& settings,
                      const std::vector<f_t>& crushed_x,
                      const std::vector<f_t>& crushed_y,
                      const std::vector<f_t>& crushed_z,
                      std::vector<f_t>& uncrushed_x,
                      std::vector<f_t>& uncrushed_y,
                      std::vector<f_t>& uncrushed_z);

}  // namespace cuopt::linear_programming::dual_simplex
