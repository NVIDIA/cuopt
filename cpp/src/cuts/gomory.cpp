/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuts/cuts.hpp>
#include <dual_simplex/basis_solves.hpp>
#include <math_optimization/tic_toc.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

using simplex::basis_update_mpf_t;
using simplex::lp_problem_t;
using simplex::simplex_solver_settings_t;
using simplex::variable_type_t;

template <typename i_t, typename f_t>
void cut_generation_t<i_t, f_t>::generate_gomory_cuts(
  const lp_problem_t<i_t, f_t>& lp,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  csr_matrix_t<i_t, f_t>& Arow,
  const std::vector<i_t>& new_slacks,
  const std::vector<variable_type_t>& var_types,
  basis_update_mpf_t<i_t, f_t>& basis_update,
  const std::vector<f_t>& xstar,
  const std::vector<i_t>& basic_list,
  const std::vector<i_t>& nonbasic_list,
  f_t start_time)
{
  tableau_equality_t<i_t, f_t> tableau(lp, basis_update, nonbasic_list);
  mixed_integer_gomory_cut_t<i_t, f_t> gomory_cut;
  complemented_mixed_integer_rounding_cut_t<i_t, f_t> complemented_mir(lp, settings, new_slacks);
  simplex_solver_settings_t<i_t, f_t> variable_settings = settings;
  variable_settings.inside_submip                       = 1;
  variable_bounds_t<i_t, f_t> variable_bounds(lp, variable_settings, var_types, Arow, new_slacks);
  strong_cg_cut_t<i_t, f_t> cg(lp, var_types, xstar);
  std::vector<f_t> transformed_xstar;
  complemented_mir.bound_substitution(lp, variable_bounds, var_types, xstar, transformed_xstar);

  for (i_t i = 0; i < lp.num_rows; i++) {
    if (toc(start_time) >= settings.time_limit) { break; }
    inequality_t<i_t, f_t> inequality(lp.num_cols);
    const i_t j = basic_list[i];
    if (var_types[j] != variable_type_t::INTEGER) { continue; }
    const f_t x_j = xstar[j];
    if (fractional_part(x_j) < 0.05 || fractional_part(x_j) > 0.95) { continue; }

    i_t tableau_status = tableau.generate_base_equality(
      lp, settings, Arow, var_types, basis_update, xstar, basic_list, nonbasic_list, i, inequality);
    if (tableau_status == 0) {
      // Generate a CG cut
      const bool generate_cg_cut = settings.strong_chvatal_gomory_cuts != 0;
      if (generate_cg_cut) {
        // Try to generate a CG cut
        inequality_t<i_t, f_t> cg_inequality = inequality;
        if (fractional_part(inequality.rhs) < 0.5) {
          // Multiply by -1 to force the fractional part to be greater than 0.5
          cg_inequality.negate();
        }
        inequality_t<i_t, f_t> cg_cut(lp.num_cols);
        i_t cg_status =
          cg.generate_strong_cg_cut(lp, settings, var_types, cg_inequality, xstar, cg_cut);
        if (cg_status == 0) { cut_pool_.add_cut(cut_type_t::CHVATAL_GOMORY, cg_cut); }
      }

      if (settings.mixed_integer_gomory_cuts == 0) { continue; }

      // Transform the inequality
      inequality_t<i_t, f_t> transformed_inequality = inequality;
      complemented_mir.transform_inequality(variable_bounds, var_types, transformed_inequality);

      // Generate a MIR cut from the transformed inequality
      inequality_t<i_t, f_t> cut_A_float(lp.num_cols);
      bool cut_ok = complemented_mir.generate_cut_nonnegative_maintain_indicies(
        transformed_inequality, var_types, cut_A_float);

      // Transform the cut back to the original variables
      complemented_mir.untransform_inequality(variable_bounds, var_types, cut_A_float);
      complemented_mir.remove_small_coefficients(lp.lower, lp.upper, cut_A_float);

      inequality_t<i_t, f_t> cut_A(lp.num_cols);
      if (cut_ok) {
        cut_ok = rational_coefficients(
          var_types, lp.lower, lp.upper, inequality_sense_t::GREATER_EQUAL, cut_A_float, cut_A);
      }

      // See if the inequality is violated by the original relaxation solution
      f_t cut_A_violation = complemented_mir.compute_violation(cut_A, xstar);
      bool A_valid        = false;
      f_t cut_A_distance  = 0.0;
      if (cut_ok && cut_A_violation > 1e-6) {
        if (cut_A.size() == 0) { continue; }
        complemented_mir.substitute_slacks(lp, Arow, cut_A);
        complemented_mir.remove_small_coefficients(lp.lower, lp.upper, cut_A);
        if (cut_A.size() == 0) {
          A_valid = false;
        } else {
          // Check that the cut is violated
          f_t dot      = cut_A.vector.dot(xstar);
          f_t cut_norm = cut_A.vector.norm2_squared();
          if (dot >= cut_A.rhs) { continue; }
          cut_A_distance = (cut_A.rhs - dot) / std::sqrt(cut_norm);
          A_valid        = true;
        }
      }

      // Negate the base inequality
      inequality.negate();

      inequality_t<i_t, f_t> cut_B_float(lp.num_cols);

      transformed_inequality = inequality;
      complemented_mir.transform_inequality(variable_bounds, var_types, transformed_inequality);

      cut_ok = complemented_mir.generate_cut_nonnegative_maintain_indicies(
        transformed_inequality, var_types, cut_B_float);
      // Transform the cut back to the original variables
      complemented_mir.untransform_inequality(variable_bounds, var_types, cut_B_float);
      complemented_mir.remove_small_coefficients(lp.lower, lp.upper, cut_B_float);

      inequality_t<i_t, f_t> cut_B(lp.num_cols);
      if (cut_ok) {
        cut_ok = rational_coefficients(
          var_types, lp.lower, lp.upper, inequality_sense_t::GREATER_EQUAL, cut_B_float, cut_B);
      }

      bool B_valid        = false;
      f_t cut_B_distance  = 0.0;
      f_t cut_B_violation = complemented_mir.compute_violation(cut_B, xstar);
      if (cut_ok && cut_B_violation > 1e-6) {
        if (cut_B.size() == 0) { continue; }
        complemented_mir.substitute_slacks(lp, Arow, cut_B);
        complemented_mir.remove_small_coefficients(lp.lower, lp.upper, cut_B);
        if (cut_B.size() == 0) {
          B_valid = false;
        } else {
          // Check that the cut is violated
          f_t dot      = cut_B.vector.dot(xstar);
          f_t cut_norm = cut_B.vector.norm2_squared();
          if (dot >= cut_B.rhs) { continue; }
          cut_B_distance = (cut_B.rhs - dot) / std::sqrt(cut_norm);
          B_valid        = true;
        }
      }

      if ((cut_A_distance > cut_B_distance) && A_valid) {
        cut_pool_.add_cut(cut_type_t::MIXED_INTEGER_GOMORY, cut_A);
      } else if (B_valid) {
        cut_pool_.add_cut(cut_type_t::MIXED_INTEGER_GOMORY, cut_B);
      }
    }
  }
}

template <typename i_t, typename f_t>
i_t tableau_equality_t<i_t, f_t>::generate_base_equality(
  const lp_problem_t<i_t, f_t>& lp,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  csr_matrix_t<i_t, f_t>& Arow,
  const std::vector<variable_type_t>& var_types,
  basis_update_mpf_t<i_t, f_t>& basis_update,
  const std::vector<f_t>& xstar,
  const std::vector<i_t>& basic_list,
  const std::vector<i_t>& nonbasic_list,
  i_t i,
  inequality_t<i_t, f_t>& inequality)
{
  // Let's look for Gomory cuts
  const i_t j = basic_list[i];
  if (var_types[j] != variable_type_t::INTEGER) { return -1; }
  const f_t x_j = xstar[j];
  if (std::abs(x_j - std::round(x_j)) < settings.integer_tol) { return -1; }
#ifdef PRINT_CUT_INFO
  settings_.log.printf("Generating cut for variable %d relaxed value %e row %d\n", j, x_j, i);
#endif

  // Solve B^T u_bar = e_i
  sparse_vector_t<i_t, f_t> e_i(lp.num_rows, 1);
  e_i.i[0] = i;
  e_i.x[0] = 1.0;
  sparse_vector_t<i_t, f_t> u_bar(lp.num_rows, 0);
  basis_update.b_transpose_solve(e_i, u_bar);

#ifdef CHECK_B_TRANSPOSE_SOLVE
  std::vector<f_t> u_bar_dense(lp.num_rows);
  u_bar.to_dense(u_bar_dense);

  std::vector<f_t> BTu_bar(lp.num_rows);
  simplex::b_transpose_multiply(lp, basic_list, u_bar_dense, BTu_bar);
  for (i_t k = 0; k < lp.num_rows; k++) {
    if (k == i) {
      settings.log.printf("BTu_bar %d error %e\n", k, std::abs(BTu_bar[k] - 1.0));
      if (std::abs(BTu_bar[k] - 1.0) > 1e-10) {
        settings.log.printf("BTu_bar[%d] = %e i %d\n", k, BTu_bar[k], i);
        assert(false);
      }
    } else {
      settings.log.printf("BTu_bar %d error %e\n", k, std::abs(BTu_bar[k]));
      if (std::abs(BTu_bar[k]) > 1e-10) {
        settings.log.printf("BTu_bar[%d] = %e i %d\n", k, BTu_bar[k], i);
        assert(false);
      }
    }
  }
#endif

  // Compute a_bar = N^T u_bar
  // TODO: This is similar to a function in phase2 of dual simplex. See if it can be reused.
  const i_t nz_ubar = u_bar.i.size();
  std::vector<i_t> abar_indices;
  abar_indices.reserve(nz_ubar);
  for (i_t k = 0; k < nz_ubar; k++) {
    const i_t ii        = u_bar.i[k];
    const f_t u_bar_i   = u_bar.x[k];
    const i_t row_start = Arow.row_start[ii];
    const i_t row_end   = Arow.row_start[ii + 1];
    for (i_t p = row_start; p < row_end; p++) {
      const i_t jj = Arow.j[p];
      if (nonbasic_mark_[jj] == 1) {
        const f_t val    = u_bar_i * Arow.x[p];
        const f_t y      = val - c_workspace_[jj];
        const f_t t      = x_workspace_[jj] + y;
        c_workspace_[jj] = (t - x_workspace_[jj]) - y;
        x_workspace_[jj] = t;
        if (!x_mark_[jj]) {
          x_mark_[jj] = 1;
          abar_indices.push_back(jj);
        }
      }
    }
  }
  // TODO: abar has lots of small coefficients. Double check that
  // we do not accidently create a base (in)equality
  // that cuts off an integer solution, when we drop the small coefficients.

  i_t small_coeff              = 0;
  const f_t drop_tol           = 1e-12;
  const bool drop_coefficients = true;
  sparse_vector_t<i_t, f_t> a_bar(lp.num_cols, 0);
  a_bar.i.reserve(abar_indices.size() + 1);
  a_bar.x.reserve(abar_indices.size() + 1);
  for (i_t k = 0; k < abar_indices.size(); k++) {
    const i_t jj = abar_indices[k];
    if (drop_coefficients && std::abs(x_workspace_[jj]) < drop_tol) {
      small_coeff++;
    } else {
      a_bar.i.push_back(jj);
      a_bar.x.push_back(x_workspace_[jj]);
    }
  }
  const bool verbose = false;
  if (verbose && small_coeff > 0) { settings.log.printf("Small coeff dropped %d\n", small_coeff); }

  // Clear the workspace
  for (i_t jj : abar_indices) {
    x_workspace_[jj] = 0.0;
    x_mark_[jj]      = 0;
    c_workspace_[jj] = 0.0;
  }
  abar_indices.clear();

  // We should now have the base inequality
  // x_j + a_bar^T x_N >= b_bar_i
  // We add x_j into a_bar so that everything is in a single sparse_vector_t
  a_bar.i.push_back(j);
  a_bar.x.push_back(1.0);

  // Check that the tableau equality is satisfied
  const f_t tableau_tol = 1e-6;
  f_t a_bar_dot_xstar   = a_bar.dot(xstar);
  if (std::abs(a_bar_dot_xstar - b_bar_[i]) > tableau_tol) {
    settings.log.debug("bad tableau equality. error %e\n", std::abs(a_bar_dot_xstar - b_bar_[i]));
    return -1;
  }

  // We have that x_j + a_bar^T x_N == b_bar_i
  // So x_j + a_bar^T x_N >= b_bar_i
  // And x_j + a_bar^T x_N <= b_bar_i
  // Or -x_j - a_bar^T x_N >= -b_bar_i

  // Skip cuts that are shallow
  const f_t shallow_tol = 1e-2;
  if (std::abs(x_j - std::round(x_j)) < shallow_tol) {
    // Skip cuts where integer variable has small fractional part
    return -1;
  }

#ifdef PRINT_BASE_INEQUALITY
  // Print out the base inequality
  for (i_t k = 0; k < a_bar.i.size(); k++) {
    const i_t jj = a_bar.i[k];
    const f_t aj = a_bar.x[k];
    settings_.log.printf("a_bar[%d] = %e\n", k, aj);
  }
  settings_.log.printf("b_bar[%d] = %e\n", i, b_bar[i]);
#endif

  inequality.vector = a_bar;
  inequality.rhs    = b_bar_[i];

  return 0;
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE
template class tableau_equality_t<int, double>;

template void cut_generation_t<int, double>::generate_gomory_cuts(
  const lp_problem_t<int, double>& lp,
  const simplex_solver_settings_t<int, double>& settings,
  csr_matrix_t<int, double>& Arow,
  const std::vector<int>& new_slacks,
  const std::vector<variable_type_t>& var_types,
  basis_update_mpf_t<int, double>& basis_update,
  const std::vector<double>& xstar,
  const std::vector<int>& basic_list,
  const std::vector<int>& nonbasic_list,
  double start_time);
#endif

}  // namespace cuopt::mathematical_optimization::mip
