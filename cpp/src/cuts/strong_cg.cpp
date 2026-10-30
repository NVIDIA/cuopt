/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuts/cuts.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace cuopt::mathematical_optimization::mip {

using simplex::lp_problem_t;
using simplex::simplex_solver_settings_t;
using simplex::variable_type_t;

template <typename i_t, typename f_t>
strong_cg_cut_t<i_t, f_t>::strong_cg_cut_t(const lp_problem_t<i_t, f_t>& lp,
                                           const std::vector<variable_type_t>& var_types,
                                           const std::vector<f_t>& xstar)
  : transformed_variables_(lp.num_cols, 0)
{
  // Determine the substition for the integer variables
  for (i_t j = 0; j < lp.num_cols; j++) {
    if (var_types[j] == variable_type_t::INTEGER) {
      const f_t l_j = lp.lower[j];
      const f_t u_j = lp.upper[j];
      if (l_j != 0.0) {
        // We need to transform the variable
        // Check the distance to each bound
        const f_t dist_to_lower = std::max(0.0, xstar[j] - l_j);
        const f_t dist_to_upper = std::max(0.0, u_j - xstar[j]);
        if (dist_to_upper >= dist_to_lower || u_j >= inf) {
          // We are closer to the lower bound.
          transformed_variables_[j] = -1;
        } else if (u_j < inf) {
          // We are closer to the finite upper bound
          transformed_variables_[j] = 1;
        }
      }
    }
  }
}

template <typename i_t, typename f_t>
i_t strong_cg_cut_t<i_t, f_t>::remove_continuous_variables_integers_nonnegative(
  const lp_problem_t<i_t, f_t>& lp,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  const std::vector<variable_type_t>& var_types,
  inequality_t<i_t, f_t>& inequality)
{
  const bool verbose = false;
  // Count the number of continuous variables in the inequality
  i_t num_continuous = 0;
  const i_t nz       = inequality.size();
  for (i_t k = 0; k < nz; k++) {
    const i_t j = inequality.index(k);
    if (var_types[j] == variable_type_t::CONTINUOUS) { num_continuous++; }
  }

  if (verbose) { settings.log.printf("num_continuous %d\n", num_continuous); }
  // We assume the inequality is of the form sum_j a_j x_j <= rhs

  for (i_t k = 0; k < nz; k++) {
    const i_t j   = inequality.index(k);
    const f_t l_j = lp.lower[j];
    const f_t u_j = lp.upper[j];
    const f_t a_j = inequality.coeff(k);
    if (var_types[j] == variable_type_t::CONTINUOUS) {
      if (a_j == 0.0) { continue; }

      if (a_j > 0.0 && l_j > -inf) {
        // v_j = x_j - l_j >= 0
        // x_j = v_j + l_j
        // sum_{k != j} a_k x_k + a_j x_j <= rhs
        // sum_{k != j} a_k x_k + a_j (v_j + l_j) <= rhs
        // sum_{k != j} a_k x_k + a_j v_j <= rhs - a_j l_j
        inequality.rhs -= a_j * l_j;
        transformed_variables_[j] = -1;

        // We now have a_j * v_j with a_j, v_j >= 0
        // So we have sum_{k != j} a_k x_k <= sum_{k != j} a_k x_k + a_j v_j <= rhs - a_j l_j
        // So we can now drop the continuous variable v_j
        inequality.vector.x[k] = 0.0;

      } else if (a_j < 0.0 && u_j < inf) {
        // w_j = u_j - x_j >= 0
        // x_j = u_j - w_j
        // sum_{k != j} a_k x_k + a_j x_j <= rhs
        // sum_{k != j} a_k x_k + a_j (u_j - w_j) <= rhs
        // sum_{k != j} a_k x_k - a_j w_j <= rhs - a_j u_j
        inequality.rhs -= a_j * u_j;
        transformed_variables_[j] = 1;

        // We now have a_j * w_j with a_j, w_j >= 0
        // So we have sum_{k != j} a_k x_k <= sum_{k != j} a_k x_k + a_j w_j <= rhs - a_j u_j
        // So we can now drop the continuous variable w_j
        inequality.vector.x[k] = 0.0;
      } else {
        // We can't keep the coefficient of the continuous variable positive
        // This means we can't eliminate the continuous variable
        if (verbose) { settings.log.printf("x%d ak: %e lo: %e up: %e\n", j, a_j, l_j, u_j); }
        return -1;
      }
    } else {
      // The variable is integer. We just need to ensure it is nonnegative
      if (transformed_variables_[j] == -1) {
        // We are closer to the lower bound.
        // v_j = x_j - l_j >= 0
        // x_j = v_j + l_j
        // sum_{k != j} a_k x_k + a_j x_j <= rhs
        // sum_{k != j} a_k x_k + a_j (v_j + l_j) <= rhs
        // sum_{k != j} a_k x_k + a_j v_j <= rhs - a_j l_j
        inequality.rhs -= a_j * l_j;
      } else if (transformed_variables_[j] == 1) {
        // We are closer to the finite upper bound
        // w_j = u_j - x_j >= 0
        // x_j = u_j - w_j
        // sum_{k != j} a_k x_k + a_j x_j <= rhs
        // sum_{k != j} a_k x_k + a_j (u_j - w_j) <= rhs
        // sum_{k != j} a_k x_k - a_j w_j <= rhs - a_j u_j
        inequality.rhs -= a_j * u_j;
        inequality.vector.x[k] *= -1.0;
      }
    }
  }

  // Squeeze out the zero coefficents
  sparse_vector_t<i_t, f_t> new_inequality_vector(inequality.vector.n, 0);
  inequality.vector.squeeze(new_inequality_vector);
  inequality.vector = new_inequality_vector;
  return 0;
}

template <typename i_t, typename f_t>
void strong_cg_cut_t<i_t, f_t>::to_original_integer_variables(const lp_problem_t<i_t, f_t>& lp,
                                                              inequality_t<i_t, f_t>& cut)
{
  // We expect a cut of the form sum_j a_j y_j <= rhs
  // where y_j >= 0 is a transformed variable
  // We need to convert it back into a cut on the original variables

  for (i_t k = 0; k < cut.size(); k++) {
    const i_t j   = cut.index(k);
    const f_t a_j = cut.coeff(k);
    if (transformed_variables_[j] == -1) {
      // sum_{k != j} a_k x_k + a_j v_j <= rhs
      // v_j = x_j - l_j >= 0,
      // sum_{k != j} a_k x_k + a_j (x_j - l_j) <= rhs
      // sum_{k != j} a_k x_k + a_j x_j <= rhs + a_j l_j
      cut.rhs += a_j * lp.lower[j];
    } else if (transformed_variables_[j] == 1) {
      // sum_{k != j} a_k x_k + a_j w_j <= rhs
      // w_j = u_j - x_j >= 0
      // sum_{k != j} a_k x_k + a_j (u_j - x_j) <= rhs
      // sum_{k != j} a_k x_k - a_j x_j <= rhs - a_j u_j
      cut.rhs -= a_j * lp.upper[j];
      cut.vector.x[k] *= -1.0;
    }
  }
}

template <typename i_t, typename f_t>
i_t strong_cg_cut_t<i_t, f_t>::generate_strong_cg_cut_integer_only(
  const simplex_solver_settings_t<i_t, f_t>& settings,
  const std::vector<variable_type_t>& var_types,
  const inequality_t<i_t, f_t>& inequality,
  inequality_t<i_t, f_t>& cut)
{
  // We expect an inequality of the form sum_j a_j x_j <= rhs
  // where all the variables x_j are integer and nonnegative

  // We then apply the CG cut:
  // sum_j floor(a_j) x_j <= floor(rhs)
  cut.reserve(inequality.size());
  cut.clear();

  f_t a_0   = inequality.rhs;
  f_t f_a_0 = fractional_part(a_0);

  if (f_a_0 == 0.0) {
    // f(a_0) == 0.0 so we do a weak CG cut
    cut.reserve(inequality.size());
    cut.clear();
    for (i_t k = 0; k < inequality.size(); k++) {
      const i_t j   = inequality.index(k);
      const f_t a_j = inequality.coeff(k);
      if (var_types[j] == variable_type_t::INTEGER) {
        cut.push_back(j, std::floor(a_j));
      } else {
        return -1;
      }
    }
    cut.rhs = std::floor(inequality.rhs);
  } else {
    return generate_strong_cg_cut_helper(
      inequality.vector.i, inequality.vector.x, inequality.rhs, var_types, cut);
  }
  return 0;
}

template <typename i_t, typename f_t>
i_t strong_cg_cut_t<i_t, f_t>::generate_strong_cg_cut_helper(
  const std::vector<i_t>& indicies,
  const std::vector<f_t>& coefficients,
  f_t rhs,
  const std::vector<variable_type_t>& var_types,
  inequality_t<i_t, f_t>& cut)
{
  const bool verbose = false;
  const i_t nz       = indicies.size();
  const f_t f_a_0    = fractional_part(rhs);

  const f_t min_fractional_part = 1e-2;
  if (f_a_0 < min_fractional_part) { return -1; }
  if (f_a_0 > 1 - min_fractional_part) { return -1; }

  // We will try to generat a strong CG cut.
  // Find the unique integer k such that
  // 1/(k+1) <= f(a_0) < 1/k
  const f_t k_upper = 1.0 / f_a_0;
  i_t k             = static_cast<i_t>(std::ceil(k_upper)) - 1;

  const f_t alpha = 1.0 - f_a_0;
  f_t lower       = 1.0 / static_cast<f_t>(k + 1);
  f_t upper       = 1.0 / static_cast<f_t>(k);
  if (verbose) { printf("f_a_0 %e lower %e upper %e alpha %e\n", f_a_0, lower, upper, alpha); }
  if (f_a_0 >= lower && f_a_0 < upper) {
    cut.reserve(nz);
    cut.clear();
    for (i_t q = 0; q < nz; q++) {
      const i_t j   = indicies[q];
      const f_t a_j = coefficients[q];
      if (var_types[j] == variable_type_t::INTEGER) {
        const f_t f_a_j = fractional_part(a_j);
        const f_t tol   = 1e-4;
        if (f_a_j <= f_a_0 + tol) {
          cut.push_back(j, (k + 1.0) * std::floor(a_j));
          if (verbose) { printf("j %d a_j %e f_a_j %e k %d\n", j, a_j, f_a_j, k); }
        } else {
          // Find p such that p <= k * f(a_j) < p + 1
          i_t p = static_cast<i_t>(std::floor(k * f_a_j));
          // If f(a_j) > f(a_0) + p /k (1 - f(a_0)) then we can increase the cofficient by 1
          const f_t rhs_j = f_a_0 + static_cast<f_t>(p) / static_cast<f_t>(k) * alpha;
          const i_t coeff = (k + 1) * static_cast<i_t>(std::floor(a_j)) + p;
          if (f_a_j > rhs_j + tol) {
            cut.push_back(j, static_cast<f_t>(coeff + 1));
          } else {
            cut.push_back(j, static_cast<f_t>(coeff));
          }
        }
      } else {
        return -1;
      }
    }
  } else {
    if (verbose) { printf("Error: k %d lower %e f(a_0) %e upper %e\n", k, lower, f_a_0, upper); }
    return -1;
  }
  cut.rhs = (k + 1.0) * std::floor(rhs);
  if (verbose) {
    printf("Generated strong CG cut: k %d f_a_0 %e cut_rhs %e\n", k, f_a_0, cut.rhs);
    for (i_t q = 0; q < cut.size(); q++) {
      if (cut.vector.x[q] != 0.0) { printf("%.16e x%d ", cut.vector.x[q], cut.vector.i[q]); }
    }
    printf("\n");
    printf("Original inequality rhs %e nz %ld\n", rhs, coefficients.size());
    for (i_t q = 0; q < nz; q++) {
      printf("%e x%d ", coefficients[q], indicies[q]);
    }
    printf("\n");
  }
  return 0;
}

template <typename i_t, typename f_t>
i_t strong_cg_cut_t<i_t, f_t>::generate_strong_cg_cut(
  const lp_problem_t<i_t, f_t>& lp,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  const std::vector<variable_type_t>& var_types,
  const inequality_t<i_t, f_t>& inequality,
  const std::vector<f_t>& xstar,
  inequality_t<i_t, f_t>& cut)
{
#ifdef PRINT_INEQUALITY_INFO
  for (i_t k = 0; k < inequality.i.size(); k++) {
    printf("%e %c%d ",
           inequality.x[k],
           var_types[inequality.i[k]] == variable_type_t::CONTINUOUS ? 'x' : 'y',
           inequality.i[k]);
  }
  printf("CG inequality rhs %e\n", inequality_rhs);
#endif
  // Try to remove continuous variables from the inequality
  // and transform integer variables to be nonnegative

  // Copy the inequality since remove continuous variables will modify it
  inequality_t<i_t, f_t> cg_inequality = inequality;
  i_t status =
    remove_continuous_variables_integers_nonnegative(lp, settings, var_types, cg_inequality);

  if (status != 0) {
    // Try negating the equality and see if that helps
    cg_inequality = inequality;
    cg_inequality.negate();

    status =
      remove_continuous_variables_integers_nonnegative(lp, settings, var_types, cg_inequality);
  }

  if (status == 0) {
    // We have an inequality with no continuous variables

    // Generate a CG cut
    status = generate_strong_cg_cut_integer_only(settings, var_types, cg_inequality, cut);
    if (status != 0) { return -1; }

    // Convert the CG cut back to the original variables
    to_original_integer_variables(lp, cut);

    // Check for violation
    f_t dot = cut.vector.dot(xstar);
    // If the cut is violated we will have: sum_j a_j xstar_j > rhs
    f_t violation                     = dot - cut.rhs;
    const f_t min_violation_threshold = 1e-6;
    if (violation > min_violation_threshold) {
      //  Note that no slacks are currently present. Since slacks are currently treated as
      //  continuous. However, this may change. We may need to substitute out the slacks here

      // The CG cut is in the form: sum_j a_j x_j <= rhs
      // The cut pool wants the cut in the form: sum_j a_j x_j >= rhs
      cut.negate();
      return 0;
    }
  }
  return -1;
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE
template class strong_cg_cut_t<int, double>;
#endif

}  // namespace cuopt::mathematical_optimization::mip
