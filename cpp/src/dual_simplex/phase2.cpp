/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include <dual_simplex/basis_solves.hpp>
#include <dual_simplex/basis_updates.hpp>
#include <dual_simplex/initial_basis.hpp>
#include <dual_simplex/phase1.hpp>
#include <dual_simplex/phase2.hpp>
#include <dual_simplex/solve.hpp>
#include <dual_simplex/sparse_matrix.hpp>
#include <dual_simplex/tic_toc.hpp>

#include <cassert>
#include <cstdio>
#include <iterator>
#include <limits>
#include <list>

namespace cuopt::linear_programming::dual_simplex {

namespace phase2 {

template <typename i_t, typename f_t>
void compute_dual_solution_from_basis(const lp_problem_t<i_t, f_t>& lp,
                                      basis_update_t<i_t, f_t>& ft,
                                      const std::vector<i_t>& basic_list,
                                      const std::vector<i_t>& nonbasic_list,
                                      std::vector<f_t>& y,
                                      std::vector<f_t>& z)
{
  const i_t m = lp.num_rows;
  const i_t n = lp.num_cols;

  y.resize(m);
  std::vector<f_t> cB(m);
  for (i_t k = 0; k < m; ++k) {
    const i_t j = basic_list[k];
    cB[k]       = lp.objective[j];
  }
  ft.b_transpose_solve(cB, y);

  // We want A'y + z = c
  // A = [ B N ]
  // B' y = c_B, z_B = 0
  // N' y + z_N = c_N
  z.resize(n);
  // zN = cN - N'*y
  for (i_t k = 0; k < n - m; k++) {
    const i_t j = nonbasic_list[k];
    // z_j <- c_j
    z[j] = lp.objective[j];

    // z_j <- z_j - A(:, j)'*y
    const i_t col_start = lp.A.col_start[j];
    const i_t col_end   = lp.A.col_start[j + 1];
    f_t dot             = 0.0;
    for (i_t p = col_start; p < col_end; ++p) {
      dot += lp.A.x[p] * y[lp.A.i[p]];
    }
    z[j] -= dot;
  }
  // zB = 0
  for (i_t k = 0; k < m; ++k) {
    z[basic_list[k]] = 0.0;
  }
}

template <typename i_t, typename f_t>
i_t steepest_edge_pricing(const lp_problem_t<i_t, f_t>& lp,
                          const simplex_solver_settings_t<i_t, f_t>& settings,
                          const std::vector<f_t>& x,
                          const std::vector<f_t>& dy_steepest_edge,
                          const std::vector<i_t>& basic_list,
                          i_t& direction,
                          i_t& basic_leaving,
                          f_t& primal_inf,
                          f_t& max_val)
{
  const i_t m          = lp.num_rows;
  max_val              = 0.0;
  i_t leaving_index    = -1;
  const f_t primal_tol = settings.primal_tol;
  primal_inf           = 0;
  i_t num_candidates   = 0;
  for (i_t k = 0; k < m; ++k) {
    const i_t j = basic_list[k];
    if (x[j] < lp.lower[j] - primal_tol) {
      num_candidates++;
      // x_j < l_j => -x_j > -l_j => -x_j + l_j > 0
      const f_t infeas = -x[j] + lp.lower[j];
      primal_inf += infeas;
      const f_t val = (infeas * infeas) / dy_steepest_edge[j];
#ifdef DEBUG_PRICE
      settings.log.printf("price %d x %e lo %e infeas %e val %e se %e\n",
                          j,
                          x[j],
                          lp.lower[j],
                          infeas,
                          val,
                          dy_steepest_edge[j]);
#endif
      assert(val > 0.0);
      if (val > max_val) {
        max_val       = val;
        leaving_index = j;
        basic_leaving = k;
        direction     = 1;
      }
    }
    if (x[j] > lp.upper[j] + primal_tol) {
      num_candidates++;
      // x_j > u_j => x_j - u_j > 0
      const f_t infeas = x[j] - lp.upper[j];
      primal_inf += infeas;
      const f_t val = (infeas * infeas) / dy_steepest_edge[j];
#ifdef DEBUG_PRICE
      settings.log.printf("price %d x %e up %e infeas %e val %e se %e\n",
                          j,
                          x[j],
                          lp.upper[j],
                          infeas,
                          val,
                          dy_steepest_edge[j]);
#endif
      assert(val > 0.0);
      if (val > max_val) {
        max_val       = val;
        leaving_index = j;
        basic_leaving = k;
        direction     = -1;
      }
    }
  }
  return leaving_index;
}

// Maximum infeasibility
template <typename i_t, typename f_t>
i_t phase2_pricing(const lp_problem_t<i_t, f_t>& lp,
                   const simplex_solver_settings_t<i_t, f_t>& settings,
                   const std::vector<f_t>& x,
                   const std::vector<i_t>& basic_list,
                   i_t& direction,
                   i_t& basic_leaving,
                   f_t& primal_inf)
{
  const i_t m          = lp.num_rows;
  f_t max_val          = 0.0;
  i_t leaving_index    = -1;
  const f_t primal_tol = settings.primal_tol / 10;
  primal_inf           = 0;
  for (i_t k = 0; k < m; ++k) {
    const i_t j = basic_list[k];
    if (x[j] < lp.lower[j] - primal_tol) {
      // x_j < l_j => -x_j > -l_j => -x_j + l_j > 0
      const f_t val = -x[j] + lp.lower[j];
      assert(val > 0.0);
      primal_inf += val;
      if (val > max_val) {
        max_val       = val;
        leaving_index = j;
        basic_leaving = k;
        direction     = 1;
      }
    }
    if (x[j] > lp.upper[j] + primal_tol) {
      // x_j > u_j => x_j - u_j > 0
      const f_t val = x[j] - lp.upper[j];
      assert(val > 0.0);
      primal_inf += val;
      if (val > max_val) {
        max_val       = val;
        leaving_index = j;
        basic_leaving = k;
        direction     = -1;
      }
    }
  }
  return leaving_index;
}

template <typename i_t, typename f_t>
f_t first_stage_harris(const lp_problem_t<i_t, f_t>& lp,
                       const std::vector<variable_status_t>& vstatus,
                       const std::vector<i_t>& nonbasic_list,
                       std::vector<f_t>& z,
                       std::vector<f_t>& delta_z)
{
  const i_t n             = lp.num_cols;
  const i_t m             = lp.num_rows;
  constexpr f_t pivot_tol = 1e-7;
  constexpr f_t dual_tol  = 1e-7;
  f_t min_val             = inf;
  f_t step_length         = -inf;

  for (i_t k = 0; k < n - m; ++k) {
    const i_t j = nonbasic_list[k];
    if (vstatus[j] == variable_status_t::NONBASIC_LOWER && delta_z[j] < -pivot_tol) {
      const f_t ratio = (-dual_tol - z[j]) / delta_z[j];
      if (ratio < min_val) {
        min_val     = ratio;
        step_length = ratio;
      }
    }
    if (vstatus[j] == variable_status_t::NONBASIC_UPPER && delta_z[j] > pivot_tol) {
      const f_t ratio = (dual_tol - z[j]) / delta_z[j];
      if (ratio < min_val) {
        min_val     = ratio;
        step_length = ratio;
      }
    }
  }
  return step_length;
}

template <typename i_t, typename f_t>
i_t second_stage_harris(const lp_problem_t<i_t, f_t>& lp,
                        const std::vector<variable_status_t>& vstatus,
                        const std::vector<i_t>& nonbasic_list,
                        const std::vector<f_t>& z,
                        const std::vector<f_t>& delta_z,
                        f_t max_step_length,
                        f_t& step_length,
                        i_t& nonbasic_entering)
{
  const i_t n        = lp.num_cols;
  const i_t m        = lp.num_rows;
  i_t entering_index = -1;
  f_t max_val        = 0;
  for (i_t k = 0; k < n - m; ++k) {
    const i_t j = nonbasic_list[k];
    if (vstatus[j] == variable_status_t::NONBASIC_LOWER && delta_z[j] < 0) {
      // z_j + alpha delta_z_j >= 0, delta_z_j < 0
      // alpha delta_z_j >= -z_j
      // alpha <= -z_j/delta_z_j
      const f_t ratio = -z[j] / delta_z[j];
      if (ratio < max_step_length && std::abs(delta_z[j]) > max_val) {
        step_length       = ratio;
        max_val           = std::abs(delta_z[j]);
        entering_index    = j;
        nonbasic_entering = k;
      }
    } else if (vstatus[j] == variable_status_t::NONBASIC_UPPER && delta_z[j] > 0) {
      // z_j + alpha delta_z_j <= 0, delta_z_j > 0
      // alpha <= -z_j/delta_z_j
      const f_t ratio = -z[j] / delta_z[j];
      if (ratio < max_step_length && std::abs(delta_z[j]) > max_val) {
        step_length       = ratio;
        max_val           = std::abs(delta_z[j]);
        entering_index    = j;
        nonbasic_entering = k;
      }
    }
  }
  return entering_index;
}

template <typename i_t, typename f_t>
i_t phase2_ratio_test(const lp_problem_t<i_t, f_t>& lp,
                      const simplex_solver_settings_t<i_t, f_t>& settings,
                      const std::vector<variable_status_t>& vstatus,
                      const std::vector<i_t>& nonbasic_list,
                      std::vector<f_t>& z,
                      std::vector<f_t>& delta_z,
                      f_t& step_length,
                      i_t& nonbasic_entering)
{
  i_t entering_index  = -1;
  const i_t n         = lp.num_cols;
  const i_t m         = lp.num_rows;
  const f_t pivot_tol = settings.pivot_tol;
  const f_t dual_tol  = settings.dual_tol / 10;
  const f_t zero_tol  = settings.zero_tol;
  f_t min_val         = inf;

  for (i_t k = 0; k < n - m; ++k) {
    const i_t j = nonbasic_list[k];
    if (vstatus[j] == variable_status_t::NONBASIC_FIXED) { continue; }
    if (vstatus[j] == variable_status_t::NONBASIC_LOWER && delta_z[j] < -pivot_tol) {
      const f_t ratio = (-dual_tol - z[j]) / delta_z[j];
      if (ratio < min_val) {
        min_val           = ratio;
        entering_index    = j;
        step_length       = ratio;
        nonbasic_entering = k;
      } else if (ratio < min_val + zero_tol && std::abs(z[j]) > std::abs(z[entering_index])) {
        min_val           = ratio;
        entering_index    = j;
        step_length       = ratio;
        nonbasic_entering = k;
      }
    }
    if (vstatus[j] == variable_status_t::NONBASIC_UPPER && delta_z[j] > pivot_tol) {
      const f_t ratio = (dual_tol - z[j]) / delta_z[j];
      if (ratio < min_val) {
        min_val           = ratio;
        entering_index    = j;
        step_length       = ratio;
        nonbasic_entering = k;
      } else if (ratio < min_val + zero_tol && std::abs(z[j]) > std::abs(z[entering_index])) {
        min_val           = ratio;
        entering_index    = j;
        step_length       = ratio;
        nonbasic_entering = k;
      }
    }
  }
  return entering_index;
}

template <typename i_t, typename f_t>
i_t bound_flipping_ratio_test(const lp_problem_t<i_t, f_t>& lp,
                              const simplex_solver_settings_t<i_t, f_t>& settings,
                              const std::vector<variable_status_t>& vstatus,
                              const std::vector<i_t>& nonbasic_list,
                              const std::vector<f_t>& x,
                              std::vector<f_t>& z,
                              std::vector<f_t>& delta_z,
                              i_t direction,
                              i_t leaving_index,
                              f_t& step_length,
                              i_t& nonbasic_entering)
{
  const i_t n = lp.num_cols;
  const i_t m = lp.num_rows;

  f_t slope = direction == 1 ? (lp.lower[leaving_index] - x[leaving_index])
                             : (x[leaving_index] - lp.upper[leaving_index]);
  assert(slope > 0);

  const f_t pivot_tol         = settings.pivot_tol;
  const f_t relaxed_pivot_tol = settings.pivot_tol;
  const f_t zero_tol          = settings.zero_tol;
  std::list<i_t> q_pos;
  assert(nonbasic_list.size() == n - m);
  for (i_t k = 0; k < n - m; ++k) {
    const i_t j = nonbasic_list[k];
    if (vstatus[j] == variable_status_t::NONBASIC_FIXED) { continue; }
    if (vstatus[j] == variable_status_t::NONBASIC_LOWER && delta_z[j] < -pivot_tol) {
      q_pos.push_back(k);
    } else if (vstatus[j] == variable_status_t::NONBASIC_UPPER && delta_z[j] > pivot_tol) {
      q_pos.push_back(k);
    }
  }
  i_t entering_index = -1;
  step_length        = inf;
  const f_t dual_tol = settings.dual_tol / 10;
  while (q_pos.size() > 0 && slope > 0) {
    // Find the minimum ratio for nonbasic variables in q_pos
    f_t min_val = inf;
    typename std::list<i_t>::iterator q_index;
    i_t candidate = -1;
    for (typename std::list<i_t>::iterator it = q_pos.begin(); it != q_pos.end(); ++it) {
      const i_t k = *it;
      const i_t j = nonbasic_list[k];
      f_t ratio   = inf;
      if (vstatus[j] == variable_status_t::NONBASIC_LOWER && delta_z[j] < -pivot_tol) {
        ratio = (-dual_tol - z[j]) / delta_z[j];
      } else if (vstatus[j] == variable_status_t::NONBASIC_UPPER && delta_z[j] > pivot_tol) {
        ratio = (dual_tol - z[j]) / delta_z[j];
      } else if (min_val != inf) {
        // We've already found something just continue;
      } else if (vstatus[j] == variable_status_t::NONBASIC_LOWER) {
        ratio = (-dual_tol - z[j]) / delta_z[j];
      } else if (vstatus[j] == variable_status_t::NONBASIC_UPPER) {
        ratio = (dual_tol - z[j]) / delta_z[j];
      } else {
        assert(1 == 0);
      }

      ratio = std::max(ratio, 0.0);

      if (ratio < min_val) {
        min_val = ratio;
        q_index = it;  // Save the iterator so we can remove the element it
                       // points to from the q_pos list later (if it corresponds
                       // to a bounded variable)
        candidate = j;
      } else if (ratio < min_val + zero_tol &&
                 std::abs(delta_z[j]) > std::abs(delta_z[candidate])) {
        min_val   = ratio;
        q_index   = it;
        candidate = j;
      }
    }
    step_length       = min_val;  // Save the step length
    nonbasic_entering = *q_index;
    const i_t j = entering_index = nonbasic_list[nonbasic_entering];
    if (lp.lower[j] > -inf && lp.upper[j] < inf && lp.lower[j] != lp.upper[j]) {
      const f_t interval    = lp.upper[j] - lp.lower[j];
      const f_t delta_slope = std::abs(delta_z[j]) * interval;
#ifdef BOUND_FLIP_DEBUG
      if (slope - delta_slope > 0) {
        log.printf(
          "Bound flip %d slope change %e prev slope %e slope %e. curr step "
          "length %e\n",
          j,
          delta_slope,
          slope,
          slope - delta_slope,
          step_length);
      }
#endif
      slope -= delta_slope;
      q_pos.erase(q_index);
    } else {
      // we hit a variable that is not bounded. Exit
      break;
    }
  }
  // step_length, nonbasic_entering, and entering_index are defined after the
  // while loop
  assert(step_length >= 0);

  return entering_index;
}

template <typename i_t, typename f_t>
i_t flip_bounds(const lp_problem_t<i_t, f_t>& lp,
                const simplex_solver_settings_t<i_t, f_t>& settings,
                const std::vector<f_t>& objective,
                const std::vector<f_t>& z,
                const std::vector<i_t>& nonbasic_list,
                i_t entering_index,
                std::vector<variable_status_t>& vstatus,
                std::vector<f_t>& delta_x,
                std::vector<f_t>& atilde)
{
  f_t delta_obj = 0;
  for (i_t j : nonbasic_list) {
    if (j == entering_index) { continue; }
    const bool bounded =
      (lp.lower[j] > -inf) && (lp.upper[j] < inf) && (lp.lower[j] != lp.upper[j]);
    if (!bounded) { continue; }
    // x_j is now a nonbasic bounded variable that will not enter the basis this
    // iteration
    const f_t dual_tol =
      settings.dual_tol;  // lower to 1e-7 or less will cause 25fv47 and d2q06c to cycle
    if (vstatus[j] == variable_status_t::NONBASIC_LOWER && z[j] < -dual_tol) {
      const f_t delta = lp.upper[j] - lp.lower[j];
      scatter_dense(lp.A, j, -delta, atilde);
      delta_obj += delta * objective[j];
      delta_x[j] += delta;
      vstatus[j] = variable_status_t::NONBASIC_UPPER;
#ifdef BOUND_FLIP_DEBUG
      settings.log.printf(
        "Flipping nonbasic %d from lo %e to up %e. z %e\n", j, lp.lower[j], lp.upper[j], z[j]);
#endif
    } else if (vstatus[j] == variable_status_t::NONBASIC_UPPER && z[j] > dual_tol) {
      const f_t delta = lp.lower[j] - lp.upper[j];
      scatter_dense(lp.A, j, -delta, atilde);
      delta_obj += delta * objective[j];
      delta_x[j] += delta;
      vstatus[j] = variable_status_t::NONBASIC_LOWER;
#ifdef BOUND_FLIP_DEBUG
      settings.log.printf(
        "Flipping nonbasic %d from up %e to lo %e. z %e\n", j, lp.upper[j], lp.lower[j], z[j]);
#endif
    }
  }
  return 0;
}

template <typename i_t, typename f_t>
i_t initialize_steepest_edge_norms(const simplex_solver_settings_t<i_t, f_t>& settings,
                                   const f_t start_time,
                                   const std::vector<i_t>& basic_list,
                                   const basis_update_t<i_t, f_t>& ft,
                                   std::vector<f_t>& delta_y_steepest_edge)
{
  // TODO: Skip this initialization when starting from a slack basis
  //       Or skip individual columns corresponding to slack variables
  const i_t m  = basic_list.size();
  f_t last_log = tic();
  for (i_t k = 0; k < m; ++k) {
    std::vector<f_t> ei(m);
    std::vector<f_t> dy(m);
    const i_t j = basic_list[k];
    ei[k]       = -1.0;
    ft.b_transpose_solve(ei, dy);
    ei[k]          = 0.0;
    const f_t init = vector_norm2_squared<i_t, f_t>(dy);
    assert(init > 0);
    delta_y_steepest_edge[j] = init;

    f_t now            = toc(start_time);
    f_t time_since_log = toc(last_log);
    if (time_since_log > 10) {
      last_log = tic();
      settings.log.printf("Initialized %d of %d steepest edge norms in %.2fs\n", k, m, now);
    }
    if (toc(start_time) > settings.time_limit) { return -1; }
    if (settings.concurrent_halt != nullptr &&
        settings.concurrent_halt->load(std::memory_order_acquire) == 1) {
      return -1;
    }
  }
  return 0;
}

template <typename i_t, typename f_t>
i_t update_steepest_edge_norms(const simplex_solver_settings_t<i_t, f_t>& settings,
                               const std::vector<i_t>& basic_list,
                               const basis_update_t<i_t, f_t>& ft,
                               i_t direction,
                               const std::vector<f_t>& delta_y,
                               const std::vector<f_t>& scaled_delta_xB,
                               i_t basic_leaving_index,
                               i_t entering_index,
                               std::vector<f_t>& delta_y_steepest_edge)
{
  i_t m = delta_y.size();
  std::vector<f_t> v(m);
  // B^T delta_y = - direction * e_basic_leaving_index
  // We want B v =  - B^{-T} e_basic_leaving_index
  ft.b_solve(delta_y, v);
  // if direction = -1 we need to scale v
  if (direction == -1) {
    for (i_t k = 0; k < m; ++k) {
      v[k] *= -1;
    }
  }
  const f_t dy_norm_squared      = vector_norm2_squared<i_t, f_t>(delta_y);
  const i_t leaving_index        = basic_list[basic_leaving_index];
  const f_t prev_dy_norm_squared = delta_y_steepest_edge[leaving_index];
#ifdef STEEPEST_EDGE_DEBUG
  const f_t err = std::abs(dy_norm_squared - prev_dy_norm_squared) / (1.0 + dy_norm_squared);
  if (err > 1e-3) {
    settings.log.printf("i %d j %d leaving norm error %e computed %e previous estimate %e\n",
                        basic_leaving_index,
                        leaving_index,
                        err,
                        dy_norm_squared,
                        prev_dy_norm_squared);
  }
#endif

  // B*w = A(:, leaving_index)
  // B*scaled_delta_xB = -A(:, leaving_index) so w = -scaled_delta_xB
  const f_t wr = -scaled_delta_xB[basic_leaving_index];
  if (wr == 0) { return -1; }
  const f_t omegar = dy_norm_squared / (wr * wr);

  for (i_t k = 0; k < m; ++k) {
    const i_t j = basic_list[k];
    if (k == basic_leaving_index) {
      const f_t w_squared      = scaled_delta_xB[k] * scaled_delta_xB[k];
      delta_y_steepest_edge[j] = (1.0 / w_squared) * dy_norm_squared;
    } else {
      const f_t wk = -scaled_delta_xB[k];
      f_t new_val  = delta_y_steepest_edge[j] + wk * (2.0 * v[k] / wr + wk * omegar);
      new_val      = std::max(new_val, 1e-4);
#ifdef STEEPEST_EDGE_DEBUG
      if (!(new_val >= 0)) {
        settings.log.printf("new val %e\n", new_val);
        settings.log.printf("k %d j %d norm old %e wk %e vk %e wr %e omegar %e\n",
                            k,
                            j,
                            delta_y_steepest_edge[j],
                            wk,
                            v[k],
                            wr,
                            omegar);
      }
#endif
      assert(new_val >= 0.0);
      delta_y_steepest_edge[j] = new_val;
    }
  }

  return 0;
}

// Compute steepest edge info for entering variable
template <typename i_t, typename f_t>
i_t compute_steepest_edge_norm_entering(const simplex_solver_settings_t<i_t, f_t>& setttings,
                                        i_t m,
                                        const basis_update_t<i_t, f_t>& ft,
                                        i_t basic_leaving_index,
                                        i_t entering_index,
                                        std::vector<f_t>& steepest_edge_norms)
{
  std::vector<f_t> es(m);
  es[basic_leaving_index] = -1.0;
  std::vector<f_t> delta_ys(m);
  ft.b_transpose_solve(es, delta_ys);
  steepest_edge_norms[entering_index] = vector_norm2_squared<i_t, f_t>(delta_ys);
#ifdef STEEPEST_EDGE_DEBUG
  settings.log.printf("Steepest edge norm %e for entering j %d at i %d\n",
                      steepest_edge_norms[entering_index],
                      entering_index,
                      basic_leaving_index);
#endif
  return 0;
}

template <typename i_t, typename f_t>
i_t check_steepest_edge_norms(const simplex_solver_settings_t<i_t, f_t>& settings,
                              const std::vector<i_t>& basic_list,
                              const basis_update_t<i_t, f_t>& ft,
                              const std::vector<f_t>& delta_y_steepest_edge)
{
  const i_t m = basic_list.size();
  for (i_t k = 0; k < m; ++k) {
    const i_t j = basic_list[k];
    std::vector<f_t> ei(m);
    ei[k] = -1.0;
    std::vector<f_t> delta_yi(m);
    ft.b_transpose_solve(ei, delta_yi);
    const f_t computed_norm = vector_norm2_squared(delta_yi);
    const f_t updated_norm  = delta_y_steepest_edge[j];
    const f_t err = std::abs(computed_norm - updated_norm) / (1 + std::abs(computed_norm));
    if (err > 1e-3) {
      settings.log.printf(
        "i %d j %d computed %e updated %e err %e\n", k, j, computed_norm, updated_norm, err);
    }
  }
  return 0;
}

template <typename i_t, typename f_t>
i_t compute_perturbation(const lp_problem_t<i_t, f_t>& lp,
                         const simplex_solver_settings_t<i_t, f_t>& settings,
                         std::vector<f_t>& z,
                         std::vector<f_t>& objective,
                         f_t& sum_perturb)
{
  const i_t n         = lp.num_cols;
  const i_t m         = lp.num_rows;
  const f_t tight_tol = settings.tight_tol;
  i_t num_perturb     = 0;
  sum_perturb         = 0.0;
  for (i_t j = 0; j < n; ++j) {
    if (lp.upper[j] == inf && lp.lower[j] > -inf && z[j] < -tight_tol) {
      const f_t violation = -z[j];
      z[j] += violation;  // z[j] <- 0
      objective[j] += violation;
      num_perturb++;
      sum_perturb += violation;
#ifdef PERTURBATION_DEBUG
      if (violation > 1e-1) {
        settings.log.printf(
          "perturbation: violation %e j %d lower %e\n", violation, j, lp.lower[j]);
      }
#endif
    } else if (lp.lower[j] == -inf && lp.upper[j] < inf && z[j] > tight_tol) {
      const f_t violation = z[j];
      z[j] -= violation;  // z[j] <- 0
      objective[j] -= violation;
      num_perturb++;
      sum_perturb += violation;
#ifdef PERTURBATION_DEWBUG
      if (violation > 1e-1) {
        settings.log.printf(
          "perturbation: violation %e j %d upper %e\n", violation, j, lp.upper[j]);
      }
#endif
    }
  }
#ifdef PERTURBATION_DEBUG
  if (num_perturb > 0) {
    settings.log.printf("Perturbed %d dual variables by %e\n", num_perturb, sum_perturb);
  }
#endif
  return 0;
}

template <typename i_t, typename f_t>
f_t dual_infeasibility(const lp_problem_t<i_t, f_t>& lp,
                       const simplex_solver_settings_t<i_t, f_t>& settings,
                       const std::vector<variable_status_t>& vstatus,
                       const std::vector<f_t>& z,
                       f_t tight_tol,
                       f_t dual_tol)
{
  const i_t n             = lp.num_cols;
  const i_t m             = lp.num_rows;
  i_t num_infeasible      = 0;
  f_t sum_infeasible      = 0.0;
  i_t lower_bound_inf     = 0;
  i_t upper_bound_inf     = 0;
  i_t free_inf            = 0;
  i_t non_basic_lower_inf = 0;
  i_t non_basic_upper_inf = 0;

  for (i_t j = 0; j < n; ++j) {
    if (vstatus[j] == variable_status_t::NONBASIC_FIXED) { continue; }
    if (lp.upper[j] == inf && lp.lower[j] > -inf && z[j] < -tight_tol) {
      // -inf < l_j <= x_j < inf, so need z_j > 0 to be feasible
      num_infeasible++;
      sum_infeasible += std::abs(z[j]);
      lower_bound_inf++;
      settings.log.debug("lower_bound_inf %d lower %e upper %e z %e vstatus %d\n",
                         j,
                         lp.lower[j],
                         lp.upper[j],
                         z[j],
                         static_cast<int>(vstatus[j]));
    } else if (lp.lower[j] == -inf && lp.upper[j] < inf && z[j] > tight_tol) {
      // -inf < x_j <= u_j < inf, so need z_j < 0 to be feasible
      num_infeasible++;
      sum_infeasible += std::abs(z[j]);
      upper_bound_inf++;
      settings.log.debug("upper_bound_inf %d upper %e lower %e z %e vstatus %d\n",
                         j,
                         lp.upper[j],
                         lp.lower[j],
                         z[j],
                         static_cast<int>(vstatus[j]));
    } else if (lp.lower[j] == -inf && lp.upper[j] == inf && z[j] > tight_tol) {
      // -inf < x_j < inf, so need z_j = 0 to be feasible
      num_infeasible++;
      sum_infeasible += std::abs(z[j]);
      free_inf++;
    } else if (lp.lower[j] == -inf && lp.upper[j] == inf && z[j] < -tight_tol) {
      // -inf < x_j < inf, so need z_j = 0 to be feasible
      num_infeasible++;
      sum_infeasible += std::abs(z[j]);
      free_inf++;
    } else if (vstatus[j] == variable_status_t::NONBASIC_LOWER && z[j] < -dual_tol) {
      num_infeasible++;
      sum_infeasible += std::abs(z[j]);
      non_basic_lower_inf++;
    } else if (vstatus[j] == variable_status_t::NONBASIC_UPPER && z[j] > dual_tol) {
      num_infeasible++;
      sum_infeasible += std::abs(z[j]);
      non_basic_upper_inf++;
    }
  }

#ifdef DUAL_INFEASIBILE_DEBUG
  if (num_infeasible > 0) {
    settings.log.printf(
      "Infeasibilities %e: lower %d upper %d free %d nonbasic lower %d "
      "nonbasic upper %d\n",
      sum_infeasible,
      lower_bound_inf,
      upper_bound_inf,
      free_inf,
      non_basic_lower_inf,
      non_basic_upper_inf);
    settings.log.printf("num infeasible %d\n", num_infeasible);
  }
#endif
  return sum_infeasible;
}

template <typename i_t, typename f_t>
f_t primal_infeasibility(const lp_problem_t<i_t, f_t>& lp,
                         const simplex_solver_settings_t<i_t, f_t>& settings,
                         const std::vector<variable_status_t>& vstatus,
                         const std::vector<f_t>& x)
{
  const i_t n    = lp.num_cols;
  f_t primal_inf = 0;
  for (i_t j = 0; j < n; ++j) {
    if (x[j] < lp.lower[j]) {
      // x_j < l_j => -x_j > -l_j => -x_j + l_j > 0
      const f_t infeas = -x[j] + lp.lower[j];
      primal_inf += infeas;
#ifdef PRIMAL_INFEASIBLE_DEBUG
      if (infeas > settings.primal_tol) {
        settings.log.printf("x %d infeas %e lo %e val %e up %e vstatus %d\n",
                            j,
                            infeas,
                            lp.lower[j],
                            x[j],
                            lp.upper[j],
                            static_cast<int>(vstatus[j]));
      }
#endif
    }
    if (x[j] > lp.upper[j]) {
      // x_j > u_j => x_j - u_j > 0
      const f_t infeas = x[j] - lp.upper[j];
      primal_inf += infeas;
#ifdef PRIMAL_INFEASIBLE_DEBUG
      if (infeas > settings.primal_tol) {
        settings.log.printf("x %d infeas %e lo %e val %e up %e vstatus %d\n",
                            j,
                            infeas,
                            lp.lower[j],
                            x[j],
                            lp.upper[j],
                            static_cast<int>(vstatus[j]));
      }
#endif
    }
  }
  return primal_inf;
}

template <typename i_t, typename f_t>
void bound_info(const lp_problem_t<i_t, f_t>& lp,
                const simplex_solver_settings_t<i_t, f_t>& settings)
{
  i_t n                 = lp.num_cols;
  i_t num_free          = 0;
  i_t num_boxed         = 0;
  i_t num_lower_bounded = 0;
  i_t num_upper_bounded = 0;
  i_t num_fixed         = 0;
  for (i_t j = 0; j < n; ++j) {
    if (lp.lower[j] == lp.upper[j]) {
      num_fixed++;
    } else if (lp.lower[j] > -inf && lp.upper[j] < inf) {
      num_boxed++;
    } else if (lp.lower[j] > -inf && lp.upper[j] == inf) {
      num_lower_bounded++;
    } else if (lp.lower[j] == -inf && lp.upper[j] < inf) {
      num_upper_bounded++;
    } else if (lp.lower[j] == -inf && lp.upper[j] == inf) {
      num_free++;
    }
  }
  settings.log.debug("Fixed %d Free %d Boxed %d Lower %d Upper %d\n",
                     num_fixed,
                     num_free,
                     num_boxed,
                     num_lower_bounded,
                     num_upper_bounded);
}

template <typename i_t, typename f_t>
void set_primal_variables_on_bounds(const lp_problem_t<i_t, f_t>& lp,
                                    const simplex_solver_settings_t<i_t, f_t>& settings,
                                    const std::vector<f_t>& z,
                                    std::vector<variable_status_t>& vstatus,
                                    std::vector<f_t>& x)
{
  const i_t n = lp.num_cols;
  for (i_t j = 0; j < n; ++j) {
    // We set z_j = 0 for basic variables
    // But we explicitally skip setting basic variables here
    if (vstatus[j] == variable_status_t::BASIC) { continue; }
    // We will flip the status of variables between nonbasic lower and nonbasic
    // upper here to improve dual feasibility
    const f_t fixed_tolerance = settings.fixed_tol;
    if (std::abs(lp.lower[j] - lp.upper[j]) < fixed_tolerance) {
      if (vstatus[j] != variable_status_t::NONBASIC_FIXED) {
        settings.log.debug("Setting fixed variable %d to %e (current %e). vstatus %d\n",
                           j,
                           lp.lower[j],
                           x[j],
                           static_cast<int>(vstatus[j]));
      }
      x[j]       = lp.lower[j];
      vstatus[j] = variable_status_t::NONBASIC_FIXED;
    } else if (z[j] == 0 && lp.lower[j] > -inf && vstatus[j] == variable_status_t::NONBASIC_LOWER) {
      x[j] = lp.lower[j];
    } else if (z[j] == 0 && lp.upper[j] < inf && vstatus[j] == variable_status_t::NONBASIC_UPPER) {
      x[j] = lp.upper[j];
    } else if (z[j] >= 0 && lp.lower[j] > -inf) {
      if (vstatus[j] != variable_status_t::NONBASIC_LOWER) {
        settings.log.debug(
          "Setting nonbasic lower variable (zj %e) %d to %e (current %e). vstatus %d\n",
          z[j],
          j,
          lp.lower[j],
          x[j],
          static_cast<int>(vstatus[j]));
      }
      x[j]       = lp.lower[j];
      vstatus[j] = variable_status_t::NONBASIC_LOWER;
    } else if (z[j] <= 0 && lp.upper[j] < inf) {
      if (vstatus[j] != variable_status_t::NONBASIC_UPPER) {
        settings.log.debug(
          "Setting nonbasic upper variable (zj %e) %d to %e (current %e). vstatus %d\n",
          z[j],
          j,
          lp.upper[j],
          x[j],
          static_cast<int>(vstatus[j]));
      }
      x[j]       = lp.upper[j];
      vstatus[j] = variable_status_t::NONBASIC_UPPER;
    } else if (lp.upper[j] == inf && lp.lower[j] > -inf && z[j] < 0) {
      // dual infeasible
      if (vstatus[j] != variable_status_t::NONBASIC_LOWER) {
        settings.log.debug("Setting nonbasic lower variable %d to %e (current %e). vstatus %d\n",
                           j,
                           lp.lower[j],
                           x[j],
                           static_cast<int>(vstatus[j]));
      }
      x[j]       = lp.lower[j];
      vstatus[j] = variable_status_t::NONBASIC_LOWER;
    } else if (lp.lower[j] == -inf && lp.upper[j] < inf && z[j] > 0) {
      // dual infeasible
      if (vstatus[j] != variable_status_t::NONBASIC_UPPER) {
        settings.log.debug("Setting nonbasic upper variable %d to %e (current %e). vstatus %d\n",
                           j,
                           lp.upper[j],
                           x[j],
                           static_cast<int>(vstatus[j]));
      }
      x[j]       = lp.upper[j];
      vstatus[j] = variable_status_t::NONBASIC_UPPER;
    } else if (lp.lower[j] == -inf && lp.upper[j] == inf) {
      x[j] = 0;  // Set nonbasic free variables to 0 this overwrites previous lines
      if (vstatus[j] != variable_status_t::NONBASIC_FREE) {
        settings.log.debug(
          "Setting free variable %d to %e. vstatus %d\n", j, 0, static_cast<int>(vstatus[j]));
      }
      vstatus[j] = variable_status_t::NONBASIC_FREE;
      settings.log.printf("Setting free variable %d as nonbasic at 0\n", j);
    } else {
      assert(1 == 0);
    }
  }
}

template <typename i_t, typename f_t>
void prepare_optimality(const lp_problem_t<i_t, f_t>& lp,
                        const simplex_solver_settings_t<i_t, f_t>& settings,
                        basis_update_t<i_t, f_t>& ft,
                        const std::vector<f_t>& objective,
                        const std::vector<i_t>& basic_list,
                        const std::vector<i_t>& nonbasic_list,
                        const std::vector<variable_status_t>& vstatus,
                        int phase,
                        f_t start_time,
                        f_t max_val,
                        i_t iter,
                        const std::vector<f_t>& x,
                        std::vector<f_t>& y,
                        std::vector<f_t>& z,
                        lp_solution_t<i_t, f_t>& sol)
{
  const i_t m = lp.num_rows;
  const i_t n = lp.num_cols;

  sol.objective      = compute_objective(lp, sol.x);
  sol.user_objective = compute_user_objective(lp, sol.objective);
  f_t perturbation   = 0.0;
  for (i_t j = 0; j < n; ++j) {
    perturbation += std::abs(lp.objective[j] - objective[j]);
  }

  if (perturbation > 1e-6 && phase == 2) {
    // Try to remove perturbation
    std::vector<f_t> unperturbed_y(m);
    std::vector<f_t> unperturbed_z(n);
    phase2::compute_dual_solution_from_basis(
      lp, ft, basic_list, nonbasic_list, unperturbed_y, unperturbed_z);
    {
      const f_t dual_infeas = phase2::dual_infeasibility(
        lp, settings, vstatus, unperturbed_z, settings.tight_tol, settings.dual_tol);
      if (dual_infeas <= settings.dual_tol) {
        settings.log.printf("Removed perturbation of %.2e.\n", perturbation);
        z            = unperturbed_z;
        y            = unperturbed_y;
        perturbation = 0.0;
      }
    }
  }

  const f_t dual_infeas   = phase2::dual_infeasibility(lp, settings, vstatus, z, 0.0, 0.0);
  const f_t primal_infeas = phase2::primal_infeasibility(lp, settings, vstatus, x);
  if (phase == 1 && iter > 0) {
    settings.log.printf("Dual phase I complete. Iterations %d. Time %.2f\n", iter, toc(start_time));
  }
  if (phase == 2) {
    if (!settings.inside_mip) {
      settings.log.printf("\n");
      settings.log.printf(
        "Optimal solution found in %d iterations and %.2fs\n", iter, toc(start_time));
      settings.log.printf("Objective %+.8e\n", sol.user_objective);
      settings.log.printf("\n");
      settings.log.printf("Primal infeasibility (abs): %.2e\n", primal_infeas);
      settings.log.printf("Dual infeasibility (abs):   %.2e\n", dual_infeas);
      settings.log.printf("Perturbation:               %.2e\n", perturbation);
      settings.log.printf("Max steepest edge norm:     %.2e\n", max_val);
    } else {
      settings.log.printf("\n");
      settings.log.printf(
        "Root relaxation solution found in %d iterations and %.2fs\n", iter, toc(start_time));
      settings.log.printf("Root relaxation objective %+.8e\n", sol.user_objective);
      settings.log.printf("\n");
    }
  }
}

}  // namespace phase2

template <typename i_t, typename f_t>
dual::status_t dual_phase2(i_t phase,
                           i_t slack_basis,
                           f_t start_time,
                           const lp_problem_t<i_t, f_t>& lp,
                           const simplex_solver_settings_t<i_t, f_t>& settings,
                           std::vector<variable_status_t>& vstatus,
                           lp_solution_t<i_t, f_t>& sol,
                           i_t& iter,
                           std::vector<f_t>& delta_y_steepest_edge)
{
  const i_t m = lp.num_rows;
  const i_t n = lp.num_cols;
  assert(m <= n);
  assert(vstatus.size() == n);
  assert(lp.A.m == m);
  assert(lp.A.n == n);
  assert(lp.objective.size() == n);
  assert(lp.lower.size() == n);
  assert(lp.upper.size() == n);
  assert(lp.rhs.size() == m);
  std::vector<i_t> basic_list(m);
  std::vector<i_t> nonbasic_list;
  std::vector<i_t> superbasic_list;

  std::vector<f_t>& x = sol.x;
  std::vector<f_t>& y = sol.y;
  std::vector<f_t>& z = sol.z;

  dual::status_t status = dual::status_t::UNSET;

  // Perturbed objective
  std::vector<f_t> objective = lp.objective;

  settings.log.printf("Dual Simplex Phase %d\n", phase);
  std::vector<variable_status_t> vstatus_old = vstatus;
  std::vector<f_t> z_old                     = z;

  phase2::bound_info(lp, settings);
  get_basis_from_vstatus(m, vstatus, basic_list, nonbasic_list, superbasic_list);
  assert(superbasic_list.size() == 0);
  assert(nonbasic_list.size() == n - m);

  // Compute L*U = A(p, basic_list)
  csc_matrix_t<i_t, f_t> L(m, m, 1);
  csc_matrix_t<i_t, f_t> U(m, m, 1);
  std::vector<i_t> pinv(m);
  std::vector<i_t> p;
  std::vector<i_t> q;
  std::vector<i_t> deficient;
  std::vector<i_t> slacks_needed;

  if (factorize_basis(lp.A, settings, basic_list, L, U, p, pinv, q, deficient, slacks_needed) ==
      -1) {
    settings.log.debug("Initial factorization failed\n");
    basis_repair(lp.A, settings, deficient, slacks_needed, basic_list, nonbasic_list, vstatus);
    if (factorize_basis(lp.A, settings, basic_list, L, U, p, pinv, q, deficient, slacks_needed) ==
        -1) {
      return dual::status_t::NUMERICAL;
    }
    settings.log.printf("Basis repaired\n");
  }
  if (toc(start_time) > settings.time_limit) { return dual::status_t::TIME_LIMIT; }
  assert(q.size() == m);
  reorder_basic_list(q, basic_list);
  basis_update_t ft(L, U, p);

  std::vector<f_t> c_basic(m);
  for (i_t k = 0; k < m; ++k) {
    const i_t j = basic_list[k];
    c_basic[k]  = objective[j];
  }

  // Solve B'*y = cB
  ft.b_transpose_solve(c_basic, y);
  if (toc(start_time) > settings.time_limit) { return dual::status_t::TIME_LIMIT; }
  constexpr bool print_norms = false;
  if (print_norms) {
    settings.log.printf(
      "|| y || %e || cB || %e\n", vector_norm_inf<i_t, f_t>(y), vector_norm_inf<i_t, f_t>(c_basic));
  }

  // zN = cN - N'*y
  for (i_t k = 0; k < n - m; k++) {
    const i_t j = nonbasic_list[k];
    // z_j <- c_j
    z[j] = objective[j];

    // z_j <- z_j - A(:, j)'*y
    const i_t col_start = lp.A.col_start[j];
    const i_t col_end   = lp.A.col_start[j + 1];
    f_t dot             = 0.0;
    for (i_t p = col_start; p < col_end; ++p) {
      dot += lp.A.x[p] * y[lp.A.i[p]];
    }
    z[j] -= dot;
  }
  // zB = 0
  for (i_t k = 0; k < m; ++k) {
    z[basic_list[k]] = 0.0;
  }
  if (print_norms) { settings.log.printf("|| z || %e\n", vector_norm_inf<i_t, f_t>(z)); }

#ifdef COMPUTE_DUAL_RESIDUAL
  // || A'*y + z  - c||_inf
  std::vector<f_t> dual_res1 = z;
  for (i_t j = 0; j < n; ++j) {
    dual_res1[j] -= objective[j];
  }
  matrix_transpose_vector_multiply(lp.A, 1.0, y, 1.0, dual_res1);
  f_t dual_res_norm = vector_norm_inf<i_t, f_t>(dual_res1);
  if (1 || dual_res_norm > settings.tight_tol) {
    settings.log.printf("|| A'*y + z - c || %e\n", dual_res_norm);
  }
  assert(dual_res_norm < 1e-3);
#endif

  phase2::set_primal_variables_on_bounds(lp, settings, z, vstatus, x);

#ifdef PRINT_VSTATUS_CHANGES
  i_t num_vstatus_changes = 0;
  i_t num_z_changes       = 0;
  for (i_t j = 0; j < n; ++j) {
    if (vstatus[j] != vstatus_old[j]) { num_vstatus_changes++; }
    if (std::abs(z[j] - z_old[j]) > 1e-6) { num_z_changes++; }
  }

  printf("Number of vstatus changes %d\n", num_vstatus_changes);
  printf("Number of z changes %d\n", num_z_changes);
#endif

  const f_t init_dual_inf =
    phase2::dual_infeasibility(lp, settings, vstatus, z, settings.tight_tol, settings.dual_tol);
  if (init_dual_inf > settings.dual_tol) {
    settings.log.printf("Initial dual infeasibility %e\n", init_dual_inf);
  }

  for (i_t j = 0; j < n; ++j) {
    if (lp.lower[j] == -inf && lp.upper[j] == inf && vstatus[j] != variable_status_t::BASIC) {
      settings.log.printf("Free variable %d vstatus %d\n", j, vstatus[j]);
    }
  }

  std::vector<f_t> rhs = lp.rhs;
  // rhs = b - sum_{j : x_j = l_j} A(:, j) l(j) - sum_{j : x_j = u_j} A(:, j) *
  // u(j)
  for (i_t k = 0; k < n - m; ++k) {
    const i_t j         = nonbasic_list[k];
    const i_t col_start = lp.A.col_start[j];
    const i_t col_end   = lp.A.col_start[j + 1];
    const f_t xj        = x[j];
    if (std::abs(xj) < settings.tight_tol * 10) continue;
    for (i_t p = col_start; p < col_end; ++p) {
      rhs[lp.A.i[p]] -= xj * lp.A.x[p];
    }
  }

  std::vector<f_t> xB(m);
  ft.b_solve(rhs, xB);
  if (toc(start_time) > settings.time_limit) { return dual::status_t::TIME_LIMIT; }

  for (i_t k = 0; k < m; ++k) {
    const i_t j = basic_list[k];
    x[j]        = xB[k];
  }
  if (print_norms) { settings.log.printf("|| x || %e\n", vector_norm2<i_t, f_t>(x)); }

#ifdef COMPUTE_PRIMAL_RESIDUAL
  std::vector<f_t> residual = lp.rhs;
  matrix_vector_multiply(lp.A, 1.0, x, -1.0, residual);
  f_t primal_residual = vector_norm_inf<i_t, f_t>(residual);
  if (primal_residual > settings.primal_tol) {
    settings.log.printf("|| A*x - b || %e\n", primal_residual);
  }
#endif

  if (delta_y_steepest_edge.size() == 0) {
    delta_y_steepest_edge.resize(n);
    if (slack_basis) {
      for (i_t k = 0; k < m; ++k) {
        const i_t j              = basic_list[k];
        delta_y_steepest_edge[j] = 1.0;
      }
      for (i_t k = 0; k < n - m; ++k) {
        const i_t j              = nonbasic_list[k];
        delta_y_steepest_edge[j] = 1e-4;
      }
    } else {
      std::fill(delta_y_steepest_edge.begin(), delta_y_steepest_edge.end(), -1);
      if (phase2::initialize_steepest_edge_norms(
            settings, start_time, basic_list, ft, delta_y_steepest_edge) == -1) {
        return dual::status_t::TIME_LIMIT;
      }
    }
  } else {
    settings.log.printf("using exisiting steepest edge %e\n",
                        vector_norm2<i_t, f_t>(delta_y_steepest_edge));
  }

  if (phase == 2) { settings.log.printf(" Iter     Objective   Primal Infeas  Perturb  Time\n"); }

  const i_t iter_limit = settings.iteration_limit;
  std::vector<f_t> delta_y(m);
  std::vector<f_t> delta_z(n);
  std::vector<f_t> delta_x(n);
  const i_t start_iter = iter;
  while (iter < iter_limit) {
    // Pricing
    i_t direction;
    i_t basic_leaving_index;
    f_t primal_infeasibility;
    i_t leaving_index = -1;
    f_t max_val;
    if (settings.use_steepest_edge_pricing) {
      leaving_index = phase2::steepest_edge_pricing(lp,
                                                    settings,
                                                    x,
                                                    delta_y_steepest_edge,
                                                    basic_list,
                                                    direction,
                                                    basic_leaving_index,
                                                    primal_infeasibility,
                                                    max_val);
    } else {
      // Max infeasibility pricing
      leaving_index = phase2::phase2_pricing(
        lp, settings, x, basic_list, direction, basic_leaving_index, primal_infeasibility);
    }
    if (leaving_index == -1) {
      phase2::prepare_optimality(lp,
                                 settings,
                                 ft,
                                 objective,
                                 basic_list,
                                 nonbasic_list,
                                 vstatus,
                                 phase,
                                 start_time,
                                 max_val,
                                 iter,
                                 x,
                                 y,
                                 z,
                                 sol);
      status = dual::status_t::OPTIMAL;
      break;
    }

    // BTran
    // TODO: replace with sparse solve.
    std::vector<f_t> ei(m, 0.0);
    std::vector<f_t> delta_y(m);
    ei[basic_leaving_index] = -direction;
    // BT*delta_y = -delta_zB = -sigma*ei
    ft.b_transpose_solve(ei, delta_y);

    const f_t steepest_edge_norm_check = vector_norm2_squared<i_t, f_t>(delta_y);
    if (delta_y_steepest_edge[leaving_index] <
        settings.steepest_edge_ratio * steepest_edge_norm_check) {
      constexpr bool verbose = false;
      if (verbose) {
        settings.log.printf(
          "iteration restart due to steepest edge. Leaving %d. Actual %.2e "
          "from update %.2e\n",
          leaving_index,
          steepest_edge_norm_check,
          delta_y_steepest_edge[leaving_index]);
      }
      delta_y_steepest_edge[leaving_index] = steepest_edge_norm_check;
      continue;
    }

#ifdef COMPUTE_BTRANSPOSE_RESIDUAL
    {
      std::vector<f_t> res(m);
      b_transpose_multiply(lp, basic_list, delta_y, res);
      for (Int k = 0; k < m; k++) {
        const f_t err = std::abs(res[k] - ei[k]);
        if (err > 1e-4) { settings.log.printf("BT err %d %e\n", k, err); }
        assert(err < 1e-4);
      }
    }
#endif

    // delta_zB = sigma*ei
    for (i_t k = 0; k < m; k++) {
      const i_t j = basic_list[k];
      delta_z[j]  = 0;
    }
    delta_z[leaving_index] = direction;
    // delta_zN = -N'*delta_y
    for (i_t k = 0; k < n - m; k++) {
      const i_t j = nonbasic_list[k];
      // z_j <- -A(:, j)'*delta_y
      const i_t col_start = lp.A.col_start[j];
      const i_t col_end   = lp.A.col_start[j + 1];
      f_t dot             = 0.0;
      for (i_t p = col_start; p < col_end; ++p) {
        dot += lp.A.x[p] * delta_y[lp.A.i[p]];
      }
      delta_z[j] = -dot;
    }

#ifdef COMPUTE_DUAL_RESIDUAL
    std::vector<f_t> dual_residual = delta_z;
    // || A'*delta_y + delta_z ||_inf
    matrix_transpose_vector_multiply(lp.A, 1.0, delta_y, 1.0, dual_residual);
    f_t dual_residual_norm = vector_norm_inf<i_t, f_t>(dual_residual);
    settings.log.printf("|| A'*dy - dz || %e\n", dual_residual_norm);
#endif

    // Ratio test
    f_t step_length;
    i_t entering_index          = -1;
    i_t nonbasic_entering_index = -1;
    const bool harris_ratio     = settings.use_harris_ratio;
    const bool bound_flip_ratio = settings.use_bound_flip_ratio;
    if (harris_ratio) {
      f_t max_step_length = phase2::first_stage_harris(lp, vstatus, nonbasic_list, z, delta_z);
      entering_index      = phase2::second_stage_harris(lp,
                                                   vstatus,
                                                   nonbasic_list,
                                                   z,
                                                   delta_z,
                                                   max_step_length,
                                                   step_length,
                                                   nonbasic_entering_index);
    } else if (bound_flip_ratio) {
      entering_index = phase2::bound_flipping_ratio_test(lp,
                                                         settings,
                                                         vstatus,
                                                         nonbasic_list,
                                                         x,
                                                         z,
                                                         delta_z,
                                                         direction,
                                                         leaving_index,
                                                         step_length,
                                                         nonbasic_entering_index);
    } else {
      entering_index = phase2::phase2_ratio_test(
        lp, settings, vstatus, nonbasic_list, z, delta_z, step_length, nonbasic_entering_index);
    }
    if (entering_index == -1) {
      if (primal_infeasibility > settings.primal_tol &&
          max_val < settings.steepest_edge_primal_tol) {
        // We could be done
        settings.log.printf("Exiting due to small primal infeasibility se %e\n", max_val);
        phase2::prepare_optimality(lp,
                                   settings,
                                   ft,
                                   objective,
                                   basic_list,
                                   nonbasic_list,
                                   vstatus,
                                   phase,
                                   start_time,
                                   max_val,
                                   iter,
                                   x,
                                   y,
                                   z,
                                   sol);
        status = dual::status_t::OPTIMAL;
        break;
      }
      const f_t dual_infeas =
        phase2::dual_infeasibility(lp, settings, vstatus, z, settings.tight_tol, settings.dual_tol);
      settings.log.printf("Dual infeasibility %e\n", dual_infeas);
      const f_t primal_inf = phase2::primal_infeasibility(lp, settings, vstatus, x);
      settings.log.printf("Primal infeasibility %e\n", primal_inf);
      if (dual_infeas > settings.dual_tol) {
        settings.log.printf(
          "Numerical issues encountered. No entering variable found with large infeasibility.\n");
        return dual::status_t::NUMERICAL;
      }
      return dual::status_t::DUAL_UNBOUNDED;
    }

    // Update dual variables
    // y <- y + steplength * delta_y
    for (i_t i = 0; i < m; ++i) {
      y[i] += step_length * delta_y[i];
    }

    // z <- z + steplength * delta_z
    for (i_t j = 0; j < n; ++j) {
      z[j] += step_length * delta_z[j];
    }

#ifdef COMPUTE_DUAL_RESIDUAL
    dual_res1 = z;
    for (i_t j = 0; j < n; ++j) {
      dual_res1[j] -= objective[j];
    }
    matrix_transpose_vector_multiply(lp.A, 1.0, y, 1.0, dual_res1);
    f_t dual_res_norm = vector_norm_inf<i_t, f_t>(dual_res1);
    if (dual_res_norm > settings.dual_tol) {
      settings.log.printf("|| A'*y + z - c || %e steplength %e\n", dual_res_norm, step_length);
    }
#endif

    // Update primal variable
    std::vector<f_t> atilde(m);
    std::vector<f_t> delta_x_flip(n);
    phase2::flip_bounds(
      lp, settings, objective, z, nonbasic_list, entering_index, vstatus, delta_x_flip, atilde);

    // B*delta_xB_0 = atilde
    std::vector<f_t> delta_xB_0(m);
    ft.b_solve(atilde, delta_xB_0);
    for (i_t k = 0; k < m; ++k) {
      const i_t j = basic_list[k];
      x[j] += delta_xB_0[k];
    }
    for (i_t k = 0; k < n - m; ++k) {
      const i_t j = nonbasic_list[k];
      x[j] += delta_x_flip[j];
    }

    f_t delta_x_leaving;
    if (direction == 1) {
      delta_x_leaving = lp.lower[leaving_index] - x[leaving_index];
    } else {
      delta_x_leaving = lp.upper[leaving_index] - x[leaving_index];
    }
    // B*w = -A(:, entering)
    std::vector<f_t> scaled_delta_xB(m);
    std::fill(rhs.begin(), rhs.end(), 0.0);
    lp.A.load_a_column(entering_index, rhs);
    std::vector<f_t> utilde(m);
    ft.b_solve(rhs, scaled_delta_xB, utilde);
    for (i_t i = 0; i < m; ++i) {
      scaled_delta_xB[i] *= -1.0;
    }

#ifdef COMPUTE_BSOLVE_RESIDUAL
    {
      std::vector<f_t> residual_B(m);
      b_multiply(lp, basic_list, scaled_delta_xB, residual_B);
      f_t err_max = 0;
      for (Int k = 0; k < m; ++k) {
        const f_t err = std::abs(rhs[k] - residual_B[k]);
        if (err >= 1e-5) {
          settings.log.printf(
            "Bsolve diff %d %e rhs %e residual %e\n", k, err, rhs[k], residual_B[k]);
        }
        err_max = std::max(err_max, err);
      }
      assert(err_max < 1e-4);
    }
#endif

    f_t primal_step_length = delta_x_leaving / scaled_delta_xB[basic_leaving_index];
    for (i_t k = 0; k < m; ++k) {
      const i_t j = basic_list[k];
      delta_x[j]  = primal_step_length * scaled_delta_xB[k];
    }
    delta_x[leaving_index] = delta_x_leaving;
    for (i_t k = 0; k < n - m; k++) {
      const i_t j = nonbasic_list[k];
      delta_x[j]  = 0.0;
    }
    delta_x[entering_index] = primal_step_length;

#ifdef COMPUTE_PRIMAL_STEP_RESIDUAL
    matrix_vector_multiply(lp.A, 1.0, delta_x, 1.0, residual);
    f_t primal_step_err = vector_norm_inf(residual);
    if (primal_step_err > 1e-4) { settings.log.printf("|| A * dx || %e\n", primal_step_err); }
#endif

    const i_t steepest_edge_status = phase2::update_steepest_edge_norms(settings,
                                                                        basic_list,
                                                                        ft,
                                                                        direction,
                                                                        delta_y,
                                                                        scaled_delta_xB,
                                                                        basic_leaving_index,
                                                                        entering_index,
                                                                        delta_y_steepest_edge);
#ifdef STEEPEST_EDGE_DEBUG
    if (steepest_edge_status == -1) {
      settings.log.printf("Num updates %d\n", ft.num_updates());
      settings.log.printf(" Primal step length %e\n", primal_step_length);
      settings.log.printf("|| delta_xB || %e\n", vector_norm_inf(scaled_delta_xB));
      settings.log.printf("|| rhs || %e\n", vector_norm_inf(rhs));
    }
#endif
    assert(steepest_edge_status == 0);

    // x <- x + delta_x
    for (i_t j = 0; j < n; ++j) {
      x[j] += delta_x[j];
    }
#ifdef COMPUTE_PRIMAL_RESIDUAL
    residual = lp.rhs;
    matrix_vector_multiply(lp.A, 1.0, x, -1.0, residual);
    primal_residual = vector_norm_inf<i_t, f_t>(residual);
    if (iter % 100 == 0 && primal_residual > 10 * settings.primal_tol) {
      settings.log.printf("|| A*x - b || %e\n", primal_residual);
    }
#endif

    f_t sum_perturb = 0.0;
    phase2::compute_perturbation(lp, settings, z, objective, sum_perturb);

    // Update basis
    vstatus[entering_index] = variable_status_t::BASIC;
    if (lp.lower[leaving_index] != lp.upper[leaving_index]) {
      vstatus[leaving_index] = static_cast<variable_status_t>(-direction);
    } else {
      vstatus[leaving_index] = variable_status_t::NONBASIC_FIXED;
    }
    basic_list[basic_leaving_index]        = entering_index;
    nonbasic_list[nonbasic_entering_index] = leaving_index;

    // Refactor or Update
    bool should_refactor = ft.num_updates() > settings.refactor_frequency;
    if (!should_refactor) {
      i_t recommend_refactor = ft.update(utilde, basic_leaving_index);
#ifdef CHECK_FT
      {
        csc_matrix_t Btest(m, m, 1);
        ft.multiply_lu(Btest);
        {
          csc_matrix_t B(m, m, 1);
          form_b(lp, basic_list, B);
          csc_matrix_t Diff(m, m, 1);
          add(Btest, B, 1.0, -1.0, Diff);
          const f_t err = Diff.norm1();
          if (err > settings.primal_tol) {
            settings.log.printf("|| B - L*U || %e\n", Diff.norm1());
          }
          assert(err < settings.primal_tol);
        }
      }
#endif
      should_refactor = recommend_refactor == 1;
    }

    if (should_refactor) {
      if (factorize_basis(lp.A, settings, basic_list, L, U, p, pinv, q, deficient, slacks_needed) ==
          -1) {
        basis_repair(lp.A, settings, deficient, slacks_needed, basic_list, nonbasic_list, vstatus);
        if (factorize_basis(
              lp.A, settings, basic_list, L, U, p, pinv, q, deficient, slacks_needed) == -1) {
          return dual::status_t::NUMERICAL;
        }
      }
      reorder_basic_list(q, basic_list);
      ft.reset(L, U, p);
    }

    phase2::compute_steepest_edge_norm_entering(
      settings, m, ft, basic_leaving_index, entering_index, delta_y_steepest_edge);

#ifdef STEEPEST_EDGE_DEBUG
    if (iter < 100 || iter % 100 == 0))
        {
            phase2::check_steepest_edge_norms(settings, basic_list, ft, delta_y_steepest_edge);
        }
#endif

    iter++;

    const f_t obj = compute_objective(lp, x);
    f_t now       = toc(start_time);
    if ((iter - start_iter) < settings.first_iteration_log ||
        (iter % settings.iteration_log_frequency) == 0) {
      if (phase == 1 && iter == 1) {
        settings.log.printf(" Iter     Objective   Primal Infeas  Perturb  Time\n");
      }
      settings.log.printf("%5d %+.8e %.8e %.2e %.2f\n",
                          iter,
                          compute_user_objective(lp, obj),
                          primal_infeasibility,
                          sum_perturb,
                          now);
    }

    if (obj >= settings.cut_off) {
      settings.log.printf("Solve cutoff. Current objecive %e. Cutoff %e\n", obj, settings.cut_off);
      return dual::status_t::CUTOFF;
    }

    if (now > settings.time_limit) { return dual::status_t::TIME_LIMIT; }

    if (settings.concurrent_halt != nullptr &&
        settings.concurrent_halt->load(std::memory_order_acquire) == 1) {
      return dual::status_t::CONCURRENT_LIMIT;
    }
  }
  if (iter >= iter_limit) { status = dual::status_t::ITERATION_LIMIT; }
  return status;
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE

template dual::status_t dual_phase2<int, double>(
  int phase,
  int slack_basis,
  double start_time,
  const lp_problem_t<int, double>& lp,
  const simplex_solver_settings_t<int, double>& settings,
  std::vector<variable_status_t>& vstatus,
  lp_solution_t<int, double>& sol,
  int& iter,
  std::vector<double>& steepest_edge_norms);

#endif

}  // namespace cuopt::linear_programming::dual_simplex
