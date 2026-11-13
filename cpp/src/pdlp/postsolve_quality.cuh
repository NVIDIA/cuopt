/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <thrust/fill.h>
#include <algorithm>
#include <cmath>
#include <cuopt/mathematical_optimization/optimization_problem.hpp>
#include <cuopt/mathematical_optimization/pdlp/solver_solution.hpp>
#include <limits>
#include <mip_heuristics/utils.hpp>
#include <pdlp/cusparse_view.hpp>
#include <utilities/copy_helpers.hpp>
#include <utilities/logger.hpp>
#include <utilities/timer.hpp>
#include <vector>

namespace cuopt::mathematical_optimization::detail {

// Use the original device matrix; final logging must not copy all of A to the host.
template <typename i_t, typename f_t>
void postsolve_spmv(const raft::handle_t* handle,
                    i_t rows,
                    i_t cols,
                    const rmm::device_uvector<i_t>& offsets,
                    const rmm::device_uvector<i_t>& indices,
                    const rmm::device_uvector<f_t>& values,
                    const rmm::device_uvector<f_t>& x,
                    rmm::device_uvector<f_t>& result,
                    bool transpose = false)
{
  auto stream = handle->get_stream();
  thrust::fill(rmm::exec_policy(stream), result.begin(), result.end(), f_t(0));
  if (rows == 0 || cols == 0 || values.is_empty()) { return; }
  auto matrix = pdlp::make_csr(rows,
                               cols,
                               values.size(),
                               const_cast<i_t*>(offsets.data()),
                               const_cast<i_t*>(indices.data()),
                               const_cast<f_t*>(values.data()));
  auto input  = pdlp::make_dnvec(transpose ? rows : cols, const_cast<f_t*>(x.data()));
  auto output = pdlp::make_dnvec(transpose ? cols : rows, result.data());
  static constexpr f_t one_value = 1, zero_value = 0;
  rmm::device_scalar<f_t> one(one_value, stream), zero(zero_value, stream);
  auto sparse    = handle->get_cusparse_handle();
  auto operation = transpose ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
  auto type      = std::is_same_v<f_t, double> ? CUDA_R_64F : CUDA_R_32F;
  RAFT_CUSPARSE_TRY(cusparseSetPointerMode(sparse, CUSPARSE_POINTER_MODE_DEVICE));
  RAFT_CUSPARSE_TRY(cusparseSetStream(sparse, stream.get()));
  size_t bytes = 0;
  RAFT_CUSPARSE_TRY(cusparseSpMV_bufferSize(sparse,
                                            operation,
                                            one.data(),
                                            matrix.get(),
                                            input.get(),
                                            zero.data(),
                                            output.get(),
                                            type,
                                            CUSPARSE_SPMV_CSR_ALG2,
                                            &bytes));
  rmm::device_buffer scratch(bytes, stream);
  RAFT_CUSPARSE_TRY(cusparseSpMV(sparse,
                                 operation,
                                 one.data(),
                                 matrix.get(),
                                 input.get(),
                                 zero.data(),
                                 output.get(),
                                 type,
                                 CUSPARSE_SPMV_CSR_ALG2,
                                 scratch.data()));
}

template <typename f_t>
struct postsolve_quality_t {
  f_t row_l2{0};
  f_t bound_l2{0};
  f_t primal_relative{0};
  f_t dual_l2{std::numeric_limits<f_t>::quiet_NaN()};
  f_t dual_relative{std::numeric_limits<f_t>::quiet_NaN()};
  f_t dual_projection_l2{std::numeric_limits<f_t>::quiet_NaN()};
  bool dual_available{false};
};

template <typename f_t>
f_t postsolve_interval_violation(f_t value, f_t lower, f_t upper)
{
  if (!std::isfinite(value) || std::isnan(lower) || std::isnan(upper)) {
    return std::numeric_limits<f_t>::infinity();
  }
  return std::max(f_t(0), std::max(lower - value, value - upper));
}

template <typename f_t>
f_t project_postsolve_multiplier(f_t value, f_t lower, f_t upper, f_t sense)
{
  if ((sense * value > 0 && !std::isfinite(lower)) ||
      (sense * value < 0 && !std::isfinite(upper))) {
    return 0;
  }
  return value;
}

// HPR-style postsolve diagnostics for the ORIGINAL model:
//   min c^T x + (1/2)x^T Hx,  row_lower <= Ax <= row_upper, lower <= x <= upper.
// v_row = Ax - projection_[row_lower,row_upper](Ax), v_bound = x-projection_[lower,upper](x).
// b_i = max(|finite row_lower_i|, |finite row_upper_i|), treating infinite endpoints as 0.
// rp = max(||v_row||_2, ||v_bound||_2) / (1 + ||b||_2).
// Let y_hat,z_hat be COPIES projected onto admissible multiplier signs. Then
// rd = ||Hx+c-A^T y_hat-z_hat||_2 / (1+||c||_2).
// The projection distance ||(y-y_hat,z-z_hat)||_2 is reported too. A small rd alone
// does not certify the returned unprojected duals. Neither x/y/z nor API fields change.
template <typename i_t, typename f_t>
postsolve_quality_t<f_t> evaluate_postsolve_quality(const optimization_problem_t<i_t, f_t>& model,
                                                    const rmm::device_uvector<f_t>& device_x,
                                                    const rmm::device_uvector<f_t>& device_y,
                                                    const rmm::device_uvector<f_t>& device_z,
                                                    bool dual_available)
{
  postsolve_quality_t<f_t> result;
  auto handle = model.get_handle_ptr();
  auto stream = handle->get_stream();
  const i_t n = model.get_n_variables(), m = model.get_n_constraints();
  const f_t inf = std::numeric_limits<f_t>::infinity();
  if (device_x.size() != static_cast<size_t>(n)) {
    result.row_l2 = result.bound_l2 = result.primal_relative = inf;
    return result;
  }
  const auto x     = cuopt::host_copy(device_x, stream);
  auto row_lower   = model.get_constraint_lower_bounds_host();
  auto row_upper   = model.get_constraint_upper_bounds_host();
  const auto lower = model.get_variable_lower_bounds_host();
  const auto upper = model.get_variable_upper_bounds_host();
  if (row_lower.size() != static_cast<size_t>(m) || row_upper.size() != static_cast<size_t>(m)) {
    const auto rhs   = model.get_constraint_bounds_host();
    const auto types = model.get_row_types_host();
    row_lower.resize(m);
    row_upper.resize(m);
    for (i_t row = 0; row < m; ++row) {
      const char type = types.empty() ? 'E' : types[row];
      row_lower[row]  = type == 'L' ? -inf : rhs[row];
      row_upper[row]  = type == 'G' ? inf : rhs[row];
    }
  }
  rmm::device_uvector<f_t> d_ax(m, stream);
  postsolve_spmv(handle,
                 m,
                 n,
                 model.get_constraint_matrix_offsets(),
                 model.get_constraint_matrix_indices(),
                 model.get_constraint_matrix_values(),
                 device_x,
                 d_ax);
  const auto ax = cuopt::host_copy(d_ax, stream);
  f_t rhs_norm  = 0;
  for (i_t row = 0; row < m; ++row) {
    result.row_l2 = std::hypot(
      result.row_l2, postsolve_interval_violation(ax[row], row_lower[row], row_upper[row]));
    const f_t lo = std::isfinite(row_lower[row]) ? std::abs(row_lower[row]) : 0;
    const f_t hi = std::isfinite(row_upper[row]) ? std::abs(row_upper[row]) : 0;
    rhs_norm     = std::hypot(rhs_norm, std::max(lo, hi));
  }
  for (i_t col = 0; col < n; ++col) {
    result.bound_l2 =
      std::hypot(result.bound_l2, postsolve_interval_violation(x[col], lower[col], upper[col]));
  }
  result.primal_relative =
    std::isfinite(rhs_norm) ? std::max(result.row_l2, result.bound_l2) / (1 + rhs_norm) : inf;
  if (!dual_available || device_y.size() != static_cast<size_t>(m) || device_z.size() != x.size()) {
    return result;
  }
  auto y = cuopt::host_copy(device_y, stream);
  auto z = cuopt::host_copy(device_z, stream);
  for (auto value : y) {
    if (!std::isfinite(value)) { return result; }
  }
  for (auto value : z) {
    if (!std::isfinite(value)) { return result; }
  }
  const f_t sense           = model.get_sense() ? -1 : 1;
  result.dual_projection_l2 = 0;
  for (i_t row = 0; row < m; ++row) {
    const f_t projected =
      project_postsolve_multiplier(y[row], row_lower[row], row_upper[row], sense);
    result.dual_projection_l2 = std::hypot(result.dual_projection_l2, y[row] - projected);
    y[row]                    = projected;
  }
  for (i_t col = 0; col < n; ++col) {
    const f_t projected       = project_postsolve_multiplier(z[col], lower[col], upper[col], sense);
    result.dual_projection_l2 = std::hypot(result.dual_projection_l2, z[col] - projected);
    z[col]                    = projected;
  }
  const auto d_y = cuopt::device_copy(y, stream);
  rmm::device_uvector<f_t> d_aty(n, stream);
  postsolve_spmv(handle,
                 m,
                 n,
                 model.get_constraint_matrix_offsets(),
                 model.get_constraint_matrix_indices(),
                 model.get_constraint_matrix_values(),
                 d_y,
                 d_aty,
                 true);
  const auto aty     = cuopt::host_copy(d_aty, stream);
  const auto c       = model.get_objective_coefficients_host();
  const auto& qo     = model.get_quadratic_objective_offsets();
  const auto& qi     = model.get_quadratic_objective_indices();
  const auto& qv     = model.get_quadratic_objective_values();
  f_t objective_norm = 0;
  result.dual_l2     = 0;
  for (i_t col = 0; col < n; ++col) {
    // The quadratic setter has already stored the full Hessian H=Q+Q^T.
    const f_t hx       = !qv.empty() && static_cast<size_t>(col + 1) < qo.size()
                           ? mip::compensated_dot2_csr(qo.data(), qi.data(), qv.data(), x.data(), col)
                           : 0;
    const f_t residual = hx + c[col] - aty[col] - z[col];
    result.dual_l2     = std::hypot(result.dual_l2, std::isfinite(residual) ? residual : inf);
    objective_norm     = std::hypot(objective_norm, c[col]);
  }
  result.dual_relative =
    std::isfinite(objective_norm) ? result.dual_l2 / (1 + objective_norm) : inf;
  result.dual_available = true;
  return result;
}

template <typename i_t, typename f_t>
void print_postsolve_quality(const optimization_problem_t<i_t, f_t>& model,
                             optimization_problem_solution_t<i_t, f_t>& solution,
                             bool dual_available)
{
  if (solution.get_additional_termination_informations().size() != 1) { return; }
  const auto status = solution.get_termination_status();
  const auto& info  = solution.get_additional_termination_information();
  if ((status != pdlp_termination_status_t::Optimal &&
       status != pdlp_termination_status_t::PrimalFeasible &&
       status != pdlp_termination_status_t::TimeLimit &&
       status != pdlp_termination_status_t::IterationLimit &&
       status != pdlp_termination_status_t::ConcurrentLimit) ||
      (info.solved_by != method_t::PDLP && info.solved_by != method_t::Unset &&
       !std::isfinite(info.primal_objective))) {
    CUOPT_LOG_INFO("Final original-model residuals unavailable: no solution iterate was returned.");
    return;
  }
  if (model.has_quadratic_constraints()) {
    CUOPT_LOG_INFO("Final original-model residuals unavailable for quadratic constraints.");
    return;
  }
  const auto metrics = evaluate_postsolve_quality(model,
                                                  solution.get_primal_solution(),
                                                  solution.get_dual_solution(),
                                                  solution.get_reduced_cost(),
                                                  dual_available);
  CUOPT_LOG_INFO("Final solution quality (original model; L2 norms):");
  CUOPT_LOG_INFO("  Row violation (L2):                     %.8e", metrics.row_l2);
  CUOPT_LOG_INFO("  Variable-bound violation (L2):          %.8e", metrics.bound_l2);
  CUOPT_LOG_INFO("  Primal residual (relative):             %.8e", metrics.primal_relative);
  if (metrics.dual_available) {
    CUOPT_LOG_INFO("  Dual residual (projected y/z, abs/rel): %.8e / %.8e",
                   metrics.dual_l2,
                   metrics.dual_relative);
    CUOPT_LOG_INFO("  Dual sign projection distance (L2):    %.8e", metrics.dual_projection_l2);
  } else {
    CUOPT_LOG_INFO("  Dual residual unavailable: missing or non-finite dual information.");
  }
}

// This is a log-only summary. Evaluate the original objective from restored x:
// P = scale * (c^T x + (1/2)x^T Hx + offset), with H = Q + Q^T.
// A non-Optimal result deliberately displays NaN, even if it has a feasible iterate.
// The independent timer starts at solve entry. Synchronize before sampling it so
// Time includes initialization, presolve, solving, unscale, postsolve and diagnostics.
template <typename i_t, typename f_t>
void print_final_status(const optimization_problem_t<i_t, f_t>& model,
                        optimization_problem_solution_t<i_t, f_t>& solution,
                        const cuopt::timer_t& timer)
{
  if (solution.get_additional_termination_informations().size() != 1) { return; }
  f_t objective = std::numeric_limits<f_t>::quiet_NaN();
  if (solution.get_termination_status() == pdlp_termination_status_t::Optimal &&
      solution.get_primal_solution().size() == static_cast<size_t>(model.get_n_variables())) {
    const auto x =
      cuopt::host_copy(solution.get_primal_solution(), model.get_handle_ptr()->get_stream());
    const auto c = model.get_objective_coefficients_host();
    if (c.size() == x.size() &&
        std::all_of(x.begin(), x.end(), [](f_t value) { return std::isfinite(value); })) {
      const auto& offsets = model.get_quadratic_objective_offsets();
      const auto& indices = model.get_quadratic_objective_indices();
      const auto& values  = model.get_quadratic_objective_values();
      std::vector<f_t> half_hx(x.size(), 0);
      if (!values.empty()) {
        for (size_t row = 0; row + 1 < offsets.size(); ++row) {
          half_hx[row] = f_t(0.5) * mip::compensated_dot2_csr(
                                      offsets.data(), indices.data(), values.data(), x.data(), row);
        }
      }
      objective =
        model.get_objective_scaling_factor() *
        (mip::compensated_dot2(c.data(), x.data(), x.size()) +
         mip::compensated_dot2(half_hx.data(), x.data(), x.size()) + model.get_objective_offset());
    }
  }
  model.get_handle_ptr()->sync_stream();
  CUOPT_LOG_INFO("Final Status: %s  Objective: %.8e  Iterations: %d  Time: %.3fs",
                 solution.get_termination_status_string().c_str(),
                 objective,
                 solution.get_additional_termination_information().number_of_steps_taken,
                 timer.elapsed_time());
}
}  // namespace cuopt::mathematical_optimization::detail
