/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <dual_simplex/scaling.hpp>
#include <linear_algebra/sparse_matrix.hpp>

#include <cmath>

namespace cuopt::mathematical_optimization::simplex {

template <typename i_t, typename f_t>
i_t scaling(const lp_problem_t<i_t, f_t>& unscaled,
            const simplex_solver_settings_t<i_t, f_t>& settings,
            lp_problem_t<i_t, f_t>& scaled,
            std::vector<f_t>& column_scaling,
            std::vector<f_t>& row_scaling)
{
  scaled = unscaled;
  i_t m  = scaled.num_rows;
  i_t n  = scaled.num_cols;

  row_scaling.assign(m, 1.0);

  // =========================================================================
  // Ruiz equilibration for SOCP (and QP) problems
  // =========================================================================
  // For SOCP problems, apply Ruiz equilibration: alternating row and column
  // infinity-norm scaling to bring the constraint matrix close to equilibrium.
  // This dramatically improves the conditioning of the augmented KKT system.
  // Applied only when the constraint matrix has a large row-norm imbalance.
  if (!unscaled.second_order_cone_dims.empty() || unscaled.Q.n > 0) {
    // col_scale and row_scale accumulate reciprocal scale factors during Ruiz iterations.
    std::vector<f_t> col_scale(n, 1.0);

    // Decide whether Ruiz scaling is needed by checking row-norm imbalance.
    // If max_row_norm / min_row_norm is small, the matrix is already well-conditioned
    // and scaling can hurt (e.g. by amplifying tiny noise coefficients).
    csr_matrix_t<i_t, f_t> Arow_check(0, 0, 0);
    scaled.A.to_compressed_row(Arow_check);
    f_t max_row_norm = 0;
    f_t min_row_norm = std::numeric_limits<f_t>::max();
    for (i_t i = 0; i < m; ++i) {
      f_t row_norm = 0;
      for (i_t p = Arow_check.row_start[i]; p < Arow_check.row_start[i + 1]; ++p) {
        f_t a = std::abs(Arow_check.x[p]);
        if (a > row_norm) row_norm = a;
      }
      if (row_norm > 0) {
        max_row_norm = std::max(max_row_norm, row_norm);
        min_row_norm = std::min(min_row_norm, row_norm);
      }
    }
    f_t row_norm_ratio = (min_row_norm > 0) ? max_row_norm / min_row_norm : 1.0;

    if (row_norm_ratio < 100.0) {
      settings.log.printf("Skipping Ruiz equilibration (row norm ratio %.1f < 100)\n",
                          row_norm_ratio);
      column_scaling.assign(n, 1.0);
      return 0;
    }

    // Apply Ruiz equilibration
    csr_matrix_t<i_t, f_t> Arow(0, 0, 0);
    scaled.A.to_compressed_row(Arow);

    constexpr i_t max_ruiz_iterations = 10;
    for (i_t iter = 0; iter < max_ruiz_iterations; ++iter) {
      f_t max_deviation = 0.0;

      // --- Row scaling: scale each row by 1/sqrt(max|a_ij|) ---
      std::vector<f_t> r(m);
      for (i_t i = 0; i < m; ++i) {
        f_t rm = 0.0;
        for (i_t p = Arow.row_start[i]; p < Arow.row_start[i + 1]; ++p) {
          f_t a = std::abs(Arow.x[p]);
          if (a > rm) rm = a;
        }
        r[i]          = rm > 0 ? 1.0 / std::sqrt(rm) : 1.0;
        max_deviation = std::max(max_deviation, std::abs(rm - 1.0));
      }
      for (i_t j = 0; j < n; ++j) {
        for (i_t p = scaled.A.col_start[j]; p < scaled.A.col_start[j + 1]; ++p) {
          scaled.A.x[p] *= r[scaled.A.i[p]];
        }
      }
      for (i_t i = 0; i < m; ++i) {
        for (i_t p = Arow.row_start[i]; p < Arow.row_start[i + 1]; ++p) {
          Arow.x[p] *= r[i];
        }
        scaled.rhs[i] *= r[i];
        row_scaling[i] *= r[i];
      }

      // --- Column scaling: scale each column by 1/sqrt(max|a_ij|) ---
      // For cone variables, use a uniform scale per cone block to preserve SOC structure.
      std::vector<f_t> c(n);
      const i_t cone_start = unscaled.second_order_cone_dims.empty() ? n : unscaled.cone_var_start;

      // Linear columns: scale independently
      for (i_t j = 0; j < cone_start; ++j) {
        f_t cm = 0.0;
        for (i_t p = scaled.A.col_start[j]; p < scaled.A.col_start[j + 1]; ++p) {
          f_t a = std::abs(scaled.A.x[p]);
          if (a > cm) cm = a;
        }
        c[j]          = cm > 0 ? 1.0 / std::sqrt(cm) : 1.0;
        max_deviation = std::max(max_deviation, std::abs(cm - 1.0));
      }

      // Cone columns: uniform scale per cone block
      i_t cone_off = cone_start;
      for (i_t k = 0; k < static_cast<i_t>(unscaled.second_order_cone_dims.size()); ++k) {
        i_t q_k = unscaled.second_order_cone_dims[k];
        // Find max column inf-norm across all columns in this cone
        f_t cone_max = 0.0;
        for (i_t j = cone_off; j < cone_off + q_k; ++j) {
          for (i_t p = scaled.A.col_start[j]; p < scaled.A.col_start[j + 1]; ++p) {
            f_t a = std::abs(scaled.A.x[p]);
            if (a > cone_max) cone_max = a;
          }
        }
        f_t cone_scale = cone_max > 0 ? 1.0 / std::sqrt(cone_max) : 1.0;
        max_deviation  = std::max(max_deviation, std::abs(cone_max - 1.0));
        for (i_t j = cone_off; j < cone_off + q_k; ++j) {
          c[j] = cone_scale;
        }
        cone_off += q_k;
      }
      for (i_t j = 0; j < n; ++j) {
        for (i_t p = scaled.A.col_start[j]; p < scaled.A.col_start[j + 1]; ++p) {
          scaled.A.x[p] *= c[j];
        }
      }
      for (i_t i = 0; i < m; ++i) {
        for (i_t p = Arow.row_start[i]; p < Arow.row_start[i + 1]; ++p) {
          Arow.x[p] *= c[Arow.j[p]];
        }
      }
      for (i_t j = 0; j < n; ++j) {
        scaled.objective[j] *= c[j];
        col_scale[j] *= c[j];
      }
      // Bounds use +/-inf for unbounded sides (see types.hpp). Use +/-1e20 as a practical
      // sentinel: we do not expect finite bounds beyond this magnitude, and skipping scale
      // on |bound| >= 1e20 avoids overflow when dividing very large limits by small c[j].
      for (i_t j = 0; j < n; ++j) {
        if (scaled.lower[j] > -1e20) scaled.lower[j] /= c[j];
        if (scaled.upper[j] < 1e20) scaled.upper[j] /= c[j];
      }
      if (scaled.Q.n > 0) {
        for (i_t row = 0; row < scaled.Q.m; ++row) {
          for (i_t p = scaled.Q.row_start[row]; p < scaled.Q.row_start[row + 1]; ++p) {
            i_t col = scaled.Q.j[p];
            scaled.Q.x[p] *= c[row] * c[col];
          }
        }
      }
      if (max_deviation < 0.1) break;
    }

    // Ruiz col_scale/row_scaling accumulate reciprocals (c[j] = 1/sqrt(norm)).
    // Invert to match the output convention: C(j,j) = 1/column_scaling[j],
    // R(i,i) = 1/row_scaling[i].
    column_scaling.resize(n);
    for (i_t j = 0; j < n; ++j) {
      column_scaling[j] = f_t(1) / col_scale[j];
    }
    for (i_t i = 0; i < m; ++i) {
      row_scaling[i] = f_t(1) / row_scaling[i];
    }

    f_t a_min = std::numeric_limits<f_t>::max();
    f_t a_max = 0;
    for (i_t p = 0; p < scaled.A.col_start[n]; ++p) {
      f_t a = std::abs(scaled.A.x[p]);
      if (a > 0) {
        a_min = std::min(a_min, a);
        a_max = std::max(a_max, a);
      }
    }
    settings.log.printf("Ruiz equilibration: coefficient range [%e, %e]\n", a_min, a_max);
    return 0;
  }

  if (!settings.scale_columns) {
    settings.log.printf("Skipping column scaling\n");
    column_scaling.resize(n, 1.0);
    return 0;
  }

  column_scaling.resize(n);
  f_t max = 0;
  f_t min = std::numeric_limits<f_t>::max();
  for (i_t j = 0; j < n; ++j) {
    const i_t col_start = scaled.A.col_start[j];
    const i_t col_end   = scaled.A.col_start[j + 1];
    f_t sum             = 0.0;
    for (i_t p = col_start; p < col_end; ++p) {
      const f_t x = scaled.A.x[p];
      sum += x * x;
    }
    f_t col_norm_j = column_scaling[j] = sum > 0 ? std::sqrt(sum) : 1.0;
    max                                = std::max(col_norm_j, max);
    min                                = std::min(col_norm_j, min);
  }
  settings.log.printf("Scaling matrix. Maximum column norm %e, minimum column norm %e\n", max, min);
  // C(j, j) = 1/column_scaling(j)

  // scaled_A = unscaled_A * C
  for (i_t j = 0; j < n; ++j) {
    const i_t col_start = scaled.A.col_start[j];
    const i_t col_end   = scaled.A.col_start[j + 1];
    for (i_t p = col_start; p < col_end; ++p) {
      scaled.A.x[p] /= column_scaling[j];
    }
  }
  // scaled_obj = C*unscaled_obj
  for (i_t j = 0; j < n; ++j) {
    scaled.objective[j] /= column_scaling[j];
  }
  // scaled_lower = C^{-1} * unscaled_lower
  // scaled_upper = C^{-1} * unscaled_upper
  for (i_t j = 0; j < n; ++j) {
    scaled.lower[j] *= column_scaling[j];
    scaled.upper[j] *= column_scaling[j];
  }

  for (i_t i = 0; i < unscaled.Q.n; ++i) {
    const i_t row_start = unscaled.Q.row_start[i];
    const i_t row_end   = unscaled.Q.row_start[i + 1];
    i_t row             = i;
    for (i_t p = row_start; p < row_end; ++p) {
      i_t col       = unscaled.Q.j[p];
      scaled.Q.x[p] = unscaled.Q.x[p] / (column_scaling[row] * column_scaling[col]);
    }
  }
  return 0;
}

template <typename i_t, typename f_t>
void unscale_solution(const std::vector<f_t>& column_scaling,
                      const std::vector<f_t>& row_scaling,
                      const std::vector<f_t>& scaled_x,
                      const std::vector<f_t>& scaled_y,
                      const std::vector<f_t>& scaled_z,
                      std::vector<f_t>& unscaled_x,
                      std::vector<f_t>& unscaled_y,
                      std::vector<f_t>& unscaled_z)
{
  const i_t n = scaled_x.size();
  unscaled_x.resize(n);
  unscaled_z.resize(n);
  for (i_t j = 0; j < n; ++j) {
    unscaled_x[j] = scaled_x[j] / column_scaling[j];
    unscaled_z[j] = scaled_z[j] * column_scaling[j];
  }

  const i_t m = scaled_y.size();
  unscaled_y.resize(m);
  // R(i,i) = 1/row_scaling[i], so y_orig = y_scaled / row_scaling
  for (i_t i = 0; i < m; ++i) {
    unscaled_y[i] = scaled_y[i] / row_scaling[i];
  }
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE

template int scaling<int, double>(const lp_problem_t<int, double>& unscaled,
                                  const simplex_solver_settings_t<int, double>& settings,
                                  lp_problem_t<int, double>& scaled,
                                  std::vector<double>& column_scaling,
                                  std::vector<double>& row_scaling);

template void unscale_solution<int, double>(const std::vector<double>& column_scaling,
                                            const std::vector<double>& row_scaling,
                                            const std::vector<double>& scaled_x,
                                            const std::vector<double>& scaled_y,
                                            const std::vector<double>& scaled_z,
                                            std::vector<double>& unscaled_x,
                                            std::vector<double>& unscaled_y,
                                            std::vector<double>& unscaled_z);

#endif

}  // namespace cuopt::mathematical_optimization::simplex
