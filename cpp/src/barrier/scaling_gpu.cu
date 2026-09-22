/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

// GPU port of Ruiz equilibration for the barrier/SOCP-QP path (see scaling.cpp's
// `scaling()` for the CPU reference implementation this mirrors step-for-step). Only
// the Ruiz-equilibration branch is implemented here; callers must only invoke this
// for problems with second-order cones or a quadratic objective (the same condition
// `scaling()` uses to select the Ruiz branch over the plain geometric-mean scaling).

#include <barrier/scaling_gpu.cuh>

#include <dual_simplex/presolve.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>

#include <barrier/device_sparse_matrix.cuh>
#include <utilities/copy_helpers.hpp>
#include <utilities/reduce_ops.cuh>

#include <raft/util/cuda_utils.cuh>

#include <cub/cub.cuh>

#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace cuopt::mathematical_optimization::simplex {

namespace {

using cuopt::mathematical_optimization::barrier::device_csc_matrix_t;
using cuopt::mathematical_optimization::barrier::device_csr_matrix_t;

// row_norm[i] = max_j |A(i,j)|, computed straight off CSC: A.i[p] is the row of nonzero p,
// so the per-row maxima need no row-contiguous (CSR) copy of the matrix. Mirrors
// compute_row_inf_norms in scaling.cpp.
template <typename i_t, typename f_t>
void compute_row_inf_norms(const device_csc_matrix_t<i_t, f_t>& A,
                           rmm::device_uvector<f_t>& row_norm,
                           rmm::cuda_stream_view stream)
{
  row_norm.resize(A.m, stream);
  thrust::fill(rmm::exec_policy(stream), row_norm.begin(), row_norm.end(), f_t(0));
  thrust::for_each(rmm::exec_policy(stream),
                   thrust::make_counting_iterator(i_t(0)),
                   thrust::make_counting_iterator(A.nz_max),
                   [x = A.x.data(), row = A.i.data(), rmax = row_norm.data()] __device__(i_t p) {
                     const f_t a = raft::abs(x[p]);
                     // Read first: most nonzeros lose to the running max, so the CAS loop
                     // inside myAtomicMax is usually skipped entirely. Collisions are rare
                     // anyway -- a CSC column cannot repeat a row, so consecutive threads
                     // within a column target distinct accumulators.
                     if (a > rmax[row[p]]) { raft::myAtomicMax(&rmax[row[p]], a); }
                   });
}

// max_i |transform(input[i])| over the whole array (single segment); used for the
// one-shot imbalance-ratio heuristic, not the per-iteration row/column reduces.
template <typename f_t, typename InputIt>
f_t whole_array_abs_max(InputIt input, size_t n, rmm::cuda_stream_view stream)
{
  if (n == 0) return f_t(0);
  auto abs_it = thrust::make_transform_iterator(input, cuopt::abs_value_transform_t<f_t>{});
  return thrust::reduce(
    rmm::exec_policy(stream), abs_it, abs_it + n, f_t(0), cuopt::max_op_t<f_t>{});
}

// min_i (|transform(input[i])| > 0 ? |value| : sentinel) over the whole array, sentinel ==
// std::numeric_limits<f_t>::max() when nothing is nonzero -- mirrors the host code's
// "ignore exact zeros" min-norm loop (scaling.cpp:45-46,53-56) bit for bit, including the
// degenerate all-zero case where the ratio check below reduces to `min_row_norm > 0`.
template <typename f_t, typename InputIt>
f_t whole_array_nonzero_abs_min(InputIt input, size_t n, rmm::cuda_stream_view stream)
{
  const f_t sentinel = std::numeric_limits<f_t>::max();
  if (n == 0) return sentinel;
  auto nz_it = thrust::make_transform_iterator(input, [sentinel] __device__(f_t value) -> f_t {
    const f_t abs_value = raft::abs(value);
    return abs_value > f_t(0) ? abs_value : sentinel;
  });
  return thrust::reduce(
    rmm::exec_policy(stream), nz_it, nz_it + n, sentinel, cuopt::min_op_t<f_t>{});
}

// max/min per-segment |value| via cub::DeviceSegmentedReduce over an arbitrary offsets
// pair (row_start for CSR, col_start for CSC); segments with no nonzeros reduce to 0.
template <typename i_t, typename f_t, typename OffsetBeginIt, typename OffsetEndIt>
void segmented_abs_max(const f_t* values,
                       OffsetBeginIt begin_offsets,
                       OffsetEndIt end_offsets,
                       i_t num_segments,
                       f_t* out,
                       rmm::device_buffer& temp_storage,
                       rmm::cuda_stream_view stream)
{
  if (num_segments == 0) return;
  auto abs_it  = thrust::make_transform_iterator(values, cuopt::abs_value_transform_t<f_t>{});
  size_t bytes = 0;
  cub::DeviceSegmentedReduce::Reduce(nullptr,
                                     bytes,
                                     abs_it,
                                     out,
                                     num_segments,
                                     begin_offsets,
                                     end_offsets,
                                     cuopt::max_op_t<f_t>{},
                                     f_t(0),
                                     stream);
  temp_storage.resize(bytes, stream);
  cub::DeviceSegmentedReduce::Reduce(temp_storage.data(),
                                     bytes,
                                     abs_it,
                                     out,
                                     num_segments,
                                     begin_offsets,
                                     end_offsets,
                                     cuopt::max_op_t<f_t>{},
                                     f_t(0),
                                     stream);
}

}  // namespace

template <typename i_t, typename f_t>
i_t scaling_ruiz_gpu(const lp_problem_t<i_t, f_t>& unscaled,
                     const simplex_solver_settings_t<i_t, f_t>& settings,
                     lp_problem_t<i_t, f_t>& scaled,
                     std::vector<f_t>& column_scaling,
                     std::vector<f_t>& row_scaling,
                     std::unique_ptr<device_csc_matrix_t<i_t, f_t>>& device_A,
                     std::unique_ptr<device_csc_matrix_t<i_t, f_t>>& device_Q)
{
  scaled     = unscaled;
  i_t m      = scaled.num_rows;
  i_t n      = scaled.num_cols;
  bool has_q = unscaled.Q.n > 0;

  rmm::cuda_stream_view stream = unscaled.handle_ptr->get_stream();

  device_A.reset();
  device_Q.reset();
  row_scaling.assign(m, 1.0);

  // --- Upload only what the skip heuristic needs; the rest of the setup is deferred
  // until after the decision, so a skipped problem pays almost nothing. ---
  device_csc_matrix_t<i_t, f_t> dA(scaled.A, stream);
  device_csr_matrix_t<i_t, f_t> dQ(scaled.Q, stream);

  // --- One-shot imbalance heuristic (mirrors scaling.cpp:43-116) ---
  rmm::device_buffer scratch;
  // Holds the raw row inf-norms, both for the heuristic here and in each Ruiz iteration
  // below, where it is then converted in place into that iteration's row scale factors.
  rmm::device_uvector<f_t> r(0, stream);
  compute_row_inf_norms(dA, r, stream);
  f_t max_row_norm   = whole_array_abs_max<f_t>(r.data(), m, stream);
  f_t min_row_norm   = whole_array_nonzero_abs_min<f_t>(r.data(), m, stream);
  f_t row_norm_ratio = (min_row_norm > 0) ? max_row_norm / min_row_norm : f_t(1.0);

  rmm::device_uvector<f_t> col_max_full(n, stream);
  segmented_abs_max<i_t, f_t>(dA.x.data(),
                              dA.col_start.data(),
                              dA.col_start.data() + 1,
                              n,
                              col_max_full.data(),
                              scratch,
                              stream);
  f_t max_col_norm   = whole_array_abs_max<f_t>(col_max_full.data(), n, stream);
  f_t min_col_norm   = whole_array_nonzero_abs_min<f_t>(col_max_full.data(), n, stream);
  f_t col_norm_ratio = (min_col_norm > 0) ? max_col_norm / min_col_norm : f_t(1.0);

  f_t q_ratio = f_t(1.0);
  if (has_q) {
    f_t max_q = whole_array_abs_max<f_t>(dQ.x.data(), dQ.nz_max, stream);
    f_t min_q = whole_array_nonzero_abs_min<f_t>(dQ.x.data(), dQ.nz_max, stream);
    if (min_q <= max_q) { q_ratio = max_q / min_q; }
  }

  const i_t ruiz_mode  = settings.qcqp_ruiz_equilibration;
  const bool balanced  = row_norm_ratio < 100.0 && col_norm_ratio < 5e4 && q_ratio < 100.0;
  const bool skip_ruiz = (ruiz_mode == 0) || (ruiz_mode < 0 && balanced);

  if (skip_ruiz) {
    if (ruiz_mode == 0) {
      settings.log.printf("Skipping Ruiz equilibration (qcqp_hyper_ruiz_equilibration = 0)\n");
    } else {
      settings.log.printf(
        "Skipping Ruiz equilibration (row norm ratio %.1f, column norm ratio %.1f < 5e4, Q coeff "
        "ratio %.1f < 100)\n",
        row_norm_ratio,
        col_norm_ratio,
        q_ratio);
    }
    column_scaling.assign(n, 1.0);
    return 0;
  }
  if (ruiz_mode > 0) {
    settings.log.printf(
      "Applying Ruiz equilibration (qcqp_hyper_ruiz_equilibration = 1, row norm ratio %.1f, "
      "column norm ratio %.1f, Q coeff ratio %.1f) [GPU]\n",
      row_norm_ratio,
      col_norm_ratio,
      q_ratio);
  }

  // --- Ruiz is actually going to run: upload the rest and build the index arrays. ---
  std::vector<f_t> col_scale_host(n, 1.0);
  rmm::device_uvector<f_t> d_rhs       = cuopt::device_copy(scaled.rhs, stream);
  rmm::device_uvector<f_t> d_objective = cuopt::device_copy(scaled.objective, stream);
  rmm::device_uvector<f_t> d_lower     = cuopt::device_copy(scaled.lower, stream);
  rmm::device_uvector<f_t> d_upper     = cuopt::device_copy(scaled.upper, stream);
  rmm::device_uvector<f_t> d_row_scale = cuopt::device_copy(row_scaling, stream);
  rmm::device_uvector<f_t> d_col_scale = cuopt::device_copy(col_scale_host, stream);

  // Per-nonzero column ids for A, built once (sparsity pattern is fixed across iterations).
  dA.form_col_index(stream);  // dA.col_index[p] = column of nonzero p

  const i_t cone_start = unscaled.second_order_cone_dims.empty() ? n : unscaled.cone_var_start;
  const i_t num_cones  = static_cast<i_t>(unscaled.second_order_cone_dims.size());
  // Column boundaries of each cone (in the global column index space) and, for every
  // cone column, which cone it belongs to -- both built once, used every iteration.
  std::vector<i_t> cone_col_offsets_host(num_cones + 1, cone_start);
  for (i_t k = 0; k < num_cones; ++k) {
    cone_col_offsets_host[k + 1] = cone_col_offsets_host[k] + unscaled.second_order_cone_dims[k];
  }
  std::vector<i_t> col_cone_id_host(n - cone_start);
  for (i_t k = 0; k < num_cones; ++k) {
    for (i_t j = cone_col_offsets_host[k]; j < cone_col_offsets_host[k + 1]; ++j) {
      col_cone_id_host[j - cone_start] = k;
    }
  }
  rmm::device_uvector<i_t> d_cone_col_offsets = cuopt::device_copy(cone_col_offsets_host, stream);
  rmm::device_uvector<i_t> d_col_cone_id      = cuopt::device_copy(col_cone_id_host, stream);

  // --- Ruiz iteration loop (mirrors scaling.cpp:123-224) ---
  constexpr i_t max_ruiz_iterations = 10;
  rmm::device_uvector<f_t> c(n, stream);
  rmm::device_uvector<f_t> col_max_linear(cone_start, stream);
  rmm::device_uvector<f_t> qrow_max_linear(has_q ? cone_start : 0, stream);
  rmm::device_uvector<f_t> cone_max(num_cones, stream);

  for (i_t iter = 0; iter < max_ruiz_iterations; ++iter) {
    f_t max_deviation = 0.0;

    // --- Row scaling: r[i] = 1/sqrt(max_j |A(i,j)|) ---
    // On the first pass r still holds the row inf-norms computed for the skip heuristic
    // above, and A has not been touched since, so only recompute once A has been scaled.
    if (iter > 0) { compute_row_inf_norms(dA, r, stream); }
    max_deviation = std::max(
      max_deviation,
      whole_array_abs_max<f_t>(thrust::make_transform_iterator(
                                 r.data(), [] __device__(f_t v) -> f_t { return v - f_t(1); }),
                               m,
                               stream));
    thrust::transform(
      rmm::exec_policy(stream), r.data(), r.data() + m, r.data(), [] __device__(f_t rm) {
        return rm > 0 ? f_t(1) / std::sqrt(rm) : f_t(1);
      });

    thrust::for_each(
      rmm::exec_policy(stream),
      thrust::make_counting_iterator(i_t(0)),
      thrust::make_counting_iterator(dA.nz_max),
      [x = dA.x.data(), row = dA.i.data(), r = r.data()] __device__(i_t p) { x[p] *= r[row[p]]; });
    thrust::transform(rmm::exec_policy(stream),
                      d_rhs.data(),
                      d_rhs.data() + m,
                      r.data(),
                      d_rhs.data(),
                      cuda::std::multiplies<f_t>{});
    thrust::transform(rmm::exec_policy(stream),
                      d_row_scale.data(),
                      d_row_scale.data() + m,
                      r.data(),
                      d_row_scale.data(),
                      cuda::std::multiplies<f_t>{});

    // --- Column scaling: linear columns [0, cone_start) combine A and Q; cone columns
    // use one uniform scale per cone. ---
    if (cone_start > 0) {
      segmented_abs_max<i_t, f_t>(dA.x.data(),
                                  dA.col_start.data(),
                                  dA.col_start.data() + 1,
                                  cone_start,
                                  col_max_linear.data(),
                                  scratch,
                                  stream);
      if (has_q) {
        segmented_abs_max<i_t, f_t>(dQ.x.data(),
                                    dQ.row_start.data(),
                                    dQ.row_start.data() + 1,
                                    cone_start,
                                    qrow_max_linear.data(),
                                    scratch,
                                    stream);
        thrust::transform(rmm::exec_policy(stream),
                          col_max_linear.data(),
                          col_max_linear.data() + cone_start,
                          qrow_max_linear.data(),
                          col_max_linear.data(),
                          cuopt::max_op_t<f_t>{});
      }
      max_deviation =
        std::max(max_deviation,
                 whole_array_abs_max<f_t>(
                   thrust::make_transform_iterator(
                     col_max_linear.data(), [] __device__(f_t v) -> f_t { return v - f_t(1); }),
                   cone_start,
                   stream));
      thrust::transform(rmm::exec_policy(stream),
                        col_max_linear.data(),
                        col_max_linear.data() + cone_start,
                        c.data(),
                        [] __device__(f_t cm) { return cm > 0 ? f_t(1) / std::sqrt(cm) : f_t(1); });
    }
    if (num_cones > 0) {
      auto begin_it =
        thrust::make_permutation_iterator(dA.col_start.data(), d_cone_col_offsets.data());
      auto end_it =
        thrust::make_permutation_iterator(dA.col_start.data(), d_cone_col_offsets.data() + 1);
      segmented_abs_max<i_t, f_t>(
        dA.x.data(), begin_it, end_it, num_cones, cone_max.data(), scratch, stream);
      max_deviation =
        std::max(max_deviation,
                 whole_array_abs_max<f_t>(
                   thrust::make_transform_iterator(
                     cone_max.data(), [] __device__(f_t v) -> f_t { return v - f_t(1); }),
                   num_cones,
                   stream));
      thrust::transform(rmm::exec_policy(stream),
                        cone_max.data(),
                        cone_max.data() + num_cones,
                        cone_max.data(),
                        [] __device__(f_t cm) { return cm > 0 ? f_t(1) / std::sqrt(cm) : f_t(1); });
      thrust::gather(rmm::exec_policy(stream),
                     d_col_cone_id.data(),
                     d_col_cone_id.data() + (n - cone_start),
                     cone_max.data(),
                     c.data() + cone_start);
    }

    thrust::for_each(rmm::exec_policy(stream),
                     thrust::make_counting_iterator(i_t(0)),
                     thrust::make_counting_iterator(dA.nz_max),
                     [x = dA.x.data(), col = dA.col_index.data(), c = c.data()] __device__(i_t p) {
                       x[p] *= c[col[p]];
                     });
    thrust::transform(rmm::exec_policy(stream),
                      d_objective.data(),
                      d_objective.data() + n,
                      c.data(),
                      d_objective.data(),
                      cuda::std::multiplies<f_t>{});
    thrust::transform(rmm::exec_policy(stream),
                      d_col_scale.data(),
                      d_col_scale.data() + n,
                      c.data(),
                      d_col_scale.data(),
                      cuda::std::multiplies<f_t>{});
    thrust::for_each(
      rmm::exec_policy(stream),
      thrust::make_counting_iterator(i_t(0)),
      thrust::make_counting_iterator(n),
      [lower = d_lower.data(), upper = d_upper.data(), c = c.data()] __device__(i_t j) {
        if (lower[j] > f_t(-1e20)) lower[j] /= c[j];
        if (upper[j] < f_t(1e20)) upper[j] /= c[j];
      });
    if (has_q) {
      // Row-parallel so the row index comes from the iteration variable, as in scaling.cpp.
      // Deriving it per nonzero from row_start instead would mis-attribute every nonzero
      // after an empty row, and Q has an empty row for every variable with no quadratic term.
      thrust::for_each(
        rmm::exec_policy(stream),
        thrust::make_counting_iterator(i_t(0)),
        thrust::make_counting_iterator(dQ.m),
        [x = dQ.x.data(), rs = dQ.row_start.data(), col = dQ.j.data(), c = c.data()] __device__(
          i_t i) {
          for (i_t p = rs[i]; p < rs[i + 1]; ++p) {
            x[p] *= c[i] * c[col[p]];
          }
        });
    }

    if (max_deviation < 0.1) break;
  }

  // --- Finalize: invert accumulated reciprocal scales (mirrors scaling.cpp:226-235) ---
  thrust::transform(rmm::exec_policy(stream),
                    d_col_scale.data(),
                    d_col_scale.data() + n,
                    d_col_scale.data(),
                    [] __device__(f_t v) { return f_t(1) / v; });
  thrust::transform(rmm::exec_policy(stream),
                    d_row_scale.data(),
                    d_row_scale.data() + m,
                    d_row_scale.data(),
                    [] __device__(f_t v) { return f_t(1) / v; });

  const f_t a_max = whole_array_abs_max<f_t>(dA.x.data(), dA.nz_max, stream);
  const f_t a_min = whole_array_nonzero_abs_min<f_t>(dA.x.data(), dA.nz_max, stream);

  // --- Download scaled problem and scale vectors back to host ---
  // SOCP goes straight to the barrier's augmented path, where no host code reads A's values, so
  // A stays on device rather than being downloaded and immediately uploaded again.
  if (!unscaled.second_order_cone_dims.empty()) {
    scaled.A.x.clear();
    scaled.A.x.shrink_to_fit();
    device_A = std::make_unique<device_csc_matrix_t<i_t, f_t>>(std::move(dA));
  } else {
    scaled.A = dA.to_host(stream);
  }
  scaled.Q = dQ.to_host(stream);
  // Symmetric Q: its CSR arrays are also its CSC arrays.
  if (dQ.nz_max > 0) {
    auto dQ_csc       = std::make_unique<device_csc_matrix_t<i_t, f_t>>(stream);
    dQ_csc->m         = dQ.m;
    dQ_csc->n         = dQ.m;
    dQ_csc->nz_max    = dQ.nz_max;
    dQ_csc->col_start = std::move(dQ.row_start);
    dQ_csc->i         = std::move(dQ.j);
    dQ_csc->x         = std::move(dQ.x);
    device_Q          = std::move(dQ_csc);
  }
  scaled.rhs       = cuopt::host_copy(d_rhs, stream);
  scaled.objective = cuopt::host_copy(d_objective, stream);
  scaled.lower     = cuopt::host_copy(d_lower, stream);
  scaled.upper     = cuopt::host_copy(d_upper, stream);
  column_scaling   = cuopt::host_copy(d_col_scale, stream);
  row_scaling      = cuopt::host_copy(d_row_scale, stream);

  settings.log.printf("Ruiz equilibration: coefficient range [%e, %e] [GPU]\n", a_min, a_max);
  return 0;
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE

template int scaling_ruiz_gpu<int, double>(
  const lp_problem_t<int, double>& unscaled,
  const simplex_solver_settings_t<int, double>& settings,
  lp_problem_t<int, double>& scaled,
  std::vector<double>& column_scaling,
  std::vector<double>& row_scaling,
  std::unique_ptr<device_csc_matrix_t<int, double>>& device_A,
  std::unique_ptr<device_csc_matrix_t<int, double>>& device_Q);

#endif

}  // namespace cuopt::mathematical_optimization::simplex
