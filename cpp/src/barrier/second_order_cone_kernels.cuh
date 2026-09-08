/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <barrier/second_order_cone_reduction.cuh>

#include <utilities/copy_helpers.hpp>
#include <utilities/cuda_helpers.cuh>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <raft/core/device_span.hpp>

#include <thrust/binary_search.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>
#include <thrust/zip_function.h>

#include <concepts>
#include <cstddef>
#include <numeric>
#include <span>
#include <utility>

// =============================================================================
// SOC (second-order cone) kernels for the cuOpt barrier solver.
//
//   x_soc     : cone primal block
//   z_soc     : cone dual block
//   W, W^{-1} : Nesterov-Todd scaling matrix and inverse. W is symmetric for
//               SOC, so W^{T} = W
//   H         : S^{T} S = S^{2}, the cone KKT block added to the
//               primal-reduced system
//   eta       : sqrt(z_J / x_J), where x_J = sqrt(det_J(x_soc))
//   w         : NT scaling direction with det_J(w) = 1 and
//               w[head] = sqrt(1 + ||w_tail||^2)
//
// Cone vectors are packed flat:
// entries [cone_offsets[i], cone_offsets[i + 1]) belong to cone i.
// =============================================================================

namespace cuopt::mathematical_optimization::barrier {

inline constexpr int soc_block_size = 256;

/**
 * Tail aggregates for the cone step-length reduction (CUB value type).
 */
template <std::floating_point f_t>
struct step_tail_sums_t {
  f_t du_tail_sq{};
  f_t u_tail_du_tail{};
  f_t u_tail_sq{};

  __host__ __device__ step_tail_sums_t operator+(step_tail_sums_t o) const
  {
    return {du_tail_sq + o.du_tail_sq, u_tail_du_tail + o.u_tail_du_tail, u_tail_sq + o.u_tail_sq};
  }
};

/**
 * Reusable device workspace for second-order cone kernels.
 *
 * The scratch object owns only temporary storage. Kernels may reuse the scalar
 * slots and `temp_cone` sequentially inside a higher-level operation, but no
 * persistent NT scaling or iterate state is stored here.
 */
template <std::integral i_t, std::floating_point f_t, int n_slots = 3>
struct cone_scratch_t {
  i_t n_cones;            // number of SOC blocks
  size_t n_cone_entries;  // total packed cone dimension

  rmm::device_uvector<f_t> slots;  // [n_slots * n_cones]

  // Per-cone step candidates before the final min reduction.
  rmm::device_uvector<f_t> step_alpha_primal;  // [n_cones]
  rmm::device_uvector<f_t> step_alpha_dual;    // [n_cones]

  // Large-cone step-length CUB outputs, one packed slot per large cone.
  rmm::device_uvector<step_tail_sums_t<f_t>> step_large_tail_sums;  // [n_large]

  // TODO: Consider moving this out to the barrier layer when we wire it in
  rmm::device_uvector<f_t> temp_cone;  // [n_cone_entries]

  cone_scratch_t(i_t n_cones_in,
                 size_t n_cone_entries_in,
                 size_t n_large,
                 rmm::cuda_stream_view stream)
    : n_cones(n_cones_in),
      n_cone_entries(n_cone_entries_in),
      slots(0, stream),
      step_alpha_primal(0, stream),
      step_alpha_dual(0, stream),
      step_large_tail_sums(0, stream),
      temp_cone(0, stream)
  {
    const size_t n_cones_size = static_cast<size_t>(n_cones);

    slots.resize(n_cones_size * static_cast<size_t>(n_slots), stream);
    step_alpha_primal.resize(n_cones_size, stream);
    step_alpha_dual.resize(n_cones_size, stream);
    if (n_large > 0) { step_large_tail_sums.resize(n_large, stream); }
    temp_cone.resize(n_cone_entries, stream);
  }

  template <int slot_idx>
  raft::device_span<const f_t> get_slot() const
  {
    static_assert(slot_idx >= 0 && slot_idx < n_slots, "scratch slot index out of range");
    const size_t n_cones_size = static_cast<size_t>(n_cones);
    const size_t begin        = static_cast<size_t>(slot_idx) * n_cones_size;
    const size_t end          = begin + n_cones_size;
    return cuopt::make_span(slots, begin, end);
  }

  template <int slot_idx>
  raft::device_span<f_t> get_slot()
  {
    const auto const_slot = static_cast<cone_scratch_t const&>(*this).template get_slot<slot_idx>();
    return raft::device_span<f_t>(const_cast<f_t*>(const_slot.data()), const_slot.size());
  }
};

struct to_size_t_t {
  template <typename value_t>
  HD size_t operator()(value_t value) const
  {
    return value;
  }
};

/**
 * Device storage for second-order cone topology, NT scaling, and iterate views.
 *
 * Flat arrays are packed by cone: entries
 * [cone_offsets[i], cone_offsets[i + 1]) belong to cone i, whose dimension is
 * cone_dimensions[i].
 *
 * The primal/dual cone vectors are non-owning spans over the SOC slice of the
 * solver's global x/z vectors. The caller must keep the underlying storage
 * alive for the lifetime of this object.
 */
template <std::integral i_t, std::floating_point f_t>
struct cone_data_t {
  // Topology. This is immutable after construction.
  i_t n_cones;            // number of SOC blocks
  size_t n_cone_entries;  // total packed cone dimension = sum(cone_dimensions)

  rmm::device_uvector<size_t> cone_offsets;  // [n_cones + 1], prefix sum of dimensions
  rmm::device_uvector<i_t> cone_dimensions;  // [n_cones], dimension q_i of each cone
  // Owning cone per entry for upcoming flat per-entry SOC kernels.
  rmm::device_uvector<i_t> element_cone_ids;  // [n_cone_entries]
  segmented_sum_t<i_t> segmented_sum;

  // Non-owning iterate views over the cone portion of x/z.
  raft::device_span<f_t> x;  // [n_cone_entries], SOC primal block
  raft::device_span<f_t> z;  // [n_cone_entries], SOC dual block

  // Persistent Nesterov-Todd scaling state, recomputed from x/z each iteration.
  rmm::device_uvector<f_t> eta;     // [n_cones], sqrt(|z|_J / |x|_J)
  rmm::device_uvector<f_t> w;       // [n_cone_entries], unit-J-norm NT direction
  rmm::device_uvector<f_t> lambda;  // [n_cone_entries], NT point lambda = W^{-T} z

  // Sparse SOC rank-2 scaling (cones with dimension > soc_threshold).
  i_t soc_threshold;
  i_t n_sparse_cones;
  size_t n_sparse_cone_entries;               // sum of sparse cone dimensions
  rmm::device_uvector<i_t> sparse_cone_ids;   // [n_sparse_cones], indices into cone arrays
  rmm::device_uvector<i_t> sparse_cone_dims;  // [n_sparse_cones], dimension of each sparse cone
  rmm::device_uvector<f_t> d;                 // [n_sparse_cones], corner of rank-2 diagonal block
  rmm::device_uvector<f_t> sparse_v;  // [n_sparse_cone_entries], rank-2 vector v per sparse cone
  rmm::device_uvector<f_t> sparse_u;  // [n_sparse_cone_entries], rank-2 vector u per sparse cone
  rmm::device_uvector<i_t>
    sparse_entry_offsets;  // [n_sparse_cones + 1], packed prefix offsets of sparse cone entries
  rmm::device_uvector<i_t> cone_is_sparse;  // [n_cones], 1 if cone uses sparse KKT expansion

  cone_scratch_t<i_t, f_t> scratch;

  cone_data_t(std::span<const i_t> cone_dimensions_host,
              raft::device_span<f_t> x_in,
              raft::device_span<f_t> z_in,
              rmm::cuda_stream_view stream,
              i_t soc_threshold_in = 100)
    : n_cones(cone_dimensions_host.size()),
      n_cone_entries(
        std::reduce(cone_dimensions_host.begin(), cone_dimensions_host.end(), size_t{0})),
      cone_offsets(n_cones + 1, stream),
      cone_dimensions(n_cones, stream),
      element_cone_ids(n_cone_entries, stream),
      segmented_sum(cone_dimensions_host, cuopt::make_span(cone_offsets), stream),
      x(x_in),
      z(z_in),
      eta(n_cones, stream),
      w(n_cone_entries, stream),
      lambda(n_cone_entries, stream),
      sparse_cone_ids(0, stream),
      sparse_cone_dims(0, stream),
      d(0, stream),
      sparse_v(0, stream),
      sparse_u(0, stream),
      sparse_entry_offsets(0, stream),
      cone_is_sparse(n_cones, stream),
      scratch(n_cones, n_cone_entries, segmented_sum.large_cone_ids.size(), stream),
      soc_threshold(soc_threshold_in),
      n_sparse_cones(0),
      n_sparse_cone_entries(0)
  {
    thrust::fill(rmm::exec_policy(stream), cone_is_sparse.begin(), cone_is_sparse.end(), 0);

    std::vector<i_t> sparse_cone_ids_host;
    std::vector<i_t> sparse_cone_dims_host;
    sparse_cone_ids_host.reserve(n_cones);
    sparse_cone_dims_host.reserve(n_cones);
    for (i_t cone = 0; cone < n_cones; ++cone) {
      if (cone_dimensions_host[cone] > soc_threshold) {
        sparse_cone_ids_host.push_back(cone);
        sparse_cone_dims_host.push_back(cone_dimensions_host[cone]);
      }
    }
    n_sparse_cones = static_cast<i_t>(sparse_cone_ids_host.size());
    n_sparse_cone_entries =
      std::reduce(sparse_cone_dims_host.begin(), sparse_cone_dims_host.end(), size_t{0});

    if (n_sparse_cones > 0) {
      sparse_cone_ids.resize(n_sparse_cones, stream);
      sparse_cone_dims.resize(n_sparse_cones, stream);
      d.resize(n_sparse_cones, stream);
      sparse_v.resize(n_sparse_cone_entries, stream);
      sparse_u.resize(n_sparse_cone_entries, stream);
      sparse_entry_offsets.resize(n_sparse_cones + 1, stream);
      raft::copy(sparse_cone_ids.data(), sparse_cone_ids_host.data(), n_sparse_cones, stream);
      raft::copy(sparse_cone_dims.data(), sparse_cone_dims_host.data(), n_sparse_cones, stream);
      std::vector<i_t> cone_is_sparse_host(n_cones, 0);
      for (i_t cone : sparse_cone_ids_host) {
        cone_is_sparse_host[cone] = 1;
      }
      raft::copy(cone_is_sparse.data(), cone_is_sparse_host.data(), n_cones, stream);

      // Packed prefix offsets of sparse cone entries.
      std::vector<i_t> sparse_entry_offsets_host(n_sparse_cones + 1, 0);
      for (i_t s = 0; s < n_sparse_cones; ++s) {
        sparse_entry_offsets_host[s + 1] = sparse_entry_offsets_host[s] + sparse_cone_dims_host[s];
      }
      raft::copy(
        sparse_entry_offsets.data(), sparse_entry_offsets_host.data(), n_sparse_cones + 1, stream);
    }

    raft::copy(cone_dimensions.data(), cone_dimensions_host.data(), n_cones, stream);
    cone_offsets.set_element_to_zero_async(0, stream);
    auto policy = rmm::exec_policy(stream);

    auto cone_dimensions_as_offsets =
      thrust::make_transform_iterator(cone_dimensions.begin(), to_size_t_t{});
    thrust::inclusive_scan(policy,
                           cone_dimensions_as_offsets,
                           cone_dimensions_as_offsets + n_cones,
                           cone_offsets.begin() + 1,
                           cuda::std::plus<size_t>{});

    thrust::upper_bound(policy,
                        cone_offsets.begin() + 1,
                        cone_offsets.end(),
                        thrust::make_counting_iterator<size_t>(0),
                        thrust::make_counting_iterator<size_t>(n_cone_entries),
                        element_cone_ids.begin());
    segmented_sum.template prepare_workspace<f_t, step_tail_sums_t<f_t>>(stream);
  }

  // True when at least one cone is large enough (dim > soc_threshold) to use the
  // rank-2 sparse KKT expansion instead of a dense q x q block.
  bool has_sparse_cones() const { return n_sparse_cones > 0; }

  // Number of cones kept dense (dim <= soc_threshold).
  i_t n_dense_cones() const { return n_cones - n_sparse_cones; }

  // Extra augmented-system columns/rows: two expansion variables (v, u) per sparse cone.
  i_t expansion_var_count() const { return 2 * n_sparse_cones; }
};

template <std::integral i_t, std::floating_point f_t>
__global__ void __launch_bounds__(soc_block_size)
  nt_finalize_scaling_scalars_kernel(raft::device_span<const f_t> x,
                                     raft::device_span<const f_t> z,
                                     raft::device_span<f_t> x_scale,
                                     raft::device_span<f_t> z_scale,
                                     raft::device_span<f_t> eta,
                                     raft::device_span<const size_t> cone_offsets,
                                     i_t n_cones)
{
  const i_t cone = static_cast<i_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (cone >= n_cones) { return; }

  const size_t off      = cone_offsets[cone];
  const f_t x_tail_norm = sqrt(x_scale[cone]);
  const f_t z_tail_norm = sqrt(z_scale[cone]);
  const f_t x_det       = (x[off] - x_tail_norm) * (x[off] + x_tail_norm);
  const f_t z_det       = (z[off] - z_tail_norm) * (z[off] + z_tail_norm);

  x_scale[cone] = sqrt(x_det);
  z_scale[cone] = sqrt(z_det);
  eta[cone]     = sqrt(z_scale[cone] / x_scale[cone]);
}

template <std::integral i_t, std::floating_point f_t>
__global__ void __launch_bounds__(soc_block_size)
  nt_finalize_w_scale_kernel(raft::device_span<const f_t> w,
                             raft::device_span<const f_t> tail_sq,
                             raft::device_span<f_t> w_scale,
                             raft::device_span<const size_t> cone_offsets,
                             i_t n_cones)
{
  const i_t cone = static_cast<i_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (cone >= n_cones) { return; }

  const size_t cone_off = cone_offsets[cone];
  const f_t head        = w[cone_off];
  const f_t tail_norm   = sqrt(tail_sq[cone]);
  const f_t residual    = (head - tail_norm) * (head + tail_norm);
  w_scale[cone]         = sqrt(residual);
}

/**
 * Write unnormalized w:
 *
 *   w_0 = z_0 / z_scale + x_0 / x_scale
 *   w_tail = z_tail / z_scale - x_tail / x_scale.
 */
template <std::integral i_t, std::floating_point f_t>
__global__ void __launch_bounds__(soc_block_size)
  nt_write_w_kernel(raft::device_span<const f_t> x,
                    raft::device_span<const f_t> z,
                    raft::device_span<const f_t> x_scale,
                    raft::device_span<const f_t> z_scale,
                    raft::device_span<f_t> w,
                    raft::device_span<const size_t> cone_offsets,
                    raft::device_span<const i_t> element_cone_ids)
{
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= w.size()) { return; }

  const i_t cone        = element_cone_ids[idx];
  const size_t cone_off = cone_offsets[cone];
  if (idx == cone_off) {
    w[idx] = z[idx] / z_scale[cone] + x[idx] / x_scale[cone];
    return;
  }

  w[idx] = z[idx] / z_scale[cone] - x[idx] / x_scale[cone];
}

template <std::integral i_t, std::floating_point f_t>
__global__ void __launch_bounds__(soc_block_size)
  nt_normalize_w_kernel(raft::device_span<f_t> w,
                        raft::device_span<const f_t> w_scale,
                        raft::device_span<const i_t> element_cone_ids)
{
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= w.size()) { return; }

  const i_t cone = element_cone_ids[idx];
  w[idx] /= w_scale[cone];
}

template <std::integral i_t, std::floating_point f_t>
__global__ void __launch_bounds__(soc_block_size)
  nt_finalize_head_kernel(raft::device_span<f_t> w,
                          raft::device_span<const f_t> normalized_tail_sq,
                          raft::device_span<const size_t> cone_offsets,
                          i_t n_cones)
{
  const i_t cone = static_cast<i_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (cone >= n_cones) { return; }

  w[cone_offsets[cone]] = sqrt(1 + normalized_tail_sq[cone]);
}

template <std::integral i_t, std::floating_point f_t>
__global__ void __launch_bounds__(soc_block_size)
  nt_write_lambda_kernel(raft::device_span<const f_t> x,
                         raft::device_span<const f_t> z,
                         raft::device_span<const f_t> x_scale,
                         raft::device_span<const f_t> z_scale,
                         raft::device_span<const f_t> w_scale,
                         raft::device_span<f_t> lambda,
                         raft::device_span<const size_t> cone_offsets,
                         raft::device_span<const i_t> element_cone_ids)
{
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= lambda.size()) { return; }

  const i_t cone         = element_cone_ids[idx];
  const size_t cone_off  = cone_offsets[cone];
  const size_t local_idx = idx - cone_off;

  const f_t x_scale_cone = x_scale[cone];
  const f_t z_scale_cone = z_scale[cone];
  const f_t gamma        = static_cast<f_t>(0.5) * w_scale[cone];
  const f_t head_scale   = sqrt(x_scale_cone * z_scale_cone);

  if (local_idx == 0) {
    lambda[idx] = gamma * head_scale;
    return;
  }

  const f_t x_head  = x[cone_off];
  const f_t z_head  = z[cone_off];
  const f_t denom   = z_head / z_scale_cone + x_head / x_scale_cone + static_cast<f_t>(2) * gamma;
  const f_t coeff_z = (gamma + x_head / x_scale_cone) / z_scale_cone;
  const f_t coeff_x = (gamma + z_head / z_scale_cone) / x_scale_cone;

  const f_t lambda_tail = (coeff_z * z[idx] + coeff_x * x[idx]) / denom;
  lambda[idx]           = lambda_tail * head_scale;
}

/**
 * Build Nesterov-Todd scaling for packed SOC blocks.
 *
 * Given interior cone primal/dual blocks x and z:
 *
 *   det_J(x) = x_0^2 - ||x_tail||^2
 *   det_J(z) = z_0^2 - ||z_tail||^2
 *   x_scale = sqrt(det_J(x)), z_scale = sqrt(det_J(z))
 *   eta     = sqrt(z_scale / x_scale)
 *   w_tmp_0 = z_0 / z_scale + x_0 / x_scale
 *   w_tmp_tail = z_tail / z_scale - x_tail / x_scale
 *   w_scale = sqrt(det_J(w_tmp))
 *   w = w_tmp / w_scale
 *   w_0 = sqrt(1 + ||w_tail||^2) to re-impose det_J(w) = 1
 *
 * Scratch slots:
 *   0: ||x_tail||^2 -> x_scale
 *   1: ||z_tail||^2 -> z_scale
 */
template <std::integral i_t, std::floating_point f_t>
void launch_nt_scaling(cone_data_t<i_t, f_t>& cones, rmm::cuda_stream_view stream)
{
  auto x_scale = cones.scratch.template get_slot<0>();
  auto z_scale = cones.scratch.template get_slot<1>();
  auto w_scale = cones.scratch.template get_slot<2>();

  const auto span_x           = cones.x;
  const auto span_z           = cones.z;
  const auto cone_offsets     = cuopt::make_span(cones.cone_offsets);
  const auto element_cone_ids = cuopt::make_span(cones.element_cone_ids);

  auto x_tail_sq_terms = thrust::make_transform_iterator(
    thrust::make_counting_iterator<size_t>(0),
    [span_x, cone_offsets, element_cone_ids] HD(size_t idx) -> f_t {
      const i_t cone = element_cone_ids[idx];
      return idx == cone_offsets[cone] ? 0 : span_x[idx] * span_x[idx];
    });
  cones.segmented_sum(x_tail_sq_terms, x_scale, stream);

  auto z_tail_sq_terms = thrust::make_transform_iterator(
    thrust::make_counting_iterator<size_t>(0),
    [span_z, cone_offsets, element_cone_ids] HD(size_t idx) -> f_t {
      const i_t cone = element_cone_ids[idx];
      return idx == cone_offsets[cone] ? 0 : span_z[idx] * span_z[idx];
    });
  cones.segmented_sum(z_tail_sq_terms, z_scale, stream);

  const size_t cone_grid_dim =
    raft::ceildiv<size_t>(static_cast<size_t>(cones.n_cones), soc_block_size);
  nt_finalize_scaling_scalars_kernel<i_t, f_t><<<cone_grid_dim, soc_block_size, 0, stream.get()>>>(
    cones.x, cones.z, x_scale, z_scale, cuopt::make_span(cones.eta), cone_offsets, cones.n_cones);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  const size_t element_grid_dim = raft::ceildiv<size_t>(cones.n_cone_entries, soc_block_size);

  auto w = cuopt::make_span(cones.w);
  nt_write_w_kernel<i_t, f_t><<<element_grid_dim, soc_block_size, 0, stream.get()>>>(
    cones.x, cones.z, x_scale, z_scale, w, cone_offsets, element_cone_ids);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  auto unnormalized_tail_sq_terms =
    thrust::make_transform_iterator(thrust::make_counting_iterator<size_t>(0),
                                    [cone_offsets, element_cone_ids, w] HD(size_t idx) -> f_t {
                                      const i_t cone = element_cone_ids[idx];
                                      return idx == cone_offsets[cone] ? 0 : w[idx] * w[idx];
                                    });
  cones.segmented_sum(unnormalized_tail_sq_terms, w_scale, stream);

  nt_finalize_w_scale_kernel<i_t, f_t><<<cone_grid_dim, soc_block_size, 0, stream.get()>>>(
    w, w_scale, w_scale, cone_offsets, cones.n_cones);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  nt_normalize_w_kernel<i_t, f_t>
    <<<element_grid_dim, soc_block_size, 0, stream.get()>>>(w, w_scale, element_cone_ids);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  // Persist lambda while w_scale still stores sqrt(det_J(w_tmp)).
  nt_write_lambda_kernel<i_t, f_t>
    <<<element_grid_dim, soc_block_size, 0, stream.get()>>>(cones.x,
                                                            cones.z,
                                                            x_scale,
                                                            z_scale,
                                                            w_scale,
                                                            cuopt::make_span(cones.lambda),
                                                            cone_offsets,
                                                            element_cone_ids);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  // w_scale is overwritten from here
  auto normalized_tail_terms =
    thrust::make_transform_iterator(thrust::make_counting_iterator<size_t>(0),
                                    [cone_offsets, element_cone_ids, w] HD(size_t idx) -> f_t {
                                      const i_t cone = element_cone_ids[idx];
                                      return idx == cone_offsets[cone] ? 0 : w[idx] * w[idx];
                                    });
  cones.segmented_sum(normalized_tail_terms, w_scale, stream);

  nt_finalize_head_kernel<i_t, f_t><<<cone_grid_dim, soc_block_size, 0, stream.get()>>>(
    cuopt::make_span(cones.w), w_scale, cone_offsets, cones.n_cones);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

// One block per sparse cone. Recompute the rank-2 factors (corner d and the
// vectors v, u, both scaled by eta^2) from the current NT direction w so that
// the implicit block reproduces the dense H = eta^2 (2 w w^T - J).
template <std::integral i_t, std::floating_point f_t>
__global__ void update_scaling_sparse_kernel(raft::device_span<const f_t> w,
                                             raft::device_span<const f_t> eta,
                                             raft::device_span<f_t> d,
                                             raft::device_span<f_t> sparse_v,
                                             raft::device_span<f_t> sparse_u,
                                             raft::device_span<const size_t> cone_offsets,
                                             raft::device_span<const i_t> sparse_cone_dims,
                                             raft::device_span<const i_t> sparse_cone_ids,
                                             raft::device_span<const i_t> sparse_entry_offsets,
                                             i_t n_sparse_cones)
{
  const i_t sparse_idx = blockIdx.x;
  if (sparse_idx >= n_sparse_cones) { return; }

  __shared__ f_t s_mem[4];

  const i_t cone_idx    = sparse_cone_ids[sparse_idx];
  const size_t cone_off = cone_offsets[cone_idx];
  const i_t cone_dim    = sparse_cone_dims[sparse_idx];
  const i_t block_start = sparse_entry_offsets[sparse_idx];

  if (threadIdx.x == 0) {
    const f_t alpha    = f_t(2) * w[cone_off];
    const f_t wsq      = f_t(2) * w[cone_off] * w[cone_off] - f_t(1);
    const f_t wsq_safe = f_t(0.5) * (wsq + sqrt(wsq * wsq + f_t(1)));
    const f_t wsqinv   = f_t(1) / wsq_safe;
    const f_t di       = f_t(0.5) * wsqinv;
    d[sparse_idx]      = di;
    const f_t radicand = wsq_safe - di;
    const f_t u0       = sqrt(max(radicand, f_t(0)));
    const f_t u1       = (u0 > f_t(0)) ? alpha / u0 : f_t(0);
    const f_t v0       = f_t(0);
    const f_t denom    = f_t(2) * wsq_safe - wsqinv;
    const f_t v1_arg   = (abs(denom) > f_t(1e-12)) ? f_t(2) * (f_t(2) + wsqinv) / denom : f_t(2);
    const f_t v1       = sqrt(max(v1_arg, f_t(0)));
    const f_t eta_sq   = eta[cone_idx] * eta[cone_idx];
    s_mem[0]           = eta_sq * u0;
    s_mem[1]           = eta_sq * u1;
    s_mem[2]           = eta_sq * v0;
    s_mem[3]           = eta_sq * v1;
  }
  __syncthreads();

  const f_t scaled_u0 = s_mem[0];
  const f_t scaled_u1 = s_mem[1];
  const f_t scaled_v0 = s_mem[2];
  const f_t scaled_v1 = s_mem[3];

  for (i_t j = threadIdx.x; j < cone_dim; j += blockDim.x) {
    if (j == 0) {
      sparse_u[block_start + j] = scaled_u0;
      sparse_v[block_start + j] = scaled_v0;
    } else {
      sparse_u[block_start + j] = scaled_u1 * w[cone_off + j];
      sparse_v[block_start + j] = scaled_v1 * w[cone_off + j];
    }
  }
}

/**
 * Refresh the rank-2 NT factors (d, v, u) of every sparse cone from the current
 * scaling, so the implicit sparse block matches the dense Hessian for this
 * iteration. Call after `launch_nt_scaling` has updated w and eta.
 */
template <std::integral i_t, std::floating_point f_t>
void launch_update_scaling_sparse(cone_data_t<i_t, f_t>& cones, rmm::cuda_stream_view stream)
{
  if (!cones.has_sparse_cones()) { return; }

  const i_t n_sparse = cones.n_sparse_cones;
  update_scaling_sparse_kernel<i_t, f_t>
    <<<n_sparse, soc_block_size, 0, stream.get()>>>(cuopt::make_span(cones.w),
                                                    cuopt::make_span(cones.eta),
                                                    cuopt::make_span(cones.d),
                                                    cuopt::make_span(cones.sparse_v),
                                                    cuopt::make_span(cones.sparse_u),
                                                    cuopt::make_span(cones.cone_offsets),
                                                    cuopt::make_span(cones.sparse_cone_dims),
                                                    cuopt::make_span(cones.sparse_cone_ids),
                                                    cuopt::make_span(cones.sparse_entry_offsets),
                                                    n_sparse);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <std::integral i_t, std::floating_point f_t>
__global__ void __launch_bounds__(soc_block_size)
  apply_w_inv_write_kernel(raft::device_span<const f_t> v,
                           raft::device_span<f_t> out,
                           raft::device_span<const f_t> w,
                           raft::device_span<const f_t> eta,
                           raft::device_span<const f_t> tail_dot,
                           raft::device_span<const size_t> cone_offsets,
                           raft::device_span<const i_t> element_cone_ids)
{
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= out.size()) { return; }

  const i_t cone         = element_cone_ids[idx];
  const size_t cone_off  = cone_offsets[cone];
  const size_t local_idx = idx - cone_off;

  const f_t w0      = w[cone_off];
  const f_t zeta    = tail_dot[cone];
  const f_t v0      = v[cone_off];
  const f_t inv_eta = f_t(1) / eta[cone];

  if (local_idx == 0) {
    out[idx] = inv_eta * (w0 * v0 - zeta);
    return;
  }

  const f_t coeff = -v0 + zeta / (f_t(1) + w0);
  out[idx]        = inv_eta * (v[idx] + coeff * w[idx]);
}

template <std::integral i_t, std::floating_point f_t>
__global__ void __launch_bounds__(soc_block_size)
  apply_w_write_kernel(raft::device_span<const f_t> v,
                       raft::device_span<f_t> out,
                       raft::device_span<const f_t> w,
                       raft::device_span<const f_t> eta,
                       raft::device_span<const f_t> tail_dot,
                       raft::device_span<const size_t> cone_offsets,
                       raft::device_span<const i_t> element_cone_ids)
{
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= out.size()) { return; }

  const i_t cone         = element_cone_ids[idx];
  const size_t cone_off  = cone_offsets[cone];
  const size_t local_idx = idx - cone_off;

  const f_t w0       = w[cone_off];
  const f_t zeta     = tail_dot[cone];
  const f_t v0       = v[cone_off];
  const f_t cone_eta = eta[cone];

  if (local_idx == 0) {
    out[idx] = cone_eta * (w0 * v0 + zeta);
    return;
  }

  const f_t coeff = v0 + zeta / (f_t(1) + w0);
  out[idx]        = cone_eta * (v[idx] + coeff * w[idx]);
}

template <std::integral i_t, std::floating_point f_t>
__global__ void __launch_bounds__(soc_block_size)
  apply_hessian_kernel(raft::device_span<const f_t> v,
                       raft::device_span<f_t> out,
                       raft::device_span<const f_t> w,
                       raft::device_span<const f_t> eta,
                       raft::device_span<const f_t> wv_dot,
                       raft::device_span<const size_t> cone_offsets,
                       raft::device_span<const i_t> element_cone_ids,
                       raft::device_span<const i_t> cone_is_sparse,
                       bool dense_cones_only,
                       raft::device_span<const f_t> bias,
                       f_t output_scale,
                       f_t bias_scale)
{
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= out.size()) { return; }

  const i_t cone = element_cone_ids[idx];
  if (dense_cones_only && !cone_is_sparse.empty() && cone_is_sparse[cone] != 0) { return; }
  const size_t cone_off  = cone_offsets[cone];
  const size_t local_idx = idx - cone_off;

  const f_t eta_sq  = (eta[cone] * eta[cone]);
  const f_t coeff   = 2 * wv_dot[cone] * eta_sq;
  const int sign    = (local_idx == 0) * 2 - 1;
  const f_t value   = coeff * w[idx] - eta_sq * v[idx] * sign;
  const f_t h_value = output_scale * value;

  out[idx] = bias.empty() ? h_value : bias_scale * bias[idx] + h_value;
}

template <std::integral i_t, std::floating_point f_t>
__global__ void __launch_bounds__(soc_block_size)
  gather_cone_heads_kernel(raft::device_span<const f_t> values,
                           raft::device_span<f_t> heads,
                           raft::device_span<const size_t> cone_offsets,
                           i_t n_cones)
{
  const i_t cone = static_cast<i_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (cone >= n_cones) { return; }

  heads[cone] = values[cone_offsets[cone]];
}

/**
 * Build the Mehrotra corrector shift:
 *
 *   d = (W dx_aff) o (W^{-T} dz_aff) - sigma_mu e.
 *
 * On entry, `scaled_dx` is W dx_aff and `scaled_dz` is W^{-T} dz_aff. The
 * cone head uses the full dot product, and tail entries use the SOC Jordan
 * product:
 *
 *   d_0    = <scaled_dx, scaled_dz> - sigma_mu
 *   d_tail = scaled_dx_0 * scaled_dz_tail + scaled_dz_0 * scaled_dx_tail.
 */
template <std::integral i_t, std::floating_point f_t>
__global__ void __launch_bounds__(soc_block_size)
  combined_cone_shift_write_kernel(raft::device_span<f_t> shift,
                                   raft::device_span<const f_t> scaled_dx,
                                   raft::device_span<const f_t> scaled_dz,
                                   raft::device_span<const f_t> full_dot,
                                   raft::device_span<const f_t> scaled_dx_head,
                                   raft::device_span<const f_t> scaled_dz_head,
                                   raft::device_span<const size_t> cone_offsets,
                                   raft::device_span<const i_t> element_cone_ids,
                                   f_t sigma_mu)
{
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= shift.size()) { return; }

  const i_t cone         = element_cone_ids[idx];
  const size_t cone_off  = cone_offsets[cone];
  const size_t local_idx = idx - cone_off;

  if (local_idx == 0) {
    shift[idx] = full_dot[cone] - sigma_mu;
    return;
  }

  shift[idx] = scaled_dx_head[cone] * scaled_dz[idx] + scaled_dz_head[cone] * scaled_dx[idx];
}

/**
 * Per-cone scalar stage for p = lambda \ d:
 *
 *   p_0 = (lambda_0 d_0 - <lambda_tail, d_tail>) / det_J(lambda)
 *   inv_lambda_0 = 1 / lambda_0.
 *
 * A second flat kernel writes `-p`, which lets the final W^{-1} call produce
 * q = -W^{-1} p without adding an output-scale argument to W^{-1}.
 */
template <std::integral i_t, std::floating_point f_t>
__global__ void __launch_bounds__(soc_block_size)
  jordan_divide_by_lambda_scalar_kernel(raft::device_span<const f_t> shift,
                                        raft::device_span<const f_t> nt_point,
                                        raft::device_span<const f_t> lambda_tail_dot,
                                        raft::device_span<const f_t> lambda_tail_sq,
                                        raft::device_span<f_t> p0,
                                        raft::device_span<f_t> inv_lambda0,
                                        raft::device_span<const size_t> cone_offsets,
                                        i_t n_cones)
{
  const i_t cone = static_cast<i_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (cone >= n_cones) { return; }

  const size_t cone_off      = cone_offsets[cone];
  const f_t lambda0          = nt_point[cone_off];
  const f_t lambda_tail_norm = sqrt(lambda_tail_sq[cone]);
  const f_t det_lambda       = (lambda0 - lambda_tail_norm) * (lambda0 + lambda_tail_norm);

  // repurpose the heads in lambda_tail_dot, lambda_tail_sq for each cone
  p0[cone]          = (lambda0 * shift[cone_off] - lambda_tail_dot[cone]) / det_lambda;
  inv_lambda0[cone] = 1 / lambda0;
}

template <std::integral i_t, std::floating_point f_t>
__global__ void __launch_bounds__(soc_block_size)
  jordan_divide_by_lambda_write_kernel(raft::device_span<const f_t> shift,
                                       raft::device_span<const f_t> nt_point,
                                       raft::device_span<const f_t> p0,
                                       raft::device_span<const f_t> inv_lambda0,
                                       raft::device_span<const size_t> cone_offsets,
                                       raft::device_span<const i_t> element_cone_ids,
                                       raft::device_span<f_t> out)
{
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= out.size()) { return; }

  const i_t cone         = element_cone_ids[idx];
  const size_t cone_off  = cone_offsets[cone];
  const size_t local_idx = idx - cone_off;

  if (local_idx == 0) {
    out[idx] = -p0[cone];
    return;
  }

  out[idx] = (p0[cone] * nt_point[idx] - shift[idx]) * inv_lambda0[cone];
}

/**
 * Apply the Nesterov-Todd scaling matrix: out = W^{-1} v.
 *
 * For each cone:
 *   zeta = <w_tail, v_tail>
 *   (W^{-1}v)_0 = inv_eta * (w_0 v_0 - zeta)
 *   (W^{-1}v)_tail = inv_eta * (v_tail + (-v_0 + zeta / (1 + w_0)) w_tail)
 */
template <std::integral i_t, std::floating_point f_t>
void apply_w_inv(raft::device_span<const f_t> v,
                 raft::device_span<f_t> out,
                 cone_data_t<i_t, f_t>& cones,
                 rmm::cuda_stream_view stream)
{
  auto w                = cuopt::make_span(cones.w);
  auto eta              = cuopt::make_span(cones.eta);
  auto cone_offsets     = cuopt::make_span(cones.cone_offsets);
  auto element_cone_ids = cuopt::make_span(cones.element_cone_ids);
  auto tail_dot         = cones.scratch.template get_slot<0>();

  auto tail_terms =
    thrust::make_transform_iterator(thrust::make_counting_iterator<size_t>(0),
                                    [v, w, cone_offsets, element_cone_ids] HD(size_t idx) -> f_t {
                                      const i_t cone = element_cone_ids[idx];
                                      return idx == cone_offsets[cone] ? 0 : w[idx] * v[idx];
                                    });
  cones.segmented_sum(tail_terms, tail_dot, stream);

  const size_t grid_dim = raft::ceildiv<size_t>(out.size(), soc_block_size);
  apply_w_inv_write_kernel<i_t, f_t><<<grid_dim, soc_block_size, 0, stream.get()>>>(
    v, out, w, eta, tail_dot, cone_offsets, element_cone_ids);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

/**
 * Apply the multiplication of Nesterov-Todd scaling matrix:
 * out = W v.
 *
 * For each cone,
 *   zeta = <w_tail, v_tail>
 *   (W * v)_0 = eta * (w_0 v_0 + zeta)
 *   (W * v)_tail =
 *     eta * (v_tail + (v_0 + zeta / (1 + w_0)) w_tail)
 */
template <std::integral i_t, std::floating_point f_t>
void apply_w(raft::device_span<const f_t> v,
             raft::device_span<f_t> out,
             cone_data_t<i_t, f_t>& cones,
             rmm::cuda_stream_view stream)
{
  auto w                = cuopt::make_span(cones.w);
  auto eta              = cuopt::make_span(cones.eta);
  auto cone_offsets     = cuopt::make_span(cones.cone_offsets);
  auto element_cone_ids = cuopt::make_span(cones.element_cone_ids);
  auto tail_dot         = cones.scratch.template get_slot<0>();

  auto tail_terms =
    thrust::make_transform_iterator(thrust::make_counting_iterator<size_t>(0),
                                    [v, w, cone_offsets, element_cone_ids] HD(size_t idx) -> f_t {
                                      const i_t cone = element_cone_ids[idx];
                                      return idx == cone_offsets[cone] ? 0 : w[idx] * v[idx];
                                    });
  cones.segmented_sum(tail_terms, tail_dot, stream);

  const size_t grid_dim = raft::ceildiv<size_t>(out.size(), soc_block_size);
  apply_w_write_kernel<i_t, f_t><<<grid_dim, soc_block_size, 0, stream.get()>>>(
    v, out, w, eta, tail_dot, cone_offsets, element_cone_ids);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

/**
 * Apply the cone KKT block H = S^T S = S^2.
 *
 * With rho = <w, v>:
 *   (Hv)_0 = eta^{2} (2 w_0 rho - v_0)
 *   (Hv)_tail = eta^{2} (2 w_tail rho + v_tail)
 */
template <std::integral i_t, std::floating_point f_t>
void apply_hessian(raft::device_span<const f_t> v,
                   raft::device_span<f_t> out,
                   cone_data_t<i_t, f_t>& cones,
                   rmm::cuda_stream_view stream,
                   f_t output_scale                  = 1,
                   raft::device_span<const f_t> bias = {},
                   f_t bias_scale                    = 0,
                   bool dense_cones_only             = false)
{
  auto w                = cuopt::make_span(cones.w);
  auto eta              = cuopt::make_span(cones.eta);
  auto cone_offsets     = cuopt::make_span(cones.cone_offsets);
  auto element_cone_ids = cuopt::make_span(cones.element_cone_ids);
  auto wv_dot           = cones.scratch.template get_slot<0>();

  auto wv_terms =
    thrust::make_transform_iterator(thrust::make_counting_iterator<size_t>(0),
                                    [v, w] HD(size_t idx) -> f_t { return w[idx] * v[idx]; });
  cones.segmented_sum(wv_terms, wv_dot, stream);

  const size_t grid_dim = raft::ceildiv<size_t>(out.size(), soc_block_size);
  apply_hessian_kernel<i_t, f_t>
    <<<grid_dim, soc_block_size, 0, stream.get()>>>(v,
                                                    out,
                                                    w,
                                                    eta,
                                                    wv_dot,
                                                    cone_offsets,
                                                    element_cone_ids,
                                                    cuopt::make_span(cones.cone_is_sparse),
                                                    dense_cones_only,
                                                    bias,
                                                    output_scale,
                                                    bias_scale);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

/**
 * Recover the SOC dual direction after the reduced KKT solve.
 *
 * The reduced solve gives `dx`; the cone equation supplies the target RHS.
 * This function applies the cone block H = S^2 and writes:
 *   dz = cone_target - H dx.
 */
template <std::integral i_t, std::floating_point f_t>
void recover_cone_dz_from_target(raft::device_span<const f_t> dx,
                                 cone_data_t<i_t, f_t>& cones,
                                 raft::device_span<const f_t> cone_target,
                                 raft::device_span<f_t> dz,
                                 rmm::cuda_stream_view stream)
{
  apply_hessian<i_t, f_t>(dx, dz, cones, stream, -1, cone_target, 1);
}

/**
 * Accumulate the dense SOC cone-block matvec into an existing output vector:
 *   out += H x, where H = S^2, applied to dense cones only.
 */
template <std::integral i_t, std::floating_point f_t>
void launch_dense_hessian_matvec(raft::device_span<const f_t> x,
                                 cone_data_t<i_t, f_t>& cones,
                                 raft::device_span<f_t> out,
                                 rmm::cuda_stream_view stream)
{
  auto out_input = raft::device_span<const f_t>(out.data(), out.size());
  apply_hessian<i_t, f_t>(x, out, cones, stream, 1, out_input, 1, true);
}

// Bucket index owning `entry` via upper-bound search on the exclusive prefix ends in
// offsets[1..n_buckets].
template <std::integral i_t, typename offset_t>
__device__ i_t bucket_index(raft::device_span<const offset_t> offsets,
                            offset_t entry,
                            i_t n_buckets)
{
  i_t lo = 0;
  i_t hi = n_buckets;
  while (lo < hi) {
    const i_t mid = lo + (hi - lo) / 2;
    if (offsets[mid + 1] <= entry) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

template <std::integral i_t, std::floating_point f_t>
__global__ void __launch_bounds__(soc_block_size) scatter_sparse_hessian_into_augmented_kernel(
  raft::device_span<f_t> augmented_x,
  raft::device_span<f_t> Hs_diag,
  raft::device_span<const f_t> eta,
  raft::device_span<const f_t> d,
  raft::device_span<const i_t> sparse_cone_ids,
  raft::device_span<const i_t> sparse_entry_offsets,
  i_t n_sparse_cones,
  raft::device_span<const i_t> hessian_diag_csr_indices,
  raft::device_span<const f_t> q_values,
  raft::device_span<const f_t> sparse_v,
  raft::device_span<const f_t> sparse_u,
  raft::device_span<const i_t> exp_v_col,
  raft::device_span<const i_t> exp_u_col,
  raft::device_span<const i_t> exp_v_row,
  raft::device_span<const i_t> exp_u_row,
  raft::device_span<const i_t> sparse_expansion_D,
  f_t dual_perturb)
{
  const size_t e = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (e >= Hs_diag.size()) { return; }

  const i_t e_i        = static_cast<i_t>(e);  // single cast, reused below
  const i_t sparse_idx = bucket_index(sparse_entry_offsets, e_i, n_sparse_cones);
  const i_t cone_idx   = sparse_cone_ids[sparse_idx];
  const bool is_head   = e_i == sparse_entry_offsets[sparse_idx];
  const f_t eta_val    = eta[cone_idx];
  const f_t eta_sq     = eta_val * eta_val;
  const f_t hs_val     = is_head ? eta_sq * d[sparse_idx] : eta_sq;
  Hs_diag[e]           = hs_val;

  augmented_x[hessian_diag_csr_indices[e]] = -hs_val - q_values[e] - dual_perturb;

  const f_t v_val           = sparse_v[e];
  const f_t u_val           = sparse_u[e];
  augmented_x[exp_v_col[e]] = v_val;
  augmented_x[exp_u_col[e]] = u_val;
  augmented_x[exp_v_row[e]] = v_val;
  augmented_x[exp_u_row[e]] = u_val;

  if (is_head) {
    const f_t perturbed_eta_sq                          = eta_sq + dual_perturb;
    augmented_x[sparse_expansion_D[2 * sparse_idx]]     = -perturbed_eta_sq;
    augmented_x[sparse_expansion_D[2 * sparse_idx + 1]] = perturbed_eta_sq;
  }
}

/**
 * Sparse SOC augmented-system assembly: writing `Hs_diag`, the Hessian CSR scatter, and the four
 * rank-2 couplings, with each cone's unique head thread additionally applying the rank-2
 * corner scale and writing that cone's two expansion diagonals.
 *
 * `Hs_diag` is left populated for downstream matrix-free matvec / iterative refinement use.
 */
template <std::integral i_t, std::floating_point f_t>
void scatter_sparse_hessian_into_augmented(cone_data_t<i_t, f_t>& cones,
                                           rmm::device_uvector<f_t>& augmented_x,
                                           rmm::device_uvector<f_t>& Hs_diag,
                                           const rmm::device_uvector<i_t>& hessian_diag_csr_indices,
                                           const rmm::device_uvector<f_t>& q_values,
                                           const rmm::device_uvector<i_t>& exp_v_col,
                                           const rmm::device_uvector<i_t>& exp_u_col,
                                           const rmm::device_uvector<i_t>& exp_v_row,
                                           const rmm::device_uvector<i_t>& exp_u_row,
                                           const rmm::device_uvector<i_t>& sparse_expansion_D,
                                           rmm::cuda_stream_view stream,
                                           f_t dual_perturb)
{
  if (!cones.has_sparse_cones()) { return; }

  const i_t n_sparse      = cones.n_sparse_cones;
  const size_t E          = cones.n_sparse_cone_entries;
  const size_t entry_grid = raft::ceildiv<size_t>(E, soc_block_size);
  scatter_sparse_hessian_into_augmented_kernel<i_t, f_t>
    <<<entry_grid, soc_block_size, 0, stream.get()>>>(cuopt::make_span(augmented_x),
                                                      cuopt::make_span(Hs_diag),
                                                      cuopt::make_span(cones.eta),
                                                      cuopt::make_span(cones.d),
                                                      cuopt::make_span(cones.sparse_cone_ids),
                                                      cuopt::make_span(cones.sparse_entry_offsets),
                                                      n_sparse,
                                                      cuopt::make_span(hessian_diag_csr_indices),
                                                      cuopt::make_span(q_values),
                                                      cuopt::make_span(cones.sparse_v),
                                                      cuopt::make_span(cones.sparse_u),
                                                      cuopt::make_span(exp_v_col),
                                                      cuopt::make_span(exp_u_col),
                                                      cuopt::make_span(exp_v_row),
                                                      cuopt::make_span(exp_u_row),
                                                      cuopt::make_span(sparse_expansion_D),
                                                      dual_perturb);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

/**
 * Accumulate the sparse-SOC expanded KKT block into a matrix-free product.
 *
 * For each sparse cone this adds to the primal cone rows (before augmented_multiply negates r1):
 *   r1 += Hs_diag .* x_cone - v * x_exp_v - u * x_exp_u
 * so y1 = -r1 matches the explicit CSR coupling (+v, +u on primal rows).
 * and to the expansion rows:
 *   y_exp[2s]   += -eta^2 * x_exp_v + v^T x_cone
 *   y_exp[2s+1] += +eta^2 * x_exp_u + u^T x_cone
 *
 * `v` and `u` are the rank-2 vectors in cones.sparse_v / cones.sparse_u.
 */
template <std::integral i_t, std::floating_point f_t>
__global__ void sparse_augmented_matvec_kernel(raft::device_span<const f_t> x,
                                               raft::device_span<f_t> r1,
                                               raft::device_span<f_t> y_exp,
                                               raft::device_span<const f_t> Hs_diag,
                                               raft::device_span<const f_t> sparse_v,
                                               raft::device_span<const f_t> sparse_u,
                                               raft::device_span<const f_t> eta,
                                               raft::device_span<const i_t> sparse_cone_ids,
                                               raft::device_span<const i_t> sparse_cone_dims,
                                               raft::device_span<const i_t> sparse_entry_offsets,
                                               raft::device_span<const size_t> cone_offsets,
                                               i_t cone_var_start,
                                               i_t n_primal,
                                               i_t m_constraints,
                                               i_t n_sparse_cones)
{
  const i_t sparse_idx = blockIdx.x;
  if (sparse_idx >= n_sparse_cones) { return; }

  const i_t cone    = sparse_cone_ids[sparse_idx];
  const i_t q       = sparse_cone_dims[sparse_idx];
  const i_t flat    = sparse_entry_offsets[sparse_idx];
  const size_t off  = cone_offsets[cone];
  const i_t base    = cone_var_start + static_cast<i_t>(off);
  const i_t exp_v   = n_primal + m_constraints + 2 * sparse_idx;
  const i_t exp_u   = exp_v + 1;
  const f_t x_exp_v = x[exp_v];
  const f_t x_exp_u = x[exp_u];
  const f_t eta_sq  = eta[cone] * eta[cone];

  f_t partial_dot_v = f_t(0);
  f_t partial_dot_u = f_t(0);
  for (i_t j = threadIdx.x; j < q; j += blockDim.x) {
    const f_t xj = x[base + j];
    const f_t vj = sparse_v[flat + j];
    const f_t uj = sparse_u[flat + j];
    partial_dot_v += vj * xj;
    partial_dot_u += uj * xj;
    r1[base + j] += Hs_diag[flat + j] * xj - vj * x_exp_v - uj * x_exp_u;
  }

  __shared__ f_t s_dot_v[soc_block_size];
  __shared__ f_t s_dot_u[soc_block_size];
  s_dot_v[threadIdx.x] = partial_dot_v;
  s_dot_u[threadIdx.x] = partial_dot_u;
  __syncthreads();

  for (i_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      s_dot_v[threadIdx.x] += s_dot_v[threadIdx.x + stride];
      s_dot_u[threadIdx.x] += s_dot_u[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    y_exp[2 * sparse_idx] += -eta_sq * x_exp_v + s_dot_v[0];
    y_exp[2 * sparse_idx + 1] += eta_sq * x_exp_u + s_dot_u[0];
  }
}

/**
 * Matrix-free product of the implicit sparse cone blocks (see
 * `sparse_augmented_matvec_kernel`). Accumulates the cone-row contribution into
 * `r1` and the expansion-row contribution into `y_exp`, one block per sparse cone.
 */
template <std::integral i_t, std::floating_point f_t>
void launch_sparse_augmented_matvec(raft::device_span<const f_t> x,
                                    raft::device_span<f_t> r1,
                                    raft::device_span<f_t> y_exp,
                                    cone_data_t<i_t, f_t>& cones,
                                    raft::device_span<const f_t> Hs_diag,
                                    i_t cone_var_start,
                                    i_t n_primal,
                                    i_t m_constraints,
                                    rmm::cuda_stream_view stream)
{
  if (!cones.has_sparse_cones()) { return; }

  const i_t n_sparse = cones.n_sparse_cones;
  cuopt_assert(Hs_diag.size() == cones.n_sparse_cone_entries, "Hs_diag size mismatch");
  cuopt_assert(y_exp.size() == static_cast<size_t>(cones.expansion_var_count()),
               "expansion output size mismatch");

  sparse_augmented_matvec_kernel<i_t, f_t>
    <<<n_sparse, soc_block_size, 0, stream.get()>>>(x,
                                                    r1,
                                                    y_exp,
                                                    Hs_diag,
                                                    cuopt::make_span(cones.sparse_v),
                                                    cuopt::make_span(cones.sparse_u),
                                                    cuopt::make_span(cones.eta),
                                                    cuopt::make_span(cones.sparse_cone_ids),
                                                    cuopt::make_span(cones.sparse_cone_dims),
                                                    cuopt::make_span(cones.sparse_entry_offsets),
                                                    cuopt::make_span(cones.cone_offsets),
                                                    cone_var_start,
                                                    n_primal,
                                                    m_constraints,
                                                    n_sparse);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

// Scatter one nonzero of a dense cone's q x q NT Hessian block into the
// augmented value buffer. Only cones with dim <= soc_threshold take this path.
template <std::integral i_t, std::floating_point f_t>
__global__ void __launch_bounds__(soc_block_size)
  scatter_dense_hessian_into_augmented_kernel(raft::device_span<f_t> augmented_x,
                                              raft::device_span<const i_t> csr_indices,
                                              raft::device_span<const f_t> q_values,
                                              raft::device_span<const f_t> w,
                                              raft::device_span<const f_t> eta,
                                              raft::device_span<const size_t> cone_offsets,
                                              raft::device_span<const size_t> dense_block_offsets,
                                              raft::device_span<const i_t> dense_cone_ids,
                                              i_t n_dense_cones,
                                              f_t dual_perturb_value)
{
  const size_t e = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (e >= csr_indices.size()) { return; }

  const i_t dense_idx  = bucket_index(dense_block_offsets, e, n_dense_cones);
  const i_t cone       = dense_cone_ids[dense_idx];
  const size_t off     = cone_offsets[cone];
  const size_t q       = cone_offsets[cone + 1] - off;
  const size_t blk_off = dense_block_offsets[dense_idx];
  const size_t local   = e - blk_off;
  const size_t r       = local / q;
  const size_t c       = local % q;

  const f_t eta_sq = eta[cone] * eta[cone];
  const f_t w0     = w[off];
  const f_t u_r    = (r == 0) ? w0 : w[off + r];
  const f_t u_c    = (c == 0) ? w0 : w[off + c];
  const f_t val    = f_t{2} * u_r * eta_sq * u_c;

  f_t entry = -val - q_values[e];
  if (r == c) {
    const f_t diag_correction = (r == 0) ? -eta_sq : eta_sq;
    entry -= diag_correction + dual_perturb_value;
  }
  augmented_x[csr_indices[e]] = entry;
}

/**
 * Write the full q x q NT Hessian blocks of the dense cones (dim <= soc_threshold)
 * into the augmented system value buffer at the precomputed CSR positions.
 */
template <std::integral i_t, std::floating_point f_t>
void scatter_dense_hessian_into_augmented(const cone_data_t<i_t, f_t>& cones,
                                          rmm::device_uvector<f_t>& augmented_x,
                                          const rmm::device_uvector<i_t>& csr_indices,
                                          const rmm::device_uvector<f_t>& q_values,
                                          const rmm::device_uvector<size_t>& dense_block_offsets,
                                          const rmm::device_uvector<i_t>& dense_cone_ids,
                                          rmm::cuda_stream_view stream,
                                          f_t dual_perturb_value)
{
  const size_t count = csr_indices.size();
  if (count == 0) { return; }
  cuopt_assert(count == q_values.size(), "dense cone CSR index and Q-value arrays must match");

  const i_t n_dense = cones.n_dense_cones();
  const size_t grid = raft::ceildiv<size_t>(count, soc_block_size);
  scatter_dense_hessian_into_augmented_kernel<i_t, f_t>
    <<<grid, soc_block_size, 0, stream.get()>>>(cuopt::make_span(augmented_x),
                                                cuopt::make_span(csr_indices),
                                                cuopt::make_span(q_values),
                                                cuopt::make_span(cones.w),
                                                cuopt::make_span(cones.eta),
                                                cuopt::make_span(cones.cone_offsets),
                                                cuopt::make_span(dense_block_offsets),
                                                cuopt::make_span(dense_cone_ids),
                                                n_dense,
                                                dual_perturb_value);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

// =============================================================================
// Cone step length
//
// Max alpha keeping u + alpha du in the SOC for each packed cone. Size-aware
// path: one pass over each cone's tail accumulates
// (||du_tail||^2, <u_tail, du_tail>, ||u_tail||^2), then solves the quadratic
// boundary. Small/medium/large partitions come from segmented_sum_t.
// =============================================================================

template <std::floating_point f_t>
HD f_t cone_step_length_from_scalars(
  f_t u0, f_t du0, f_t du_tail_sq, f_t u_tail_du_tail, f_t u_tail_sq, f_t alpha_max)
{
  const f_t a     = du0 * du0 - du_tail_sq;
  const f_t b     = u0 * du0 - u_tail_du_tail;
  const f_t c_raw = u0 * u0 - u_tail_sq;
  const f_t c     = c_raw > 0 ? c_raw : 0;
  const f_t disc  = b * b - a * c;
  f_t alpha       = alpha_max;

  if (u0 >= 0 && du0 < 0) { alpha = cuda::std::min(alpha, -u0 / du0); }

  if ((a > 0 && b > 0) || disc < 0) { return alpha; }

  if (a == 0) {
    return alpha;
  } else if (c == 0) {
    alpha = a >= 0 ? alpha : 0;
  } else {
    const f_t t = -(b + copysign(sqrt(disc), b));
    f_t r1      = c / t;
    f_t r2      = t / a;
    if (r1 < 0) { r1 = alpha; }
    if (r2 < 0) { r2 = alpha; }
    alpha = cuda::std::min(alpha, cuda::std::min(r1, r2));
  }

  return alpha;
}

/**
 * One warp per small cone: accumulate the three tail scalars, then solve for
 * alpha[cone]. Tail-only (local index 0 is the SOC head).
 */
template <std::integral i_t, std::floating_point f_t, int warps_per_cta = 8>
__global__ void __launch_bounds__(warps_per_cta* raft::WarpSize)
  step_length_small_kernel(raft::device_span<const f_t> u,
                           raft::device_span<const f_t> du,
                           raft::device_span<f_t> alpha,
                           raft::device_span<const i_t> small_cone_ids,
                           raft::device_span<const size_t> cone_offsets,
                           f_t alpha_max)
{
  static_assert(warps_per_cta > 0);
  static_assert(warps_per_cta * raft::WarpSize <= 1024);

  using warp_reduce_t = cub::WarpReduce<step_tail_sums_t<f_t>, raft::WarpSize>;
  __shared__ typename warp_reduce_t::TempStorage temp_storage[warps_per_cta];

  const auto lane_id  = raft::laneId();
  const auto warp_idx = threadIdx.x / raft::WarpSize;
  const auto slot     = blockIdx.x * warps_per_cta + warp_idx;
  if (slot >= small_cone_ids.size()) { return; }

  const i_t cone   = small_cone_ids[slot];
  const size_t off = cone_offsets[cone];
  const size_t dim = cone_offsets[cone + 1] - off;

  step_tail_sums_t<f_t> tail_sums{};
  for (size_t i = lane_id + 1; i < dim; i += raft::WarpSize) {
    const f_t ui  = u[off + i];
    const f_t dui = du[off + i];
    tail_sums.du_tail_sq += dui * dui;
    tail_sums.u_tail_du_tail += ui * dui;
    tail_sums.u_tail_sq += ui * ui;
  }

  tail_sums = warp_reduce_t(temp_storage[warp_idx]).Sum(tail_sums);

  if (lane_id == 0) {
    alpha[cone] = cone_step_length_from_scalars(u[off],
                                                du[off],
                                                tail_sums.du_tail_sq,
                                                tail_sums.u_tail_du_tail,
                                                tail_sums.u_tail_sq,
                                                alpha_max);
  }
}

/**
 * One block per medium cone: three-scalar tail reduction + step solve.
 */
template <std::integral i_t, std::floating_point f_t, int block_dim = 256>
__global__ void __launch_bounds__(block_dim)
  step_length_medium_kernel(raft::device_span<const f_t> u,
                            raft::device_span<const f_t> du,
                            raft::device_span<f_t> alpha,
                            raft::device_span<const i_t> medium_cone_ids,
                            raft::device_span<const size_t> cone_offsets,
                            f_t alpha_max)
{
  static_assert(block_dim > 0);
  static_assert(block_dim <= 1024);

  constexpr int items_per_thread = 4;

  using block_reduce_t = cub::BlockReduce<f_t, block_dim>;
  __shared__ typename block_reduce_t::TempStorage temp_storage;

  const auto slot = blockIdx.x;
  if (slot >= medium_cone_ids.size()) { return; }

  const i_t cone   = medium_cone_ids[slot];
  const size_t off = cone_offsets[cone];
  const size_t dim = cone_offsets[cone + 1] - off;

  f_t acc_du_sq[items_per_thread]{};
  f_t acc_u_du[items_per_thread]{};
  f_t acc_u_sq[items_per_thread]{};

  const size_t tile = static_cast<size_t>(block_dim) * items_per_thread;
  for (size_t tile_start = 0; tile_start < dim; tile_start += tile) {
#pragma unroll
    for (int k = 0; k < items_per_thread; ++k) {
      const size_t idx = tile_start + threadIdx.x + static_cast<size_t>(k) * block_dim;
      if (idx > 0 && idx < dim) {
        const f_t ui  = u[off + idx];
        const f_t dui = du[off + idx];
        acc_du_sq[k] += dui * dui;
        acc_u_du[k] += ui * dui;
        acc_u_sq[k] += ui * ui;
      }
    }
  }

  f_t du_sq = f_t{0};
  f_t u_du  = f_t{0};
  f_t u_sq  = f_t{0};
#pragma unroll
  for (int k = 0; k < items_per_thread; ++k) {
    du_sq += acc_du_sq[k];
    u_du += acc_u_du[k];
    u_sq += acc_u_sq[k];
  }

  du_sq = block_reduce_t(temp_storage).Sum(du_sq);
  __syncthreads();
  u_du = block_reduce_t(temp_storage).Sum(u_du);
  __syncthreads();
  u_sq = block_reduce_t(temp_storage).Sum(u_sq);

  if (threadIdx.x == 0) {
    alpha[cone] = cone_step_length_from_scalars(u[off], du[off], du_sq, u_du, u_sq, alpha_max);
  }
}

template <std::integral i_t, std::floating_point f_t>
__global__ void step_length_large_solve_kernel(raft::device_span<const f_t> u,
                                               raft::device_span<const f_t> du,
                                               raft::device_span<f_t> alpha,
                                               raft::device_span<const step_tail_sums_t<f_t>> sums,
                                               raft::device_span<const i_t> large_cone_ids,
                                               raft::device_span<const size_t> cone_offsets,
                                               f_t alpha_max)
{
  const size_t slot = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (slot >= large_cone_ids.size()) { return; }

  const i_t cone   = large_cone_ids[slot];
  const size_t off = cone_offsets[cone];
  const auto& s    = sums[slot];
  alpha[cone]      = cone_step_length_from_scalars(
    u[off], du[off], s.du_tail_sq, s.u_tail_du_tail, s.u_tail_sq, alpha_max);
}

/**
 * Size-aware step length: one pass over each cone's tail, then write
 * alpha[cone].
 */
template <std::integral i_t, std::floating_point f_t>
void launch_cone_step_length(segmented_sum_t<i_t>& partitions,
                             raft::device_span<const f_t> u,
                             raft::device_span<const f_t> du,
                             raft::device_span<f_t> alpha,
                             raft::device_span<step_tail_sums_t<f_t>> large_sums,
                             f_t alpha_max,
                             rmm::cuda_stream_view stream)
{
  constexpr int warps_per_cta = 8;
  if (!partitions.small_cone_ids.is_empty()) {
    const auto n_small = partitions.small_cone_ids.size();
    const auto grid    = (n_small + warps_per_cta - 1) / warps_per_cta;
    step_length_small_kernel<i_t, f_t, warps_per_cta>
      <<<grid, warps_per_cta * raft::WarpSize, 0, stream.get()>>>(
        u,
        du,
        alpha,
        cuopt::make_span(partitions.small_cone_ids),
        partitions.cone_offsets,
        alpha_max);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }

  if (!partitions.medium_cone_ids.is_empty()) {
    constexpr int medium_block_dim = 256;
    step_length_medium_kernel<i_t, f_t, medium_block_dim>
      <<<partitions.medium_cone_ids.size(), medium_block_dim, 0, stream.get()>>>(
        u,
        du,
        alpha,
        cuopt::make_span(partitions.medium_cone_ids),
        partitions.cone_offsets,
        alpha_max);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }

  if (!partitions.large_cone_ids.empty()) {
    const auto n_large = partitions.large_cone_ids.size();

    for (std::size_t i = 0; i < n_large; ++i) {
      const size_t off       = partitions.large_cone_offsets[i];
      const i_t dim          = partitions.large_cone_dimensions[i];
      std::size_t temp_bytes = partitions.cub_workspace_bytes;

      auto input =
        thrust::make_transform_iterator(thrust::make_counting_iterator<size_t>(0),
                                        [u, du, off] HD(size_t local) -> step_tail_sums_t<f_t> {
                                          if (local == 0) { return {}; }
                                          const f_t ui  = u[off + local];
                                          const f_t dui = du[off + local];
                                          return {dui * dui, ui * dui, ui * ui};
                                        });

      RAFT_CUDA_TRY(cub::DeviceReduce::Sum(partitions.cub_workspace.data(),
                                           temp_bytes,
                                           input,
                                           large_sums.data() + i,
                                           dim,
                                           stream.get()));
    }

    raft::device_span<const step_tail_sums_t<f_t>> large_sums_c(large_sums.data(),
                                                                large_sums.size());
    constexpr int large_solve_block_dim = 256;
    const auto grid = raft::ceildiv<size_t>(n_large, static_cast<size_t>(large_solve_block_dim));
    step_length_large_solve_kernel<i_t, f_t><<<grid, large_solve_block_dim, 0, stream.get()>>>(
      u,
      du,
      alpha,
      large_sums_c,
      cuopt::make_span(partitions.large_cone_ids_device),
      partitions.cone_offsets,
      alpha_max);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }
}

/**
 * Combined (primal and dual) maximum step length keeping SOC blocks feasible:
 *
 *   x + alpha dx in Q,  z + alpha dz in Q,  alpha <= alpha_max.
 */
template <std::integral i_t, std::floating_point f_t>
f_t compute_cone_step_length(cone_data_t<i_t, f_t>& cones,
                             raft::device_span<const f_t> dx,
                             raft::device_span<const f_t> dz,
                             f_t alpha_max,
                             rmm::cuda_stream_view stream)
{
  auto alpha_primal = cuopt::make_span(cones.scratch.step_alpha_primal);
  auto alpha_dual   = cuopt::make_span(cones.scratch.step_alpha_dual);
  raft::device_span<const f_t> x(cones.x.data(), cones.x.size());
  raft::device_span<const f_t> z(cones.z.data(), cones.z.size());

  auto large_sums = cuopt::make_span(cones.scratch.step_large_tail_sums);
  launch_cone_step_length(cones.segmented_sum, x, dx, alpha_primal, large_sums, alpha_max, stream);
  launch_cone_step_length(cones.segmented_sum, z, dz, alpha_dual, large_sums, alpha_max, stream);

  return thrust::transform_reduce(
    rmm::exec_policy(stream),
    thrust::make_zip_iterator(alpha_primal.begin(), alpha_dual.begin()),
    thrust::make_zip_iterator(alpha_primal.end(), alpha_dual.end()),
    [] HD(const thrust::tuple<f_t, f_t>& t) -> f_t {
      return cuda::std::min(thrust::get<0>(t), thrust::get<1>(t));
    },
    alpha_max,
    thrust::minimum<f_t>());
}

/**
 * Build the SOC corrector target for the reduced KKT solve.
 *
 * Mehrotra's corrector uses affine cone directions to form
 *
 *   d = (W dx_aff) o (W^{-T} dz_aff) - sigma_mu e,
 *
 * where `o` is the SOC Jordan product and `e = (1, 0, ..., 0)` per cone.
 * The reduced KKT solve needs the cone target
 *
 *   q = -W * p,  where p = lambda \ d and lambda = W^{-T} z.
 *
 * On return, `out` holds `q`. Internally, `out` is reused for `W^{-T} dz_aff` and
 * then `d`; `scratch.temp_cone` is reused for `W dx_aff`, then `-p`.
 */
template <std::integral i_t, std::floating_point f_t>
void compute_combined_cone_rhs_term(raft::device_span<const f_t> dx_aff,
                                    raft::device_span<const f_t> dz_aff,
                                    cone_data_t<i_t, f_t>& cones,
                                    f_t sigma_mu,
                                    raft::device_span<f_t> out,
                                    rmm::cuda_stream_view stream)
{
  auto cone_offsets     = cuopt::make_span(cones.cone_offsets);
  auto element_cone_ids = cuopt::make_span(cones.element_cone_ids);

  auto scratch_cone = cuopt::make_span(cones.scratch.temp_cone);
  auto scaled_dx    = raft::device_span<const f_t>(scratch_cone.data(), scratch_cone.size());
  auto scaled_dz    = raft::device_span<const f_t>(out.data(), out.size());
  auto slot_0       = cones.scratch.template get_slot<0>();
  auto slot_1       = cones.scratch.template get_slot<1>();
  auto slot_2       = cones.scratch.template get_slot<2>();

  apply_w(dx_aff, scratch_cone, cones, stream);
  apply_w_inv(dz_aff, out, cones, stream);

  auto full_product_terms = thrust::make_transform_iterator(
    thrust::make_zip_iterator(scaled_dx.begin(), scaled_dz.begin()),
    thrust::make_zip_function([] HD(f_t dx, f_t dz) -> f_t { return dx * dz; }));
  cones.segmented_sum(full_product_terms, slot_0, stream);

  // `out` currently aliases W^{-T} dz_aff and is about to be overwritten with d.
  // Stage both head vectors first because every tail entry needs them.
  const size_t cone_grid_dim =
    raft::ceildiv<size_t>(static_cast<size_t>(cones.n_cones), soc_block_size);
  gather_cone_heads_kernel<i_t, f_t><<<cone_grid_dim, soc_block_size, 0, stream.get()>>>(
    scaled_dx, slot_1, cone_offsets, cones.n_cones);
  gather_cone_heads_kernel<i_t, f_t><<<cone_grid_dim, soc_block_size, 0, stream.get()>>>(
    scaled_dz, slot_2, cone_offsets, cones.n_cones);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  const size_t element_grid_dim = raft::ceildiv<size_t>(cones.n_cone_entries, soc_block_size);
  combined_cone_shift_write_kernel<i_t, f_t><<<element_grid_dim, soc_block_size, 0, stream.get()>>>(
    out, scaled_dx, scaled_dz, slot_0, slot_1, slot_2, cone_offsets, element_cone_ids, sigma_mu);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  auto shift    = raft::device_span<const f_t>(out.data(), out.size());
  auto nt_point = raft::device_span<const f_t>(cones.lambda.data(), cones.lambda.size());

  // compute W *(-(\lambda inv_circ shift))
  auto lambda_tail_dot_terms = thrust::make_transform_iterator(
    thrust::make_counting_iterator<size_t>(0),
    [shift, nt_point, cone_offsets, element_cone_ids] HD(size_t idx) -> f_t {
      const i_t cone = element_cone_ids[idx];
      return idx == cone_offsets[cone] ? 0 : nt_point[idx] * shift[idx];
    });
  cones.segmented_sum(lambda_tail_dot_terms, slot_0, stream);

  auto lambda_tail_sq_terms = thrust::make_transform_iterator(
    thrust::make_counting_iterator<size_t>(0),
    [nt_point, cone_offsets, element_cone_ids] HD(size_t idx) -> f_t {
      const i_t cone = element_cone_ids[idx];
      return idx == cone_offsets[cone] ? 0 : nt_point[idx] * nt_point[idx];
    });
  cones.segmented_sum(lambda_tail_sq_terms, slot_1, stream);

  jordan_divide_by_lambda_scalar_kernel<i_t, f_t>
    <<<cone_grid_dim, soc_block_size, 0, stream.get()>>>(
      shift, nt_point, slot_0, slot_1, slot_0, slot_1, cone_offsets, cones.n_cones);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  // Note that we implicitly multiply by -1 here since we are writing -p.
  jordan_divide_by_lambda_write_kernel<i_t, f_t>
    <<<element_grid_dim, soc_block_size, 0, stream.get()>>>(
      shift, nt_point, slot_0, slot_1, cone_offsets, element_cone_ids, scratch_cone);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  apply_w<i_t, f_t>(scratch_cone, out, cones, stream);
}

}  // namespace cuopt::mathematical_optimization::barrier
