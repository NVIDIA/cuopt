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

  // TODO: Consider moving this out to the barrier layer when we wire it in
  rmm::device_uvector<f_t> temp_cone;  // [n_cone_entries]

  cone_scratch_t(i_t n_cones_in, size_t n_cone_entries_in, rmm::cuda_stream_view stream)
    : n_cones(n_cones_in),
      n_cone_entries(n_cone_entries_in),
      slots(0, stream),
      step_alpha_primal(0, stream),
      step_alpha_dual(0, stream),
      temp_cone(0, stream)
  {
    const size_t n_cones_size = static_cast<size_t>(n_cones);

    slots.resize(n_cones_size * static_cast<size_t>(n_slots), stream);
    step_alpha_primal.resize(n_cones_size, stream);
    step_alpha_dual.resize(n_cones_size, stream);
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

template <std::integral i_t, std::floating_point f_t>
__global__ void __launch_bounds__(soc_block_size)
  step_length_single_kernel(raft::device_span<const f_t> u,
                            raft::device_span<const f_t> du,
                            raft::device_span<f_t> alpha,
                            raft::device_span<const f_t> du_tail_sq,
                            raft::device_span<const f_t> u_tail_du_tail,
                            raft::device_span<const f_t> u_tail_sq,
                            raft::device_span<const size_t> cone_offsets,
                            f_t alpha_max,
                            i_t n_cones)
{
  const i_t cone = static_cast<i_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (cone >= n_cones) { return; }

  const size_t off = cone_offsets[cone];
  alpha[cone]      = cone_step_length_from_scalars(
    u[off], du[off], du_tail_sq[cone], u_tail_du_tail[cone], u_tail_sq[cone], alpha_max);
}

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

  cone_scratch_t<i_t, f_t> scratch;

  cone_data_t(std::span<const i_t> cone_dimensions_host,
              raft::device_span<f_t> x_in,
              raft::device_span<f_t> z_in,
              rmm::cuda_stream_view stream)
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
      scratch(n_cones, n_cone_entries, stream)
  {
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
    segmented_sum.template prepare_workspace<f_t>(stream);
  }
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
  nt_finalize_scaling_scalars_kernel<i_t, f_t>
    <<<cone_grid_dim, soc_block_size, 0, stream.value()>>>(
      cones.x, cones.z, x_scale, z_scale, cuopt::make_span(cones.eta), cone_offsets, cones.n_cones);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  const size_t element_grid_dim = raft::ceildiv<size_t>(cones.n_cone_entries, soc_block_size);

  auto w = cuopt::make_span(cones.w);
  nt_write_w_kernel<i_t, f_t><<<element_grid_dim, soc_block_size, 0, stream.value()>>>(
    cones.x, cones.z, x_scale, z_scale, w, cone_offsets, element_cone_ids);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  auto unnormalized_tail_sq_terms =
    thrust::make_transform_iterator(thrust::make_counting_iterator<size_t>(0),
                                    [cone_offsets, element_cone_ids, w] HD(size_t idx) -> f_t {
                                      const i_t cone = element_cone_ids[idx];
                                      return idx == cone_offsets[cone] ? 0 : w[idx] * w[idx];
                                    });
  cones.segmented_sum(unnormalized_tail_sq_terms, w_scale, stream);

  nt_finalize_w_scale_kernel<i_t, f_t><<<cone_grid_dim, soc_block_size, 0, stream.value()>>>(
    w, w_scale, w_scale, cone_offsets, cones.n_cones);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  nt_normalize_w_kernel<i_t, f_t>
    <<<element_grid_dim, soc_block_size, 0, stream.value()>>>(w, w_scale, element_cone_ids);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  // Persist lambda while w_scale still stores sqrt(det_J(w_tmp)).
  nt_write_lambda_kernel<i_t, f_t>
    <<<element_grid_dim, soc_block_size, 0, stream.value()>>>(cones.x,
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

  nt_finalize_head_kernel<i_t, f_t><<<cone_grid_dim, soc_block_size, 0, stream.value()>>>(
    cuopt::make_span(cones.w), w_scale, cone_offsets, cones.n_cones);
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
  apply_hessian_write_kernel(raft::device_span<const f_t> v,
                             raft::device_span<f_t> out,
                             raft::device_span<const f_t> w,
                             raft::device_span<const f_t> eta,
                             raft::device_span<const f_t> wv_dot,
                             raft::device_span<const size_t> cone_offsets,
                             raft::device_span<const i_t> element_cone_ids,
                             raft::device_span<const f_t> bias,
                             f_t output_scale,
                             f_t bias_scale)
{
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= out.size()) { return; }

  const i_t cone         = element_cone_ids[idx];
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
  apply_w_inv_write_kernel<i_t, f_t><<<grid_dim, soc_block_size, 0, stream.value()>>>(
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
  apply_w_write_kernel<i_t, f_t><<<grid_dim, soc_block_size, 0, stream.value()>>>(
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
                   f_t bias_scale                    = 0)
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
  apply_hessian_write_kernel<i_t, f_t><<<grid_dim, soc_block_size, 0, stream.value()>>>(
    v, out, w, eta, wv_dot, cone_offsets, element_cone_ids, bias, output_scale, bias_scale);
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
 * Accumulate the SOC cone-block matvec into an existing output vector.
 *
 * Used by matrix-free products with the primal-reduced KKT block:
 *   out += H x, where H = S^2.
 */
template <std::integral i_t, std::floating_point f_t>
void accumulate_cone_hessian_matvec(raft::device_span<const f_t> x,
                                    cone_data_t<i_t, f_t>& cones,
                                    raft::device_span<f_t> out,
                                    rmm::cuda_stream_view stream)
{
  auto out_input = raft::device_span<const f_t>(out.data(), out.size());
  apply_hessian<i_t, f_t>(x, out, cones, stream, 1, out_input, 1);
}

template <std::integral i_t, std::floating_point f_t>
__global__ void __launch_bounds__(soc_block_size)
  scatter_hessian_into_augmented_kernel(raft::device_span<f_t> augmented_x,
                                        raft::device_span<const i_t> csr_indices,
                                        raft::device_span<const f_t> q_values,
                                        raft::device_span<const f_t> w,
                                        raft::device_span<const f_t> eta,
                                        raft::device_span<const size_t> cone_offsets,
                                        raft::device_span<const size_t> block_offsets,
                                        i_t n_cones,
                                        f_t dual_perturb_value)
{
  const size_t e = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (e >= csr_indices.size()) { return; }

  i_t lo = 0;
  i_t hi = n_cones;
  while (lo < hi) {
    const i_t mid = lo + (hi - lo) / 2;
    if (block_offsets[mid + 1] <= e) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }

  const i_t cone       = lo;
  const size_t off     = cone_offsets[cone];
  const size_t q       = cone_offsets[cone + 1] - off;
  const size_t blk_off = block_offsets[cone];
  const size_t local   = e - blk_off;
  const size_t r       = local / q;
  const size_t c       = local % q;

  const f_t eta_sq          = eta[cone] * eta[cone];
  const f_t w0              = w[off];
  const f_t u_r             = (r == 0) ? w0 : w[off + r];
  const f_t u_c             = (c == 0) ? w0 : w[off + c];
  f_t val                   = f_t{2} * u_r * eta_sq * u_c;
  const f_t diag_correction = (r == 0) ? -eta_sq : eta_sq;
  if (r == c) { val += diag_correction; }

  augmented_x[csr_indices[e]] = -val - q_values[e];
}

template <std::integral i_t, std::floating_point f_t>
void scatter_hessian_into_augmented(const cone_data_t<i_t, f_t>& cones,
                                    rmm::device_uvector<f_t>& augmented_x,
                                    const rmm::device_uvector<i_t>& csr_indices,
                                    const rmm::device_uvector<f_t>& q_values,
                                    rmm::cuda_stream_view stream,
                                    f_t dual_perturb_value)
{
  const size_t count = csr_indices.size();
  if (count == 0) { return; }
  cuopt_assert(count == q_values.size(), "cone CSR index and Q-value arrays must match");

  // TODO: This offset calculation should be done in the barrier layer,
  // because it is already done in the barrier layer for the augmented system, see
  // cone_block_offsets_host.
  rmm::device_uvector<size_t> block_offsets(cones.n_cones + 1, stream);
  block_offsets.set_element_to_zero_async(0, stream);

  auto block_sizes = thrust::make_transform_iterator(
    cones.cone_dimensions.begin(), [] HD(i_t q) -> size_t { return static_cast<size_t>(q) * q; });
  thrust::inclusive_scan(
    rmm::exec_policy(stream), block_sizes, block_sizes + cones.n_cones, block_offsets.begin() + 1);

  // TODO: use dual_perturb_value for regularization
  const size_t grid = raft::ceildiv<size_t>(count, soc_block_size);
  scatter_hessian_into_augmented_kernel<i_t, f_t>
    <<<grid, soc_block_size, 0, stream.value()>>>(cuopt::make_span(augmented_x),
                                                  cuopt::make_span(csr_indices),
                                                  cuopt::make_span(q_values),
                                                  cuopt::make_span(cones.w),
                                                  cuopt::make_span(cones.eta),
                                                  cuopt::make_span(cones.cone_offsets),
                                                  cuopt::make_span(block_offsets),
                                                  cones.n_cones,
                                                  dual_perturb_value);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

/**
 * Compute the maximum primal and dual step lengths that keep SOC blocks
 * feasible:
 *
 *   x + alpha dx in Q,  z + alpha dz in Q,  alpha <= alpha_max.
 *
 * For one cone u + alpha du, feasibility is
 *
 *   u_0 + alpha du_0 >= ||u_tail + alpha du_tail||.
 *
 * Squaring gives the quadratic
 *
 *   c + 2 b alpha + a alpha^2 >= 0,
 *
 * where c = det_J(u), b = u_0 du_0 - <u_tail, du_tail>, and
 * a = det_J(du). The per-cone kernel below solves for the first boundary
 * crossing, and the final reductions take the global minimum over cones.
 */
template <std::integral i_t, std::floating_point f_t>
std::pair<f_t, f_t> compute_cone_step_length(cone_data_t<i_t, f_t>& cones,
                                             raft::device_span<const f_t> dx,
                                             raft::device_span<const f_t> dz,
                                             f_t alpha_max,
                                             rmm::cuda_stream_view stream)
{
  auto cone_offsets     = cuopt::make_span(cones.cone_offsets);
  auto element_cone_ids = cuopt::make_span(cones.element_cone_ids);
  auto slot_0           = cones.scratch.template get_slot<0>();
  auto slot_1           = cones.scratch.template get_slot<1>();
  auto slot_2           = cones.scratch.template get_slot<2>();

  auto run_pass = [&](raft::device_span<const f_t> u,
                      raft::device_span<const f_t> du,
                      raft::device_span<f_t> alpha) {
    auto du_tail_sq_terms =
      thrust::make_transform_iterator(thrust::make_counting_iterator<size_t>(0),
                                      [du, cone_offsets, element_cone_ids] HD(size_t idx) -> f_t {
                                        const i_t cone = element_cone_ids[idx];
                                        return idx == cone_offsets[cone] ? 0 : du[idx] * du[idx];
                                      });
    cones.segmented_sum(du_tail_sq_terms, slot_0, stream);

    auto u_tail_du_tail_terms = thrust::make_transform_iterator(
      thrust::make_counting_iterator<size_t>(0),
      [u, du, cone_offsets, element_cone_ids] HD(size_t idx) -> f_t {
        const i_t cone = element_cone_ids[idx];
        return idx == cone_offsets[cone] ? 0 : u[idx] * du[idx];
      });
    cones.segmented_sum(u_tail_du_tail_terms, slot_1, stream);

    auto u_tail_sq_terms =
      thrust::make_transform_iterator(thrust::make_counting_iterator<size_t>(0),
                                      [u, cone_offsets, element_cone_ids] HD(size_t idx) -> f_t {
                                        const i_t cone = element_cone_ids[idx];
                                        return idx == cone_offsets[cone] ? 0 : u[idx] * u[idx];
                                      });
    cones.segmented_sum(u_tail_sq_terms, slot_2, stream);

    const size_t grid_dim =
      raft::ceildiv<size_t>(static_cast<size_t>(cones.n_cones), soc_block_size);
    step_length_single_kernel<i_t, f_t><<<grid_dim, soc_block_size, 0, stream.value()>>>(
      u, du, alpha, slot_0, slot_1, slot_2, cone_offsets, alpha_max, cones.n_cones);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  };

  auto alpha_primal = cuopt::make_span(cones.scratch.step_alpha_primal);
  auto alpha_dual   = cuopt::make_span(cones.scratch.step_alpha_dual);

  run_pass(cones.x, dx, alpha_primal);
  run_pass(cones.z, dz, alpha_dual);

  const f_t primal = thrust::reduce(rmm::exec_policy(stream),
                                    alpha_primal.begin(),
                                    alpha_primal.end(),
                                    alpha_max,
                                    thrust::minimum<f_t>());
  const f_t dual   = thrust::reduce(rmm::exec_policy(stream),
                                  alpha_dual.begin(),
                                  alpha_dual.end(),
                                  alpha_max,
                                  thrust::minimum<f_t>());

  return {primal, dual};
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
  gather_cone_heads_kernel<i_t, f_t><<<cone_grid_dim, soc_block_size, 0, stream.value()>>>(
    scaled_dx, slot_1, cone_offsets, cones.n_cones);
  gather_cone_heads_kernel<i_t, f_t><<<cone_grid_dim, soc_block_size, 0, stream.value()>>>(
    scaled_dz, slot_2, cone_offsets, cones.n_cones);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  const size_t element_grid_dim = raft::ceildiv<size_t>(cones.n_cone_entries, soc_block_size);
  combined_cone_shift_write_kernel<i_t, f_t>
    <<<element_grid_dim, soc_block_size, 0, stream.value()>>>(
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
    <<<cone_grid_dim, soc_block_size, 0, stream.value()>>>(
      shift, nt_point, slot_0, slot_1, slot_0, slot_1, cone_offsets, cones.n_cones);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  // Note that we implicitly multiply by -1 here since we are writing -p.
  jordan_divide_by_lambda_write_kernel<i_t, f_t>
    <<<element_grid_dim, soc_block_size, 0, stream.value()>>>(
      shift, nt_point, slot_0, slot_1, cone_offsets, element_cone_ids, scratch_cone);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  apply_w<i_t, f_t>(scratch_cone, out, cones, stream);
}

}  // namespace cuopt::mathematical_optimization::barrier
