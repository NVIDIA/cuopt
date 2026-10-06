/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <pdlp/cusparse_view.hpp>

#include <cuda/stream>
#include <raft/core/handle.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

namespace cuopt::mathematical_optimization::pdlp {

/**
 * @brief Adaptive column-reduced operator for single-problem reflected PDLP.
 *
 * Variables whose projected value is exactly at a bound are omitted from the
 * sparse matrix. Their contribution to A*x is advanced analytically until a
 * full refresh iteration allows them to leave the bound again.
 *
 * This design follows the column-reduction strategy in HPR-LP-C:
 * https://github.com/PolyU-IOR/HPR-LP-C
 */
template <typename i_t, typename f_t>
class reduced_matrix_t {
 public:
  reduced_matrix_t(raft::handle_t const* handle_ptr,
                   mip::problem_t<i_t, f_t>& problem,
                   bool enabled);

  bool active() const { return active_; }
  // The original structure outlives this operator; numerical values stay scaled.
  void redirect_csr_structure(const mip::problem_t<i_t, f_t>& original_problem)
  {
    structure_problem_ptr_ = &original_problem;
  }
  bool refresh_pending() const { return refresh_pending_; }
  bool request_refresh(bool update_mask,
                       rmm::device_uvector<f_t>& current_primal,
                       const rmm::device_uvector<f_t>& restart_primal);

  /** Update activation state at a convergence checkpoint. Returns true if CUDA graphs changed. */
  bool update_mode(f_t relative_kkt,
                   bool restarted,
                   const rmm::device_uvector<f_t>& projected_primal,
                   const rmm::device_uvector<f_t>& current_primal,
                   const rmm::device_uvector<f_t>& restart_primal);

  /** Rebuild after the one full iteration requested by update_mode(). */
  bool finish_refresh(const rmm::device_uvector<f_t>& projected_primal,
                      const rmm::device_uvector<f_t>& current_primal,
                      const rmm::device_uvector<f_t>& restart_primal);

  void compute_At_y(const rmm::device_uvector<f_t>& dual);

  /** Advance only free columns. Full vectors are materialized before a full refresh step. */
  void update_primal(const rmm::device_uvector<f_t>& primal_step_size,
                     const rmm::device_scalar<f_t>& halpern_weight);

  void compute_A_x(rmm::device_uvector<f_t>& dual_gradient,
                   const rmm::device_scalar<f_t>& halpern_weight);

  i_t free_count() const { return free_count_; }

 private:
  bool rebuild(const rmm::device_uvector<f_t>& projected_primal,
               const rmm::device_uvector<f_t>& current_primal,
               const rmm::device_uvector<f_t>& restart_primal,
               bool initialize_mask);
  void clear_operator();
  void initialize_mask(const rmm::device_uvector<f_t>& projected_primal);
  void release_mask(const rmm::device_uvector<f_t>& projected_primal);
  void build_compact_operator();
  void build_fixed_activities(const rmm::device_uvector<f_t>& current_primal,
                              const rmm::device_uvector<f_t>& initial_primal);
  void materialize_primal(rmm::device_uvector<f_t>& current_primal,
                          const rmm::device_uvector<f_t>& initial_primal);
  void initialize_spmv();
  void run_spmv(cusparse_sp_mat_descr_view matrix,
                cusparse_dn_vec_descr_view input,
                cusparse_dn_vec_descr_view output,
                rmm::device_uvector<uint8_t>& buffer);

  raft::handle_t const* handle_ptr_;
  cuda::stream_ref stream_view_;
  mip::problem_t<i_t, f_t>* problem_ptr_;
  const mip::problem_t<i_t, f_t>* structure_problem_ptr_;
  bool enabled_;
  bool active_{false};
  bool refresh_pending_{false};
  bool refresh_mask_pending_{false};
  bool mask_started_{false};
  int stable_checks_{0};
  i_t free_count_{0};
  i_t last_free_count_{-1};

  rmm::device_uvector<uint8_t> mask_;
  rmm::device_uvector<i_t> free_to_original_;

  rmm::device_uvector<i_t> reduced_At_offsets_;
  rmm::device_uvector<i_t> reduced_At_indices_;
  rmm::device_uvector<f_t> reduced_At_values_;
  rmm::device_uvector<i_t> reduced_A_offsets_;
  rmm::device_uvector<i_t> reduced_A_indices_;
  rmm::device_uvector<f_t> reduced_A_values_;

  rmm::device_uvector<f_t> reduced_At_y_;
  // The reflected vector is the input to the compact A*x product.
  rmm::device_uvector<f_t> reduced_x_;
  rmm::device_uvector<f_t> reduced_current_;
  rmm::device_uvector<f_t> reduced_initial_;
  rmm::device_uvector<f_t> reduced_objective_;
  rmm::device_uvector<typename type_2<f_t>::type> reduced_bounds_;
  // For omitted columns, x = b + alpha*(x_at_refresh-b) + beta*(x_at_restart-b).
  // This avoids advancing the full primal vector on every reduced iteration.
  rmm::device_scalar<f_t> fixed_alpha_;
  rmm::device_scalar<f_t> fixed_beta_;
  rmm::device_uvector<f_t> fixed_bound_activity_;
  rmm::device_uvector<f_t> fixed_current_activity_;
  rmm::device_uvector<f_t> fixed_initial_activity_;

  cusparse_sp_mat_uptr reduced_A_descr_;
  cusparse_sp_mat_uptr reduced_At_descr_;
  cusparse_dn_vec_uptr dual_descr_;
  cusparse_dn_vec_uptr reduced_At_y_descr_;
  cusparse_dn_vec_uptr reduced_x_descr_;
  cusparse_dn_vec_uptr dual_gradient_descr_;
  rmm::device_uvector<uint8_t> A_buffer_;
  rmm::device_uvector<uint8_t> At_buffer_;
  const rmm::device_scalar<f_t> one_;
  const rmm::device_scalar<f_t> zero_;
};

}  // namespace cuopt::mathematical_optimization::pdlp
