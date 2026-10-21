/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/mathematical_optimization/constants.h>
#include <cuda/stream>
#include <cuopt/export.hpp>
#include <cuopt/mathematical_optimization/cpu_pdlp_warm_start_data.hpp>
#include <cuopt/mathematical_optimization/pdlp/pdlp_hyper_params.cuh>
#include <cuopt/mathematical_optimization/pdlp/pdlp_parameters.hpp>
#include <cuopt/mathematical_optimization/pdlp/pdlp_warm_start_data.hpp>
#include <cuopt/mathematical_optimization/utilities/internals.hpp>
#include <cuopt/mathematical_optimization/utilities/opaque_ptr.hpp>
#include <optional>
#include <raft/core/device_span.hpp>
#include <rmm/device_uvector.hpp>

#include <atomic>
#include <tuple>

#include <cuda/std/span>

namespace cuopt {
namespace CUOPT_EXPORT mathematical_optimization {

template <typename i_t, typename f_t>
class barrier_cache_t;

// Forward declare solver_settings_t for friend class
template <typename i_t, typename f_t>
class solver_settings_t;

template <typename i_t, typename f_t>
class pdlp_solver_settings_t : public pdlp_parameters_t<i_t, f_t> {
 public:
  using typename pdlp_parameters_t<i_t, f_t>::tolerances_t;

  pdlp_solver_settings_t() = default;

  /**
   * @brief Set an initial primal solution.
   *
   * @note Default value is all 0.
   *
   * @param[in] initial_primal_solution Device or host memory pointer to a
   * floating point array of size size. cuOpt copies this data. Copy happens on
   * the stream of the raft:handler passed to the problem.
   * @param size Size of the initial_primal_solution array.
   */
  void set_initial_primal_solution(const f_t* initial_primal_solution,
                                   i_t size,
                                   cuda::stream_ref stream = cuda::stream_ref{
                                     cudaStream_t{cudaStreamDefault}});

  /**
   * @brief Set an initial dual solution.
   *
   * @note Default value is all 0.
   *
   * @param[in] initial_dual_solution Device or host memory pointer to a
   * floating point array of size size. cuOpt copies this data. Copy happens on
   * the stream of the raft:handler passed to the problem.
   * @param size Size of the initial_dual_solution array.
   */
  void set_initial_dual_solution(const f_t* initial_dual_solution,
                                 i_t size,
                                 cuda::stream_ref stream = cuda::stream_ref{
                                   cudaStream_t{cudaStreamDefault}});

  /** TODO batch mode: tmp
   * @brief Set an initial step size.
   *
   * @param[in] initial_step_size Initial step size.
   */
  // TODO batch mode: tmp
  void set_initial_step_size(f_t initial_step_size);
  /**
   * @brief Set an initial primal weight.
   *
   * @param[in] initial_primal_weight Initial primal weight.
   */
  void set_initial_primal_weight(f_t initial_primal_weight);
  /**
   * @brief Set an initial pdlp iteration.
   *
   * @param[in] initial_pdlp_iteration Initial pdlp iteration.
   */
  void set_initial_pdlp_iteration(i_t initial_pdlp_iteration);

  /**
   * @brief Set the pdlp warm start data. This allows to restart PDLP with a
   * previous solution
   *
   * @note Interface for the C++ side. Only Stable2 and Fast1 are supported.
   *
   * @param pdlp_warm_start_data_view Pdlp warm start data from your solution
   * object to warm start from
   * @param var_mapping Variables indices to scatter to in case the new problem
   * has less variables
   * @param constraint_mapping Constraints indices to scatter to in case the new
   * problem has less constraints
   */
  void set_pdlp_warm_start_data(
    pdlp_warm_start_data_t<i_t, f_t>& pdlp_warm_start_data_view,
    const rmm::device_uvector<i_t>& var_mapping =
      rmm::device_uvector<i_t>{0, cuda::stream_ref{cudaStream_t{cudaStreamDefault}}},
    const rmm::device_uvector<i_t>& constraint_mapping = rmm::device_uvector<i_t>{
      0, cuda::stream_ref{cudaStream_t{cudaStreamDefault}}});

  // Same but for the Cython interface
  void set_pdlp_warm_start_data(const f_t* current_primal_solution,
                                const f_t* current_dual_solution,
                                const f_t* initial_primal_average,
                                const f_t* initial_dual_average,
                                const f_t* current_ATY,
                                const f_t* sum_primal_solutions,
                                const f_t* sum_dual_solutions,
                                const f_t* last_restart_duality_gap_primal_solution,
                                const f_t* last_restart_duality_gap_dual_solution,
                                i_t primal_size,
                                i_t dual_size,
                                f_t initial_primal_weight_,
                                f_t initial_step_size_,
                                i_t total_pdlp_iterations_,
                                i_t total_pdhg_iterations_,
                                f_t last_candidate_kkt_score_,
                                f_t last_restart_kkt_score_,
                                f_t sum_solution_weight_,
                                i_t iterations_since_last_restart_);
  /**
   * @brief Get the pdlp warm start data
   *
   * @return pdlp warm start data
   */
  const pdlp_warm_start_data_t<i_t, f_t>& get_pdlp_warm_start_data() const noexcept;
  pdlp_warm_start_data_t<i_t, f_t>& get_pdlp_warm_start_data();
  const pdlp_warm_start_data_view_t<i_t, f_t>& get_pdlp_warm_start_data_view() const noexcept;

  /**
   * @brief Get the CPU-backed PDLP warm start data (for remote execution)
   * @return Const reference to cpu_pdlp_warm_start_data_t
   * @note Used when the solver runs on a CPU-only host via remote execution.
   *       The data is std::vector-backed rather than device_uvector-backed.
   */
  const cpu_pdlp_warm_start_data_t<i_t, f_t>& get_cpu_pdlp_warm_start_data() const noexcept;

  /**
   * @brief Get mutable CPU-backed PDLP warm start data (for remote execution)
   * @return Mutable reference to cpu_pdlp_warm_start_data_t
   */
  cpu_pdlp_warm_start_data_t<i_t, f_t>& get_cpu_pdlp_warm_start_data() noexcept;
  // TODO batch mode: tmp
  std::optional<f_t> get_initial_step_size() const;
  // TODO batch mode: tmp
  std::optional<f_t> get_initial_primal_weight() const;
  // TODO batch mode: tmp
  std::optional<i_t> get_initial_pdlp_iteration() const;

  const rmm::device_uvector<f_t>& get_initial_primal_solution() const;
  const rmm::device_uvector<f_t>& get_initial_dual_solution() const;

  bool has_initial_primal_solution() const;
  bool has_initial_dual_solution() const;

  template <typename U, typename V>
  friend class problem_checking_t;

  bool inside_mip{false};
  // For concurrent termination
  std::atomic<int>* concurrent_halt{nullptr};
  // Shared strong branching solved flags for cooperative DS + PDLP
  cuda::std::span<std::atomic<int>> shared_sb_solved;
  static constexpr f_t minimal_absolute_tolerance = 1.0e-12;
  /** Non-owning cache pointer set by ``call_solve`` for barrier cache reuse. */
  barrier_cache_t<i_t, f_t>* barrier_cache{nullptr};

 private:
  /** Initial primal solution */
  std::shared_ptr<rmm::device_uvector<f_t>> initial_primal_solution_;
  /** Initial dual solution */
  std::shared_ptr<rmm::device_uvector<f_t>> initial_dual_solution_;
  /** Initial step size */
  // TODO batch mode: tmp
  std::optional<f_t> initial_step_size_;
  /** Initial primal weight */
  // TODO batch mode: tmp
  std::optional<f_t> initial_primal_weight_;
  /** Initial pdlp iteration */
  // TODO batch mode: tmp
  std::optional<i_t> initial_pdlp_iteration_;
  /** GPU-backed warm start data (device_uvector), used by C++ API and local GPU solves. Built on
   * first use by the GPU library, so this class needs no GPU symbol to construct, copy or destroy.
   */
  mutable opaque_ptr_t<pdlp_warm_start_data_t<i_t, f_t>> gpu_warm_start_;
  /** Warm start data as spans over external memory, used by Cython/Python interface */
  pdlp_warm_start_data_view_t<i_t, f_t> pdlp_warm_start_data_view_;
  /** CPU-backed warm start data (std::vector), used for remote execution on CPU-only hosts */
  cpu_pdlp_warm_start_data_t<i_t, f_t> cpu_pdlp_warm_start_data_;

  /** Creates the device warm start on first use. Defined in the GPU library. */
  pdlp_warm_start_data_t<i_t, f_t>& gpu_warm_start() const;

  friend class solver_settings_t<i_t, f_t>;
};

}  // namespace CUOPT_EXPORT mathematical_optimization
}  // namespace cuopt
