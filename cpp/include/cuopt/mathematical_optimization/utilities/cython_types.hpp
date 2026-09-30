/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/export.hpp>
#include <cuopt/mathematical_optimization/mip/solver_solution.hpp>
#include <cuopt/mathematical_optimization/pdlp/solver_solution.hpp>
#include <cuopt/mathematical_optimization/utilities/internals.hpp>

#include <cassert>
#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace cuopt {
namespace CUOPT_EXPORT mathematical_optimization {
// Forward declared, not included: these structs are also compiled into cuopt_client, which
// is CPU-only and cannot link the GPU-side barrier_cache_t destructor.
class barrier_cache_t;
}  // namespace CUOPT_EXPORT mathematical_optimization

namespace CUOPT_EXPORT cython {

using cpu_buffer = std::vector<double>;

// Held through an opaque pointer with a runtime deleter so destroying the variant never
// needs rmm, which keeps cuopt_client CPU-only (#1890). Resolving the deleter at run time
// rather than by name also avoids a cuopt_client -> cuopt_mathopt edge.
// Defined in cython_types_gpu.hpp.
struct lp_gpu_solutions_t;
struct mip_gpu_solution_t;

template <typename T>
struct gpu_holder_deleter_t {
  void (*destroy)(T*) = nullptr;
  void operator()(T* p) const noexcept
  {
    // Only the factories in cython_types_gpu.hpp build these, so a null deleter with a
    // live pointer is a bug. Aborting beats leaking it silently.
    if (p == nullptr) { return; }
    assert(destroy != nullptr);
    destroy(p);
  }
};

using lp_gpu_ptr  = std::unique_ptr<lp_gpu_solutions_t, gpu_holder_deleter_t<lp_gpu_solutions_t>>;
using mip_gpu_ptr = std::unique_ptr<mip_gpu_solution_t, gpu_holder_deleter_t<mip_gpu_solution_t>>;

// LP solution struct — GPU and CPU solutions use the same struct, differing only in the
// vector storage type (device_buffer vs std::vector).  The solutions_ variant holds all
// buffer/vector fields; shared scalar fields live directly on the struct.
struct linear_programming_ret_t {
  struct cpu_solutions_t {
    cpu_buffer primal_solution_;
    cpu_buffer dual_solution_;
    cpu_buffer reduced_cost_;
    cpu_buffer current_primal_solution_;
    cpu_buffer current_dual_solution_;
    cpu_buffer initial_primal_average_;
    cpu_buffer initial_dual_average_;
    cpu_buffer current_ATY_;
    cpu_buffer sum_primal_solutions_;
    cpu_buffer sum_dual_solutions_;
    cpu_buffer last_restart_duality_gap_primal_solution_;
    cpu_buffer last_restart_duality_gap_dual_solution_;
  };

  std::variant<lp_gpu_ptr, cpu_solutions_t> solutions_;

  /* -- PDLP Warm Start Scalars -- */
  double initial_primal_weight_{};
  double initial_step_size_{};
  int total_pdlp_iterations_{};
  int total_pdhg_iterations_{};
  double last_candidate_kkt_score_{};
  double last_restart_kkt_score_{};
  double sum_solution_weight_{};
  int iterations_since_last_restart_{};
  /* -- /PDLP Warm Start Scalars -- */

  mathematical_optimization::pdlp_termination_status_t termination_status_{};
  error_type_t error_status_{};
  std::string error_message_;

  /*Termination stats*/
  double l2_primal_residual_{};
  double l2_dual_residual_{};
  double primal_objective_{};
  double dual_objective_{};
  double gap_{};
  int nb_iterations_{};
  double solve_time_{};
  mathematical_optimization::method_t solved_by_{};

  /** GPU barrier cache (stream + handle + iteration workspace), non-owning. call_solve hands
   * ownership to the caller, which wraps it in a Python capsule and deletes it there.
   */
  mathematical_optimization::barrier_cache_t* barrier_cache{nullptr};

  bool is_gpu() const { return std::holds_alternative<lp_gpu_ptr>(solutions_); }
};

// MIP solution struct — GPU and CPU solutions use the same struct, differing only in the
// solution vector storage type.
struct mip_ret_t {
  std::variant<mip_gpu_ptr, cpu_buffer> solution_;

  mathematical_optimization::mip_termination_status_t termination_status_{};
  error_type_t error_status_{};
  std::string error_message_;

  /*Termination stats*/
  double objective_{};
  double mip_gap_{};
  double solution_bound_{};
  double total_solve_time_{};
  double presolve_time_{};
  double max_constraint_violation_{};
  double max_int_violation_{};
  double max_variable_bound_violation_{};
  int nodes_{};
  int simplex_iterations_{};

  bool is_gpu() const { return std::holds_alternative<mip_gpu_ptr>(solution_); }
};

}  // namespace CUOPT_EXPORT cython
}  // namespace cuopt
