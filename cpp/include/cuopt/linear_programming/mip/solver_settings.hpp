/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <vector>

#include <cuopt/linear_programming/constants.h>
#include <cuopt/linear_programming/pdlp/pdlp_hyper_params.cuh>
#include <cuopt/linear_programming/utilities/internals.hpp>

#include <raft/core/device_span.hpp>
#include <rmm/device_uvector.hpp>

#include <vector>

namespace cuopt::linear_programming {

struct benchmark_info_t {
  double last_improvement_of_best_feasible    = 0;
  double last_improvement_after_recombination = 0;
  double objective_of_initial_population      = std::numeric_limits<double>::max();
};

// Forward declare solver_settings_t for friend class
template <typename i_t, typename f_t>
class solver_settings_t;

template <typename i_t, typename f_t>
class mip_solver_settings_t {
 public:
  mip_solver_settings_t() = default;

  /**
   * @brief Set the callback for the user solution
   *
   * @param[in] callback - Callback handler for user solutions.
   * @param[in] user_data - Pointer to user-defined data forwarded to the callback.
   */
  void set_mip_callback(internals::base_solution_callback_t* callback = nullptr,
                        void* user_data                               = nullptr);

  /**
   * @brief Add an primal solution.
   *
   * @note This function can be called multiple times to add more solutions.
   *
   * @param[in] initial_solution Device or host memory pointer to a floating
   * point array of size size. cuOpt copies this data. Copy happens on the
   * stream of the raft:handler passed to the problem.
   * @param size Size of the initial_solution array.
   */
  void add_initial_solution(const f_t* initial_solution,
                            i_t size,
                            rmm::cuda_stream_view stream = rmm::cuda_stream_default);

  /**
   * @brief Get the callback for the user solution
   *
   * @return callback pointer
   */
  const std::vector<internals::base_solution_callback_t*> get_mip_callbacks() const;

  struct tolerances_t {
    f_t presolve_absolute_tolerance = 1.0e-6;
    f_t absolute_tolerance          = 1.0e-6;
    f_t relative_tolerance          = 1.0e-12;
    f_t integrality_tolerance       = 1.0e-5;
    f_t absolute_mip_gap            = 1.0e-10;
    f_t relative_mip_gap            = 1.0e-4;
  };

  /**
   * @brief Get the tolerance settings as a single structure.
   */
  tolerances_t get_tolerances() const noexcept;

  template <typename U, typename V>
  friend class problem_checking_t;
  tolerances_t tolerances;

  f_t time_limit                = std::numeric_limits<f_t>::infinity();
  i_t node_limit                = std::numeric_limits<i_t>::max();
  bool heuristics_only          = false;
  i_t reliability_branching     = -1;
  i_t num_cpu_threads           = -1;  // -1 means use default number of threads in branch and bound
  i_t max_cut_passes            = 10;  // number of cut passes to make
  i_t mir_cuts                  = -1;
  i_t mixed_integer_gomory_cuts = -1;
  i_t knapsack_cuts             = -1;
  i_t strong_chvatal_gomory_cuts      = -1;
  i_t reduced_cost_strengthening      = -1;
  f_t cut_change_threshold            = 1e-3;
  f_t cut_min_orthogonality           = 0.5;
  i_t mip_batch_pdlp_strong_branching = 0;
  i_t num_gpus                        = 1;
  bool log_to_console                 = true;

  std::string log_file;
  std::string sol_file;
  std::string user_problem_file;

  /** Initial primal solutions */
  std::vector<std::shared_ptr<rmm::device_uvector<f_t>>> initial_solutions;
  bool mip_scaling = false;
  presolver_t presolver{presolver_t::Default};
  // this is for extracting info from different places of the solver during
  // benchmarks
  benchmark_info_t* benchmark_info_ptr = nullptr;

  // TODO check with Akif and Alice
  pdlp_hyper_params::pdlp_hyper_params_t hyper_params;

 private:
  std::vector<internals::base_solution_callback_t*> mip_callbacks_;

  friend class solver_settings_t<i_t, f_t>;
};

}  // namespace cuopt::linear_programming
