/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuda/stream>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include <cuopt/mathematical_optimization/io/mps_data_model.hpp>
#include <cuopt/mathematical_optimization/optimization_problem.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>
#include <dual_simplex/user_problem.hpp>
#include <mip_heuristics/presolve/presolve_budget_policy.hpp>

#include <PSLP/PSLP_API.h>

namespace papilo {
template <typename T>
class PostsolveStorage;

template <typename T>
class Problem;
}  // namespace papilo

namespace cuopt::mathematical_optimization::mip {

template <typename f_t>
struct papilo_postsolve_deleter {
  void operator()(papilo::PostsolveStorage<f_t>* ptr) const;
};

enum class third_party_presolve_status_t {
  INFEASIBLE,
  UNBOUNDED,
  UNBNDORINFEAS,
  OPTIMAL,
  REDUCED,
  UNCHANGED,
};

// Features of the problem as the user handed it in, i.e. before any reduction. This is what Papilo
// itself will work on, so its budget is derived from these rather than from the reduced problem.
template <typename i_t, typename f_t>
presolve_features_t papilo_presolve_features(optimization_problem_t<i_t, f_t> const& op_problem);

template <typename i_t, typename f_t, typename ProblemT>
struct third_party_presolve_result_t {
  third_party_presolve_status_t status;
  ProblemT reduced_problem;
  std::vector<i_t> implied_integer_indices;
  std::vector<i_t> reduced_to_original_map;
  std::vector<i_t> original_to_reduced_map;
  // clique info, etc...
};

template <typename i_t, typename f_t>
using third_party_presolve_device_result_t =
  third_party_presolve_result_t<i_t, f_t, optimization_problem_t<i_t, f_t>>;

template <typename i_t, typename f_t>
using third_party_presolve_host_result_t =
  third_party_presolve_result_t<i_t, f_t, io::mps_data_model_t<i_t, f_t>>;

template <typename i_t, typename f_t>
class third_party_presolve_t {
 public:
  third_party_presolve_t() = default;

  // Delete copy constructor, copy assignment operator, move constructor, and move assignment
  // operator This is because we are using PSLP pointers
  third_party_presolve_t(const third_party_presolve_t&)            = delete;
  third_party_presolve_t& operator=(const third_party_presolve_t&) = delete;
  third_party_presolve_t(third_party_presolve_t&&)                 = delete;
  third_party_presolve_t& operator=(third_party_presolve_t&&)      = delete;

  // Device entry: takes an optimization_problem_t and returns a device-side
  // reduced optimization_problem_t. This is a wrapper around apply_presolve_from_mps_data.
  third_party_presolve_device_result_t<i_t, f_t> apply_presolve_from_op_problem(
    optimization_problem_t<i_t, f_t> const& op_problem,
    problem_category_t category,
    cuopt::mathematical_optimization::presolver_t presolver,
    bool dual_postsolve,
    f_t absolute_tolerance,
    f_t relative_tolerance,
    double time_limit,
    i_t num_cpu_threads = 0,
    i_t max_rounds      = -1,
    i_t max_badgesize   = -1);

  // Host entry: takes an mps_data_model_t and returns a host-side reduced
  // mps_data_model_t. Pure-host throughout
  third_party_presolve_host_result_t<i_t, f_t> apply_presolve_from_mps_data(
    io::mps_data_model_t<i_t, f_t> const& mps_problem,
    problem_category_t category,
    cuopt::mathematical_optimization::presolver_t presolver,
    bool dual_postsolve,
    f_t absolute_tolerance,
    f_t relative_tolerance,
    double time_limit,
    i_t num_cpu_threads = 0,
    i_t max_rounds      = -1,
    i_t max_badgesize   = -1);

  // If set, only Papilo methods whose getName() is listed are registered
  void set_reduction_allowlist(std::optional<std::unordered_set<std::string>> allowlist)
  {
    reduction_allowlist_ = std::move(allowlist);
  }

  // Apply the presolve on an simplex::user_problem in-place. Used in sub MIP and (in the future)
  // restarts.
  third_party_presolve_status_t apply_to_subproblem(
    simplex::user_problem_t<i_t, f_t>& problem,
    const simplex::simplex_solver_settings_t<i_t, f_t>& settings,
    f_t time_limit,
    i_t num_threads);

  // Undo the presolve from a device-side optimization_problem_t.
  // This is a wrapper around undo().
  void undo_from_device(rmm::device_uvector<f_t>& primal_solution,
                        rmm::device_uvector<f_t>& dual_solution,
                        rmm::device_uvector<f_t>& reduced_costs,
                        problem_category_t category,
                        bool status_to_skip,
                        bool dual_postsolve,
                        cuda::stream_ref stream_view);

  // Host-only postsolve. Resizes the vectors to original-problem dimensions.
  void undo(std::vector<f_t>& primal_solution,
            std::vector<f_t>& dual_solution,
            std::vector<f_t>& reduced_costs,
            problem_category_t category,
            bool status_to_skip,
            bool dual_postsolve);

  void uncrush_primal_solution(const std::vector<f_t>& reduced_primal,
                               std::vector<f_t>& full_primal,
                               bool check_postsolve = true) const;

  void crush_primal_solution(const optimization_problem_t<i_t, f_t>& reduced_problem,
                             const std::vector<f_t>& original_primal,
                             std::vector<f_t>& reduced_primal) const;

  void crush_primal_solution(const simplex::user_problem_t<i_t, f_t>& reduced_problem,
                             const std::vector<f_t>& original_primal,
                             std::vector<f_t>& reduced_primal) const;

  void crush_primal_dual_solution(const std::vector<f_t>& x_original,
                                  const std::vector<f_t>& y_original,
                                  std::vector<f_t>& x_reduced,
                                  std::vector<f_t>& y_reduced,
                                  const std::vector<f_t>& z_original,
                                  std::vector<f_t>& z_reduced,
                                  const std::vector<f_t>& A_values,
                                  const std::vector<i_t>& A_indices,
                                  const std::vector<i_t>& A_offsets) const;
  const std::vector<i_t>& get_reduced_to_original_map() const { return reduced_to_original_map_; }
  const std::vector<i_t>& get_original_to_reduced_map() const { return original_to_reduced_map_; }

  const std::vector<f_t>& get_original_objective_coefficients() const
  {
    return original_objective_coefficients_;
  }
  f_t get_original_objective_offset() const { return original_objective_offset_; }
  f_t get_original_objective_scaling_factor() const { return original_objective_scaling_factor_; }

  ~third_party_presolve_t();

 private:
  third_party_presolve_status_t apply_pslp(io::mps_data_model_t<i_t, f_t> const& mps,
                                           double time_limit);

  third_party_presolve_status_t apply_papilo(papilo::Problem<f_t>& papilo_problem,
                                             problem_category_t category,
                                             bool dual_postsolve,
                                             f_t absolute_tolerance,
                                             f_t relative_tolerance,
                                             double time_limit,
                                             i_t num_cpu_threads,
                                             i_t max_rounds,
                                             i_t max_badgesize);

  // Host-only per-backend postsolve helpers. Both resize their vector args
  // to original-problem dimensions.
  void undo_pslp(std::vector<f_t>& primal_solution,
                 std::vector<f_t>& dual_solution,
                 std::vector<f_t>& reduced_costs);

  void undo_papilo(std::vector<f_t>& primal_solution,
                   std::vector<f_t>& dual_solution,
                   std::vector<f_t>& reduced_costs,
                   bool dual_postsolve);

  bool maximize_ = false;
  cuopt::mathematical_optimization::presolver_t presolver_ =
    cuopt::mathematical_optimization::presolver_t::PSLP;
  // PSLP settings
  Settings* pslp_stgs_{nullptr};
  Presolver* pslp_presolver_{nullptr};

  // Necessary due to a nvcc bug due to papilo's constexpr functions
  // Keep the papilo includes in the .cpp to avoid bringing them
  // into any .cu context
  std::unique_ptr<papilo::PostsolveStorage<f_t>, papilo_postsolve_deleter<f_t>>
    papilo_post_solve_storage_;

  std::vector<i_t> reduced_to_original_map_{};
  std::vector<i_t> original_to_reduced_map_{};

  std::vector<f_t> original_objective_coefficients_{};
  f_t original_objective_offset_{0};
  f_t original_objective_scaling_factor_{1};

  std::optional<std::unordered_set<std::string>> reduction_allowlist_{};
};

// Just for testing the conversion: user_problem -> Papilo problem -> user_problem.
template <typename i_t, typename f_t>
void papilo_round_trip(simplex::user_problem_t<i_t, f_t>& problem);

}  // namespace cuopt::mathematical_optimization::mip
