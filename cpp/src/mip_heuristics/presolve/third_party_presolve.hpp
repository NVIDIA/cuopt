/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <memory>
#include <optional>
#include <vector>

#include <cuopt/mathematical_optimization/io/mps_data_model.hpp>
#include <cuopt/mathematical_optimization/optimization_problem.hpp>

#include <PSLP/PSLP_API.h>

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc++11-narrowing"
#pragma clang diagnostic ignored "-Wimplicit-const-int-float-conversion"
#else
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstringop-overflow"
#pragma GCC diagnostic ignored "-Wnarrowing"
#endif
#include <papilo/core/Presolve.hpp>
#include <papilo/core/ProblemBuilder.hpp>
#if defined(__clang__)
#pragma clang diagnostic pop
#else
#pragma GCC diagnostic pop
#endif

namespace papilo {
template <typename T>
class PostsolveStorage;
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

template <typename i_t, typename f_t>
struct third_party_presolve_device_result_t {
  third_party_presolve_status_t status;
  optimization_problem_t<i_t, f_t> reduced_problem;
  std::vector<i_t> implied_integer_indices;
  std::vector<i_t> reduced_to_original_map;
  std::vector<i_t> original_to_reduced_map;
  // clique info, etc...
};

// Host counterpart of third_party_presolve_device_result_t: the reduced
// problem is an mps_data_model_t (host) instead of an optimization_problem_t
// (device). Produced by apply_presolve_from_mps_data.
template <typename i_t, typename f_t>
struct third_party_presolve_host_result_t {
  third_party_presolve_status_t status;
  io::mps_data_model_t<i_t, f_t> reduced_problem;
  std::vector<i_t> implied_integer_indices;
  std::vector<i_t> reduced_to_original_map;
  std::vector<i_t> original_to_reduced_map;
};

// Host-side PSLP input: every buffer PSLP's C API needs, plus dimensions.
template <typename i_t, typename f_t>
struct pslp_input_t {
  std::vector<f_t> coefficients;
  std::vector<i_t> indices;
  std::vector<i_t> offsets;
  std::vector<f_t> obj_coeffs;
  std::vector<f_t> var_lb;
  std::vector<f_t> var_ub;
  std::vector<f_t> constr_lb;
  std::vector<f_t> constr_ub;
  i_t n_rows{0};
  i_t n_cols{0};
  i_t nnz{0};
};

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
  // reduced optimization_problem_t.
  third_party_presolve_device_result_t<i_t, f_t> apply_presolve_from_op_problem(
    optimization_problem_t<i_t, f_t> const& op_problem,
    problem_category_t category,
    cuopt::mathematical_optimization::presolver_t presolver,
    bool dual_postsolve,
    f_t absolute_tolerance,
    f_t relative_tolerance,
    double time_limit,
    i_t num_cpu_threads = 0);

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
    i_t num_cpu_threads = 0);

  void undo(rmm::device_uvector<f_t>& primal_solution,
            rmm::device_uvector<f_t>& dual_solution,
            rmm::device_uvector<f_t>& reduced_costs,
            problem_category_t category,
            bool status_to_skip,
            bool dual_postsolve,
            rmm::cuda_stream_view stream_view);

  // Host-only postsolve. Resizes the vectors to original-problem dimensions.
  // The device-side `undo` above is a thin shim around this method.
  void undo_host(std::vector<f_t>& primal_solution,
                 std::vector<f_t>& dual_solution,
                 std::vector<f_t>& reduced_costs,
                 problem_category_t category,
                 bool status_to_skip,
                 bool dual_postsolve);

  void uncrush_primal_solution(const std::vector<f_t>& reduced_primal,
                               std::vector<f_t>& full_primal) const;

  void crush_primal_solution(const std::vector<f_t>& original_primal,
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

  ~third_party_presolve_t();

 private:
  third_party_presolve_status_t apply_pslp(pslp_input_t<i_t, f_t>& arrays, double time_limit);

  third_party_presolve_status_t apply_papilo(papilo::Problem<f_t>& papilo_problem,
                                             problem_category_t category,
                                             bool dual_postsolve,
                                             f_t absolute_tolerance,
                                             f_t relative_tolerance,
                                             double time_limit,
                                             i_t num_cpu_threads);

  // Host-only per-backend postsolve helpers. Both resize their vector args
  // to original-problem dimensions.
  void undo_pslp_host(std::vector<f_t>& primal_solution,
                      std::vector<f_t>& dual_solution,
                      std::vector<f_t>& reduced_costs);

  void undo_papilo_host(std::vector<f_t>& primal_solution,
                        std::vector<f_t>& dual_solution,
                        std::vector<f_t>& reduced_costs,
                        bool dual_postsolve);

  bool maximize_ = false;
  cuopt::mathematical_optimization::presolver_t presolver_ =
    cuopt::mathematical_optimization::presolver_t::PSLP;
  // PSLP settings
  Settings* pslp_stgs_{nullptr};
  Presolver* pslp_presolver_{nullptr};

  // Necessary due to a nvcc bug due to papilo's constexpr functions.
  // Keep heavier papilo includes in the .cpp; PostsolveStorage stays opaque here.
  std::unique_ptr<papilo::PostsolveStorage<f_t>, papilo_postsolve_deleter<f_t>>
    papilo_post_solve_storage_;

  std::vector<i_t> reduced_to_original_map_{};
  std::vector<i_t> original_to_reduced_map_{};
};

}  // namespace cuopt::mathematical_optimization::mip
