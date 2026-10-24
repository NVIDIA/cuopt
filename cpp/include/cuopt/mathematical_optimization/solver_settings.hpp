/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/export.hpp>
#include <cuopt/mathematical_optimization/pdlp/pdlp_warm_start_data.hpp>

#include <cuda/stream>
#include <raft/core/device_span.hpp>

#include <rmm/device_uvector.hpp>

#include <cuopt/mathematical_optimization/constants.h>
#include <cuopt/mathematical_optimization/mip/solver_settings.hpp>
#include <cuopt/mathematical_optimization/pdlp/solver_settings.hpp>
#include <cuopt/mathematical_optimization/utilities/internals.hpp>

#include <cstddef>
#include <optional>
#include <vector>

namespace cuopt {
namespace CUOPT_EXPORT mathematical_optimization {

namespace detail {
/**
 * @brief sizeof(solver_settings_t<int, double>) as compiled into cuopt_client.
 *
 * The class is built by cuopt_mathopt and mutated by cuopt_client, so the two have to agree
 * on its layout. They are separate shared libraries, built independently, and the layout
 * depends on third-party headers, so agreement is an assumption rather than a guarantee.
 * cuOptCreateSolverSettings compares this against its own sizeof and refuses to hand back a
 * handle when they differ -- a clear error instead of writing through a wrong offset.
 */
CUOPT_EXPORT std::size_t client_solver_settings_size() noexcept;
}  // namespace detail

template <typename i_t, typename f_t>
class solver_settings_t {
 public:
  solver_settings_t();

  // Delete copy constructor
  solver_settings_t(const solver_settings_t& settings) = delete;
  // Delete assignment operator
  solver_settings_t& operator=(const solver_settings_t& settings) = delete;
  // Delete move constructor
  solver_settings_t(solver_settings_t&& settings) = delete;
  // Delete move assignment operator
  solver_settings_t& operator=(solver_settings_t&& settings) = delete;

  void set_parameter_from_string(const std::string& name, const std::string& value);

  template <typename T>
  void set_parameter(const std::string& name, T value);

  template <typename T>
  T get_parameter(const std::string& name) const;

  std::string get_parameter_as_string(const std::string& name) const;

  void set_initial_pdlp_primal_solution(const f_t* initial_primal_solution,
                                        i_t size,
                                        cuda::stream_ref stream = cuda::stream_ref{
                                          cudaStream_t{cudaStreamDefault}});
  void set_initial_pdlp_dual_solution(const f_t* initial_dual_solution,
                                      i_t size,
                                      cuda::stream_ref stream = cuda::stream_ref{
                                        cudaStream_t{cudaStreamDefault}});
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

  const rmm::device_uvector<f_t>& get_initial_pdlp_primal_solution() const;
  const rmm::device_uvector<f_t>& get_initial_pdlp_dual_solution() const;

  // MIP Settings
  void add_initial_mip_solution(const f_t* initial_solution,
                                i_t size,
                                cuda::stream_ref stream = cuda::stream_ref{
                                  cudaStream_t{cudaStreamDefault}});
  void set_mip_callback(internals::base_solution_callback_t* callback = nullptr,
                        void* user_data                               = nullptr);

  const pdlp_warm_start_data_view_t<i_t, f_t>& get_pdlp_warm_start_data_view() const noexcept;
  const std::vector<internals::base_solution_callback_t*> get_mip_callbacks() const;

  pdlp_solver_settings_t<i_t, f_t>& get_pdlp_settings();
  mip_solver_settings_t<i_t, f_t>& get_mip_settings();

  const std::vector<parameter_info_t<f_t>>& get_float_parameters() const;
  const std::vector<parameter_info_t<i_t>>& get_int_parameters() const;
  const std::vector<parameter_info_t<bool>>& get_bool_parameters() const;
  const std::vector<parameter_info_t<std::string>>& get_string_parameters() const;
  const std::vector<std::string> get_parameter_names() const;

  void load_parameters_from_file(const std::string& path);
  bool dump_parameters_to_file(const std::string& path, bool hyperparameters_only = true) const;

 private:
  pdlp_solver_settings_t<i_t, f_t> pdlp_settings;
  mip_solver_settings_t<i_t, f_t> mip_settings;

  std::vector<parameter_info_t<f_t>> float_parameters;
  std::vector<parameter_info_t<i_t>> int_parameters;
  std::vector<parameter_info_t<bool>> bool_parameters;
  std::vector<parameter_info_t<std::string>> string_parameters;
};

// Forces every consumer except solver_settings.cu (the explicit instantiation site) to bind
// to cuopt_mathopt's ctor/dtor instead of generating its own local copy.
#ifndef CUOPT_SOLVER_SETTINGS_T_EXPLICIT_INSTANTIATION
extern template class solver_settings_t<int, double>;
#endif

}  // namespace CUOPT_EXPORT mathematical_optimization
}  // namespace cuopt
