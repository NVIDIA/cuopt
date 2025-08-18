/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <rmm/device_uvector.hpp>

#include <mps_parser/utilities/span.hpp>

namespace cuopt::linear_programming {

template <typename i_t, typename f_t>
struct pdlp_warm_start_data_view_t;

// Holds everything necessary to warm start PDLP
template <typename i_t, typename f_t>
struct pdlp_warm_start_data_t {
  rmm::device_uvector<f_t>
    current_primal_solution_;  // Can't just be pulled from solution object as we might return the
                               // average as solution while we want to continue on optimize on the
                               // current, no the average
  rmm::device_uvector<f_t> current_dual_solution_;   // Same as above
  rmm::device_uvector<f_t> initial_primal_average_;  // Same as above but if current is returned
  rmm::device_uvector<f_t> initial_dual_average_;    // Same as above
  rmm::device_uvector<f_t> current_ATY_;
  rmm::device_uvector<f_t> sum_primal_solutions_;
  rmm::device_uvector<f_t> sum_dual_solutions_;
  rmm::device_uvector<f_t> last_restart_duality_gap_primal_solution_;
  rmm::device_uvector<f_t> last_restart_duality_gap_dual_solution_;
  f_t initial_primal_weight_{-1};
  f_t initial_step_size_{-1};
  i_t total_pdlp_iterations_{-1};
  i_t total_pdhg_iterations_{-1};
  f_t last_candidate_kkt_score_{-1};
  f_t last_restart_kkt_score_{-1};
  f_t sum_solution_weight_{-1};
  i_t iterations_since_last_restart_{-1};

  // Constructor when building it in the solution object
  pdlp_warm_start_data_t(rmm::device_uvector<f_t>& current_primal_solution,
                         rmm::device_uvector<f_t>& current_dual_solution,
                         rmm::device_uvector<f_t>& initial_primal_average,
                         rmm::device_uvector<f_t>& initial_dual_average,
                         rmm::device_uvector<f_t>& current_ATY,
                         rmm::device_uvector<f_t>& sum_primal_solutions,
                         rmm::device_uvector<f_t>& sum_dual_solutions,
                         rmm::device_uvector<f_t>& last_restart_duality_gap_primal_solution,
                         rmm::device_uvector<f_t>& last_restart_duality_gap_dual_solution,
                         f_t initial_primal_weight,
                         f_t initial_step_size,
                         i_t total_pdlp_iterations,
                         i_t total_pdhg_iterations,
                         f_t last_candidate_kkt_score,
                         f_t last_restart_kkt_score,
                         f_t sum_solution_weight,
                         i_t iterations_since_last_restart);

  // Empty constructor
  pdlp_warm_start_data_t();

  // Copy constructor using the view version for the cython_solver
  pdlp_warm_start_data_t(const pdlp_warm_start_data_view_t<i_t, f_t>& other,
                         rmm::cuda_stream_view stream_view);

  // Copy constructor for when copying the solver_settings object in the PDLP object
  pdlp_warm_start_data_t(const pdlp_warm_start_data_t<i_t, f_t>& other,
                         rmm::cuda_stream_view stream_view);

 private:
  // Check sizes through assertion
  void check_sizes();
};

template <typename i_t, typename f_t>
struct pdlp_warm_start_data_view_t {
  cuopt::mps_parser::span<f_t const> current_primal_solution_;
  cuopt::mps_parser::span<f_t const> current_dual_solution_;
  cuopt::mps_parser::span<f_t const> initial_primal_average_;
  cuopt::mps_parser::span<f_t const> initial_dual_average_;
  cuopt::mps_parser::span<f_t const> current_ATY_;
  cuopt::mps_parser::span<f_t const> sum_primal_solutions_;
  cuopt::mps_parser::span<f_t const> sum_dual_solutions_;
  cuopt::mps_parser::span<f_t const> last_restart_duality_gap_primal_solution_;
  cuopt::mps_parser::span<f_t const> last_restart_duality_gap_dual_solution_;
  f_t initial_primal_weight_{-1};
  f_t initial_step_size_{-1};
  i_t total_pdlp_iterations_{-1};
  i_t total_pdhg_iterations_{-1};
  f_t last_candidate_kkt_score_{-1};
  f_t last_restart_kkt_score_{-1};
  f_t sum_solution_weight_{-1};
  i_t iterations_since_last_restart_{-1};
};

}  // namespace cuopt::linear_programming
