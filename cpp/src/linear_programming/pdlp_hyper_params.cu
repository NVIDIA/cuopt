/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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

#include <cuopt/linear_programming/pdlp/pdlp_hyper_params.cuh>

namespace cuopt::linear_programming::pdlp_hyper_params {

// Scaling factor applied when computing the initial step size
double initial_step_size_scaling = 1.0;
// Number of Ruiz iterations applied during initial scaling
int default_l_inf_ruiz_iterations = 10;
// Whether to apply Pock-Chambolle scaling
bool do_pock_chambolle_scaling = true;
// Whether to apply Ruiz scaling
bool do_ruiz_scaling = true;
// Alpha parameter for Pock-Chambolle initial scaling
double default_alpha_pock_chambolle_rescaling = 1.0;
// Threshold for triggering artificial restarts
double default_artificial_restart_threshold = 0.36;
// Whether to compute initial step size before applying scaling
bool compute_initial_step_size_before_scaling = false;
// Whether to compute initial primal weight before applying scaling
bool compute_initial_primal_weight_before_scaling = false;
// Scaling factor for initial primal weight based on objective coefficients
double initial_primal_weight_c_scaling = 1.0;
// Scaling factor for initial primal weight based on constraint bounds
double initial_primal_weight_b_scaling = 1.0;
// Number of iterations between each convergence check and restart
int major_iteration = 40;
// First n min_iteration_restart iterations will have convergence checks and restart
int min_iteration_restart = 10;
// Strategy used for restarts. Check restart_strategy_t for more details
int restart_strategy = 1;
// Whether to disable restarting to average during restart
bool never_restart_to_average = false;
// Reduction exponent for adaptive step size strategy
double host_default_reduction_exponent = 0.3;
// Growth exponent for adaptive step size strategy
double host_default_growth_exponent = 0.6;
// Smoothing factor for primal weight updates
double host_default_primal_weight_update_smoothing = 0.5;
// Sufficient reduction threshold to trigger restart
double host_default_sufficient_reduction_for_restart = 0.2;
// Necessary reduction threshold to trigger restart
double host_default_necessary_reduction_for_restart = 0.8;
// Initial primal weight scaling factor
double host_primal_importance = 1.0;
// Smoothing factor for primal distance calculations
double host_primal_distance_smoothing = 0.5;
// Smoothing factor for dual distance calculations
double host_dual_distance_smoothing = 0.5;
// Whether to compute last restart before updating primal weight
bool compute_last_restart_before_new_primal_weight = true;
// Whether to allow artificial restarts in main PDLP loop
bool artificial_restart_in_main_loop = false;
// Whether to rescale the solution during restart
bool rescale_for_restart = true;
// Whether to update primal weights when an initial solution is provided
bool update_primal_weight_on_initial_solution = false;
// Whether to update step size when an initial solution is provided
bool update_step_size_on_initial_solution = false;
// Whether to treat some primal gradients as residuals for finite bounds (better with trust region
// restartbut worst with KKT restart)
bool handle_some_primal_gradients_on_finite_bounds_as_residuals = false;
// Whether to project initial primal values using variable bounds
bool project_initial_primal = true;
// Whether to use adaptive step size strategy
bool use_adaptive_step_size_strategy = true;
// All hyperparameters needed to have the same heuristics cuPDLP+
bool initial_step_size_max_singular_value  = false;
bool initial_primal_weight_combined_bounds = true;
bool bound_objective_rescaling             = false;
bool use_reflected_primal_dual             = false;
bool use_fixed_point_error                 = false;
double reflection_coefficient              = 1.0;
double restart_k_p                         = 0.99;
double restart_k_i                         = 0.01;
double restart_k_d                         = 0.0;
double restart_i_smooth                    = 0.3;
bool use_conditional_major                 = false;

}  // namespace cuopt::linear_programming::pdlp_hyper_params
