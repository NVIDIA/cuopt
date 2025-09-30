/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights
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

#pragma once

namespace cuopt::linear_programming::pdlp_hyper_params {

extern double initial_step_size_scaling;
extern int default_l_inf_ruiz_iterations;
extern bool do_pock_chambolle_scaling;
extern bool do_ruiz_scaling;
extern double default_alpha_pock_chambolle_rescaling;
extern double default_artificial_restart_threshold;
extern bool compute_initial_step_size_before_scaling;
extern bool compute_initial_primal_weight_before_scaling;
extern double initial_primal_weight_c_scaling;
extern double initial_primal_weight_b_scaling;
extern int major_iteration;
extern int min_iteration_restart;
extern int restart_strategy;
extern bool never_restart_to_average;
__constant__ double default_reduction_exponent;
__constant__ double default_growth_exponent;
__constant__ double default_primal_weight_update_smoothing;
__constant__ double default_sufficient_reduction_for_restart;
__constant__ double default_necessary_reduction_for_restart;
__constant__ double primal_importance;
__constant__ double primal_distance_smoothing;
__constant__ double dual_distance_smoothing;
extern double host_default_reduction_exponent;
extern double host_default_growth_exponent;
extern double host_default_primal_weight_update_smoothing;
extern double host_default_sufficient_reduction_for_restart;
extern double host_default_necessary_reduction_for_restart;
extern double host_primal_importance;
extern double host_primal_distance_smoothing;
extern double host_dual_distance_smoothing;
extern bool compute_last_restart_before_new_primal_weight;
extern bool artificial_restart_in_main_loop;
extern bool rescale_for_restart;
extern bool update_primal_weight_on_initial_solution;
extern bool update_step_size_on_initial_solution;
extern bool handle_some_primal_gradients_on_finite_bounds_as_residuals;
extern bool project_initial_primal;
extern bool use_adaptive_step_size_strategy;
extern bool initial_step_size_max_singular_value;
extern bool initial_primal_weight_combined_bounds;
extern bool bound_objective_rescaling;
extern bool use_reflected_primal_dual;
extern bool use_fixed_point_error;
extern double reflection_coefficient;
extern double restart_k_p;
extern double restart_k_i;
extern double restart_k_d;
extern double restart_i_smooth;
extern bool use_conditional_major;

}  // namespace cuopt::linear_programming::pdlp_hyper_params
