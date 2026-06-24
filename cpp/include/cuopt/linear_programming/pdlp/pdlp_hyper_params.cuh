/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

namespace cuopt::linear_programming::pdlp_hyper_params {

struct pdlp_hyper_params_t {
  double initial_step_size_scaling                  = 1.0;
  int default_l_inf_ruiz_iterations                 = 10;
  bool do_pock_chambolle_scaling                    = true;
  bool do_ruiz_scaling                              = true;
  double default_alpha_pock_chambolle_rescaling     = 1.0;
  double default_artificial_restart_threshold       = 0.36;
  bool compute_initial_step_size_before_scaling     = false;
  bool compute_initial_primal_weight_before_scaling = true;
  double initial_primal_weight_c_scaling            = 1.0;
  double initial_primal_weight_b_scaling            = 1.0;
  int major_iteration                               = 200;
  int min_iteration_restart                         = 0;
  int restart_strategy                              = 3;
  bool never_restart_to_average                     = true;

  double reduction_exponent               = 0.3;
  double growth_exponent                  = 0.6;
  double primal_weight_update_smoothing   = 0.5;
  double sufficient_reduction_for_restart = 0.2;
  double necessary_reduction_for_restart  = 0.8;
  double primal_importance                = 1.0;
  double primal_distance_smoothing        = 0.5;
  double dual_distance_smoothing          = 0.5;

  bool compute_last_restart_before_new_primal_weight              = true;
  bool artificial_restart_in_main_loop                            = false;
  bool rescale_for_restart                                        = true;
  bool update_primal_weight_on_initial_solution                   = false;
  bool update_step_size_on_initial_solution                       = false;
  bool handle_some_primal_gradients_on_finite_bounds_as_residuals = false;
  bool project_initial_primal                                     = true;
  bool use_adaptive_step_size_strategy                            = false;
  bool initial_step_size_max_singular_value                       = true;
  bool initial_primal_weight_combined_bounds                      = false;
  bool bound_objective_rescaling                                  = true;
  bool use_reflected_primal_dual                                  = true;
  bool use_fixed_point_error                                      = true;
  double reflection_coefficient                                   = 1.0;
  double restart_k_p                                              = 0.99;
  double restart_k_i                                              = 0.01;
  double restart_k_d                                              = 0.0;
  double restart_i_smooth                                         = 0.3;
  bool use_conditional_major                                      = true;
};

// TODO most likely we want to get rid of pdlp_solver_mode and just have prebuilt
// constpexr version of each (Stable2, Stable1, Methodical1, Fast1, Stable3...)

}  // namespace cuopt::linear_programming::pdlp_hyper_params
