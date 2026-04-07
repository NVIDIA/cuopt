/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <mip_heuristics/mip_constants.hpp>

#include <cuopt/linear_programming/pdlp/pdlp_hyper_params.cuh>
#include <cuopt/linear_programming/pdlp/solver_settings.hpp>

namespace cuopt::linear_programming {

// Corresponds to the first good general settings we found
// It's what was used for the GTC results
static void set_Stable1(pdlp_hyper_params::pdlp_hyper_params_t& hyper_params)
{
  hyper_params.initial_step_size_scaling                                  = 1.6;
  hyper_params.default_l_inf_ruiz_iterations                              = 1;
  hyper_params.do_pock_chambolle_scaling                                  = true;
  hyper_params.do_ruiz_scaling                                            = true;
  hyper_params.default_alpha_pock_chambolle_rescaling                     = 1.3;
  hyper_params.default_artificial_restart_threshold                       = 0.5;
  hyper_params.compute_initial_step_size_before_scaling                   = false;
  hyper_params.compute_initial_primal_weight_before_scaling               = true;
  hyper_params.initial_primal_weight_c_scaling                            = 2.2;
  hyper_params.initial_primal_weight_b_scaling                            = 4.6;
  hyper_params.major_iteration                                            = 52;
  hyper_params.min_iteration_restart                                      = 0;
  hyper_params.restart_strategy                                           = 1;
  hyper_params.never_restart_to_average                                   = false;
  hyper_params.reduction_exponent                                         = 0.5;
  hyper_params.growth_exponent                                            = 0.9;
  hyper_params.primal_weight_update_smoothing                             = 0.3;
  hyper_params.sufficient_reduction_for_restart                           = 0.2;
  hyper_params.necessary_reduction_for_restart                            = 0.5;
  hyper_params.primal_importance                                          = 1.8;
  hyper_params.primal_distance_smoothing                                  = 0.6;
  hyper_params.dual_distance_smoothing                                    = 0.2;
  hyper_params.compute_last_restart_before_new_primal_weight              = false;
  hyper_params.artificial_restart_in_main_loop                            = false;
  hyper_params.rescale_for_restart                                        = false;
  hyper_params.update_primal_weight_on_initial_solution                   = false;
  hyper_params.update_step_size_on_initial_solution                       = false;
  hyper_params.handle_some_primal_gradients_on_finite_bounds_as_residuals = true;
  hyper_params.project_initial_primal                                     = false;
  hyper_params.use_adaptive_step_size_strategy                            = true;
  hyper_params.initial_step_size_max_singular_value                       = false;
  hyper_params.initial_primal_weight_combined_bounds                      = true;
  hyper_params.bound_objective_rescaling                                  = false;
  hyper_params.use_reflected_primal_dual                                  = false;
  hyper_params.use_fixed_point_error                                      = false;
  hyper_params.reflection_coefficient = 1.0;
  hyper_params.use_conditional_major  = false;
}

// Even better general setting due to proper primal gradient handling for KKT restart and initial
// projection
static void set_Stable2(pdlp_hyper_params::pdlp_hyper_params_t& hyper_params)
{
  hyper_params.initial_step_size_scaling                                  = 1.0;
  hyper_params.default_l_inf_ruiz_iterations                              = 10;
  hyper_params.do_pock_chambolle_scaling                                  = true;
  hyper_params.do_ruiz_scaling                                            = true;
  hyper_params.default_alpha_pock_chambolle_rescaling                     = 1.0;
  hyper_params.default_artificial_restart_threshold                       = 0.36;
  hyper_params.compute_initial_step_size_before_scaling                   = false;
  hyper_params.compute_initial_primal_weight_before_scaling               = false;
  hyper_params.initial_primal_weight_c_scaling                            = 1.0;
  hyper_params.initial_primal_weight_b_scaling                            = 1.0;
  hyper_params.major_iteration                                            = 40;
  hyper_params.min_iteration_restart                                      = 10;
  hyper_params.restart_strategy                                           = 1;
  hyper_params.never_restart_to_average                                   = false;
  hyper_params.reduction_exponent                                         = 0.3;
  hyper_params.growth_exponent                                            = 0.6;
  hyper_params.primal_weight_update_smoothing                             = 0.5;
  hyper_params.sufficient_reduction_for_restart                           = 0.2;
  hyper_params.necessary_reduction_for_restart                            = 0.8;
  hyper_params.primal_importance                                          = 1.0;
  hyper_params.primal_distance_smoothing                                  = 0.5;
  hyper_params.dual_distance_smoothing                                    = 0.5;
  hyper_params.compute_last_restart_before_new_primal_weight              = true;
  hyper_params.artificial_restart_in_main_loop                            = false;
  hyper_params.rescale_for_restart                                        = true;
  hyper_params.update_primal_weight_on_initial_solution                   = false;
  hyper_params.update_step_size_on_initial_solution                       = false;
  hyper_params.handle_some_primal_gradients_on_finite_bounds_as_residuals = false;
  hyper_params.project_initial_primal                                     = true;
  hyper_params.use_adaptive_step_size_strategy                            = true;
  hyper_params.initial_step_size_max_singular_value                       = false;
  hyper_params.initial_primal_weight_combined_bounds                      = true;
  hyper_params.bound_objective_rescaling                                  = false;
  hyper_params.use_reflected_primal_dual                                  = false;
  hyper_params.use_fixed_point_error                                      = false;
  hyper_params.reflection_coefficient                                     = 1.0;
  hyper_params.use_conditional_major                                      = false;
}

/* 1 - 1 mapping of cuPDLPx(+) function from Haihao and al.
 * For more information please read:
 * @article{lu2025cupdlpx,
 *   title={cuPDLPx: A Further Enhanced GPU-Based First-Order Solver for Linear Programming},
 *   author={Lu, Haihao and Peng, Zedong and Yang, Jinwen},
 *   journal={arXiv preprint arXiv:2507.14051},
 *   year={2025}
 * }
 *
 * @article{lu2024restarted,
 *   title={Restarted Halpern PDHG for linear programming},
 *   author={Lu, Haihao and Yang, Jinwen},
 *   journal={arXiv preprint arXiv:2407.16144},
 *   year={2024}
 * }
 */
static void set_Stable3(pdlp_hyper_params::pdlp_hyper_params_t& hyper_params)
{
  hyper_params.initial_step_size_scaling                = 1.0;
  hyper_params.default_l_inf_ruiz_iterations            = 10;
  hyper_params.do_pock_chambolle_scaling                = true;
  hyper_params.do_ruiz_scaling                          = true;
  hyper_params.default_alpha_pock_chambolle_rescaling   = 1.0;
  hyper_params.default_artificial_restart_threshold     = 0.36;
  hyper_params.compute_initial_step_size_before_scaling = false;
  hyper_params.compute_initial_primal_weight_before_scaling = true;
  hyper_params.initial_primal_weight_c_scaling              = 1.0;
  hyper_params.initial_primal_weight_b_scaling              = 1.0;
  hyper_params.major_iteration                              = 200;
  hyper_params.min_iteration_restart                        = 0;
  hyper_params.restart_strategy                             = 3;
  hyper_params.never_restart_to_average                     = true;
  hyper_params.reduction_exponent                           = 0.3;
  hyper_params.growth_exponent                              = 0.6;
  hyper_params.primal_weight_update_smoothing               = 0.5;
  hyper_params.sufficient_reduction_for_restart             = 0.2;
  hyper_params.necessary_reduction_for_restart              = 0.8;
  hyper_params.primal_importance                            = 1.0;
  hyper_params.primal_distance_smoothing                    = 0.5;
  hyper_params.dual_distance_smoothing                      = 0.5;
  hyper_params.compute_last_restart_before_new_primal_weight              = true;
  hyper_params.artificial_restart_in_main_loop                            = false;
  hyper_params.rescale_for_restart                                        = true;
  hyper_params.update_primal_weight_on_initial_solution                   = false;
  hyper_params.update_step_size_on_initial_solution                       = false;
  hyper_params.handle_some_primal_gradients_on_finite_bounds_as_residuals = false;
  hyper_params.project_initial_primal                                     = true;
  hyper_params.use_adaptive_step_size_strategy                            = false;
  hyper_params.initial_step_size_max_singular_value                       = true;
  hyper_params.initial_primal_weight_combined_bounds                      = false;
  hyper_params.bound_objective_rescaling                                  = true;
  hyper_params.use_reflected_primal_dual                                  = true;
  hyper_params.use_fixed_point_error                                      = true;
  hyper_params.use_conditional_major                                      = true;
}

// Legacy/Original/Initial PDLP settings
static void set_Methodical1(pdlp_hyper_params::pdlp_hyper_params_t& hyper_params)
{
  hyper_params.initial_step_size_scaling                                  = 1.0;
  hyper_params.default_l_inf_ruiz_iterations                              = 5;
  hyper_params.do_pock_chambolle_scaling                                  = true;
  hyper_params.do_ruiz_scaling                                            = true;
  hyper_params.default_alpha_pock_chambolle_rescaling                     = 1.0;
  hyper_params.default_artificial_restart_threshold                       = 0.5;
  hyper_params.compute_initial_step_size_before_scaling                   = false;
  hyper_params.compute_initial_primal_weight_before_scaling               = false;
  hyper_params.initial_primal_weight_c_scaling                            = 1.0;
  hyper_params.initial_primal_weight_b_scaling                            = 1.0;
  hyper_params.major_iteration                                            = 64;
  hyper_params.min_iteration_restart                                      = 0;
  hyper_params.restart_strategy                                           = 2;
  hyper_params.never_restart_to_average                                   = false;
  hyper_params.reduction_exponent                                         = 0.3;
  hyper_params.growth_exponent                                            = 0.6;
  hyper_params.primal_weight_update_smoothing                             = 0.5;
  hyper_params.sufficient_reduction_for_restart                           = 0.1;
  hyper_params.necessary_reduction_for_restart                            = 0.9;
  hyper_params.primal_importance                                          = 1.0;
  hyper_params.primal_distance_smoothing                                  = 0.5;
  hyper_params.dual_distance_smoothing                                    = 0.5;
  hyper_params.compute_last_restart_before_new_primal_weight              = true;
  hyper_params.artificial_restart_in_main_loop                            = false;
  hyper_params.rescale_for_restart                                        = false;
  hyper_params.update_primal_weight_on_initial_solution                   = false;
  hyper_params.update_step_size_on_initial_solution                       = false;
  hyper_params.handle_some_primal_gradients_on_finite_bounds_as_residuals = true;
  hyper_params.project_initial_primal                                     = false;
  hyper_params.use_adaptive_step_size_strategy                            = true;
  hyper_params.initial_step_size_max_singular_value                       = false;
  hyper_params.initial_primal_weight_combined_bounds                      = true;
  hyper_params.bound_objective_rescaling                                  = false;
  hyper_params.use_reflected_primal_dual                                  = false;
  hyper_params.use_fixed_point_error                                      = false;
  hyper_params.reflection_coefficient                                     = 1.0;
  hyper_params.use_conditional_major                                      = false;
}

// Can be extremly faster but usually leads to more divergence
// Used for the blog post results
static void set_Fast1(pdlp_hyper_params::pdlp_hyper_params_t& hyper_params)
{
  hyper_params.initial_step_size_scaling                                  = 0.8;
  hyper_params.default_l_inf_ruiz_iterations                              = 6;
  hyper_params.do_pock_chambolle_scaling                                  = true;
  hyper_params.do_ruiz_scaling                                            = false;
  hyper_params.default_alpha_pock_chambolle_rescaling                     = 2.0;
  hyper_params.default_artificial_restart_threshold                       = 0.3;
  hyper_params.compute_initial_step_size_before_scaling                   = false;
  hyper_params.compute_initial_primal_weight_before_scaling               = true;
  hyper_params.initial_primal_weight_c_scaling                            = 1.2;
  hyper_params.initial_primal_weight_b_scaling                            = 1.2;
  hyper_params.major_iteration                                            = 76;
  hyper_params.min_iteration_restart                                      = 6;
  hyper_params.restart_strategy                                           = 1;
  hyper_params.never_restart_to_average                                   = true;
  hyper_params.reduction_exponent                                         = 0.4;
  hyper_params.growth_exponent                                            = 0.6;
  hyper_params.primal_weight_update_smoothing                             = 0.5;
  hyper_params.sufficient_reduction_for_restart                           = 0.3;
  hyper_params.necessary_reduction_for_restart                            = 0.9;
  hyper_params.primal_importance                                          = 0.8;
  hyper_params.primal_distance_smoothing                                  = 0.8;
  hyper_params.dual_distance_smoothing                                    = 0.3;
  hyper_params.compute_last_restart_before_new_primal_weight              = true;
  hyper_params.artificial_restart_in_main_loop                            = true;
  hyper_params.rescale_for_restart                                        = true;
  hyper_params.update_primal_weight_on_initial_solution                   = false;
  hyper_params.update_step_size_on_initial_solution                       = false;
  hyper_params.handle_some_primal_gradients_on_finite_bounds_as_residuals = true;
  hyper_params.project_initial_primal                                     = false;
  hyper_params.use_adaptive_step_size_strategy                            = true;
  hyper_params.initial_step_size_max_singular_value                       = false;
  hyper_params.initial_primal_weight_combined_bounds                      = true;
  hyper_params.bound_objective_rescaling                                  = false;
  hyper_params.use_reflected_primal_dual                                  = false;
  hyper_params.use_fixed_point_error                                      = false;
  hyper_params.reflection_coefficient                                     = 1.0;
  hyper_params.use_conditional_major                                      = false;
}

template <typename i_t, typename f_t>
void set_pdlp_solver_mode(pdlp_solver_settings_t<i_t, f_t>& settings)
{
  if (settings.pdlp_solver_mode == pdlp_solver_mode_t::Stable2)
    set_Stable2(settings.hyper_params);
  else if (settings.pdlp_solver_mode == pdlp_solver_mode_t::Stable1)
    set_Stable1(settings.hyper_params);
  else if (settings.pdlp_solver_mode == pdlp_solver_mode_t::Methodical1)
    set_Methodical1(settings.hyper_params);
  else if (settings.pdlp_solver_mode == pdlp_solver_mode_t::Fast1)
    set_Fast1(settings.hyper_params);
  else if (settings.pdlp_solver_mode == pdlp_solver_mode_t::Stable3)
    set_Stable3(settings.hyper_params);
}

#define INSTANTIATE(F_TYPE) template void set_pdlp_solver_mode(pdlp_solver_settings_t<int, F_TYPE>& settings);

#if MIP_INSTANTIATE_FLOAT
INSTANTIATE(float)
#endif

#if MIP_INSTANTIATE_DOUBLE
INSTANTIATE(double)
#endif

#undef INSTANTIATE

}  // namespace cuopt::linear_programming
