/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

namespace cuopt::mathematical_optimization::mip {

struct fj_cpu_hyper_parameters_t {
  double bigval_threshold      = 1e20;
  double start_magnitude_limit = 1e7;
  double integer_domain_limit  = 1e7;

  int32_t nnz_per_refresh_stretch = 100000;
  int32_t max_refresh_stretch     = 8;

  int32_t weight_escalate_after    = 2000;
  int32_t perturb_escalate_cap     = 24;
  int32_t weight_escalate_max      = 100;
  double weight_cap                = 1e5;
  int32_t weight_donor_samples     = 4;
  double weight_donation_floor     = 1.0;
  double obj_weight_incumbent_bump = 4.0;
  double obj_weight_incumbent_cap  = 64.0;

  int32_t restart_window_nnz_scale = 80000;
  int32_t restart_window_scale_max = 4;
  int32_t restart_window_multiple  = 4;

  int32_t two_opt_candidates = 32;

  double batch_min_class_size    = 2.0;
  double batch_max_edges_per_nnz = 32.0;
  int32_t batch_probe_attempts   = 500;
  double batch_min_yield         = 0.05;
  int32_t batch_hist_bins        = 64;

  int32_t bound_prop_rounds      = 10;
  double bound_prop_commit_scale = 1e3;

  int32_t lp_start_nnz_limit    = 8'000'000;
  double lp_pump_max_budget_s   = 2.0;
  double lp_pump_budget_share   = 0.00625;
  int32_t lp_pump_projections   = 1;
  int32_t lp_polish_nnz_limit   = 6'000'000;
  double lp_polish_budget_share = 0.20;
  double lp_polish_min_budget_s = 0.05;

  int32_t start_nnz_limit              = 8'000'000;
  double matching_budget_s             = 0.45;
  int32_t matching_max_row_width       = 20000;
  int32_t aggressive_passes            = 6;
  double aggressive_budget_s           = 0.9;
  double exact_k_tol                   = 1e-6;
  int32_t exact_k_max_width            = 20000;
  double exact_k_budget_s              = 0.5;
  int32_t anchor_repair_violated_share = 5;
  double anchor_repair_budget_s        = 0.1;
  double precedence_budget_s           = 0.05;
  int32_t precedence_passes            = 24;
  int32_t precedence_lower_num         = 9;
  int32_t precedence_lower_den         = 10;
  double covering_budget_s             = 0.4;
};

}  // namespace cuopt::mathematical_optimization::mip
