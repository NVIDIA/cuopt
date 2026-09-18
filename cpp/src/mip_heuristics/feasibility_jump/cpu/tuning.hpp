/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

namespace cuopt::mathematical_optimization::mip {

// Numeric limits.
inline constexpr double fj_bigval_threshold      = 1e20;
inline constexpr double fj_start_magnitude_limit = 1e7;
inline constexpr double fj_integer_domain_limit  = 1e7;

// Incremental-state refresh cadence.
inline constexpr int32_t fj_nnz_per_refresh_stretch = 100000;
inline constexpr int32_t fj_max_refresh_stretch     = 8;

// Row and objective weights.
inline constexpr int32_t fj_weight_escalate_after    = 2000;
inline constexpr int32_t fj_perturb_escalate_cap     = 24;
inline constexpr int32_t fj_weight_escalate_max      = 100;
inline constexpr double fj_weight_cap                = 1e5;
inline constexpr int32_t fj_weight_donor_samples     = 4;
inline constexpr double fj_weight_donation_floor     = 1.0;
inline constexpr double fj_obj_weight_incumbent_bump = 4.0;
inline constexpr double fj_obj_weight_incumbent_cap  = 64.0;

// Restarts and perturbation.
inline constexpr int32_t fj_restart_window_nnz_scale = 80000;
inline constexpr int32_t fj_restart_window_scale_max = 4;
inline constexpr int32_t fj_restart_window_multiple  = 4;

// Pair moves.
inline constexpr int32_t fj_2opt_candidates = 32;

// Move batching.
inline constexpr double fj_batch_min_class_size    = 2.0;
inline constexpr double fj_batch_max_edges_per_nnz = 32.0;
inline constexpr int64_t fj_batch_probe_attempts   = 500;
inline constexpr double fj_batch_min_yield         = 0.05;
inline constexpr int32_t fj_batch_hist_bins        = 64;

// Bound propagation.
inline constexpr int32_t fj_bound_prop_rounds      = 10;
inline constexpr double fj_bound_prop_commit_scale = 1e3;

// LP setup and polishing.
inline constexpr int64_t fj_lp_start_nnz_limit    = 8'000'000;
inline constexpr double fj_lp_pump_max_budget_s   = 2.0;
inline constexpr double fj_lp_pump_budget_share   = 0.00625;
inline constexpr int32_t fj_lp_pump_projections   = 1;
inline constexpr int64_t fj_lp_polish_nnz_limit   = 6'000'000;
inline constexpr double fj_lp_polish_budget_share = 0.20;
inline constexpr double fj_lp_polish_min_budget_s = 0.05;

// Structural starts.
inline constexpr int64_t fj_start_nnz_limit              = 8'000'000;
inline constexpr double fj_matching_budget_s             = 0.45;
inline constexpr int32_t fj_matching_max_row_width       = 20000;
inline constexpr int32_t fj_aggressive_passes            = 6;
inline constexpr double fj_aggressive_budget_s           = 0.9;
inline constexpr double fj_exact_k_tol                   = 1e-6;
inline constexpr int32_t fj_exact_k_max_width            = 20000;
inline constexpr double fj_exact_k_budget_s              = 0.5;
inline constexpr int32_t fj_anchor_repair_violated_share = 5;
inline constexpr double fj_anchor_repair_budget_s        = 0.1;
inline constexpr double fj_precedence_budget_s           = 0.05;
inline constexpr int32_t fj_precedence_passes            = 24;
inline constexpr int32_t fj_precedence_lower_num         = 9;
inline constexpr int32_t fj_precedence_lower_den         = 10;
inline constexpr double fj_covering_budget_s             = 0.4;

}  // namespace cuopt::mathematical_optimization::mip
