/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

namespace cuopt::mathematical_optimization {

/**
 * @brief Tuning knobs for the dual-simplex diving heuristics used in MIP B&B.
 *
 * Used directly as simplex_solver_settings_t::diving_settings and copied into
 * branch_and_bound_settings.diving_settings before solve. These are registered
 * in the unified parameter framework via solver_settings_t and can be loaded
 * from a config file with load_parameters_from_file().
 */
template <typename i_t, typename f_t>
struct mip_diving_hyper_params_t {
  // -1 automatic, 0 disabled, 1 enabled
  i_t line_search_diving   = -1;
  i_t pseudocost_diving    = -1;
  i_t guided_diving        = -1;
  i_t coefficient_diving   = -1;
  i_t farkas_diving        = -1;
  i_t vector_length_diving = -1;

  // The minimum depth to start diving from.
  i_t min_node_depth = 10;

  // The maximum number of nodes when performing a dive.
  i_t node_limit = 500;

  // The maximum number of dual simplex iteration allowed
  // in a single dive. This set in terms of the total number of
  // iterations in the best-first threads.
  f_t iteration_limit_factor = 0.05;

  // The maximum backtracking allowed.
  i_t backtrack_limit = 5;

  // For the Farkas diving to be effective, the coefficients in the objective function
  // must have distinct values. The low tolerance here disables Farkas diving for
  // set covering/partitioning, where all coefficients have the same value.
  f_t farkas_obj_dynamism_tol = 1E-4;

  // If a given diving heuristic found a new incumbent, show the corresponding
  // symbol in the first column of the log row. When false, every dive collapses
  // to 'D'. Otherwise,
  //   B = best-first
  //   H = heuristics
  //   C = coefficient diving
  //   L = line-search diving
  //   P = pseudocost diving
  //   G = guided diving
  //   U = unknown
  bool show_type = false;
};

}  // namespace cuopt::mathematical_optimization
