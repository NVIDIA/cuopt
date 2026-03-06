/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <dual_simplex/types.hpp>

#include <atomic>
#include <cstdint>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
class problem_t;

/**
 * Run PDLP/Barrier for root LP (used by branch-and-bound when concurrent root solve is enabled).
 * Implemented in root_lp.cu so GPU code (convert_greater_to_less, solve_lp_with_method) can run.
 */
template <typename i_t, typename f_t>
cuopt::linear_programming::dual_simplex::root_relaxation_first_solution_t<i_t, f_t>
run_pdlp_barrier_for_root_lp(problem_t<i_t, f_t>* problem,
                             f_t time_limit,
                             std::atomic<int>* concurrent_halt,
                             i_t num_gpus);

}  // namespace cuopt::linear_programming::detail
