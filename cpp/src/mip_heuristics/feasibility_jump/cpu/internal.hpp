/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <mip_heuristics/mip_constants.hpp>

#include <dual_simplex/presolve.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>
#include <dual_simplex/solve.hpp>
#include <dual_simplex/user_problem.hpp>
#include <math_optimization/tic_toc.hpp>

#include <mip_heuristics/feasibility_jump/cpu/state.hpp>
#include <mip_heuristics/feasibility_jump/cpu/tuning.hpp>
#include <mip_heuristics/feasibility_jump/fj_cpu_worker.cuh>
#include <mip_heuristics/presolve/probing_cache.hpp>
#include <mip_heuristics/utils.hpp>

#include <utilities/pcgenerator.hpp>
#include <utilities/seed_generator.cuh>

#include <raft/core/nvtx.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/tuple.h>

#include <cmath>
#include <cstdio>

#ifdef CPUFJ_NVTX_RANGES
#define CPUFJ_NVTX_RANGE(name)        raft::common::nvtx::range CPUFJ_NVTX_UNIQUE_NAME(nvtx_scope_)(name)
#define CPUFJ_NVTX_UNIQUE_NAME(base)  CPUFJ_NVTX_CONCAT(base, __LINE__)
#define CPUFJ_NVTX_CONCAT(a, b)       CPUFJ_NVTX_CONCAT_INNER(a, b)
#define CPUFJ_NVTX_CONCAT_INNER(a, b) a##b
#else
#define CPUFJ_NVTX_RANGE(name) ((void)0)
#endif

namespace cuopt::mathematical_optimization::mip {

using simplex::lp_problem_t;
using simplex::simplex_solver_settings_t;
using simplex::variable_type_t;

}  // namespace cuopt::mathematical_optimization::mip
