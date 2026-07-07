/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/mathematical_optimization/constants.h>

#define MIP_INSTANTIATE_FLOAT  CUOPT_INSTANTIATE_FLOAT
#define MIP_INSTANTIATE_DOUBLE CUOPT_INSTANTIATE_DOUBLE

#define PDLP_INSTANTIATE_FLOAT 1

/* @brief Minimimum number of threads to enable each part of the MIP Solver */
#define CUOPT_MIP_FJ_REQUIRED_THREAD_COUNT          8
#define CUOPT_MIP_EARLY_GPUFJ_REQUIRED_THREAD_COUNT 3
#define CUOPT_MIP_EARLY_CPUFJ_REQUIRED_THREAD_COUNT 2
#define CUOPT_MIP_RINS_REQUIRED_THREAD_COUNT        4
#define CUOPT_MIP_BATCH_PDLP_REQUIRED_THREAD_COUNT  3
#define CUOPT_MIP_CLIQUE_CUTS_REQUIRED_THREAD_COUNT 3

// MIP-only gate: skip the concurrent barrier when fewer threads are available than this
// (1 PDLP + 1 dual simplex + 1 barrier). Stand-alone LP always runs all three.
#define CUOPT_CONCURRENT_LP_BARRIER_REQUIRED_THREAD_COUNT 3

/* @brief Priority classes for the omp tasks. Highest value = higher priority.
 * Note that this only gives a hint to the runtime, such that the high priority
 * is not guarantee to be executed before a low priority one (i.e., do not rely on
 * these values for correctness).
 */
#define CUOPT_CRITICAL_TASK_PRIORITY 1000
#define CUOPT_HIGH_TASK_PRIORITY     100
#define CUOPT_MEDIUM_TASK_PRIORITY   10
#define CUOPT_DEFAULT_TASK_PRIORITY  1

// Default values for work stealing in B&B
#define MIP_DEFAULT_STEAL_CHANCE       0.05
#define MIP_DEFAULT_NODES_PER_STEAL    10
#define MIP_DEFAULT_MAX_STEAL_ATTEMPTS 3
