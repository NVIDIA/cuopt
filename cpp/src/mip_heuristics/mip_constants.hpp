/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/linear_programming/constants.h>

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

// Concurrent LP root solve from inside MIP: barrier runs as a third OMP task alongside PDLP
// and dual simplex. Disabled below this thread count so the barrier work doesn't overshoot
// the MIP solver's num_cpu_threads budget (need 1 PDLP + 1 dual simplex + 1 barrier).
// Stand-alone LP always runs all three concurrently regardless of this gate.
#define CUOPT_CONCURRENT_LP_BARRIER_REQUIRED_THREAD_COUNT 3
