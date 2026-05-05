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
