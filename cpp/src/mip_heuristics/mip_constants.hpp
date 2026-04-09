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

#define BB_BASE_WORK_SCALE       1.0
#define GPU_HEUR_BASE_WORK_SCALE 0.4
#define CPUFJ_BASE_WORK_SCALE    1.0
