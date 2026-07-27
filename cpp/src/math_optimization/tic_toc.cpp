/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <math_optimization/tic_toc.hpp>

#include <sys/time.h>

namespace cuopt::mathematical_optimization {

double tic()
{
  struct timeval time;
  gettimeofday(&time, 0);
  return time.tv_sec + 1e-6 * time.tv_usec;
}

double toc(double start)
{
  double now = tic();
  return (now - start);
}

}  // namespace cuopt::mathematical_optimization
