/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <dual_simplex/tic_toc.hpp>

#include <sys/time.h>

namespace cuopt::mathematical_optimization::simplex {

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

}  // namespace cuopt::mathematical_optimization::simplex
