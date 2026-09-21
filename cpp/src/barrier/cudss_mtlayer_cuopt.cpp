/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

// cuDSS threading layer implemented against the same GNU libgomp cuOpt itself links against
// (see cpp/CMakeLists.txt, ci/utils/install_modern_libgomp.py), instead of NVIDIA's prebuilt
// libcudss_mtlayer_gomp.so.0, which resolves libgomp.so.1 from the host and can end up a
// different instance than the one cuOpt uses. Adapted from cuDSS's own example
// (cudss_mtlayer_gomp.cu); see https://github.com/NVIDIA/cuopt/issues/1219.

#include <omp.h>

#include "cudss.h"
#include "cudss_threading_interface.h"

extern "C" {

int cudssGetMaxThreads() { return omp_get_max_threads(); }

void cudssParallelFor(int nthr_requested, int ntasks, void* ctx, cudss_thr_func_t f)
{
  if (nthr_requested < 0) return;
  if (nthr_requested == 1) {
    for (int task = 0; task < ntasks; task++) {
      f(task, ctx);
    }
    return;
  }
  if (nthr_requested) {
#pragma omp parallel for num_threads(nthr_requested)
    for (int task = 0; task < ntasks; task++) {
      f(task, ctx);
    }
  } else {
#pragma omp parallel for
    for (int task = 0; task < ntasks; task++) {
      f(task, ctx);
    }
  }
}

// Symbol name is fixed by cuDSS's ABI: cudssSetThreadingLayer() looks it up by this exact
// name.
cudssThreadingInterface_t cudssThreadingInterface = {cudssGetMaxThreads, cudssParallelFor};

}  // extern "C"
