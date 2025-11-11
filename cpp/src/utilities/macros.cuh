/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

// TODO have levels of debug and test assertions
// the impact can be
// 1) light
// 2) medium
// 3) heavy
#ifdef ASSERT_MODE
#include <cassert>
#define cuopt_assert(val, msg) assert(val&& msg)
#define cuopt_func_call(func)  func;
#else
#define cuopt_assert(val, msg)
#define cuopt_func_call(func) ;
#endif

#ifdef BENCHMARK
#define benchmark_call(func) func;
#else
#define benchmark_call(func) ;
#endif

// For CUDA Driver API
#define CU_CHECK(expr_to_check, err_func)                                     \
  do {                                                                        \
    CUresult result = expr_to_check;                                          \
    if (result != CUDA_SUCCESS) {                                             \
      const char* pErrStr;                                                    \
      err_func(result, &pErrStr);                                             \
      fprintf(stderr, "CUDA Error: %s:%i:%s\n", __FILE__, __LINE__, pErrStr); \
    }                                                                         \
  } while (0)
