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

#define CUOPT_SET_ERROR_MSG_NO_THROW(msg, location_prefix, fmt, ...)                             \
  do {                                                                                           \
    int size1 = std::snprintf(nullptr, 0, "%s", location_prefix);                                \
    int size2 = std::snprintf(nullptr, 0, "file=%s line=%d: ", __FILE__, __LINE__);              \
    int size3 = std::snprintf(nullptr, 0, fmt, ##__VA_ARGS__);                                   \
    if (size1 < 0 || size2 < 0 || size3 < 0) {                                                   \
      std::cerr << "Error in snprintf, cannot handle CUOPT exception." << std::endl;             \
      return;                                                                                    \
    }                                                                                            \
    auto size = size1 + size2 + size3 + 1; /* +1 for final '\0' */                               \
    std::vector<char> buf(size);                                                                 \
    std::snprintf(buf.data(), size1 + 1 /* +1 for '\0' */, "%s", location_prefix);               \
    std::snprintf(                                                                               \
      buf.data() + size1, size2 + 1 /* +1 for '\0' */, "file=%s line=%d: ", __FILE__, __LINE__); \
    std::snprintf(buf.data() + size1 + size2, size3 + 1 /* +1 for '\0' */, fmt, ##__VA_ARGS__);  \
    msg += std::string(buf.data(), buf.data() + size - 1); /* -1 to remove final '\0' */         \
  } while (0)

#define CUOPT_CUSPARSE_TRY_NO_THROW(call)                                                   \
  do {                                                                                      \
    cusparseStatus_t const status = (call);                                                 \
    if (CUSPARSE_STATUS_SUCCESS != status) {                                                \
      std::string msg{};                                                                    \
      CUOPT_SET_ERROR_MSG_NO_THROW(msg,                                                     \
                                   "cuSparse error encountered at: ",                       \
                                   "call='%s', Reason=%d:%s",                               \
                                   #call,                                                   \
                                   status,                                                  \
                                   raft::sparse::detail::cusparse_error_to_string(status)); \
    }                                                                                       \
  } while (0)
