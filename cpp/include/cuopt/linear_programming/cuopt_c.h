/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

/*
 * Compatibility forwarder.
 *
 * The cuOpt C API header moved to <cuopt/mathematical_optimization/cuopt_c.h> as part of
 * the cuopt::linear_programming -> cuopt::mathematical_optimization rename. The C API
 * itself is unchanged (same symbols, C linkage) -- only the include path moved.
 *
 * This shim keeps the legacy include path working. Please update includes to
 * <cuopt/mathematical_optimization/cuopt_c.h>; this forwarder may be removed in a future
 * release.
 */
#ifndef CUOPT_LINEAR_PROGRAMMING_C_API_COMPAT_H
#define CUOPT_LINEAR_PROGRAMMING_C_API_COMPAT_H

#pragma message( \
  "<cuopt/linear_programming/cuopt_c.h> is deprecated; include <cuopt/mathematical_optimization/cuopt_c.h> instead.")

#include <cuopt/mathematical_optimization/cuopt_c.h>

#endif /* CUOPT_LINEAR_PROGRAMMING_C_API_COMPAT_H */
