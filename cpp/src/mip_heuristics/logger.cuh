/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <utilities/logger.hpp>

namespace cuopt::mathematical_optimization::mip {

// Default to info level if not specified.
#if !defined(CUOPT_LOG_ACTIVE_LEVEL)
#define CUOPT_LOG_ACTIVE_LEVEL RAPIDS_LOGGER_LOG_LEVEL_INFO
#endif

// logger macros matching the rapids logger  enums
#if (CUOPT_LOG_ACTIVE_LEVEL <= RAPIDS_LOGGER_LOG_LEVEL_TRACE)
#define DEVICE_LOG_TRACE(...) printf(__VA_ARGS__)
#else
#define DEVICE_LOG_TRACE(...) CUOPT_LOG_DISABLED(__VA_ARGS__)
#endif

#if (CUOPT_LOG_ACTIVE_LEVEL <= RAPIDS_LOGGER_LOG_LEVEL_DEBUG)
#define DEVICE_LOG_DEBUG(...) printf(__VA_ARGS__)
#else
#define DEVICE_LOG_DEBUG(...) CUOPT_LOG_DISABLED(__VA_ARGS__)
#endif

#if (CUOPT_LOG_ACTIVE_LEVEL <= RAPIDS_LOGGER_LOG_LEVEL_INFO)
#define DEVICE_LOG_INFO(...) printf(__VA_ARGS__)
#else
#define DEVICE_LOG_INFO(...) CUOPT_LOG_DISABLED(__VA_ARGS__)
#endif

#if (CUOPT_LOG_ACTIVE_LEVEL <= RAPIDS_LOGGER_LOG_LEVEL_WARN)
#define DEVICE_LOG_WARN(...) printf(__VA_ARGS__)
#else
#define DEVICE_LOG_WARN(...) CUOPT_LOG_DISABLED(__VA_ARGS__)
#endif

#if (CUOPT_LOG_ACTIVE_LEVEL <= RAPIDS_LOGGER_LOG_LEVEL_ERROR)
#define DEVICE_LOG_ERROR(...) printf(__VA_ARGS__)
#else
#define DEVICE_LOG_ERROR(...) CUOPT_LOG_DISABLED(__VA_ARGS__)
#endif

#if (CUOPT_LOG_ACTIVE_LEVEL <= RAPIDS_LOGGER_LOG_LEVEL_CRITICAL)
#define DEVICE_LOG_CRITICAL(...) printf(__VA_ARGS__)
#else
#define DEVICE_LOG_CRITICAL(...) CUOPT_LOG_DISABLED(__VA_ARGS__)
#endif

}  // namespace cuopt::mathematical_optimization::mip
