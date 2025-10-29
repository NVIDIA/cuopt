/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <utilities/logger.hpp>

namespace cuopt::linear_programming::detail {

// Default to info level if not specified.
#if !defined(CUOPT_LOG_ACTIVE_LEVEL)
#define CUOPT_LOG_ACTIVE_LEVEL RAPIDS_LOGGER_LOG_LEVEL_INFO
#endif

// logger macros matching the rapids logger  enums
#if (CUOPT_LOG_ACTIVE_LEVEL <= RAPIDS_LOGGER_LOG_LEVEL_TRACE)
#define DEVICE_LOG_TRACE(...) printf(__VA_ARGS__)
#else
#define DEVICE_LOG_TRACE(...) void(0)
#endif

#if (CUOPT_LOG_ACTIVE_LEVEL <= RAPIDS_LOGGER_LOG_LEVEL_DEBUG)
#define DEVICE_LOG_DEBUG(...) printf(__VA_ARGS__)
#else
#define DEVICE_LOG_DEBUG(...) void(0)
#endif

#if (CUOPT_LOG_ACTIVE_LEVEL <= RAPIDS_LOGGER_LOG_LEVEL_INFO)
#define DEVICE_LOG_INFO(...) printf(__VA_ARGS__)
#else
#define DEVICE_LOG_INFO(...) void(0)
#endif

#if (CUOPT_LOG_ACTIVE_LEVEL <= RAPIDS_LOGGER_LOG_LEVEL_WARN)
#define DEVICE_LOG_WARN(...) printf(__VA_ARGS__)
#else
#define DEVICE_LOG_WARN(...) void(0)
#endif

#if (CUOPT_LOG_ACTIVE_LEVEL <= RAPIDS_LOGGER_LOG_LEVEL_ERROR)
#define DEVICE_LOG_ERROR(...) printf(__VA_ARGS__)
#else
#define DEVICE_LOG_ERROR(...) void(0)
#endif

#if (CUOPT_LOG_ACTIVE_LEVEL <= RAPIDS_LOGGER_LOG_LEVEL_CRITICAL)
#define DEVICE_LOG_CRITICAL(...) printf(__VA_ARGS__)
#else
#define DEVICE_LOG_CRITICAL(...) void(0)
#endif

}  // namespace cuopt::linear_programming::detail
