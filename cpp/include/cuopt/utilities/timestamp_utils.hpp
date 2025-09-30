/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include <string>

namespace cuopt {
namespace utilities {

/**
 * @brief Check if extra timestamps should be printed based on environment variable
 *
 * Checks the CUOPT_EXTRA_TIMESTAMPS environment variable once and caches the result.
 * Returns true if the environment variable is set to "True", "true", or "1".
 *
 * @return true if extra timestamps are enabled, false otherwise
 */
bool extraTimestamps();

/**
 * @brief Get current timestamp as seconds since epoch
 *
 * @return Current timestamp as a double representing seconds since epoch
 */
double getCurrentTimestamp();

/**
 * @brief Print a timestamp with label if extra timestamps are enabled
 *
 * @param label The label to print with the timestamp
 */
void printTimestamp(const std::string& label);

}  // namespace utilities
}  // namespace cuopt
