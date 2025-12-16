/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/utilities/timestamp_utils.hpp>

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

namespace cuopt {
namespace utilities {

bool extraTimestamps()
{
  static bool initialized      = false;
  static bool enableTimestamps = false;

  if (!initialized) {
    const char* envValue = std::getenv("CUOPT_EXTRA_TIMESTAMPS");
    if (envValue != nullptr) {
      std::string value(envValue);
      enableTimestamps = (value == "True" || value == "true" || value == "1");
    }
    initialized = true;
  }

  return enableTimestamps;
}

double getCurrentTimestamp()
{
  auto now = std::chrono::system_clock::now();
  return std::chrono::duration<double>(now.time_since_epoch()).count();
}

void printTimestamp(const std::string& label)
{
  if (extraTimestamps()) {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << std::endl << label << ": " << getCurrentTimestamp() << std::endl;
  }
}

}  // namespace utilities
}  // namespace cuopt
