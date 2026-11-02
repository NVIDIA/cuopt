/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <utilities/logger.hpp>

namespace cuopt::mathematical_optimization {

std::shared_ptr<void> configure_logging(const std::string& log_file,
                                        bool log_to_console,
                                        bool truncate)
{
  return cuopt::make_logger_config(log_file, log_to_console, truncate);
}

void set_console_log_callback(cuopt::log_console_callback_t callback)
{
  cuopt::set_console_log_callback(callback);
}

}  // namespace cuopt::mathematical_optimization
