/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <utilities/logger.hpp>

/*
 * The logger itself is header-only and hidden, so it is private to each component library.
 * This translation unit is compiled into cuopt_routing only, which is what makes the
 * functions below reach routing's instance and no other.
 */
namespace cuopt::routing {

std::shared_ptr<void> configure_logging(const std::string& log_file,
                                        bool log_to_console,
                                        bool truncate)
{
  return cuopt::make_logger_config(log_file, log_to_console, truncate);
}

}  // namespace cuopt::routing
