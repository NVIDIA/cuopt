/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <utilities/logger.hpp>

#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

/*
 * These run in the test executable, which links cuopt but has its own hidden logger, so they
 * exercise the same boundary an external caller crosses: init_component_logger_t configures a
 * logger inside the solver library, init_logger_t configures the one in this image.
 */
namespace cuopt::test {

namespace {

std::string temp_log_path(const std::string& tag)
{
  return std::string{std::tmpnam(nullptr)} + "." + tag + ".log";
}

std::string read_file(const std::string& path)
{
  std::ifstream in{path};
  std::ostringstream out;
  out << in.rdbuf();
  return out.str();
}

}  // namespace

// A second configure must not leave the logger reset back to the buffer sink. Releasing the
// previous guard after applying the new config ran ~logger_config_guard on top of it, which
// silently swallowed everything logged afterwards.
TEST(logger, reconfigure_does_not_reset_to_buffer)
{
  const auto first  = temp_log_path("first");
  const auto second = temp_log_path("second");

  {
    cuopt::init_component_logger_t outer{first, false};
    cuopt::mathematical_optimization::configure_logging(second, false, true);

    CUOPT_LOG_ERROR("after_reconfigure");
  }

  // The message must have reached a file rather than vanishing into the buffer sink.
  EXPECT_NE(read_file(first) + read_file(second), "") << "log message was swallowed";

  std::remove(first.c_str());
  std::remove(second.c_str());
}

// Overlapping configurations behave like overlapping init_logger_t instances: the inner one
// reuses the outer configuration and its destructor must not tear it down early.
TEST(logger, nested_component_loggers_keep_outer_config)
{
  const auto path = temp_log_path("nested");

  {
    cuopt::init_component_logger_t outer{path, false};
    {
      cuopt::init_component_logger_t inner{path, false};
    }
    // inner is gone; outer is still alive, so the library must still be logging to the file.
    CUOPT_LOG_ERROR("after_inner_destroyed");
  }

  EXPECT_NE(read_file(path), "") << "inner destructor tore down the outer configuration";
  std::remove(path.c_str());
}

// Two loggers on one path: the library's, configured from here, and this image's own. A
// truncating sink writes from offset 0 and would overwrite what the other appended.
TEST(logger, two_images_share_one_file_without_clobbering)
{
  const auto path = temp_log_path("shared");

  {
    cuopt::init_component_logger_t solver_log{path, false};
    cuopt::init_logger_t own_log{path, false, /*truncate=*/false};

    CUOPT_LOG_ERROR("from_this_image");
  }

  const auto contents = read_file(path);
  EXPECT_NE(contents.find("from_this_image"), std::string::npos)
    << "this image's message was overwritten by the library's sink";

  std::remove(path.c_str());
}

// truncate=true on the outermost configure clears the file, so repeated runs do not append
// to each other.
TEST(logger, truncate_clears_previous_contents)
{
  const auto path = temp_log_path("truncate");
  {
    std::ofstream seed{path};
    seed << "STALE_CONTENT_FROM_PREVIOUS_RUN\n";
  }

  {
    cuopt::init_component_logger_t solver_log{path, false};
    CUOPT_LOG_ERROR("fresh");
  }

  EXPECT_EQ(read_file(path).find("STALE_CONTENT_FROM_PREVIOUS_RUN"), std::string::npos)
    << "log file was not truncated";

  std::remove(path.c_str());
}

}  // namespace cuopt::test
