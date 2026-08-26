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
 * The logger is hidden, so this test executable has its own instance, separate from the one
 * inside libcuopt. That means CUOPT_LOG_* here reaches *this* image's logger, and only the
 * entry points that operate on this image can be observed from here:
 *
 *   - init_logger_t and configure_logging_impl configure this image's logger, so their
 *     behaviour is testable directly. They are the code the exported per-component
 *     configure_logging entry points run, just reached without crossing a library boundary.
 *   - init_component_logger_t reaches into libcuopt's logger, which only emits during a
 *     solve. Its effect on a shared file is observable here; its messages are not.
 *
 * The separation itself is checked outside this test: libcuopt exports configure_logging and
 * reset_logging and none of the logger state.
 */
namespace cuopt::test {

namespace {

int unique_id()
{
  static int counter = 0;
  return counter++;
}

std::string temp_log_path(const std::string& tag)
{
  return "cuopt_logger_test_" + tag + "_" + std::to_string(unique_id()) + ".log";
}

std::string read_file(const std::string& path)
{
  std::ifstream in{path};
  std::ostringstream out;
  out << in.rdbuf();
  return out.str();
}

// Every test must leave the depth counter at zero, or the next one's configure is treated as
// nested and silently skipped.
struct scoped_config {
  explicit scoped_config(const std::string& path, bool truncate = true)
  {
    cuopt::configure_logging_impl(path, false, truncate);
  }
  ~scoped_config() { cuopt::reset_logging_impl(); }
};

}  // namespace

// Releasing the previous guard after applying the new configuration ran
// ~logger_config_guard -- and so reset_default_logger() -- on top of the sinks just
// installed, sending everything back to the buffer sink.
TEST(logger, reconfigure_does_not_reset_to_buffer)
{
  const auto first  = temp_log_path("first");
  const auto second = temp_log_path("second");

  {
    scoped_config initial{first};
    CUOPT_LOG_ERROR("before_reconfigure");
  }
  {
    scoped_config replacement{second};
    CUOPT_LOG_ERROR("after_reconfigure");
  }

  EXPECT_NE(read_file(first).find("before_reconfigure"), std::string::npos);
  EXPECT_NE(read_file(second).find("after_reconfigure"), std::string::npos)
    << "the second configuration left the logger reset to the buffer sink";

  std::remove(first.c_str());
  std::remove(second.c_str());
}

// Overlapping configurations behave like overlapping init_logger_t instances: the inner one
// reuses the outer configuration, and its exit must not tear that configuration down.
TEST(logger, nested_config_survives_inner_exit)
{
  const auto path = temp_log_path("nested");

  {
    scoped_config outer{path};
    {
      scoped_config inner{path};
    }
    CUOPT_LOG_ERROR("after_inner_exit");
  }

  EXPECT_NE(read_file(path).find("after_inner_exit"), std::string::npos)
    << "inner exit tore down the outer configuration";

  std::remove(path.c_str());
}

// truncate clears the file up front instead of letting the sink open in truncating mode, so
// a second logger appending to the same path is not overwritten from offset 0.
TEST(logger, truncate_clears_previous_contents)
{
  const auto path = temp_log_path("truncate");
  {
    std::ofstream seed{path};
    seed << "STALE_CONTENT_FROM_PREVIOUS_RUN\n";
  }

  {
    scoped_config cfg{path};
    CUOPT_LOG_ERROR("fresh");
  }

  const auto contents = read_file(path);
  EXPECT_EQ(contents.find("STALE_CONTENT_FROM_PREVIOUS_RUN"), std::string::npos)
    << "log file was not truncated";
  EXPECT_NE(contents.find("fresh"), std::string::npos);

  std::remove(path.c_str());
}

// truncate=false leaves what is already there, which is how a second logger on the same path
// avoids clobbering the first.
TEST(logger, append_preserves_existing_contents)
{
  const auto path = temp_log_path("append");
  {
    std::ofstream seed{path};
    seed << "WRITTEN_BY_ANOTHER_LOGGER\n";
  }

  {
    scoped_config cfg{path, /*truncate=*/false};
    CUOPT_LOG_ERROR("appended");
  }

  const auto contents = read_file(path);
  EXPECT_NE(contents.find("WRITTEN_BY_ANOTHER_LOGGER"), std::string::npos)
    << "appending logger overwrote the other logger's output";
  EXPECT_NE(contents.find("appended"), std::string::npos);

  std::remove(path.c_str());
}

// The library's logger is reachable only through the exported entry point. Its messages are
// not observable here, but configuring it must clear the shared file exactly once.
TEST(logger, component_logger_truncates_shared_file)
{
  const auto path = temp_log_path("component");
  {
    std::ofstream seed{path};
    seed << "STALE\n";
  }

  {
    cuopt::init_component_logger_t solver_log{path, false};
    EXPECT_EQ(read_file(path).find("STALE"), std::string::npos)
      << "component configure did not clear the file";
  }

  std::remove(path.c_str());
}

}  // namespace cuopt::test
