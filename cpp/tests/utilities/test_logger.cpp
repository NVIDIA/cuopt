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
#include <memory>
#include <sstream>
#include <string>

/*
 * The logger is hidden, so this test binary has its own instance, separate from libcuopt's.
 * CUOPT_LOG_* here reaches this image's logger, which is what init_logger_t and
 * make_logger_config configure. That the instances really are separate is checked by
 * ci/check_symbols.sh asserting the state is absent from libcuopt's dynamic symbols, not here.
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

// Nesting under a regular file fails to open with ENOTDIR for any user, including CI's root --
// unlike a missing directory, which the sink creates.
std::string unopenable_path_under(const std::string& blocker_file)
{
  std::ofstream blocker{blocker_file};
  blocker << "not a directory";
  return blocker_file + "/child.log";
}

std::string read_file(const std::string& path)
{
  std::ifstream in{path};
  std::ostringstream out;
  out << in.rdbuf();
  return out.str();
}

// Holding the handle keeps the configuration alive; dropping it resets the logger.
struct scoped_config {
  explicit scoped_config(const std::string& path, bool truncate = true)
    : handle(cuopt::make_logger_config(path, false, truncate))
  {
  }
  std::shared_ptr<void> handle;
};

}  // namespace

// A second configuration, after the first has been released, must actually take effect and
// not leave the logger reset to the buffer sink.
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

// A configure that throws must not leave the logger wedged for later callers.
TEST(logger, failed_configure_does_not_wedge_later_ones)
{
  const auto blocker    = temp_log_path("blocker");
  const auto unopenable = unopenable_path_under(blocker);
  EXPECT_ANY_THROW(cuopt::make_logger_config(unopenable, false, true));

  const auto path = temp_log_path("recovered");
  {
    scoped_config cfg{path};
    CUOPT_LOG_ERROR("after_failed_configure");
  }

  EXPECT_NE(read_file(path).find("after_failed_configure"), std::string::npos)
    << "a failed configure left the logger wedged";

  std::remove(path.c_str());
  std::remove(blocker.c_str());
}

// Same failure for the image-local entry point: apply_logger_config clears the sinks before
// installing new ones, so a throw part way through must not leave the logger with none.
TEST(logger, failed_init_logger_restores_a_sink)
{
  const auto blocker    = temp_log_path("blocker");
  const auto unopenable = unopenable_path_under(blocker);
  EXPECT_ANY_THROW(cuopt::init_logger_t(unopenable, false, true));

  const auto path = temp_log_path("init_recovered");
  {
    scoped_config cfg{path};
    CUOPT_LOG_ERROR("after_failed_init");
  }

  EXPECT_NE(read_file(path).find("after_failed_init"), std::string::npos)
    << "a failed init_logger_t left the logger without sinks";

  std::remove(path.c_str());
  std::remove(blocker.c_str());
}

}  // namespace cuopt::test
