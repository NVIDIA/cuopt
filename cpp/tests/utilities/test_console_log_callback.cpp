/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <utilities/logger.hpp>

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace {

std::vector<std::string>& captured_lines()
{
  static std::vector<std::string> lines;
  return lines;
}

void capturing_callback(int /* level */, const char* message)
{
  captured_lines().push_back(message);
}

}  // namespace

// Covers the console-sink override added for language bindings (Java in particular) whose host
// runtime cannot safely receive a raw write to the native stdout stream -- see
// cuopt::set_console_log_callback in logger.hpp.
class console_log_callback_test : public ::testing::Test {
 protected:
  void TearDown() override
  {
    // Every test must leave the override cleared, or a later test (or a later suite entirely,
    // since the callback is process-global) would silently pick up a stale callback.
    cuopt::set_console_log_callback(nullptr);
    captured_lines().clear();
  }
};

TEST_F(console_log_callback_test, registered_callback_receives_console_output)
{
  cuopt::set_console_log_callback(&capturing_callback);
  {
    cuopt::init_logger_t guard("", /* log_to_console = */ true);
    CUOPT_LOG_INFO("hello from console_log_callback_test");
  }

  ASSERT_FALSE(captured_lines().empty());
  EXPECT_NE(captured_lines().back().find("hello from console_log_callback_test"),
            std::string::npos);
}

TEST_F(console_log_callback_test, nullptr_callback_falls_back_to_stdout_without_crashing)
{
  cuopt::set_console_log_callback(nullptr);

  EXPECT_NO_THROW({
    cuopt::init_logger_t guard("", /* log_to_console = */ true);
    CUOPT_LOG_INFO("this goes to std::cout, not a callback");
  });
  // No callback was registered, so nothing should have been captured through it.
  EXPECT_TRUE(captured_lines().empty());
}

TEST_F(console_log_callback_test, log_to_console_false_suppresses_both_sinks)
{
  cuopt::set_console_log_callback(&capturing_callback);
  {
    cuopt::init_logger_t guard("", /* log_to_console = */ false);
    CUOPT_LOG_INFO("should not reach either sink");
  }

  EXPECT_TRUE(captured_lines().empty());
}
