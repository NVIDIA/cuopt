/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/**
 * @file grpc_test_log_capture.hpp
 * @brief Test utility for capturing and verifying logs from gRPC client and server
 *
 * This utility provides a unified way to capture logs from both client and server
 * during integration tests, and provides assertion methods to verify expected log entries.
 *
 * Usage:
 * @code
 * GrpcTestLogCapture log_capture;
 *
 * // Configure client to capture debug logs
 * grpc_client_config_t config;
 * config.debug_log_callback = log_capture.client_callback();
 *
 * // Set server log file path
 * log_capture.set_server_log_path("/tmp/cuopt_test_server_19000.log");
 *
 * // ... run test ...
 *
 * // Verify client logs
 * EXPECT_TRUE(log_capture.client_log_contains("Connected to server"));
 * EXPECT_TRUE(log_capture.client_log_contains_pattern("job_id=.*-.*-.*"));
 *
 * // Verify server logs
 * EXPECT_TRUE(log_capture.server_log_contains("[Worker 0] Processing job"));
 * EXPECT_TRUE(log_capture.server_log_contains_pattern("solve_.*p done"));
 * @endcode
 */

#include <atomic>
#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace cuopt::linear_programming::testing {

/**
 * @brief Log entry with metadata
 */
struct LogEntry {
  std::string message;
  std::chrono::steady_clock::time_point timestamp;
  std::string source;  // "client" or "server"
};

/**
 * @brief Log capture and verification utility for gRPC integration tests
 *
 * This class tracks log positions to ensure tests only see logs from the current test,
 * not from previous tests. Call mark_test_start() at the beginning of each test.
 */
class GrpcTestLogCapture {
 public:
  GrpcTestLogCapture() = default;

  /**
   * @brief Clear all captured client logs and reset server log position
   *
   * Call this at the start of each test to ensure you only see logs from the current test.
   */
  void clear()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    client_logs_.clear();

    if (!server_log_path_.empty()) {
      std::ifstream file(server_log_path_, std::ios::ate);
      if (file.is_open()) { server_log_start_pos_ = file.tellg(); }
    }
    test_start_marked_ = true;
  }

  /**
   * @brief Mark the start of a test - records current server log file position
   *
   * After calling this, get_server_logs() will only return logs written after this point.
   * This ensures tests don't see log entries from previous tests.
   *
   * Call this AFTER setting the server log path and AFTER the server has started.
   */
  void mark_test_start()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    client_logs_.clear();

    // Record current end position of server log file
    if (!server_log_path_.empty()) {
      std::ifstream file(server_log_path_, std::ios::ate);
      if (file.is_open()) {
        server_log_start_pos_ = file.tellg();
      } else {
        server_log_start_pos_ = 0;
      }
    } else {
      server_log_start_pos_ = 0;
    }
    test_start_marked_ = true;
  }

  // =========================================================================
  // Client Log Capture
  // =========================================================================

  /**
   * @brief Get a callback function to capture client debug logs
   *
   * Use this with grpc_client_config_t::debug_log_callback:
   * @code
   * config.debug_log_callback = log_capture.client_callback();
   * @endcode
   */
  std::function<void(const std::string&)> client_callback()
  {
    return [this](const std::string& msg) { add_client_log(msg); };
  }

  /**
   * @brief Manually add a client log entry
   */
  void add_client_log(const std::string& message)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    LogEntry entry;
    entry.message   = message;
    entry.timestamp = std::chrono::steady_clock::now();
    entry.source    = "client";
    client_logs_.push_back(entry);
  }

  /**
   * @brief Get all captured client logs as a single string
   */
  std::string get_client_logs() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ostringstream oss;
    for (const auto& entry : client_logs_) {
      oss << entry.message << "\n";
    }
    return oss.str();
  }

  /**
   * @brief Get client log entries
   */
  std::vector<LogEntry> get_client_log_entries() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return client_logs_;
  }

  /**
   * @brief Check if client logs contain a substring
   */
  bool client_log_contains(const std::string& substring) const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& entry : client_logs_) {
      if (entry.message.find(substring) != std::string::npos) { return true; }
    }
    return false;
  }

  /**
   * @brief Check if client logs contain a pattern (regex)
   */
  bool client_log_contains_pattern(const std::string& pattern) const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    std::regex re(pattern);
    for (const auto& entry : client_logs_) {
      if (std::regex_search(entry.message, re)) { return true; }
    }
    return false;
  }

  /**
   * @brief Count occurrences of a substring in client logs
   */
  int client_log_count(const std::string& substring) const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    int count = 0;
    for (const auto& entry : client_logs_) {
      if (entry.message.find(substring) != std::string::npos) { ++count; }
    }
    return count;
  }

  // =========================================================================
  // Server Log Capture
  // =========================================================================

  /**
   * @brief Set the path to the server log file
   *
   * The server process redirects stdout/stderr to a log file. This method
   * sets the path so that server logs can be read for verification.
   *
   * Note: Call mark_test_start() after this to record the starting position.
   */
  void set_server_log_path(const std::string& path)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    server_log_path_      = path;
    server_log_start_pos_ = 0;
    test_start_marked_    = false;
  }

  /**
   * @brief Read server logs from the configured file path
   *
   * If mark_test_start() was called, this only returns logs written after that point.
   * Otherwise, returns all logs in the file.
   *
   * @param since_test_start If true (default), only return logs since mark_test_start().
   *                         If false, return all logs in the file.
   */
  std::string get_server_logs(bool since_test_start = true) const
  {
    std::string path;
    std::streampos start_pos;
    bool marked;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      path      = server_log_path_;
      start_pos = server_log_start_pos_;
      marked    = test_start_marked_;
    }
    if (path.empty()) { return ""; }

    std::ifstream file(path);
    if (!file.is_open()) { return ""; }

    if (since_test_start && marked && start_pos > 0) { file.seekg(start_pos); }

    std::ostringstream oss;
    oss << file.rdbuf();
    return oss.str();
  }

  /**
   * @brief Read all server logs (ignoring test start marker)
   *
   * Useful for debugging when you need to see the full log history.
   */
  std::string get_all_server_logs() const { return get_server_logs(false); }

  /**
   * @brief Check if server logs contain a substring
   */
  bool server_log_contains(const std::string& substring) const
  {
    std::string logs = get_server_logs();
    return logs.find(substring) != std::string::npos;
  }

  /**
   * @brief Check if server logs contain a pattern (regex)
   */
  bool server_log_contains_pattern(const std::string& pattern) const
  {
    std::string logs = get_server_logs();
    std::regex re(pattern);
    return std::regex_search(logs, re);
  }

  /**
   * @brief Count occurrences of a substring in server logs
   */
  int server_log_count(const std::string& substring) const
  {
    if (substring.empty()) { return 0; }
    std::string logs = get_server_logs();
    int count        = 0;
    size_t pos       = 0;
    while ((pos = logs.find(substring, pos)) != std::string::npos) {
      ++count;
      pos += substring.length();
    }
    return count;
  }

  /**
   * @brief Wait for a specific string to appear in server logs
   *
   * Polls the server log file until the string appears or timeout.
   * Only searches logs written after mark_test_start() was called.
   *
   * @param substring The string to wait for
   * @param timeout_ms Maximum time to wait in milliseconds
   * @param poll_interval_ms How often to check (default 100ms)
   * @return true if the string was found, false if timeout
   */
  bool wait_for_server_log(const std::string& substring,
                           int timeout_ms,
                           int poll_interval_ms = 100) const
  {
    auto start = std::chrono::steady_clock::now();
    while (true) {
      // server_log_contains() respects the test start marker
      if (server_log_contains(substring)) { return true; }

      auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start);
      if (elapsed.count() >= timeout_ms) { return false; }

      std::this_thread::sleep_for(std::chrono::milliseconds(poll_interval_ms));
    }
  }

  // =========================================================================
  // Combined Log Verification
  // =========================================================================

  /**
   * @brief Check if either client or server logs contain a substring
   */
  bool any_log_contains(const std::string& substring) const
  {
    return client_log_contains(substring) || server_log_contains(substring);
  }

  /**
   * @brief Print all captured logs for debugging
   *
   * @param include_all_server_logs If true, print all server logs (not just since test start)
   */
  void dump_logs(std::ostream& os = std::cout, bool include_all_server_logs = false) const
  {
    os << "=== Client Logs ===\n";
    os << get_client_logs();
    os << "\n=== Server Logs";
    if (test_start_marked_ && !include_all_server_logs) {
      os << " (since test start)";
    } else {
      os << " (all)";
    }
    os << " ===\n";
    os << get_server_logs(!include_all_server_logs);
    os << "\n==================\n";
  }

  /**
   * @brief Check if mark_test_start() has been called
   */
  bool is_test_start_marked() const { return test_start_marked_; }

  /**
   * @brief Get the server log file path
   */
  std::string server_log_path() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return server_log_path_;
  }

 private:
  mutable std::mutex mutex_;
  std::vector<LogEntry> client_logs_;
  std::string server_log_path_;
  std::streampos server_log_start_pos_ = 0;  // Position in server log file when test started
  std::atomic<bool> test_start_marked_{false};
};

}  // namespace cuopt::linear_programming::testing
