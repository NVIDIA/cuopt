/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifdef CUOPT_ENABLE_GRPC

#include "grpc_server_logger.hpp"
#include "grpc_server_types.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>

#include <poll.h>
#include <unistd.h>

namespace {

bool shutdown_requested()
{
  return shm_ctrl != nullptr && shm_ctrl->shutdown_requested.load(std::memory_order_acquire);
}

}  // namespace

bool write_to_pipe(int fd, const void* data, size_t size)
{
  const uint8_t* ptr = static_cast<const uint8_t*>(data);
  size_t remaining   = size;

  // Interruptible write: poll with a short timeout and abort when shutdown is
  // requested. Callers must use O_NONBLOCK write ends so write() returns
  // EAGAIN instead of blocking after poll reports POLLOUT for a partial window.
  while (remaining > 0) {
    if (shutdown_requested()) { return false; }

    struct pollfd pfd = {fd, POLLOUT, 0};
    int pr;
    do {
      pr = poll(&pfd, 1, 100);
    } while (pr < 0 && errno == EINTR);
    if (pr < 0) {
      SERVER_LOG_ERROR("[Server] poll() failed on pipe write: %s", strerror(errno));
      return false;
    }
    if (pr == 0) { continue; }
    if (pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) {
      SERVER_LOG_ERROR("[Server] Pipe write error/hangup detected");
      return false;
    }

    ssize_t written = ::write(fd, ptr, remaining);
    if (written > 0) {
      ptr += written;
      remaining -= written;
      continue;
    }
    if (written < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) { continue; }
    SERVER_LOG_ERROR("[Server] Pipe write error: %s", strerror(errno));
    return false;
  }
  return true;
}

bool read_from_pipe(int fd, void* data, size_t size, int timeout_ms)
{
  uint8_t* ptr     = static_cast<uint8_t*>(data);
  size_t remaining = size;

  // timeout_ms only bounds waiting for the *first* readable byte.
  // Once data starts flowing, the transfer is open-ended aside
  // from shutdown checks and EOF/errors. Poll in short slices so shutdown can
  // abort while waiting.
  auto first_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  bool saw_data       = false;

  while (remaining > 0) {
    if (shutdown_requested()) { return false; }

    int wait_ms = 100;
    if (!saw_data) {
      auto now = std::chrono::steady_clock::now();
      if (now >= first_deadline) {
        SERVER_LOG_ERROR("[Server] Timeout waiting for pipe data (waited %dms)", timeout_ms);
        return false;
      }
      auto remaining_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(first_deadline - now).count();
      wait_ms = static_cast<int>(std::min<long>(remaining_ms, 100));
    }

    struct pollfd pfd = {fd, POLLIN, 0};
    int pr;
    do {
      pr = poll(&pfd, 1, wait_ms);
    } while (pr < 0 && errno == EINTR);
    if (pr < 0) {
      SERVER_LOG_ERROR("[Server] poll() failed on pipe: %s", strerror(errno));
      return false;
    }
    if (pr == 0) { continue; }
    if (pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) {
      // POLLHUP with POLLIN can still have readable bytes; only fail if no POLLIN.
      if (!(pfd.revents & POLLIN)) {
        SERVER_LOG_ERROR("[Server] Pipe error/hangup detected");
        return false;
      }
    }

    ssize_t nread = ::read(fd, ptr, remaining);
    if (nread > 0) {
      saw_data = true;
      ptr += nread;
      remaining -= nread;
      continue;
    }
    if (nread == 0) {
      SERVER_LOG_ERROR("[Server] Pipe EOF (writer closed)");
      return false;
    }
    if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) { continue; }
    SERVER_LOG_ERROR("[Server] Pipe read error: %s", strerror(errno));
    return false;
  }
  return true;
}

#endif  // CUOPT_ENABLE_GRPC
