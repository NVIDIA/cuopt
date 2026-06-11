// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights
// reserved. SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

#ifdef MPS_FAST_NVTX
#include <nvtx3/nvToolsExt.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace mps_fast::nvtx {

namespace colors {
constexpr std::uint32_t generic  = 0xff8b949e;
constexpr std::uint32_t io       = 0xff58a6ff;
constexpr std::uint32_t decode   = 0xff3fb950;
constexpr std::uint32_t rows     = 0xffd29922;
constexpr std::uint32_t columns  = 0xffff7b72;
constexpr std::uint32_t rhs      = 0xffa371f7;
constexpr std::uint32_t bounds   = 0xfff0883e;
constexpr std::uint32_t ranges   = 0xff79c0ff;
constexpr std::uint32_t names    = 0xff56d364;
constexpr std::uint32_t alloc    = 0xffdb61a2;
constexpr std::uint32_t finalize = 0xffc9d1d9;
}  // namespace colors

inline std::uint32_t color_for_name(std::string_view name) noexcept
{
  if (name.find("lz4") != std::string_view::npos || name.find("read") != std::string_view::npos) {
    return colors::io;
  }
  if (name.find("decode") != std::string_view::npos ||
      name.find("decompress") != std::string_view::npos) {
    return colors::decode;
  }
  if (name.find("row") != std::string_view::npos) { return colors::rows; }
  if (name.find("column") != std::string_view::npos || name.find("csr") != std::string_view::npos) {
    return colors::columns;
  }
  if (name.find("rhs") != std::string_view::npos) { return colors::rhs; }
  if (name.find("bound") != std::string_view::npos) { return colors::bounds; }
  if (name.find("range") != std::string_view::npos) { return colors::ranges; }
  if (name.find("name") != std::string_view::npos ||
      name.find("materialize") != std::string_view::npos) {
    return colors::names;
  }
  if (name.find("alloc") != std::string_view::npos ||
      name.find("resize") != std::string_view::npos ||
      name.find("mmap") != std::string_view::npos) {
    return colors::alloc;
  }
  if (name.find("finalize") != std::string_view::npos) { return colors::finalize; }
  return colors::generic;
}

class scoped_range_t {
 public:
  explicit scoped_range_t(const char* name,
                          std::uint32_t color    = colors::generic,
                          std::uint32_t category = 0)
  {
    push(name, color, category);
  }

  explicit scoped_range_t(std::string name,
                          std::uint32_t color    = colors::generic,
                          std::uint32_t category = 0)
    : owned_name_(std::move(name))
  {
    push(owned_name_.c_str(), color, category);
  }

  ~scoped_range_t() { end(); }

  void end()
  {
#ifdef MPS_FAST_NVTX
    if (active_) {
      nvtxRangePop();
      active_ = false;
    }
#endif
  }

  scoped_range_t(const scoped_range_t&)            = delete;
  scoped_range_t& operator=(const scoped_range_t&) = delete;

 private:
  void push(const char* name, std::uint32_t color, std::uint32_t category)
  {
#ifdef MPS_FAST_NVTX
    nvtxEventAttributes_t event{};
    event.version       = NVTX_VERSION;
    event.size          = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    event.colorType     = NVTX_COLOR_ARGB;
    event.color         = color;
    event.messageType   = NVTX_MESSAGE_TYPE_ASCII;
    event.message.ascii = name;
    event.category      = category;
    nvtxRangePushEx(&event);
    active_ = true;
#else
    (void)name;
    (void)color;
    (void)category;
#endif
  }

  std::string owned_name_;
#ifdef MPS_FAST_NVTX
  bool active_ = false;
#endif
};

inline void name_current_thread(const char* name)
{
#ifdef MPS_FAST_NVTX
  nvtxNameOsThreadA((std::uint32_t)::syscall(SYS_gettid), name);
#else
  (void)name;
#endif
}

}  // namespace mps_fast::nvtx

#define MPS_FAST_NVTX_CONCAT_INNER(a, b) a##b
#define MPS_FAST_NVTX_CONCAT(a, b)       MPS_FAST_NVTX_CONCAT_INNER(a, b)
#define MPS_NVTX_RANGE(name, color) \
  ::mps_fast::nvtx::scoped_range_t MPS_FAST_NVTX_CONCAT(_mps_nvtx_range_, __LINE__)(name, color)
