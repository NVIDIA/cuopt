// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "fast_fp64_parser.hpp"

#include <cctype>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string_view>
#include <utility>

#include <simde/x86/avx2.h>
#include <simde/x86/sse4.2.h>

#ifndef __likely
#define __likely(x) __builtin_expect(!!(x), 1)
#endif

#ifndef __unlikely
#define __unlikely(x) __builtin_expect(!!(x), 0)
#endif

namespace mps_fast {

static inline void reset_number_parse_stats() {}
static inline void print_number_parse_stats() {}

static inline bool is_digit_byte(char c) noexcept { return c >= '0' && c <= '9'; }

static inline double fast_atof_core(const char*& data, const char* end)
{
  return fp64::parse_fp64_advance(data, end);
}

static inline double fast_atof(const char* data, const char* end)
{
  return fast_atof_core(data, end);
}

static inline double fast_atof_advance(const char*& ptr, const char* end)
{
  return fast_atof_core(ptr, end);
}

struct cursor_t {
  const char* start;
  const char* ptr;
  const char* end;

  cursor_t(const char* data, std::size_t size) : start(data), ptr(data), end(data + size) {}

  bool done() const { return ptr >= end; }

  std::pair<std::size_t, std::size_t> position() const
  {
    std::size_t line       = 1;
    const char* line_start = start;
    for (const char* p = start; p < ptr; ++p) {
      if (*p == '\n') {
        ++line;
        line_start = p + 1;
      }
    }
    std::size_t column = static_cast<std::size_t>(ptr - line_start) + 1;
    return {line, column};
  }

  [[noreturn]] void error(const char* msg, ...)
  {
    auto [line, col] = position();
    va_list args;
    va_start(args, msg);
    char msg_buf[512];
    std::vsnprintf(msg_buf, sizeof(msg_buf), msg, args);
    va_end(args);
    mps_parser_fail(error_type_t::ValidationError, "%zu:%zu: %s", line, col, msg_buf);
  }

  void advance(std::size_t n)
  {
    if (ptr + n > end) {
      mps_parser_fail(error_type_t::ValidationError, "cursor advanced past end of file");
    }
    ptr += n;
  }

  template <bool skip_ws_mode>
  static const char* scalar_scan(const char* p, const char* end)
  {
    while (p < end) {
      unsigned char c = static_cast<unsigned char>(*p);
      if constexpr (skip_ws_mode) {
        if (c > 32 || c == '\n') return p;
      } else {
        if (c <= 32) return p;
      }
      p++;
    }
    return end;
  }

  template <bool skip_ws_mode>
  static const char* simd_scan(const char* p, const char* end)
  {
    const simde__m256i v32 = simde_mm256_set1_epi8(32);
    const simde__m256i vnl = simde_mm256_set1_epi8('\n');

    while (p + 32 <= end) {
      simde__m256i data = simde_mm256_loadu_si256((const simde__m256i*)p);
      simde__m256i gt32 = simde_mm256_cmpgt_epi8(data, v32);

      unsigned int mask;
      if (skip_ws_mode) {
        simde__m256i is_nl = simde_mm256_cmpeq_epi8(data, vnl);
        mask = (unsigned int)simde_mm256_movemask_epi8(simde_mm256_or_si256(gt32, is_nl));
      } else {
        mask = ~(unsigned int)simde_mm256_movemask_epi8(gt32);
      }

      if (mask != 0) { return p + __builtin_ctz(mask); }
      p += 32;
    }
    return scalar_scan<skip_ws_mode>(p, end);
  }

  void skip_ws() { ptr = simd_scan<true>(ptr, end); }

  bool eol() const { return ptr < end && (*ptr == '\n' || *ptr == '\r'); }

  void consume_eol()
  {
    if (ptr < end && *ptr == '\r') {
      ptr++;
      if (ptr < end && *ptr == '\n') { ptr++; }
      return;
    }
    if (ptr < end && *ptr == '\n') { ptr++; }
  }

  void skip_comment_line()
  {
    while (!done() && *ptr != '\n' && *ptr != '\r') {
      ptr++;
    }
    consume_eol();
  }

  void skip_to_eol()
  {
    while (!done() && *ptr != '\n' && *ptr != '\r') {
      ptr++;
    }
  }

  inline __attribute__((always_inline)) std::string_view read_field()
  {
    if (__unlikely(done())) { return {}; }

    const char* field_start = ptr;
    if (__unlikely(end - ptr < 32)) {
      ptr                   = scalar_scan<false>(ptr, end);
      const char* field_end = ptr;
      if (ptr < end) { skip_ws(); }
      return std::string_view(field_start, field_end - field_start);
    }

    const simde__m256i v32 = simde_mm256_set1_epi8(32);
    const simde__m256i vnl = simde_mm256_set1_epi8('\n');

    simde__m256i data    = simde_mm256_loadu_si256((const simde__m256i*)ptr);
    simde__m256i gt32    = simde_mm256_cmpgt_epi8(data, v32);
    unsigned int ws_mask = ~(unsigned int)simde_mm256_movemask_epi8(gt32);

    if (__unlikely(ws_mask == 0)) {
      ptr                   = simd_scan<false>(ptr + 32, end);
      const char* field_end = ptr;
      if (ptr < end) { skip_ws(); }
      return std::string_view(field_start, field_end - field_start);
    }

    int field_end_off     = __builtin_ctz(ws_mask);
    const char* field_end = ptr + field_end_off;

    simde__m256i is_nl = simde_mm256_cmpeq_epi8(data, vnl);
    unsigned int stop_mask =
      (unsigned int)simde_mm256_movemask_epi8(simde_mm256_or_si256(gt32, is_nl));
    unsigned int after_field = stop_mask & ~((1u << field_end_off) - 1);

    if (__likely(after_field != 0)) {
      ptr = ptr + __builtin_ctz(after_field);
    } else {
      ptr = field_end;
      if (ptr < end) { skip_ws(); }
    }

    return std::string_view(field_start, field_end - field_start);
  }

  inline __attribute__((always_inline)) std::string_view peek_field()
  {
    if (__unlikely(done())) { return {}; }
    const char* field_end = simd_scan<false>(ptr, end);
    return std::string_view(ptr, field_end - ptr);
  }

  inline __attribute__((always_inline)) std::pair<std::string_view, std::string_view>
  read_two_fields()
  {
    if (__unlikely(end - ptr < 32)) {
      auto f1 = read_field();
      auto f2 = read_field();
      return {f1, f2};
    }

    const char* field1_start = ptr;
    const simde__m256i v32   = simde_mm256_set1_epi8(32);
    const simde__m256i vnl   = simde_mm256_set1_epi8('\n');

    simde__m256i data  = simde_mm256_loadu_si256((const simde__m256i*)ptr);
    simde__m256i gt32  = simde_mm256_cmpgt_epi8(data, v32);
    simde__m256i is_nl = simde_mm256_cmpeq_epi8(data, vnl);

    unsigned int printable_mask = (unsigned int)simde_mm256_movemask_epi8(gt32);
    unsigned int ws_mask        = ~printable_mask;
    unsigned int nl_mask        = (unsigned int)simde_mm256_movemask_epi8(is_nl);
    unsigned int stop_mask      = printable_mask | nl_mask;

    if (__unlikely(ws_mask == 0)) {
      auto f1 = read_field();
      auto f2 = read_field();
      return {f1, f2};
    }
    int field1_end_off = __builtin_ctz(ws_mask);

    unsigned int after_field1 = stop_mask & ~((1u << field1_end_off) - 1);
    if (__unlikely(after_field1 == 0)) {
      auto f1 = read_field();
      auto f2 = read_field();
      return {f1, f2};
    }
    int field2_start_off = __builtin_ctz(after_field1);

    if (__unlikely(ptr[field2_start_off] == '\n')) {
      auto f1 = read_field();
      auto f2 = read_field();
      return {f1, f2};
    }

    unsigned int ws_after_field2_start = ws_mask & ~((1u << field2_start_off) - 1);
    if (__unlikely(ws_after_field2_start == 0)) {
      auto f1 = read_field();
      auto f2 = read_field();
      return {f1, f2};
    }
    int field2_end_off = __builtin_ctz(ws_after_field2_start);

    unsigned int after_field2 = stop_mask & ~((1u << field2_end_off) - 1);
    if (__likely(after_field2 != 0)) {
      ptr = ptr + __builtin_ctz(after_field2);
    } else {
      ptr = ptr + field2_end_off;
      skip_ws();
    }

    return {std::string_view(field1_start, field1_end_off),
            std::string_view(field1_start + field2_start_off, field2_end_off - field2_start_off)};
  }
};

static inline void expect(cursor_t& cursor, const char* field)
{
  auto id = cursor.read_field();
  if (__unlikely(id != field)) { cursor.error("expected '%s', got '%s'", field, id.data()); }
}

static inline void accept_comment_line(cursor_t& cursor)
{
  for (;;) {
    while (!cursor.done() && cursor.eol()) {
      cursor.consume_eol();
    }
    if (cursor.done() || (cursor.ptr[0] != '*' && cursor.ptr[0] != '$')) { return; }
    cursor.skip_comment_line();
  }
}

static inline void expect_eol(cursor_t& cursor)
{
  if (__unlikely(!cursor.eol())) { cursor.error("expected end of line, got '%s'", cursor.ptr); }

  for (;;) {
    while (cursor.eol()) {
      cursor.consume_eol();
    }
    if (__unlikely(cursor.done())) { return; }

    if (__unlikely(cursor.ptr[0] == '*' || cursor.ptr[0] == '$')) {
      cursor.skip_comment_line();
      continue;
    }

    if (__likely(cursor.ptr[0] == ' ') && __likely(cursor.ptr + 1 < cursor.end)) {
      cursor.ptr += 1;
    }

    if (__unlikely(cursor.done())) { return; }
    if (__unlikely(!std::isalpha(static_cast<unsigned char>(cursor.ptr[0])))) {
      cursor.skip_ws();
      if (cursor.eol()) { continue; }
    }
    break;
  }
}

static inline std::string_view peek(cursor_t& cursor) { return cursor.peek_field(); }

static inline bool accept(cursor_t& cursor, const char* field)
{
  if (peek(cursor) == field) {
    expect(cursor, field);
    return true;
  }
  return false;
}

static inline void expect_section(cursor_t& cursor, const char* section)
{
  expect(cursor, section);
  expect_eol(cursor);
}

static inline double expect_number(cursor_t& cursor)
{
  auto num = cursor.read_field();
  if (num.empty()) { cursor.error("expected number, got '%s'", num.data()); }
  return fast_atof(num.data(), num.data() + num.size());
}

static inline double expect_number_fast_pm_one(cursor_t& cursor)
{
  const char* p = cursor.ptr;
  if (p[0] == '-' && p[1] == '1' && p[2] <= ' ') {
    cursor.ptr = p + 2;
    cursor.skip_ws();
    return -1.0;
  }
  if (p[0] == '1' && p[1] <= ' ') {
    cursor.ptr = p + 1;
    cursor.skip_ws();
    return 1.0;
  }
  return expect_number(cursor);
}

static inline bool accept_section(cursor_t& cursor, const char* section)
{
  if (accept(cursor, section)) {
    expect_eol(cursor);
    return true;
  }
  return false;
}

static inline bool accept_comment(cursor_t& cursor)
{
  if (__unlikely(!cursor.done() && cursor.ptr[0] == '$')) {
    cursor.skip_to_eol();
    return true;
  }
  return false;
}

}  // namespace mps_fast
