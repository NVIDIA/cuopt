// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "fast_fp64_parser.hpp"

#include <cstdarg>
#include <cstddef>
#include <utility>

#include <simde/x86/avx2.h>
#include <simde/x86/sse4.2.h>

#ifndef LIKELY
#define LIKELY(x) __builtin_expect(!!(x), 1)
#endif

#ifndef UNLIKELY
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#endif

namespace mps_fast {

enum scan_mode {
  skip_whitespace,
  until_whitespace,
};

// util to serially scan along an in-memory input buffer
// contains optimized primitives for most parsing operations
struct cursor_t {
  const char* start;
  const char* ptr;
  const char* end;

  cursor_t(const char* data, std::size_t size) : start(data), ptr(data), end(data + size) {}

  bool done() const { return ptr >= end; }

  // used in error reporting
  std::pair<std::size_t, std::size_t> linecol_position() const
  {
    std::size_t line       = 1;
    const char* line_start = start;
    for (const char* p = start; p < ptr; ++p) {
      if (*p == '\n') {
        ++line;
        line_start = p + 1;
      }
    }
    std::size_t column = (std::size_t)(ptr - line_start) + 1;
    return {line, column};
  }

  [[noreturn]] void error(const char* msg, ...)
  {
    auto [line, col] = linecol_position();
    va_list args;
    va_start(args, msg);
    char msg_buf[512];
    std::vsnprintf(msg_buf, sizeof(msg_buf), msg, args);
    va_end(args);
    mps_parser_fail(error_type_t::ValidationError, "%zu:%zu: %s", line, col, msg_buf);
  }

  void advance(std::size_t n)
  {
    if (ptr + n > end) { mps_parser_fail(error_type_t::ValidationError, "Unexpected end of file"); }
    ptr += n;
  }

  template <scan_mode mode>
  static const char* scalar_scan(const char* p, const char* end)
  {
    while (p < end) {
      unsigned char c = (unsigned char)*p;
      if constexpr (mode == skip_whitespace) {
        if (c > 32 || c == '\n') return p;
      } else {
        if (c <= 32) return p;
      }
      p++;
    }
    return end;
  }

  // scans for the first non-whitespace (or vice versa)
  template <scan_mode mode>
  static const char* simd_scan(const char* p, const char* end)
  {
    const simde__m256i v32 = simde_mm256_set1_epi8(32);  // space/control characters
    const simde__m256i vnl = simde_mm256_set1_epi8('\n');

    while (p + 32 <= end) {
      simde__m256i data = simde_mm256_loadu_si256((const simde__m256i*)p);
      simde__m256i gt32 = simde_mm256_cmpgt_epi8(data, v32);

      unsigned int mask;
      if constexpr (mode == skip_whitespace) {
        simde__m256i is_nl = simde_mm256_cmpeq_epi8(data, vnl);
        mask = (unsigned int)simde_mm256_movemask_epi8(simde_mm256_or_si256(gt32, is_nl));
      } else {
        mask = ~(unsigned int)simde_mm256_movemask_epi8(gt32);
      }

      if (mask != 0) { return p + __builtin_ctz(mask); }
      p += 32;
    }
    return scalar_scan<mode>(p, end);
  }

  void skip_ws() { ptr = simd_scan<skip_whitespace>(ptr, end); }

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

  // could be SIMD but comments are usually rare
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

  // useful for parsing NAME/OBJNAME which may span multiple "fields" according to the MPS spec
  std::string_view read_rest_of_line_trimmed()
  {
    const char* begin    = ptr;
    const char* line_end = begin;
    while (line_end < end && *line_end != '\n' && *line_end != '\r') {
      ++line_end;
    }

    while (begin < line_end && (*begin == ' ' || *begin == '\t')) {
      ++begin;
    }
    while (line_end > begin && (line_end[-1] == ' ' || line_end[-1] == '\t')) {
      --line_end;
    }
    ptr = line_end;
    return std::string_view(begin, (std::size_t)(line_end - begin));
  }

  inline __attribute__((always_inline)) std::string_view read_field()
  {
    if (UNLIKELY(done())) { return {}; }

    const char* field_start = ptr;
    if (UNLIKELY(end - ptr < 32)) {
      ptr                   = scalar_scan<until_whitespace>(ptr, end);
      const char* field_end = ptr;
      if (ptr < end) { skip_ws(); }
      return std::string_view(field_start, field_end - field_start);
    }

    const simde__m256i v32 = simde_mm256_set1_epi8(32);
    const simde__m256i vnl = simde_mm256_set1_epi8('\n');

    // all input streams provide trailing padding, so this 32B load is valid
    // whenever end - ptr >= 32
    simde__m256i data    = simde_mm256_loadu_si256((const simde__m256i*)ptr);
    simde__m256i gt32    = simde_mm256_cmpgt_epi8(data, v32);
    unsigned int ws_mask = ~(unsigned int)simde_mm256_movemask_epi8(gt32);

    if (UNLIKELY(ws_mask == 0)) {
      ptr                   = simd_scan<until_whitespace>(ptr + 32, end);
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

    if (LIKELY(after_field != 0)) {
      ptr = ptr + __builtin_ctz(after_field);
    } else {
      ptr = field_end;
      if (ptr < end) { skip_ws(); }
    }

    return std::string_view(field_start, field_end - field_start);
  }

  // read but do not consume
  inline __attribute__((always_inline)) std::string_view peek_field()
  {
    if (UNLIKELY(done())) { return {}; }
    const char* field_end = simd_scan<until_whitespace>(ptr, end);
    return std::string_view(ptr, field_end - ptr);
  }

  static inline std::string_view peek_field_at(const char* line_start, const char* section_end)
  {
    cursor_t cursor(line_start, (std::size_t)(section_end - line_start));
    cursor.skip_ws();
    return cursor.peek_field();
  }

  // usually in MPS fields go in pair. these can usually be extracted in a single 32B load
  inline __attribute__((always_inline)) std::pair<std::string_view, std::string_view>
  read_two_fields()
  {
    auto slow = [&] {
      auto f1 = read_field();
      auto f2 = read_field();
      return std::pair<std::string_view, std::string_view>{f1, f2};
    };

    if (UNLIKELY(end - ptr < 32)) { return slow(); }

    const char* field1_start = ptr;
    const simde__m256i v32   = simde_mm256_set1_epi8(32);
    const simde__m256i vnl   = simde_mm256_set1_epi8('\n');

    // Same padded-buffer contract as read_field().
    simde__m256i data = simde_mm256_loadu_si256((const simde__m256i*)ptr);
    simde__m256i gt32 = simde_mm256_cmpgt_epi8(data, v32);

    unsigned int printable_mask = (unsigned int)simde_mm256_movemask_epi8(gt32);
    unsigned int ws_mask        = ~printable_mask;

    if (UNLIKELY(ws_mask == 0)) { return slow(); }
    int field1_end_off = __builtin_ctz(ws_mask);

    unsigned int printable_after_field1 = printable_mask >> field1_end_off;
    if (UNLIKELY(printable_after_field1 == 0)) { return slow(); }
    int field2_start_off = field1_end_off + __builtin_ctz(printable_after_field1);

    if (UNLIKELY(ptr[field2_start_off] == '\n')) { return slow(); }

    unsigned int ws_after_field2_start = ws_mask >> field2_start_off;
    if (UNLIKELY(ws_after_field2_start == 0)) { return slow(); }
    int field2_end_off = field2_start_off + __builtin_ctz(ws_after_field2_start);

    simde__m256i is_nl     = simde_mm256_cmpeq_epi8(data, vnl);
    unsigned int stop_mask = printable_mask | (unsigned int)simde_mm256_movemask_epi8(is_nl);
    unsigned int stop_after_field2 = stop_mask >> field2_end_off;
    if (LIKELY(stop_after_field2 != 0)) {
      ptr = ptr + field2_end_off + __builtin_ctz(stop_after_field2);
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
  if (UNLIKELY(id != field)) {
    cursor.error("expected '%s', got '%.*s'", field, (int)id.size(), id.data());
  }
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
  if (UNLIKELY(!cursor.eol())) {
    auto got = cursor.peek_field();
    cursor.error("expected end of line, got '%.*s'", (int)got.size(), got.data());
  }

  for (;;) {
    while (cursor.eol()) {
      cursor.consume_eol();
    }
    if (UNLIKELY(cursor.done())) { return; }

    if (UNLIKELY(cursor.ptr[0] == '*' || cursor.ptr[0] == '$')) {
      cursor.skip_comment_line();
      continue;
    }

    if (LIKELY(cursor.ptr[0] == ' ') && LIKELY(cursor.ptr + 1 < cursor.end)) { cursor.ptr += 1; }

    if (UNLIKELY(cursor.done())) { return; }
    char c = cursor.ptr[0];
    if (UNLIKELY(!((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')))) {
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
  if (num.empty()) { cursor.error("expected number, got empty field"); }
  const char* p = num.data();
  return fp64::parse_fp64_advance(p, p + num.size());
}

static inline double expect_number_fast_pm_one(cursor_t& cursor)
{
  const char* p = cursor.ptr;
  if (cursor.end - p >= 3 && p[0] == '-' && p[1] == '1' && p[2] <= ' ') {
    cursor.ptr = p + 2;
    cursor.skip_ws();
    return -1.0;
  }
  if (cursor.end - p >= 2 && p[0] == '1' && p[1] <= ' ') {
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
  if (UNLIKELY(!cursor.done() && cursor.ptr[0] == '$')) {
    cursor.skip_to_eol();
    return true;
  }
  return false;
}

}  // namespace mps_fast
