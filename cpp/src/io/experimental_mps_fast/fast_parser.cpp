// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
// reserved. SPDX-License-Identifier: Apache-2.0

#include "fast_parser.hpp"
#include "fast_parse_primitives.hpp"
#include "file_reader.hpp"
#include "hash_table_smallstr.hpp"
#include "mmap_region.hpp"
#include "mps_section_scanner.hpp"
#include "nvtx_ranges.hpp"
#ifdef MPS_FAST_PERF_COUNTERS
#include "perf_counters.hpp"
#endif

#include <sys/mman.h>
#include <unistd.h>

#include <omp.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#ifndef MADV_COLLAPSE
#define MADV_COLLAPSE 25
#endif

namespace mps_fast {

static constexpr size_t COLUMN_ROW_COUNT_BLOCK_ROWS       = 4096;
static constexpr int MPS_ROWS_THREAD_CAP                  = 16;
static constexpr int MPS_COLUMNS_THREAD_CAP               = 32;
static constexpr int MPS_BOUNDS_THREAD_CAP                = 32;
static constexpr int MPS_NAMES_THREAD_CAP                 = 16;
static constexpr size_t MPS_BOUNDS_PARALLEL_INIT_MIN_VARS = 16 * 1024 * 1024;
static constexpr size_t MPS_BOUNDS_PARALLEL_MIN_BYTES     = 256ull * 1024ull * 1024ull;
static constexpr size_t MPS_COLUMNS_MIN_CHUNK_BYTES       = 1 * 1024 * 1024;

static int phase_thread_count(int phase_cap)
{
  return std::max(1, std::min(phase_cap, omp_get_max_threads()));
}

// =============================================================================
// RAII Timer for profiling with deferred output
// =============================================================================

struct TimerEntry {
  const char* name;
  double elapsed_ms;
};

static std::vector<TimerEntry>& get_timer_buffer()
{
  static std::vector<TimerEntry> buffer;
  buffer.reserve(100);
  return buffer;
}

static std::mutex& get_timer_mutex()
{
  static std::mutex mutex;
  return mutex;
}

static void flush_timers()
{
  std::lock_guard<std::mutex> lock(get_timer_mutex());
  auto& buffer = get_timer_buffer();
  for (const auto& entry : buffer) {
    std::fprintf(stderr, "[TIMER] %s: %.3f ms\n", entry.name, entry.elapsed_ms);
  }
  buffer.clear();
}

static size_t system_page_size()
{
  static size_t page_size = [] {
    long value = sysconf(_SC_PAGESIZE);
    return value > 0 ? (size_t)value : (size_t)4096;
  }();
  return page_size;
}

enum class materialize_touch_t {
  write_2mb,
  write_4kb,
};

// instanciate a range using mmap anon pages with hugepage hints, and materialize them
// by touching each to nudge the kernel into invoking its THP mechanism
static void materialize_hugepages(const char* label,
                                  void* data,
                                  size_t bytes,
                                  materialize_touch_t touch)
{
  (void)label;
  if (data == nullptr || bytes == 0) return;

  constexpr size_t two_mb = 2 * 1024 * 1024;
  size_t page_size        = system_page_size();
  uintptr_t start         = reinterpret_cast<uintptr_t>(data);
  uintptr_t end           = start + bytes;
  uintptr_t aligned_start = start & ~(uintptr_t)(page_size - 1);
  uintptr_t aligned_end   = (end + page_size - 1) & ~(uintptr_t)(page_size - 1);
  size_t aligned_bytes    = (size_t)(aligned_end - aligned_start);

  errno = 0;
  madvise(reinterpret_cast<void*>(aligned_start), aligned_bytes, MADV_HUGEPAGE);

  size_t step        = touch == materialize_touch_t::write_2mb ? two_mb : page_size;
  volatile char* ptr = reinterpret_cast<volatile char*>(data);
  for (size_t offset = 0; offset < bytes; offset += step) {
    ptr[offset] = ptr[offset];
  }
  ptr[bytes - 1] = ptr[bytes - 1];
}

template <typename T>
static void materialize_vector_hugepages(const char* label,
                                         std::vector<T>& values,
                                         materialize_touch_t touch)
{
  materialize_hugepages(label, values.data(), values.size() * sizeof(T), touch);
}

class scoped_timer_t {
 public:
  scoped_timer_t(const char* name, double* accumulator = nullptr)
    : name_(name),
      accumulator_(accumulator),
      nvtx_(name, nvtx::color_for_name(name)),
      start_(std::chrono::high_resolution_clock::now())
  {
  }

  ~scoped_timer_t()
  {
    auto end          = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start_).count();
    nvtx_.end();
    if (accumulator_) { *accumulator_ += elapsed_ms; }
    std::lock_guard<std::mutex> lock(get_timer_mutex());
    get_timer_buffer().push_back({name_, elapsed_ms});
  }

  scoped_timer_t(const scoped_timer_t&)            = delete;
  scoped_timer_t& operator=(const scoped_timer_t&) = delete;

 private:
  const char* name_;
  double* accumulator_;
  nvtx::scoped_range nvtx_;
  std::chrono::high_resolution_clock::time_point start_;
};

static inline bool section_token_matches(const char* p,
                                         const char* end,
                                         const char* token,
                                         size_t len)
{
  return (size_t)(end - p) >= len && std::memcmp(p, token, len) == 0 &&
         ((size_t)(end - p) == len || p[len] <= ' ');
}

static inline bool is_quadratic_section_start(const char* p, const char* end)
{
  return section_token_matches(p, end, "QUADOBJ", 7) ||
         section_token_matches(p, end, "QMATRIX", 7) ||
         section_token_matches(p, end, "QCMATRIX", 8);
}

static inline bool is_rhs_section_end(const char* p, const char* end)
{
  switch (p[0]) {
    case 'B': return std::memcmp(p, "BOUNDS", 6) == 0 && p[6] <= ' ';
    case 'Q': return is_quadratic_section_start(p, end);
    case 'R': return std::memcmp(p, "RANGES", 6) == 0 && p[6] <= ' ';
    case 'E': return std::memcmp(p, "ENDATA", 6) == 0 && p[6] <= ' ';
    default: return false;
  }
}

static inline void error_unknown_row(cursor_t& cursor, const char* row_start, const char* section)
{
  const char* row_end = row_start;
  while (row_end < cursor.end && *row_end > ' ') {
    row_end++;
  }
  cursor.error("unknown row name in %s: %.*s", section, (int)(row_end - row_start), row_start);
}

// =============================================================================
// Parsing state shared across section parsers
// =============================================================================

// Hash and equality for string_view keys in unordered_map
struct string_view_hash {
  size_t operator()(std::string_view sv) const { return std::hash<std::string_view>{}(sv); }
};

static inline size_t next_power_of_2(size_t n)
{
  if (n == 0) return 1;
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  n |= n >> 32;
  return n + 1;
}

enum class row_index_mode_t {
  hash,
  dense_ordered,
};

static inline bool is_decimal_digit(char c) { return (unsigned)(c - '0') <= 9; }

static inline size_t decimal_digits_u64(uint64_t value)
{
  size_t digits = 1;
  while (value >= 10) {
    value /= 10;
    digits++;
  }
  return digits;
}

static inline bool parse_trailing_u64(std::string_view name,
                                      std::string_view& prefix,
                                      uint64_t& value,
                                      size_t& suffix_width)
{
  size_t pos = name.size();
  while (pos > 0 && is_decimal_digit(name[pos - 1])) {
    pos--;
  }
  if (pos == name.size()) { return false; }

  uint64_t parsed = 0;
  for (size_t i = pos; i < name.size(); ++i) {
    uint64_t digit = (uint64_t)(name[i] - '0');
    if (parsed > (std::numeric_limits<uint64_t>::max() - digit) / 10) { return false; }
    parsed = parsed * 10 + digit;
  }

  prefix       = std::string_view(name.data(), pos);
  value        = parsed;
  suffix_width = name.size() - pos;
  return true;
}

static inline bool dense_suffix_is_zero_padded(std::string_view name, size_t suffix_width)
{
  return suffix_width > 1 && name[name.size() - suffix_width] == '0';
}

static inline bool dense_suffix_width_ok(uint64_t value,
                                         size_t suffix_width,
                                         bool zero_padded,
                                         size_t pad_width)
{
  size_t digits         = decimal_digits_u64(value);
  size_t expected_width = zero_padded ? std::max(pad_width, digits) : digits;
  return suffix_width == expected_width;
}

template <typename i_t, typename f_t>
struct parse_state_t {
  cuopt::linear_programming::io::mps_data_model_t<i_t, f_t>& problem;
  cursor_t& cursor;

  // Temporary string_view storage (points into input buffer, no allocation)
  std::vector<std::string_view> row_names_sv;
  std::vector<std::string_view> var_names_sv;
  std::string_view problem_name_sv;
  std::string_view objective_name_sv;
  std::vector<std::string_view> ignored_objective_names_sv;

  // Optional dense ordered column index for labels like V0, V1, ...
  bool col_dense_ordered = false;
  std::string_view col_dense_prefix;
  uint64_t col_dense_min_id  = 0;
  uint64_t col_dense_max_id  = 0;
  size_t col_dense_pad_width = 0;
  bool col_dense_zero_padded = false;

  // Row name hash table - sized at runtime based on row count
  size_t row_hash_buckets = 0;
  size_t row_hash_mask    = 0;  // buckets - 1, for fast modulo via &
  mmap_region_t row_hash_region;
  hash_slot_var_t* row_names_ht = nullptr;
  // Overflow map for row names longer than HASH_KEY_BYTES
  std::unordered_map<std::string_view, size_t, string_view_hash> row_names_long;

  // Optional dense ordered row index for labels like R0001, R0002, ...
  row_index_mode_t row_index_mode = row_index_mode_t::hash;
  bool row_dense_candidate        = true;
  std::string_view row_dense_prefix;
  uint64_t row_dense_min_id  = 0;
  uint64_t row_dense_max_id  = 0;
  uint64_t row_dense_base_id = 0;
  size_t row_dense_pad_width = 0;
  bool row_dense_zero_padded = false;

  // var_names still uses STL (only used in parse_bounds, not as hot)
  std::unordered_map<std::string_view, size_t, string_view_hash> var_names_map;

  parse_state_t(cuopt::linear_programming::io::mps_data_model_t<i_t, f_t>& p, cursor_t& c)
    : problem(p), cursor(c)
  {
  }

  void init_row_hash_table()
  {
    if (init_row_dense_ordered_table()) { return; }
    init_row_hash_table_impl();
  }

  bool row_dense_has_expected_width(uint64_t value, size_t suffix_width) const
  {
    return dense_suffix_width_ok(value, suffix_width, row_dense_zero_padded, row_dense_pad_width);
  }

  bool col_dense_has_expected_width(uint64_t value, size_t suffix_width) const
  {
    return dense_suffix_width_ok(value, suffix_width, col_dense_zero_padded, col_dense_pad_width);
  }

  bool is_ignored_objective_name(std::string_view name) const
  {
    return std::find(ignored_objective_names_sv.begin(), ignored_objective_names_sv.end(), name) !=
           ignored_objective_names_sv.end();
  }

  void add_ignored_objective_name(std::string_view name)
  {
    if (name == objective_name_sv || is_ignored_objective_name(name)) { return; }
    ignored_objective_names_sv.push_back(name);
  }

  void observe_objective_row_name(std::string_view name)
  {
    if (objective_name_sv.empty()) {
      objective_name_sv = name;
    } else {
      add_ignored_objective_name(name);
    }
  }

  void observe_row_name_for_dense_index(std::string_view name, size_t row_index)
  {
    if (!row_dense_candidate) { return; }

    std::string_view prefix;
    uint64_t value      = 0;
    size_t suffix_width = 0;
    if (!parse_trailing_u64(name, prefix, value, suffix_width)) {
      row_dense_candidate = false;
      return;
    }

    if (row_index == 0) {
      row_dense_prefix      = prefix;
      row_dense_min_id      = value;
      row_dense_max_id      = value;
      row_dense_base_id     = value;
      row_dense_pad_width   = suffix_width;
      row_dense_zero_padded = dense_suffix_is_zero_padded(name, suffix_width);
      return;
    }

    if (prefix != row_dense_prefix) {
      row_dense_candidate = false;
      return;
    }

    if (row_dense_base_id > std::numeric_limits<uint64_t>::max() - row_index) {
      row_dense_candidate = false;
      return;
    }

    uint64_t expected = row_dense_base_id + row_index;
    if (value != expected || !row_dense_has_expected_width(value, suffix_width)) {
      row_dense_candidate = false;
      return;
    }

    row_dense_min_id = std::min(row_dense_min_id, value);
    row_dense_max_id = std::max(row_dense_max_id, value);
  }

  bool init_row_dense_ordered_table()
  {
    scoped_timer_t timer("row_dense_finalize");
    size_t n_rows = row_names_sv.size();
    if (!row_dense_candidate || n_rows == 0) { return false; }
    if (row_dense_max_id < row_dense_min_id) { return false; }
    uint64_t dense_count = row_dense_max_id - row_dense_min_id + 1;
    if (dense_count != n_rows) { return false; }

    row_index_mode = row_index_mode_t::dense_ordered;
    return true;
  }

  void init_row_hash_table_impl()
  {
    scoped_timer_t timer("row_hash_init_total");
    size_t n_rows = row_names_sv.size();
    // load factor 50%
    row_hash_buckets          = next_power_of_2(std::max((size_t)(n_rows * 2), (size_t)64));
    row_hash_mask             = row_hash_buckets - 1;
    size_t row_hash_mmap_size = row_hash_buckets * sizeof(hash_slot_var_t);

    {
      scoped_timer_t timer("row_hash_mmap");
      // Use mmap for allocation - the OS provides zero'd pages
      row_hash_region = mmap_region_t::anonymous(
        row_hash_mmap_size, PROT_READ | PROT_WRITE, MAP_PRIVATE, "row hash table");
      row_names_ht = static_cast<hash_slot_var_t*>(row_hash_region.data());
      // Request huge pages to reduce TLB misses
      row_hash_region.advise(MADV_HUGEPAGE);
    }

    // pre-touch the 2MB huge pages to nudge the kernel into allocating them
#ifdef MPS_FAST_THP_PREFAULT
    {
      scoped_timer_t timer("row_hash_thp_prefault");
      materialize_hugepages(
        "row_names_ht", row_names_ht, row_hash_region.size(), materialize_touch_t::write_2mb);
    }
#endif

    {
      scoped_timer_t timer("row_hash_insert_all");
      for (size_t idx = 0; idx < n_rows; ++idx) {
        row_insert(row_names_sv[idx], idx);
      }
    }

    // Force the kernel to please please collapse the page range into THP pages
#ifdef MPS_FAST_MADV_COLLAPSE
    {
      scoped_timer_t timer("row_hash_madv_collapse");
      row_hash_region.advise(MADV_COLLAPSE);
    }
#endif
  }

  size_t row_lookup_dense_ordered(std::string_view name) const
  {
    std::string_view prefix;
    uint64_t value      = 0;
    size_t suffix_width = 0;
    if (!parse_trailing_u64(name, prefix, value, suffix_width)) { return SIZE_MAX; }
    if (prefix != row_dense_prefix || !row_dense_has_expected_width(value, suffix_width)) {
      return SIZE_MAX;
    }
    if (value < row_dense_min_id || value > row_dense_max_id) { return SIZE_MAX; }
    return (size_t)(value - row_dense_min_id);
  }

  size_t row_lookup(std::string_view name) const
  {
    if (__likely(row_index_mode == row_index_mode_t::dense_ordered)) {
      return row_lookup_dense_ordered(name);
    }
    return row_lookup_hash(name);
  }

  size_t read_row_lookup_dense_ordered(cursor_t& cursor) const
  {
    const char* start = cursor.ptr;
    const char* p     = start;

    size_t prefix_len = row_dense_prefix.size();
    if (prefix_len > 0) {
      if ((size_t)(cursor.end - p) < prefix_len ||
          std::memcmp(p, row_dense_prefix.data(), prefix_len) != 0) {
        cursor.read_field();
        return SIZE_MAX;
      }
      p += prefix_len;
    }

    const char* digits_start = p;
    uint64_t value           = 0;
    while (p < cursor.end && is_decimal_digit(*p)) {
      uint64_t digit = (uint64_t)(*p - '0');
      if (value > (std::numeric_limits<uint64_t>::max() - digit) / 10) {
        cursor.ptr = start;
        cursor.read_field();
        return SIZE_MAX;
      }
      value = value * 10 + digit;
      p++;
    }

    size_t suffix_width = (size_t)(p - digits_start);
    if (suffix_width == 0 || p >= cursor.end || *p > ' ' ||
        !row_dense_has_expected_width(value, suffix_width) || value < row_dense_min_id ||
        value > row_dense_max_id) {
      cursor.ptr = start;
      cursor.read_field();
      return SIZE_MAX;
    }

    cursor.ptr = p;
    cursor.skip_ws();
    return (size_t)(value - row_dense_min_id);
  }

  size_t read_row_lookup(cursor_t& cursor) const
  {
    if (__likely(row_index_mode == row_index_mode_t::dense_ordered)) {
      return read_row_lookup_dense_ordered(cursor);
    }

    auto row_name = cursor.read_field();
    return row_lookup_hash(row_name);
  }

  size_t row_lookup_hash(std::string_view name) const
  {
    if (__unlikely(name.size() > HASH_KEY_BYTES)) {
      auto it = row_names_long.find(name);
      return it != row_names_long.end() ? it->second : SIZE_MAX;
    }
    hash_key_t key               = make_key(name.data(), name.size());
    uint32_t hash                = fnv1a_hash(name.data(), name.size()) & (uint32_t)row_hash_mask;
    const hash_slot_var_t* slots = row_names_ht;
    const hash_slot_var_t* slot  = &slots[hash];

    for (size_t i = 0; i < row_hash_buckets; ++i, ++slot) {
      if (slot >= &slots[row_hash_buckets]) { slot = &slots[0]; }
      if (slot->count == 0) { return SIZE_MAX; }
      if (key_cmpeq(slot->key, key)) { return slot->count - 1; }
    }
    return SIZE_MAX;
  }

  size_t col_lookup_dense_ordered(std::string_view name) const
  {
    std::string_view prefix;
    uint64_t value      = 0;
    size_t suffix_width = 0;
    if (!parse_trailing_u64(name, prefix, value, suffix_width)) { return SIZE_MAX; }
    if (prefix != col_dense_prefix || !col_dense_has_expected_width(value, suffix_width)) {
      return SIZE_MAX;
    }
    if (value < col_dense_min_id || value > col_dense_max_id) { return SIZE_MAX; }
    return (size_t)(value - col_dense_min_id);
  }

  void dense_col_name(size_t idx, std::string& out) const
  {
    uint64_t value = col_dense_min_id + idx;
    char digits_buf[32];
    auto [digits_end, ec] = std::to_chars(digits_buf, digits_buf + sizeof(digits_buf), value);
    if (ec != std::errc()) {
      out.assign(col_dense_prefix);
      return;
    }
    size_t digits_len = (size_t)(digits_end - digits_buf);
    size_t width = col_dense_zero_padded ? std::max(col_dense_pad_width, digits_len) : digits_len;
    out.resize(col_dense_prefix.size() + width);
    std::memcpy(out.data(), col_dense_prefix.data(), col_dense_prefix.size());
    char* suffix = out.data() + col_dense_prefix.size();
    if (width > digits_len) {
      std::memset(suffix, '0', width - digits_len);
      suffix += width - digits_len;
    }
    std::memcpy(suffix, digits_buf, digits_len);
  }

  void row_insert(std::string_view name, size_t index)
  {
    if (__unlikely(name.size() > HASH_KEY_BYTES)) {
      row_names_long[name] = index;
      return;
    }
    hash_key_t key         = make_key(name.data(), name.size());
    uint32_t hash          = fnv1a_hash(name.data(), name.size()) & (uint32_t)row_hash_mask;
    hash_slot_var_t* slots = row_names_ht;
    hash_slot_var_t* slot  = &slots[hash];

    for (size_t i = 0; i < row_hash_buckets; ++i, ++slot) {
      if (slot >= &slots[row_hash_buckets]) { slot = &slots[0]; }
      if (slot->count == 0) {
        key_store(slot->key, key);            // Writes 32 bytes, including garbage in last 4
        slot->count = (uint32_t)(index + 1);  // Overwrite last 4 bytes with actual count
        return;
      }
      if (key_cmpeq(slot->key, key)) {
        slot->count = (uint32_t)(index + 1);
        return;
      }
    }
    __builtin_trap();
  }
};

// =============================================================================
// Section parsers
// =============================================================================

template <typename i_t, typename f_t>
static void parse_name_section(parse_state_t<i_t, f_t>& state)
{
  scoped_timer_t timer("parse_name");
  if (peek(state.cursor) == "ROWS") { return; }
  expect(state.cursor, "NAME");
  if (!state.cursor.eol()) {
    state.problem_name_sv = state.cursor.read_field();
    accept_comment(state.cursor);
  }
  expect_eol(state.cursor);
}

template <typename i_t, typename f_t>
static void parse_objsense_section(parse_state_t<i_t, f_t>& state)
{
  scoped_timer_t timer("parse_objsense");
  if (accept(state.cursor, "OBJSENSE")) {
    if (state.cursor.eol()) { expect_eol(state.cursor); }
    if (accept(state.cursor, "MIN")) {
      state.problem.maximize_ = false;
    } else if (accept(state.cursor, "MAX")) {
      state.problem.maximize_ = true;
    } else {
      state.cursor.error("expected MIN or MAX, got '%s'", state.cursor.read_field().data());
    }
    accept_comment(state.cursor);
    expect_eol(state.cursor);
  }
}

template <typename i_t, typename f_t>
static void parse_objname_section(parse_state_t<i_t, f_t>& state)
{
  scoped_timer_t timer("parse_objname");
  if (accept(state.cursor, "OBJNAME")) {
    if (state.cursor.eol()) { expect_eol(state.cursor); }
    state.objective_name_sv = state.cursor.read_field();
    accept_comment(state.cursor);
    expect_eol(state.cursor);
  }
}

struct RowChunkBoundary {
  const char* start;
  const char* end;
};

struct RowChunkInfo {
  size_t constraints = 0;
  bool malformed     = false;
  std::vector<std::string_view> objective_names;
  bool has_first_constraint = false;
  std::string_view first_constraint_name;
};

static const char* rows_find_next_line(const char* p, const char* end)
{
  while (p < end && *p != '\n')
    p++;
  if (p < end) p++;
  return p;
}

static bool parse_rows_line_fast(const char*& p,
                                 const char* end,
                                 char& row_type,
                                 std::string_view& row_name)
{
  while (p < end && *p <= ' ' && *p != '\n')
    p++;
  if (p >= end) { return false; }
  if (*p == '\n') {
    p++;
    return false;
  }
  if (*p == '*' || *p == '$') {
    p = rows_find_next_line(p, end);
    return false;
  }

  row_type = *p++;
  while (p < end && *p <= ' ' && *p != '\n')
    p++;

  const char* name_start = p;
  while (p < end && *p > ' ')
    p++;
  if (name_start == p) { return false; }
  row_name = std::string_view(name_start, (size_t)(p - name_start));

  // ROWS only uses fields 1-2. Fields 3-6 are ignored by the MPS spec, and
  // field 3 may start with '$' to comment the rest of the record.
  p = rows_find_next_line(p, end);
  return true;
}

static std::vector<RowChunkBoundary> compute_row_chunk_boundaries(const char* rows_start,
                                                                  const char* rows_end,
                                                                  int num_threads)
{
  scoped_timer_t timer("rows_compute_chunk_boundaries");

  std::vector<RowChunkBoundary> boundaries((size_t)num_threads);
  size_t total_size = (size_t)(rows_end - rows_start);
  size_t chunk_size = total_size / (size_t)num_threads;

  boundaries[0].start = rows_start;
  for (int t = 0; t < num_threads; ++t) {
    if (t == num_threads - 1) {
      boundaries[(size_t)t].end = rows_end;
    } else {
      const char* boundary            = rows_start + (size_t)(t + 1) * chunk_size;
      boundary                        = rows_find_next_line(boundary, rows_end);
      boundaries[(size_t)t].end       = boundary;
      boundaries[(size_t)t + 1].start = boundary;
    }
  }

  return boundaries;
}

template <typename i_t, typename f_t>
static bool parse_rows_section_parallel_impl(parse_state_t<i_t, f_t>& state,
                                             const char* rows_start,
                                             const char* rows_end,
                                             int num_threads)
{
  scoped_timer_t timer("parse_rows_parallel");

  auto boundaries = compute_row_chunk_boundaries(rows_start, rows_end, num_threads);
  std::vector<RowChunkInfo> infos((size_t)num_threads);

  {
    scoped_timer_t timer("rows_count_parallel");
#pragma omp parallel for num_threads(num_threads)
    for (int t = 0; t < num_threads; ++t) {
      MPS_NVTX_RANGE(std::string("rows_count_chunk ") + std::to_string(t), nvtx::colors::rows);
      const char* p   = boundaries[(size_t)t].start;
      const char* end = boundaries[(size_t)t].end;
      RowChunkInfo info;

      while (p < end) {
        char row_type = 0;
        std::string_view row_name;
        const char* before = p;
        if (!parse_rows_line_fast(p, end, row_type, row_name)) {
          if (p == before) {
            info.malformed = true;
            break;
          }
          continue;
        }

        if (row_type == 'N') {
          info.objective_names.push_back(row_name);
        } else {
          if (!info.has_first_constraint) {
            info.first_constraint_name = row_name;
            info.has_first_constraint  = true;
          }
          info.constraints++;
        }
      }

      infos[(size_t)t] = info;
    }
  }

  for (const auto& info : infos) {
    if (info.malformed) { return false; }
  }

  std::vector<size_t> offsets((size_t)num_threads + 1, 0);
  {
    scoped_timer_t timer("rows_prefix_sum");
    for (int t = 0; t < num_threads; ++t) {
      offsets[(size_t)t + 1] = offsets[(size_t)t] + infos[(size_t)t].constraints;
    }
  }

  size_t total_rows = offsets[(size_t)num_threads];
  {
    scoped_timer_t timer("rows_resize_outputs");
    state.row_names_sv.resize(total_rows);
    state.problem.row_types_.resize(total_rows);
  }

  if (state.objective_name_sv.empty()) {
    for (const auto& info : infos) {
      if (!info.objective_names.empty()) {
        state.objective_name_sv = info.objective_names.front();
        break;
      }
    }
  }
  for (const auto& info : infos) {
    for (std::string_view name : info.objective_names) {
      state.add_ignored_objective_name(name);
    }
  }

  bool dense_candidate = total_rows > 0;
  std::string_view dense_prefix;
  uint64_t dense_base_id = 0;
  size_t dense_pad_width = 0;
  bool dense_zero_padded = false;

  if (dense_candidate) {
    std::string_view first_name;
    for (const auto& info : infos) {
      if (info.has_first_constraint) {
        first_name = info.first_constraint_name;
        break;
      }
    }

    uint64_t first_value      = 0;
    size_t first_suffix_width = 0;
    if (!parse_trailing_u64(first_name, dense_prefix, first_value, first_suffix_width)) {
      dense_candidate = false;
    } else {
      dense_base_id     = first_value;
      dense_pad_width   = first_suffix_width;
      dense_zero_padded = dense_suffix_is_zero_padded(first_name, first_suffix_width);
    }
  }

  std::vector<uint8_t> dense_ok_by_chunk((size_t)num_threads, 1);

  {
    scoped_timer_t timer("rows_fill_parallel");
#pragma omp parallel for num_threads(num_threads)
    for (int t = 0; t < num_threads; ++t) {
      MPS_NVTX_RANGE(std::string("rows_fill_chunk ") + std::to_string(t), nvtx::colors::rows);
      const char* p   = boundaries[(size_t)t].start;
      const char* end = boundaries[(size_t)t].end;
      size_t out      = offsets[(size_t)t];

      bool local_dense_ok = dense_candidate;

      while (p < end) {
        char row_type = 0;
        std::string_view row_name;
        const char* before = p;
        if (!parse_rows_line_fast(p, end, row_type, row_name)) {
          if (p == before) {
            local_dense_ok = false;
            break;
          }
          continue;
        }

        if (row_type == 'N') { continue; }

        state.row_names_sv[out]       = row_name;
        state.problem.row_types_[out] = row_type;

        if (local_dense_ok) {
          std::string_view prefix;
          uint64_t value      = 0;
          size_t suffix_width = 0;
          uint64_t expected   = dense_base_id + out;
          local_dense_ok =
            parse_trailing_u64(row_name, prefix, value, suffix_width) && prefix == dense_prefix &&
            value == expected &&
            dense_suffix_width_ok(value, suffix_width, dense_zero_padded, dense_pad_width);
        }
        out++;
      }

      dense_ok_by_chunk[(size_t)t] = local_dense_ok ? 1 : 0;
    }
  }

  {
    scoped_timer_t timer("rows_dense_metadata");
    for (uint8_t ok : dense_ok_by_chunk) {
      dense_candidate = dense_candidate && ok;
    }
    state.row_dense_candidate = dense_candidate;
    if (dense_candidate) {
      state.row_dense_prefix      = dense_prefix;
      state.row_dense_min_id      = dense_base_id;
      state.row_dense_max_id      = dense_base_id + total_rows - 1;
      state.row_dense_base_id     = dense_base_id;
      state.row_dense_pad_width   = dense_pad_width;
      state.row_dense_zero_padded = dense_zero_padded;
    }
  }

  return true;
}

template <typename i_t, typename f_t>
static void parse_rows_section_serial_impl(parse_state_t<i_t, f_t>& state, const char* rows_end)
{
  scoped_timer_t timer("parse_rows_serial");

  while (state.cursor.ptr < rows_end) {
    auto row_type = state.cursor.ptr[0];
    state.cursor.advance(1);
    state.cursor.skip_ws();
    // if (row_type != "E" && row_type != "L" && row_type != "G" && row_type != "N") {
    //   state.cursor.error("expected E, L, G, or N, got '%s'", row_type.data());
    // }

    auto row_name = state.cursor.read_field();
    // ROWS fields after the row name are unused; tolerate annotations/comments there.
    state.cursor.skip_to_eol();

    // 'N' type is the objective row - store its name but don't add to constraints
    if (row_type == 'N') {
      state.observe_objective_row_name(row_name);
    } else {
      size_t row_idx = state.row_names_sv.size();
      state.row_names_sv.push_back(row_name);
      state.observe_row_name_for_dense_index(row_name, row_idx);
      state.problem.row_types_.push_back(row_type);
    }
    expect_eol(state.cursor);
  }
}

template <typename i_t, typename f_t>
static void parse_rows_section(parse_state_t<i_t, f_t>& state, const char* rows_end)
{
  scoped_timer_t timer("parse_rows");
  expect_section(state.cursor, "ROWS");

  {
    scoped_timer_t timer("parse_rows_scan");
    const char* rows_start = state.cursor.ptr;

    size_t rows_bytes    = (size_t)(rows_end - state.cursor.ptr);
    int num_threads      = phase_thread_count(MPS_ROWS_THREAD_CAP);
    bool parsed_parallel = false;
    if (rows_bytes >= 512ull * 1024ull * 1024ull && num_threads > 1) {
      parsed_parallel =
        parse_rows_section_parallel_impl<i_t, f_t>(state, state.cursor.ptr, rows_end, num_threads);
      if (!parsed_parallel) {
        state.row_names_sv.clear();
        state.problem.row_types_.clear();
        state.row_dense_candidate   = true;
        state.row_dense_prefix      = {};
        state.row_dense_min_id      = 0;
        state.row_dense_max_id      = 0;
        state.row_dense_base_id     = 0;
        state.row_dense_pad_width   = 0;
        state.row_dense_zero_padded = false;
        state.cursor.ptr            = rows_start;
        parse_rows_section_serial_impl(state, rows_end);
      }
    } else {
      parse_rows_section_serial_impl(state, rows_end);
    }
    state.cursor.ptr = rows_end;
  }

  state.problem.n_constraints_ = (i_t)state.row_names_sv.size();
  state.problem.b_.resize((size_t)state.problem.n_constraints_);

  {
    scoped_timer_t timer("parse_rows_hash_init");
    state.init_row_hash_table();
  }
}

// =============================================================================
// Parallel COLUMNS parser
// =============================================================================

struct MarkerInfo {
  enum Type { INTORG, INTEND };
  Type type;
  size_t after_local_var_idx;  // SIZE_MAX means "before first variable"
};

struct RowCountBlock {
  size_t block_id       = 0;
  size_t storage_offset = 0;
};

struct DenseColChunkStats {
  bool candidate = true;
  std::string_view prefix;
  uint64_t first_id = 0;
  uint64_t last_id  = 0;
  size_t pad_width  = 0;
  bool zero_padded  = false;
  size_t count      = 0;
};

struct ChunkResult {
  std::vector<double> values;
  std::vector<uint32_t> row_indices;
  std::vector<size_t> col_offsets;
  std::vector<std::string_view> var_names;
  std::vector<MarkerInfo> markers;
  std::vector<std::pair<size_t, double>> objective_entries;  // local_col_idx -> coefficient
  // Sparse per-row scratch: each touched 4096-row block stores counts after parsing,
  // then the same slots become CSR write cursors. This avoids scanning/allocating
  // chunks*n_rows entries when a chunk only touches clustered row ranges. The
  // block payloads live in one arena per chunk so scatter has hugepage-friendly
  // write-position metadata instead of many independent 32 KiB allocations.
  std::vector<int64_t> row_count_storage;
  std::vector<RowCountBlock> row_count_blocks;
  std::vector<int32_t> row_count_block_dir;
  std::string_view first_var_name;
  std::string_view last_var_name;
  DenseColChunkStats dense_col_stats;
};

struct ChunkBoundary {
  const char* start;
  const char* end;
};

struct BoundsChunkBoundary {
  const char* start;
  const char* end;
};

static inline int64_t& column_row_count_slot(ChunkResult& result, size_t row_idx)
{
  size_t block_id   = row_idx / COLUMN_ROW_COUNT_BLOCK_ROWS;
  size_t local      = row_idx - block_id * COLUMN_ROW_COUNT_BLOCK_ROWS;
  int32_t block_pos = result.row_count_block_dir[block_id];
  if (__unlikely(block_pos < 0)) {
    block_pos                            = (int32_t)result.row_count_blocks.size();
    result.row_count_block_dir[block_id] = block_pos;
    RowCountBlock block;
    block.block_id       = block_id;
    block.storage_offset = result.row_count_storage.size();
    result.row_count_storage.resize(block.storage_offset + COLUMN_ROW_COUNT_BLOCK_ROWS, 0);
    result.row_count_blocks.push_back(std::move(block));
  }
  return result
    .row_count_storage[result.row_count_blocks[(size_t)block_pos].storage_offset + local];
}

static void observe_dense_col_name(DenseColChunkStats& stats, std::string_view name)
{
  if (!stats.candidate) { return; }

  std::string_view prefix;
  uint64_t value      = 0;
  size_t suffix_width = 0;
  if (!parse_trailing_u64(name, prefix, value, suffix_width)) {
    stats.candidate = false;
    return;
  }

  if (stats.count == 0) {
    stats.prefix      = prefix;
    stats.first_id    = value;
    stats.last_id     = value;
    stats.pad_width   = suffix_width;
    stats.zero_padded = dense_suffix_is_zero_padded(name, suffix_width);
    stats.count       = 1;
    return;
  }

  if (prefix != stats.prefix) {
    stats.candidate = false;
    return;
  }
  if (stats.last_id == std::numeric_limits<uint64_t>::max() || value != stats.last_id + 1) {
    stats.candidate = false;
    return;
  }
  if (!dense_suffix_width_ok(value, suffix_width, stats.zero_padded, stats.pad_width)) {
    stats.candidate = false;
    return;
  }
  stats.last_id = value;
  stats.count++;
}

static bool dense_col_chunk_padding_compatible(const DenseColChunkStats& stats,
                                               bool global_zero_padded,
                                               size_t global_pad_width)
{
  if (global_zero_padded) {
    return stats.pad_width == global_pad_width ||
           (!stats.zero_padded && decimal_digits_u64(stats.first_id) >= global_pad_width);
  }
  return !stats.zero_padded;
}

// Read first field (column name) from a line without modifying any state
static std::string_view peek_line_column_name(const char* line_start, const char* end)
{
  const char* p = line_start;
  while (p < end && *p <= ' ' && *p != '\n')
    p++;
  const char* field_start = p;
  while (p < end && *p > ' ')
    p++;
  return std::string_view(field_start, (size_t)(p - field_start));
}

// Find the start of the next line
static const char* find_next_line(const char* p, const char* end)
{
  while (p < end && *p != '\n')
    p++;
  if (p < end) p++;
  return p;
}

static const char* find_bounds_body_end(const char* bounds_body_start, const char* parse_end)
{
  const char* p = bounds_body_start;
  while (p < parse_end) {
    if ((*p == 'E' && parse_end - p >= 6 && std::memcmp(p, "ENDATA", 6) == 0 && p[6] <= ' ') ||
        (*p == 'Q' && is_quadratic_section_start(p, parse_end)) ||
        (*p == 'R' && parse_end - p >= 6 && std::memcmp(p, "RANGES", 6) == 0 && p[6] <= ' ')) {
      return p;
    }
    p = find_next_line(p, parse_end);
  }
  return parse_end;
}

static std::vector<BoundsChunkBoundary> compute_line_chunk_boundaries(const char* section_start,
                                                                      const char* section_end,
                                                                      int num_threads)
{
  scoped_timer_t timer("bounds_compute_chunk_boundaries");

  size_t total_size = (size_t)(section_end - section_start);
  size_t chunk_size = total_size / (size_t)num_threads;

  std::vector<BoundsChunkBoundary> boundaries((size_t)num_threads);
  boundaries[0].start = section_start;
  for (int t = 0; t < num_threads; ++t) {
    if (t == num_threads - 1) {
      boundaries[(size_t)t].end = section_end;
    } else {
      const char* boundary            = section_start + (size_t)(t + 1) * chunk_size;
      boundaries[(size_t)t].end       = find_next_line(boundary, section_end);
      boundaries[(size_t)t + 1].start = boundaries[(size_t)t].end;
    }
  }
  return boundaries;
}

static std::vector<ChunkBoundary> compute_chunk_boundaries(const char* columns_start,
                                                           const char* columns_end,
                                                           int num_threads)
{
  scoped_timer_t timer("compute_chunk_boundaries");

  size_t total_size = (size_t)(columns_end - columns_start);
  size_t chunk_size = total_size / (size_t)num_threads;

  std::vector<ChunkBoundary> boundaries(num_threads);

  // Parallel boundary finding - each thread finds its own end at a column transition
  // #pragma omp parallel for
  for (int t = 0; t < num_threads; t++) {
    if (t == 0) { boundaries[t].start = columns_start; }

    if (t == num_threads - 1) {
      boundaries[t].end = columns_end;
    } else {
      // Find estimated position and align to line boundary
      const char* estimated_end = columns_start + (t + 1) * chunk_size;
      const char* line_start    = estimated_end;
      while (line_start < columns_end && *line_start != '\n')
        line_start++;
      if (line_start < columns_end) line_start++;

      // Read column name at this line
      std::string_view col_name = peek_line_column_name(line_start, columns_end);

      // Scan forward until column name changes (to avoid splitting a column)
      const char* boundary = line_start;
      while (boundary < columns_end) {
        const char* next_line = find_next_line(boundary, columns_end);
        if (next_line >= columns_end) break;

        std::string_view next_col = peek_line_column_name(next_line, columns_end);
        if (next_col != col_name && !next_col.empty() && next_col[0] != '\'') {
          // Found a column transition (and it's not a MARKER line)
          boundary = next_line;
          break;
        }
        boundary = next_line;
      }
      boundaries[t].end = boundary;
    }
  }

  // Fix up start pointers (each start is previous end)
  for (int t = 1; t < num_threads; t++) {
    boundaries[t].start = boundaries[t - 1].end;
  }

  return boundaries;
}

template <typename i_t, typename f_t>
static ChunkResult parse_columns_chunk(const char* chunk_start,
                                       const char* chunk_end,
                                       const parse_state_t<i_t, f_t>& state)
{
  ChunkResult result;

  if (chunk_start >= chunk_end) {
    result.col_offsets.push_back(0);
    return result;
  }

  size_t chunk_size     = (size_t)(chunk_end - chunk_start);
  size_t estimated_nnz  = chunk_size / 100;
  size_t estimated_cols = estimated_nnz / 10;
  if (__unlikely(state.problem.n_constraints_ > (i_t)std::numeric_limits<int32_t>::max())) {
    state.cursor.error("fast COLUMNS path requires <= INT32_MAX rows for chunk row indices");
  }
  result.values.reserve(estimated_nnz);
  result.row_indices.reserve(estimated_nnz);
  result.col_offsets.reserve(estimated_cols + 1);
  result.var_names.reserve(estimated_cols);
  result.objective_entries.reserve(estimated_cols);
  size_t n_row_blocks = ((size_t)state.problem.n_constraints_ + COLUMN_ROW_COUNT_BLOCK_ROWS - 1) /
                        COLUMN_ROW_COUNT_BLOCK_ROWS;
  result.row_count_block_dir.resize(n_row_blocks, -1);
  size_t estimated_touched_blocks = std::min(n_row_blocks, std::max<size_t>(16, estimated_nnz));
  result.row_count_blocks.reserve(estimated_touched_blocks);
  result.row_count_storage.reserve(estimated_touched_blocks * COLUMN_ROW_COUNT_BLOCK_ROWS);

  cursor_t cursor(chunk_start, (size_t)(chunk_end - chunk_start));
  std::string_view prev_var_name = "";

  cursor.skip_ws();

  while (!cursor.done()) {
    if (__unlikely(*cursor.ptr == 'R')) {
      auto next = cursor.peek_field();
      // RHS section is mandatory right after COLUMNS section
      if (next == "RHS") { break; }
    }

    auto [var_name, field2] = cursor.read_two_fields();
    if (__unlikely(!field2.empty() && field2[0] == '$')) {
      cursor.skip_to_eol();
      expect_eol(cursor);
      continue;
    }

    // Check for integer marker
    if (__unlikely(field2[0] == '\'' && field2 == "'MARKER'")) {
      auto marker_type = cursor.read_field();

      MarkerInfo marker;
      marker.after_local_var_idx =
        result.var_names.empty() ? SIZE_MAX : result.var_names.size() - 1;

      if (marker_type == "'INTORG'") {
        marker.type = MarkerInfo::INTORG;
      } else {
        marker.type = MarkerInfo::INTEND;
      }
      result.markers.push_back(marker);

      while (!cursor.done() && !cursor.eol())
        cursor.ptr++;
      if (!cursor.done()) cursor.ptr++;
      cursor.skip_ws();
      continue;
    }

    auto row_name = field2;
    // quite often in MIPs the coefficient is just a single-digit integer
    double value;
    double sign = 1.0;
    if (cursor.ptr[0] == '-') {
      sign = -1.0;
      cursor.advance(1);
    }
    if (cursor.ptr + 1 < cursor.end && is_digit_byte(cursor.ptr[0]) && cursor.ptr[1] == '\n') {
      value = sign * (cursor.ptr[0] - '0');
      cursor.advance(1);
    } else {
      value = sign * fast_atof_advance(cursor.ptr, cursor.end);
    }
    // usually EOL directly follows
    if (__unlikely(!cursor.eol())) { cursor.skip_ws(); }
    accept_comment(cursor);

    if (result.first_var_name.empty()) { result.first_var_name = var_name; }
    result.last_var_name = var_name;

    if (prev_var_name != var_name) {
      result.var_names.push_back(var_name);
      observe_dense_col_name(result.dense_col_stats, var_name);
      result.col_offsets.push_back(result.values.size());
      prev_var_name = var_name;
    }

    auto add_entry = [&](std::string_view rn, double val) {
      size_t row_idx = state.row_lookup(rn);
      if (__likely(row_idx != SIZE_MAX)) {
        assert(row_idx <= (size_t)std::numeric_limits<int32_t>::max());
        result.values.push_back(val);
        result.row_indices.push_back((uint32_t)row_idx);
        column_row_count_slot(result, row_idx)++;
      } else if (__likely(rn == state.objective_name_sv)) {
        result.objective_entries.push_back({result.var_names.size() - 1, val});
      } else if (state.is_ignored_objective_name(rn)) {
        return;
      } else {
        state.cursor.error("unknown row name in COLUMNS: %.*s", (int)rn.size(), rn.data());
      }
    };

    add_entry(row_name, value);

    // Optional second entry on same line
    if (!cursor.eol()) {
      auto row_name2 = cursor.read_field();
      if (__unlikely(!row_name2.empty() && row_name2[0] == '$')) {
        cursor.skip_to_eol();
        expect_eol(cursor);
        continue;
      }
      double value2 = fast_atof_advance(cursor.ptr, cursor.end);
      cursor.skip_ws();
      accept_comment(cursor);

      add_entry(row_name2, value2);
    }

    expect_eol(cursor);
  }

  result.col_offsets.push_back(result.values.size());

  return result;
}

// Fused merge + CSR construction: directly builds CSR from chunks without intermediate global CSC
template <typename i_t, typename f_t>
static void merge_chunk_results_to_csr(parse_state_t<i_t, f_t>& state,
                                       std::vector<ChunkResult>& chunks,
                                       int num_threads)
{
  scoped_timer_t timer("merge_chunks_to_csr");

  int num_chunks = (int)chunks.size();
  if (num_chunks == 0) return;

  i_t n_rows = state.problem.n_constraints_;

  std::vector<size_t> global_col_offset(num_chunks + 1);
  global_col_offset[0] = 0;
  size_t total_nnz     = 0;
  {
    scoped_timer_t timer("columns_global_offsets");
    for (int t = 0; t < num_chunks; t++) {
      global_col_offset[t + 1] = global_col_offset[t] + chunks[t].var_names.size();
      total_nnz += chunks[t].values.size();
    }
  }
  size_t total_cols = global_col_offset[num_chunks];
  {
    scoped_timer_t timer("columns_dense_metadata");
    bool dense_ok   = total_cols > 0;
    bool have_first = false;
    std::string_view dense_prefix;
    uint64_t expected_next_id = 0;
    uint64_t dense_min_id     = 0;
    uint64_t dense_max_id     = 0;
    size_t dense_pad_width    = 0;
    bool dense_zero_padded    = false;

    for (int t = 0; t < num_chunks && dense_ok; ++t) {
      const auto& stats = chunks[t].dense_col_stats;
      if (stats.count == 0) { continue; }
      if (!stats.candidate || stats.count != chunks[t].var_names.size()) {
        dense_ok = false;
        break;
      }
      if (!have_first) {
        have_first        = true;
        dense_prefix      = stats.prefix;
        expected_next_id  = stats.first_id;
        dense_min_id      = stats.first_id;
        dense_pad_width   = stats.pad_width;
        dense_zero_padded = stats.zero_padded;
      }
      if (stats.prefix != dense_prefix || stats.first_id != expected_next_id ||
          !dense_col_chunk_padding_compatible(stats, dense_zero_padded, dense_pad_width)) {
        dense_ok = false;
        break;
      }
      if (stats.last_id < stats.first_id || stats.last_id - stats.first_id + 1 != stats.count) {
        dense_ok = false;
        break;
      }
      dense_max_id = stats.last_id;
      if (stats.last_id == std::numeric_limits<uint64_t>::max()) {
        expected_next_id = stats.last_id;
        dense_ok         = false;
        break;
      }
      expected_next_id = stats.last_id + 1;
    }

    if (!have_first || dense_max_id < dense_min_id ||
        dense_max_id - dense_min_id + 1 != total_cols) {
      dense_ok = false;
    }

    state.col_dense_ordered = dense_ok;
    if (dense_ok) {
      state.col_dense_prefix      = dense_prefix;
      state.col_dense_min_id      = dense_min_id;
      state.col_dense_max_id      = dense_max_id;
      state.col_dense_pad_width   = dense_pad_width;
      state.col_dense_zero_padded = dense_zero_padded;
    }
  }

  // Step 2: Sum row counts (already computed during parsing) and build CSR row_offsets
  std::vector<i_t> global_row_counts((size_t)n_rows, 0);
  {
    scoped_timer_t timer("columns_sum_row_counts");
    for (int t = 0; t < num_chunks; t++) {
      for (const auto& block : chunks[t].row_count_blocks) {
        const int64_t* block_counts = chunks[t].row_count_storage.data() + block.storage_offset;
        size_t row_base             = block.block_id * COLUMN_ROW_COUNT_BLOCK_ROWS;
        size_t block_limit = std::min(COLUMN_ROW_COUNT_BLOCK_ROWS, (size_t)n_rows - row_base);
        for (size_t local = 0; local < block_limit; ++local) {
          global_row_counts[row_base + local] += (i_t)block_counts[local];
        }
      }
    }
  }
  {
    scoped_timer_t timer("columns_build_row_offsets");
    state.problem.A_offsets_.resize((size_t)n_rows + 1);
    state.problem.A_offsets_[0] = 0;
    for (i_t r = 0; r < n_rows; r++) {
      state.problem.A_offsets_[(size_t)r + 1] =
        state.problem.A_offsets_[(size_t)r] + global_row_counts[(size_t)r];
    }
  }

  {
    scoped_timer_t timer("columns_counts_to_write_positions");
    std::fill(global_row_counts.begin(), global_row_counts.end(), i_t{0});
    for (int t = 0; t < num_chunks; t++) {
      for (auto& block : chunks[t].row_count_blocks) {
        int64_t* block_counts = chunks[t].row_count_storage.data() + block.storage_offset;
        size_t row_base       = block.block_id * COLUMN_ROW_COUNT_BLOCK_ROWS;
        size_t block_limit    = std::min(COLUMN_ROW_COUNT_BLOCK_ROWS, (size_t)n_rows - row_base);
        for (size_t local = 0; local < block_limit; ++local) {
          int64_t count = block_counts[local];
          if (count == 0) continue;
          size_t row          = row_base + local;
          i_t pos             = state.problem.A_offsets_[row] + global_row_counts[row];
          block_counts[local] = (int64_t)pos;
          global_row_counts[row] += (i_t)count;
        }
      }
    }
  }

  {
    scoped_timer_t timer("columns_row_count_storage_hugepages");
#pragma omp parallel for num_threads(num_threads)
    for (int t = 0; t < num_chunks; ++t) {
      materialize_vector_hugepages(
        "column_row_count_storage", chunks[t].row_count_storage, materialize_touch_t::write_2mb);
    }
  }

  // Step 6: Allocate CSR arrays
  {
    scoped_timer_t timer("allocate_csr_arrays");

    // May be unexpectedly slow, even if already reserved() to good fit.
    // I assume the cause is probably that the pages aren't actually backed when reserve() is called
    // and the actual physical allocation only happens now

    // evil tweak until we can refactior problem_t
    // run the zero-init resize() calls in parallel

#pragma omp parallel sections num_threads(4)
    {
#pragma omp section
      {
        state.problem.A_.resize(total_nnz);
      }
#pragma omp section
      {
        state.problem.A_indices_.resize(total_nnz);
      }
#pragma omp section
      {
        if (!state.col_dense_ordered) { state.var_names_sv.resize(total_cols); }
      }
#pragma omp section
      {
        state.problem.var_types_.resize(total_cols);
      }
    }
  }

  // Step 6: Parallel scatter into CSR + copy var_names
  {
    scoped_timer_t timer("scatter_into_csr");
    {
      scoped_timer_t matrix_timer("scatter_matrix_entries");
#ifdef MPS_FAST_PERF_COUNTERS
      std::vector<perf_counter_snapshot_t> perf_snapshots((size_t)num_chunks);
#endif
#pragma omp parallel for num_threads(num_threads)
      for (int t = 0; t < num_chunks; t++) {
#ifdef MPS_FAST_PERF_COUNTERS
        thread_perf_counters_t perf_counters;
#endif
        auto& chunk = chunks[t];

        for (size_t local_col = 0; local_col < chunks[t].var_names.size(); local_col++) {
          i_t global_col = (i_t)(global_col_offset[t] + local_col);

          size_t col_start = chunks[t].col_offsets[local_col];
          size_t col_end   = chunks[t].col_offsets[local_col + 1];
          for (size_t idx = col_start; idx < col_end; idx++) {
            i_t row                        = (i_t)chunks[t].row_indices[idx];
            size_t row_idx                 = (size_t)row;
            size_t block_id                = row_idx / COLUMN_ROW_COUNT_BLOCK_ROWS;
            size_t local                   = row_idx - block_id * COLUMN_ROW_COUNT_BLOCK_ROWS;
            int32_t block_pos              = chunk.row_count_block_dir[block_id];
            RowCountBlock& block           = chunk.row_count_blocks[(size_t)block_pos];
            int64_t& write_pos             = chunk.row_count_storage[block.storage_offset + local];
            i_t dest                       = (i_t)write_pos++;
            state.problem.A_[dest]         = (f_t)chunks[t].values[idx];
            state.problem.A_indices_[dest] = global_col;
          }
        }
#ifdef MPS_FAST_PERF_COUNTERS
        perf_snapshots[(size_t)t] = perf_counters.stop();
#endif
      }
#ifdef MPS_FAST_PERF_COUNTERS
      print_perf_totals("scatter_matrix_entries", perf_snapshots);
#endif
    }

    if (!state.col_dense_ordered) {
      {
        scoped_timer_t names_timer("scatter_var_names");
#pragma omp parallel for num_threads(num_threads)
        for (int t = 0; t < num_chunks; t++) {
          for (size_t i = 0; i < chunks[t].var_names.size(); i++) {
            state.var_names_sv[global_col_offset[t] + i] = chunks[t].var_names[i];
          }
        }
      }
    } else {
      scoped_timer_t names_timer("scatter_var_names");
    }
  }

  // Step 7: Apply integer markers
  struct GlobalMarker {
    MarkerInfo::Type type;
    size_t global_var_idx;
  };
  {
    scoped_timer_t timer("columns_apply_markers");
    std::vector<GlobalMarker> all_markers;

    for (int t = 0; t < num_chunks; t++) {
      for (const auto& m : chunks[t].markers) {
        GlobalMarker gm;
        gm.type = m.type;

        if (m.after_local_var_idx == SIZE_MAX) {
          // Marker before any variable in this chunk
          gm.global_var_idx = (global_col_offset[t] > 0) ? global_col_offset[t] - 1 : SIZE_MAX;
        } else {
          gm.global_var_idx = global_col_offset[t] + m.after_local_var_idx;
        }
        all_markers.push_back(gm);
      }
    }

    std::sort(all_markers.begin(), all_markers.end(), [](const auto& a, const auto& b) {
      // SIZE_MAX means "before all variables" - should sort first
      if (a.global_var_idx == SIZE_MAX && b.global_var_idx != SIZE_MAX) return true;
      if (b.global_var_idx == SIZE_MAX && a.global_var_idx != SIZE_MAX) return false;
      return a.global_var_idx < b.global_var_idx;
    });

    bool is_integer   = false;
    size_t marker_idx = 0;

    for (size_t v = 0; v < total_cols; v++) {
      while (marker_idx < all_markers.size() &&
             (all_markers[marker_idx].global_var_idx == SIZE_MAX ||
              all_markers[marker_idx].global_var_idx < v)) {
        if (all_markers[marker_idx].type == MarkerInfo::INTORG) {
          is_integer = true;
        } else {
          is_integer = false;
        }
        marker_idx++;
      }
      state.problem.var_types_[v] = is_integer ? 'I' : 'C';
    }
  }

  // Step 8: Handle objective entries
  {
    scoped_timer_t timer("columns_objective_entries");
    state.problem.c_.resize(total_cols, f_t{0});
    for (int t = 0; t < num_chunks; t++) {
      for (const auto& [local_col, coeff] : chunks[t].objective_entries) {
        size_t global_col = global_col_offset[t] + local_col;
        if (global_col < total_cols) { state.problem.c_[global_col] = (f_t)coeff; }
      }
    }
  }

  // Store final dimensions; CSR and objective coefficients are already complete.
  state.problem.n_vars_ = (i_t)total_cols;
  state.problem.nnz_    = (i_t)total_nnz;
}

template <typename i_t, typename f_t>
static void parse_columns_section_parallel(parse_state_t<i_t, f_t>& state,
                                           int num_threads,
                                           const char* columns_end)
{
  scoped_timer_t timer("parse_columns_parallel");

  if (num_threads <= 0) { num_threads = phase_thread_count(MPS_COLUMNS_THREAD_CAP); }

  // Skip the "COLUMNS" header
  expect_section(state.cursor, "COLUMNS");

  const char* columns_start    = state.cursor.ptr;
  size_t columns_bytes         = (size_t)(columns_end - columns_start);
  size_t chunk_limited_threads = std::max<size_t>(1, columns_bytes / MPS_COLUMNS_MIN_CHUNK_BYTES);
  num_threads = std::max(1, std::min<int>(num_threads, (int)chunk_limited_threads));

  // Compute chunk boundaries
  auto chunk_bounds = compute_chunk_boundaries(columns_start, columns_end, num_threads);

  // Parse chunks in parallel
  std::vector<ChunkResult> results(num_threads);

  {
    scoped_timer_t timer("parse_columns_chunk_parallel");
#ifdef MPS_FAST_PERF_COUNTERS
    std::vector<perf_counter_snapshot_t> perf_snapshots((size_t)num_threads);
#endif
    {
#pragma omp parallel for num_threads(num_threads)
      for (int t = 0; t < num_threads; t++) {
        MPS_NVTX_RANGE(std::string("columns_chunk ") + std::to_string(t), nvtx::colors::columns);
#ifdef MPS_FAST_PERF_COUNTERS
        thread_perf_counters_t perf_counters;
#endif
        results[t] =
          parse_columns_chunk<i_t, f_t>(chunk_bounds[t].start, chunk_bounds[t].end, state);
#ifdef MPS_FAST_PERF_COUNTERS
        perf_snapshots[(size_t)t] = perf_counters.stop();
#endif
      }
    }
#ifdef MPS_FAST_PERF_COUNTERS
    print_perf_totals("parse_columns_chunk_parallel", perf_snapshots);
#endif
  }

  // Merge results directly into CSR format
  merge_chunk_results_to_csr(state, results, num_threads);

  // Update cursor to RHS section
  state.cursor.ptr = columns_end;
  state.cursor.skip_ws();
}

template <typename i_t, typename f_t>
static void parse_rhs_section(parse_state_t<i_t, f_t>& state, cursor_t& cursor)
{
  scoped_timer_t timer("parse_rhs");
  expect_section(cursor, "RHS");

  auto field_from_start = [](const char* start, const char* end) {
    const char* p = start;
    while (p < end && *p > ' ') {
      p++;
    }
    return std::string_view(start, (size_t)(p - start));
  };

  auto apply_rhs = [&](const char* row_start, size_t row_idx, f_t value) {
    if (row_idx != SIZE_MAX) {
      state.problem.b_[row_idx] = value;
      return;
    }
    std::string_view row_name = field_from_start(row_start, cursor.end);
    if (row_name == state.objective_name_sv) {
      state.problem.objective_offset_ = -value;
      return;
    }
    if (state.is_ignored_objective_name(row_name)) { return; }
    error_unknown_row(cursor, row_start, "RHS");
  };

  while (cursor.ptr < cursor.end && !is_rhs_section_end(cursor.ptr, cursor.end)) {
    auto rhs_name = cursor.read_field();
    (void)rhs_name;
    if (accept_comment(cursor)) {
      expect_eol(cursor);
      continue;
    }
    const char* row_start = cursor.ptr;
    size_t row_idx        = state.read_row_lookup(cursor);
    auto value            = expect_number_fast_pm_one(cursor);
    apply_rhs(row_start, row_idx, (f_t)value);

    accept_comment(cursor);
    if (!cursor.eol()) {
      const char* row_start2 = cursor.ptr;
      size_t row_idx2        = state.read_row_lookup(cursor);
      auto value2            = expect_number_fast_pm_one(cursor);
      apply_rhs(row_start2, row_idx2, (f_t)value2);
      accept_comment(cursor);
    }
    expect_eol(cursor);
  }
}

template <typename i_t, typename f_t>
static bool parse_bounds_section_parallel_dense(parse_state_t<i_t, f_t>& state,
                                                cursor_t& cursor,
                                                const char* bounds_body_start,
                                                const char* bounds_body_end,
                                                size_t n_vars)
{
  const size_t bounds_bytes = (size_t)(bounds_body_end - bounds_body_start);
  const int num_threads     = phase_thread_count(MPS_BOUNDS_THREAD_CAP);
  if (!state.col_dense_ordered || bounds_bytes < MPS_BOUNDS_PARALLEL_MIN_BYTES || num_threads < 2) {
    return false;
  }

  MPS_NVTX_RANGE("parse_bounds_parallel_dense", nvtx::colors::bounds);

  struct BoundsParallelStats {
    size_t lines            = 0;
    size_t dense_hits       = 0;
    size_t dense_misses     = 0;
    size_t comments         = 0;
    size_t min_var          = SIZE_MAX;
    size_t max_var          = 0;
    size_t non_strict_order = 0;
    bool saw_integer_type   = false;
    bool saw_negative_upper = false;
    const char* error_ptr   = nullptr;
    char error_msg[192]     = {};
  };

  std::vector<BoundsParallelStats> stats((size_t)num_threads);
  auto boundaries = compute_line_chunk_boundaries(bounds_body_start, bounds_body_end, num_threads);

  std::vector<uint8_t> bound_seen;
  {
    scoped_timer_t timer("bounds_parallel_seen_alloc");
    bound_seen.resize(n_vars, 0);
  }

  {
    scoped_timer_t timer("parse_bounds_parallel_dense");
    // Duplicate or non-monotone BOUNDS updates are file-order dependent. Parse
    // optimistically, then accept only if chunk summaries prove strict order.
#pragma omp parallel for schedule(static) num_threads(num_threads)
    for (int t = 0; t < num_threads; ++t) {
      auto& local = stats[(size_t)t];
      cursor_t cursor(boundaries[(size_t)t].start,
                      (size_t)(boundaries[(size_t)t].end - boundaries[(size_t)t].start));
      cursor.skip_ws();
      size_t prev_var = SIZE_MAX;
      try {
        while (cursor.ptr < cursor.end) {
          if (__unlikely(*cursor.ptr == '$')) {
            cursor.skip_to_eol();
            expect_eol(cursor);
            local.comments++;
            continue;
          }

          auto bound_type = cursor.read_field();
          if (__unlikely(bound_type.empty())) { break; }
          if (__unlikely(bound_type[0] == '$')) {
            cursor.skip_to_eol();
            expect_eol(cursor);
            local.comments++;
            continue;
          }

          auto bound_name = cursor.read_field();
          (void)bound_name;
          auto var_name = cursor.read_field();
          if (__unlikely(!var_name.empty() && var_name[0] == '$')) {
            cursor.skip_to_eol();
            expect_eol(cursor);
            local.comments++;
            continue;
          }

          size_t var_idx = state.col_lookup_dense_ordered(var_name);
          if (__unlikely(var_idx == SIZE_MAX)) {
            local.dense_misses++;
            std::snprintf(local.error_msg,
                          sizeof(local.error_msg),
                          "unknown variable name in BOUNDS: %.*s",
                          (int)var_name.size(),
                          var_name.data());
            local.error_ptr = cursor.ptr;
            break;
          }
          local.dense_hits++;
          local.lines++;
          local.min_var = std::min(local.min_var, var_idx);
          local.max_var = std::max(local.max_var, var_idx);
          if (prev_var != SIZE_MAX && var_idx <= prev_var) { local.non_strict_order++; }
          prev_var = var_idx;

          bool first_bound_for_var = bound_seen[var_idx] == 0;
          bound_seen[var_idx]      = 1;

          f_t value = 0;
          accept_comment(cursor);
          if (!cursor.eol()) {
            value = (f_t)expect_number_fast_pm_one(cursor);
            accept_comment(cursor);
          }

          if (bound_type == "LO") {
            state.problem.variable_lower_bounds_[var_idx] = value;
          } else if (bound_type == "UP") {
            state.problem.variable_upper_bounds_[var_idx] = value;
            if (first_bound_for_var && value < f_t{0}) {
              state.problem.variable_lower_bounds_[var_idx] = -std::numeric_limits<f_t>::infinity();
              local.saw_negative_upper                      = true;
            }
          } else if (bound_type == "FX") {
            state.problem.variable_lower_bounds_[var_idx] = value;
            state.problem.variable_upper_bounds_[var_idx] = value;
          } else if (bound_type == "FR") {
            state.problem.variable_lower_bounds_[var_idx] = -std::numeric_limits<f_t>::infinity();
            state.problem.variable_upper_bounds_[var_idx] = std::numeric_limits<f_t>::infinity();
          } else if (bound_type == "MI") {
            state.problem.variable_lower_bounds_[var_idx] = -std::numeric_limits<f_t>::infinity();
          } else if (bound_type == "PL") {
            state.problem.variable_upper_bounds_[var_idx] = std::numeric_limits<f_t>::infinity();
          } else if (bound_type == "BV") {
            state.problem.variable_lower_bounds_[var_idx] = 0;
            state.problem.variable_upper_bounds_[var_idx] = 1;
            state.problem.var_types_[var_idx]             = 'I';
            local.saw_integer_type                        = true;
          } else if (bound_type == "LI") {
            state.problem.variable_lower_bounds_[var_idx] = value;
            state.problem.var_types_[var_idx]             = 'I';
            local.saw_integer_type                        = true;
          } else if (bound_type == "UI") {
            state.problem.variable_upper_bounds_[var_idx] = value;
            if (first_bound_for_var && value < f_t{0}) {
              state.problem.variable_lower_bounds_[var_idx] = -std::numeric_limits<f_t>::infinity();
              local.saw_negative_upper                      = true;
            }
            state.problem.var_types_[var_idx] = 'I';
            local.saw_integer_type            = true;
          } else {
            std::snprintf(local.error_msg,
                          sizeof(local.error_msg),
                          "unknown bound type: %.*s",
                          (int)bound_type.size(),
                          bound_type.data());
            local.error_ptr = cursor.ptr;
            break;
          }

          expect_eol(cursor);
        }
      } catch (const std::exception& e) {
        std::snprintf(local.error_msg, sizeof(local.error_msg), "%s", e.what());
        local.error_ptr = cursor.ptr;
      }
    }
  }

  size_t dense_misses     = 0;
  size_t non_strict_order = 0;
  size_t overlap_chunks   = 0;
  size_t prev_max         = SIZE_MAX;
  for (int t = 0; t < num_threads; ++t) {
    const auto& local = stats[(size_t)t];
    if (local.error_ptr != nullptr) {
      cursor.ptr = local.error_ptr;
      cursor.error("%s", local.error_msg);
    }
    dense_misses += local.dense_misses;
    non_strict_order += local.non_strict_order;
    if (local.lines > 0) {
      if (prev_max != SIZE_MAX && local.min_var <= prev_max) { overlap_chunks++; }
      prev_max = local.max_var;
    }
  }

  const bool order_safe = dense_misses == 0 && non_strict_order == 0 && overlap_chunks == 0;

  if (!order_safe) {
    cursor.ptr = bounds_body_start;
    return false;
  }

  {
    scoped_timer_t timer("bounds_integer_defaults");
    for (size_t i = 0; i < n_vars; ++i) {
      if (!bound_seen[i] && state.problem.var_types_[i] == 'I') {
        state.problem.variable_lower_bounds_[i] = f_t{0};
        state.problem.variable_upper_bounds_[i] = f_t{1};
      }
    }
  }

  cursor.ptr = bounds_body_end;
  return true;
}

template <typename i_t, typename f_t>
static void parse_bounds_section(parse_state_t<i_t, f_t>& state,
                                 cursor_t& cursor,
                                 bool allow_parallel_dense = false)
{
  size_t n_vars = (size_t)state.problem.n_vars_;

  // Initialize bounds with defaults
  {
    scoped_timer_t timer("bounds_init_defaults");
    const bool parallel_init =
      n_vars >= MPS_BOUNDS_PARALLEL_INIT_MIN_VARS && omp_get_max_threads() >= 2;

    if (parallel_init) {
#pragma omp parallel sections num_threads(2)
      {
#pragma omp section
        {
          state.problem.variable_lower_bounds_.resize(n_vars, f_t{0});
        }
#pragma omp section
        {
          state.problem.variable_upper_bounds_.resize(n_vars, std::numeric_limits<f_t>::infinity());
        }
      }
    } else {
      state.problem.variable_lower_bounds_.resize(n_vars, f_t{0});
      state.problem.variable_upper_bounds_.resize(n_vars, std::numeric_limits<f_t>::infinity());
    }
  }

  {
    scoped_timer_t timer("bounds_madvise_pretouch");
    materialize_vector_hugepages("variable_lower_bounds",
                                 state.problem.variable_lower_bounds_,
                                 materialize_touch_t::write_4kb);
    materialize_vector_hugepages("variable_upper_bounds",
                                 state.problem.variable_upper_bounds_,
                                 materialize_touch_t::write_4kb);
  }

  std::vector<uint64_t> bound_seen((n_vars + 63) / 64, 0);
  auto has_bound = [&](size_t var_idx) {
    return (bound_seen[var_idx >> 6] & (uint64_t{1} << (var_idx & 63))) != 0;
  };
  auto mark_bound = [&](size_t var_idx) {
    bound_seen[var_idx >> 6] |= uint64_t{1} << (var_idx & 63);
  };
  auto apply_unspecified_integer_bounds = [&]() {
    scoped_timer_t timer("bounds_integer_defaults");
    for (size_t i = 0; i < n_vars; ++i) {
      if (!has_bound(i) && state.problem.var_types_[i] == 'I') {
        state.problem.variable_lower_bounds_[i] = f_t{0};
        state.problem.variable_upper_bounds_[i] = f_t{1};
      }
    }
  };

  if (!accept_section(cursor, "BOUNDS")) {
    apply_unspecified_integer_bounds();
    return;
  }

  const char* bounds_body_start = cursor.ptr;
  const char* bounds_body_end =
    allow_parallel_dense ? find_bounds_body_end(bounds_body_start, cursor.end) : cursor.end;
  if (allow_parallel_dense) {
    if (parse_bounds_section_parallel_dense(
          state, cursor, bounds_body_start, bounds_body_end, n_vars)) {
      return;
    }
    {
      scoped_timer_t timer("bounds_parallel_fallback_reset");
      std::fill(state.problem.variable_lower_bounds_.begin(),
                state.problem.variable_lower_bounds_.end(),
                f_t{0});
      std::fill(state.problem.variable_upper_bounds_.begin(),
                state.problem.variable_upper_bounds_.end(),
                std::numeric_limits<f_t>::infinity());
    }
  }

  size_t hint_idx = 0;
  {
    scoped_timer_t timer("parse_bounds");
    for (;;) {
      bool done = cursor.done() || peek(cursor) == "RANGES" || peek(cursor) == "ENDATA" ||
                  is_quadratic_section_start(cursor.ptr, cursor.end);
      if (done) break;

      auto bound_type = cursor.read_field();
      auto bound_name = cursor.read_field();
      (void)bound_name;
      auto var_name = cursor.read_field();
      if (__unlikely(!var_name.empty() && var_name[0] == '$')) {
        cursor.skip_to_eol();
        expect_eol(cursor);
        continue;
      }

      // optimized lookup using hint (bounds often in same order as columns)
      size_t var_idx = SIZE_MAX;
      if (__likely(state.col_dense_ordered)) {
        var_idx = state.col_lookup_dense_ordered(var_name);
        if (var_idx == SIZE_MAX) {
          cursor.error(
            "unknown variable name in BOUNDS: %.*s", (int)var_name.size(), var_name.data());
        }
      } else if (hint_idx + 1 < n_vars && state.var_names_sv[hint_idx + 1] == var_name) {
        var_idx = hint_idx + 1;
      } else if (hint_idx < n_vars && state.var_names_sv[hint_idx] == var_name) {
        var_idx = hint_idx;
      } else {
        size_t search_start = hint_idx + 2;
        size_t search_end   = n_vars;

      search_loop:
        for (size_t i = search_start; i < search_end; ++i) {
          if (state.var_names_sv[i] == var_name) {
            var_idx = i;
            goto found;
          }
        }
        if (search_start != 0) {
          search_end   = hint_idx;
          search_start = 0;
          goto search_loop;
        }
        cursor.error(
          "unknown variable name in BOUNDS: %.*s", (int)var_name.size(), var_name.data());
      }
    found:
      hint_idx                 = var_idx;
      bool first_bound_for_var = !has_bound(var_idx);

      f_t value = 0;
      accept_comment(cursor);
      if (!cursor.eol()) {
        // bounds are often just set to 0 or 1
        if (false && isdigit(cursor.ptr[0]) && cursor.ptr[1] == '\n' && cursor.ptr[2] == ' ') {
          value = cursor.ptr[0] - '0';
          cursor.ptr += 1;
        } else {
          value = (f_t)expect_number(cursor);
        }
        accept_comment(cursor);
      }

      if (bound_type == "LO") {
        state.problem.variable_lower_bounds_[var_idx] = value;
      } else if (bound_type == "UP") {
        state.problem.variable_upper_bounds_[var_idx] = value;
        if (first_bound_for_var && value < f_t{0}) {
          state.problem.variable_lower_bounds_[var_idx] = -std::numeric_limits<f_t>::infinity();
        }
      } else if (bound_type == "FX") {
        state.problem.variable_lower_bounds_[var_idx] = value;
        state.problem.variable_upper_bounds_[var_idx] = value;
      } else if (bound_type == "FR") {
        state.problem.variable_lower_bounds_[var_idx] = -std::numeric_limits<f_t>::infinity();
        state.problem.variable_upper_bounds_[var_idx] = std::numeric_limits<f_t>::infinity();
      } else if (bound_type == "MI") {
        state.problem.variable_lower_bounds_[var_idx] = -std::numeric_limits<f_t>::infinity();
      } else if (bound_type == "PL") {
        state.problem.variable_upper_bounds_[var_idx] = std::numeric_limits<f_t>::infinity();
      } else if (bound_type == "BV") {
        state.problem.variable_lower_bounds_[var_idx] = 0;
        state.problem.variable_upper_bounds_[var_idx] = 1;
        state.problem.var_types_[var_idx]             = 'I';
      } else if (bound_type == "LI") {
        state.problem.variable_lower_bounds_[var_idx] = value;
        state.problem.var_types_[var_idx]             = 'I';
      } else if (bound_type == "UI") {
        state.problem.variable_upper_bounds_[var_idx] = value;
        if (first_bound_for_var && value < f_t{0}) {
          state.problem.variable_lower_bounds_[var_idx] = -std::numeric_limits<f_t>::infinity();
        }
        state.problem.var_types_[var_idx] = 'I';
      } else {
        cursor.error("unknown bound type: %.*s", (int)bound_type.size(), bound_type.data());
      }
      mark_bound(var_idx);

      expect_eol(cursor);
    }
  }
  apply_unspecified_integer_bounds();
}

template <typename i_t, typename f_t>
static void parse_ranges_section(parse_state_t<i_t, f_t>& state, cursor_t& cursor)
{
  scoped_timer_t timer("parse_ranges");

  // Initialize constraint bounds from row_types and b_
  state.problem.constraint_lower_bounds_.resize((size_t)state.problem.n_constraints_);
  state.problem.constraint_upper_bounds_.resize((size_t)state.problem.n_constraints_);

  for (i_t i = 0; i < state.problem.n_constraints_; ++i) {
    char row_type = state.problem.row_types_[i];
    f_t b         = state.problem.b_[i];
    if (row_type == 'E') {
      state.problem.constraint_lower_bounds_[i] = b;
      state.problem.constraint_upper_bounds_[i] = b;
    } else if (row_type == 'L') {
      state.problem.constraint_lower_bounds_[i] = -std::numeric_limits<f_t>::infinity();
      state.problem.constraint_upper_bounds_[i] = b;
    } else if (row_type == 'G') {
      state.problem.constraint_lower_bounds_[i] = b;
      state.problem.constraint_upper_bounds_[i] = std::numeric_limits<f_t>::infinity();
    }
  }

  if (!accept_section(cursor, "RANGES")) { return; }

  auto apply_range = [&](std::string_view row_name, f_t range_val) {
    size_t row_idx = state.row_lookup(row_name);
    if (row_idx == SIZE_MAX) {
      cursor.error("unknown row name in RANGES: %.*s", (int)row_name.size(), row_name.data());
    }
    char row_type = state.problem.row_types_[row_idx];
    f_t abs_range = std::abs(range_val);

    if (row_type == 'E') {
      if (range_val >= 0) {
        state.problem.constraint_upper_bounds_[row_idx] =
          state.problem.constraint_lower_bounds_[row_idx] + abs_range;
      } else {
        state.problem.constraint_lower_bounds_[row_idx] =
          state.problem.constraint_upper_bounds_[row_idx] - abs_range;
      }
    } else if (row_type == 'L') {
      state.problem.constraint_lower_bounds_[row_idx] =
        state.problem.constraint_upper_bounds_[row_idx] - abs_range;
    } else if (row_type == 'G') {
      state.problem.constraint_upper_bounds_[row_idx] =
        state.problem.constraint_lower_bounds_[row_idx] + abs_range;
    }
  };

  while (cursor.ptr < cursor.end && peek(cursor) != "BOUNDS" && peek(cursor) != "ENDATA" &&
         !is_quadratic_section_start(cursor.ptr, cursor.end)) {
    auto range_name = cursor.read_field();
    (void)range_name;
    if (accept_comment(cursor)) {
      expect_eol(cursor);
      continue;
    }
    auto row_name = cursor.read_field();
    auto value    = (f_t)expect_number(cursor);
    apply_range(row_name, value);

    accept_comment(cursor);
    if (!cursor.eol()) {
      auto row_name2 = cursor.read_field();
      if (__unlikely(!row_name2.empty() && row_name2[0] == '$')) {
        cursor.skip_to_eol();
        expect_eol(cursor);
        continue;
      }
      auto value2 = (f_t)expect_number(cursor);
      apply_range(row_name2, value2);
      accept_comment(cursor);
    }
    expect_eol(cursor);
  }
}

template <typename i_t, typename f_t>
static void build_var_name_map_if_needed(parse_state_t<i_t, f_t>& state)
{
  if (state.col_dense_ordered || !state.var_names_map.empty()) { return; }
  scoped_timer_t timer("quadratic_build_var_name_map");
  state.var_names_map.reserve((size_t)state.problem.n_vars_ * 2);
  for (size_t i = 0; i < state.var_names_sv.size(); ++i) {
    state.var_names_map.emplace(state.var_names_sv[i], i);
  }
}

template <typename i_t, typename f_t>
static size_t lookup_quadratic_var(parse_state_t<i_t, f_t>& state, std::string_view name)
{
  if (state.col_dense_ordered) { return state.col_lookup_dense_ordered(name); }
  auto it = state.var_names_map.find(name);
  return it == state.var_names_map.end() ? SIZE_MAX : it->second;
}

template <typename i_t, typename f_t>
static void build_quadratic_csr(parse_state_t<i_t, f_t>& state,
                                const std::vector<std::tuple<i_t, i_t, f_t>>& entries,
                                bool symmetric_upper_triangular)
{
  scoped_timer_t timer("build_quadratic_csr");
  const size_t n_vars = (size_t)state.problem.n_vars_;
  if (entries.empty()) { return; }

  struct ExpandedEntry {
    size_t row;
    size_t col;
    size_t seq;
    f_t value;
  };

  std::vector<ExpandedEntry> expanded;
  expanded.reserve(symmetric_upper_triangular ? entries.size() * 2 : entries.size());
  size_t seq = 0;
  for (const auto& [row_i, col_i, value] : entries) {
    size_t row = (size_t)row_i;
    size_t col = (size_t)col_i;
    expanded.push_back({row, col, seq++, value});
    if (symmetric_upper_triangular && row != col) { expanded.push_back({col, row, seq++, value}); }
  }

  std::stable_sort(expanded.begin(), expanded.end(), [](const auto& a, const auto& b) {
    if (a.row != b.row) return a.row < b.row;
    if (a.col != b.col) return a.col < b.col;
    return a.seq < b.seq;
  });

  auto& values  = state.problem.Q_objective_values_;
  auto& indices = state.problem.Q_objective_indices_;
  auto& offsets = state.problem.Q_objective_offsets_;
  values.clear();
  indices.clear();
  offsets.assign(n_vars + 1, i_t{0});
  values.reserve(expanded.size());
  indices.reserve(expanded.size());

  size_t current_row = 0;
  offsets[0]         = 0;
  for (const auto& entry : expanded) {
    while (current_row < entry.row) {
      offsets[++current_row] = (i_t)values.size();
    }
    values.push_back(entry.value * f_t{0.5});
    indices.push_back((i_t)entry.col);
  }
  while (current_row < n_vars) {
    offsets[++current_row] = (i_t)values.size();
  }
}

template <typename i_t, typename f_t>
[[maybe_unused]] static void parse_quadratic_sections(parse_state_t<i_t, f_t>& state,
                                                      cursor_t& cursor)
{
  scoped_timer_t timer("parse_quadratic_sections");
  if (cursor.done() || peek(cursor) == "ENDATA") { return; }
  if (!is_quadratic_section_start(cursor.ptr, cursor.end)) { return; }

  build_var_name_map_if_needed(state);
  std::vector<std::tuple<i_t, i_t, f_t>> quadobj_entries;
  std::vector<std::tuple<i_t, i_t, f_t>> qmatrix_entries;
  std::vector<std::tuple<i_t, i_t, f_t>>* active_entries = nullptr;

  auto add_entry = [&](std::string_view var1, std::string_view var2, f_t value) {
    size_t var1_idx = lookup_quadratic_var(state, var1);
    if (var1_idx == SIZE_MAX) {
      cursor.error("unknown variable name in QUADOBJ/QMATRIX: %.*s", (int)var1.size(), var1.data());
    }
    size_t var2_idx = lookup_quadratic_var(state, var2);
    if (var2_idx == SIZE_MAX) {
      cursor.error("unknown variable name in QUADOBJ/QMATRIX: %.*s", (int)var2.size(), var2.data());
    }
    active_entries->emplace_back((i_t)var1_idx, (i_t)var2_idx, value);
  };

  while (cursor.ptr < cursor.end) {
    if (peek(cursor) == "ENDATA") { break; }
    if (accept_section(cursor, "QUADOBJ")) {
      active_entries = &quadobj_entries;
      continue;
    }
    if (accept_section(cursor, "QMATRIX")) {
      active_entries = &qmatrix_entries;
      continue;
    }
    if (active_entries == nullptr) { break; }

    auto var1 = cursor.read_field();
    if (__unlikely(var1.empty())) { break; }
    if (__unlikely(var1[0] == '$')) {
      cursor.skip_to_eol();
      expect_eol(cursor);
      continue;
    }
    auto var2 = cursor.read_field();
    if (__unlikely(!var2.empty() && var2[0] == '$')) {
      cursor.skip_to_eol();
      expect_eol(cursor);
      continue;
    }
    f_t value = (f_t)expect_number(cursor);
    add_entry(var1, var2, value);
    accept_comment(cursor);
    expect_eol(cursor);
  }

  if (!quadobj_entries.empty()) {
    build_quadratic_csr(state, quadobj_entries, true);
  } else if (!qmatrix_entries.empty()) {
    build_quadratic_csr(state, qmatrix_entries, false);
  }
}

template <typename i_t, typename f_t>
static void set_cursor_range(parse_state_t<i_t, f_t>& state, mps_phase_range_t range)
{
  state.cursor.ptr = range.begin;
  state.cursor.end = range.end;
}

template <typename i_t, typename f_t>
static void parse_header_range(parse_state_t<i_t, f_t>& state, mps_phase_range_t range)
{
  set_cursor_range(state, range);
  accept_comment_line(state.cursor);
  if (state.cursor.done()) { return; }
  parse_name_section(state);
  parse_objsense_section(state);
  parse_objname_section(state);
}

template <typename i_t, typename f_t>
static void parse_rows_range(parse_state_t<i_t, f_t>& state, mps_phase_range_t range)
{
  set_cursor_range(state, range);
  parse_rows_section(state, range.end);
}

template <typename i_t, typename f_t>
static void parse_columns_range(parse_state_t<i_t, f_t>& state,
                                mps_phase_range_t range,
                                int num_threads = 0)
{
  set_cursor_range(state, range);
  parse_columns_section_parallel(state, num_threads, range.end);
}

template <typename i_t, typename f_t>
static void parse_rhs_range(parse_state_t<i_t, f_t>& state, mps_phase_range_t range)
{
  if (!range.present) { return; }
  cursor_t cursor(range.begin, (size_t)(range.end - range.begin));
  parse_rhs_section(state, cursor);
}

template <typename i_t, typename f_t>
static void parse_bounds_range(parse_state_t<i_t, f_t>& state,
                               mps_phase_range_t range,
                               const char* fallback_ptr)
{
  if (range.present) {
    cursor_t cursor(range.begin, (size_t)(range.end - range.begin));
    parse_bounds_section(state, cursor, range.present);
  } else {
    cursor_t cursor(fallback_ptr, 16);
    parse_bounds_section(state, cursor, range.present);
  }
}

template <typename i_t, typename f_t>
static void parse_ranges_range(parse_state_t<i_t, f_t>& state,
                               mps_phase_range_t range,
                               const char* fallback_ptr)
{
  if (range.present) {
    cursor_t cursor(range.begin, (size_t)(range.end - range.begin));
    parse_ranges_section(state, cursor);
  } else {
    cursor_t cursor(fallback_ptr, 16);
    parse_ranges_section(state, cursor);
  }
}

template <typename i_t, typename f_t>
static void parse_quadratic_range(parse_state_t<i_t, f_t>& state,
                                  mps_phase_range_t range,
                                  const char* fallback_ptr)
{
  (void)state;
  if (range.present) {
    cursor_t cursor(range.begin, (size_t)(range.end - range.begin));
    if (!cursor.done() && is_quadratic_section_start(cursor.ptr, cursor.end)) {
      throw std::logic_error(
        "experimental fast MPS reader currently supports LP/MIP MPS files only; "
        "quadratic MPS sections are not supported");
    }
  } else {
    cursor_t cursor(fallback_ptr, 16);
    if (!cursor.done() && is_quadratic_section_start(cursor.ptr, cursor.end)) {
      throw std::logic_error(
        "experimental fast MPS reader currently supports LP/MIP MPS files only; "
        "quadratic MPS sections are not supported");
    }
  }
}

template <typename i_t, typename f_t>
static void materialize_problem_names(parse_state_t<i_t, f_t>& state)
{
  scoped_timer_t timer("materialize_problem_names");
  int num_threads = phase_thread_count(MPS_NAMES_THREAD_CAP);
  // Copy string_views to actual strings (this is where allocation happens)
  {
    scoped_timer_t timer("materialize_problem_scalar_names");
    state.problem.problem_name_   = std::string(state.problem_name_sv);
    state.problem.objective_name_ = std::string(state.objective_name_sv);
  }

  {
    scoped_timer_t timer("materialize_problem_row_names");
    size_t n = state.row_names_sv.size();
    state.problem.row_names_.resize(n);
    // row names are usually small enough for SSO - parallel assigns mostly don't touch the heap and
    // as such may help a lot ideally we could just allocate an arena and store non-owning string
    // views but that'd require a refactor of the problem representation
    if (n >= 1'000'000 && num_threads > 1) {
#pragma omp parallel for schedule(static) num_threads(num_threads)
      for (size_t i = 0; i < n; ++i) {
        state.problem.row_names_[i].assign(state.row_names_sv[i]);
      }
    } else {
      for (size_t i = 0; i < n; ++i) {
        state.problem.row_names_[i].assign(state.row_names_sv[i]);
      }
    }
  }

  {
    scoped_timer_t timer("materialize_problem_var_names");
    size_t n = state.col_dense_ordered ? (size_t)state.problem.n_vars_ : state.var_names_sv.size();
    state.problem.var_names_.resize(n);
    if (state.col_dense_ordered && n >= 1'000'000 && num_threads > 1) {
#pragma omp parallel for schedule(static) num_threads(num_threads)
      for (size_t i = 0; i < n; ++i) {
        state.dense_col_name(i, state.problem.var_names_[i]);
      }
    } else if (state.col_dense_ordered) {
      for (size_t i = 0; i < n; ++i) {
        state.dense_col_name(i, state.problem.var_names_[i]);
      }
    } else if (n >= 1'000'000 && num_threads > 1) {
#pragma omp parallel for schedule(static) num_threads(num_threads)
      for (size_t i = 0; i < n; ++i) {
        state.problem.var_names_[i].assign(state.var_names_sv[i]);
      }
    } else {
      for (size_t i = 0; i < n; ++i) {
        state.problem.var_names_[i].assign(state.var_names_sv[i]);
      }
    }
  }
}

template <typename Stream, typename i_t, typename f_t>
static cuopt::linear_programming::io::mps_data_model_t<i_t, f_t> parse_mps_fast_stream(
  Stream& stream, const char* total_timer_name, const char* producer_task_name)
{
  auto total_timer = std::make_unique<scoped_timer_t>(total_timer_name);
  omp_set_max_active_levels(2);

  input_stream_view_t input = stream.view();
  cuopt::linear_programming::io::mps_data_model_t<i_t, f_t> problem;
  problem.n_vars_                   = 0;
  problem.n_constraints_            = 0;
  problem.nnz_                      = 0;
  problem.maximize_                 = false;
  problem.objective_scaling_factor_ = f_t{1};
  problem.objective_offset_         = f_t{0};

  std::size_t reserve_size = std::max<std::size_t>(stream.reserve_size_hint(), 1024 * 1024);
  std::size_t reserve_dim  = std::max((size_t)1000, reserve_size / 1000);
  problem.A_offsets_.reserve(reserve_dim);
  problem.b_.reserve(reserve_dim);
  problem.variable_lower_bounds_.reserve(reserve_dim);
  problem.variable_upper_bounds_.reserve(reserve_dim);
  problem.var_types_.reserve(reserve_dim);
  problem.row_types_.reserve(reserve_dim);
  problem.row_names_.reserve(reserve_dim);
  problem.var_names_.reserve(reserve_dim);
  problem.constraint_lower_bounds_.reserve(reserve_dim);
  problem.constraint_upper_bounds_.reserve(reserve_dim);

  cursor_t cursor(input.data, 0);
  parse_state_t<i_t, f_t> state(problem, cursor);
  state.row_names_sv.reserve(reserve_dim);

  auto phase_end = [](const char*) { flush_timers(); };

  std::mutex task_error_mutex;
  std::exception_ptr first_task_error = nullptr;
  std::atomic<bool> task_failed{false};

  auto mark_task_error = [&](std::exception_ptr eptr) {
    {
      std::lock_guard<std::mutex> lock(task_error_mutex);
      if (!first_task_error) { first_task_error = eptr; }
    }
    task_failed.store(true, std::memory_order_release);
  };

  auto run_parser_task = [&](auto&& fn) {
    if (task_failed.load(std::memory_order_acquire)) { return; }
    try {
      fn();
    } catch (...) {
      mark_task_error(std::current_exception());
    }
  };

  auto unblock_phase_waiters_after_error = [&]() {
    mps_phase_range_t empty{input.data, input.data, false};
    input.registry->publish(mps_phase_kind::header, empty);
    input.registry->publish(mps_phase_kind::rows, empty);
    input.registry->publish(mps_phase_kind::columns, empty);
    input.registry->publish(mps_phase_kind::rhs, empty);
    input.registry->publish(mps_phase_kind::bounds, empty);
    input.registry->publish(mps_phase_kind::ranges, empty);
    input.registry->publish(mps_phase_kind::quadratic, empty);
  };

  int header_ready = 0, rows_ready = 0, columns_ready = 0;
  int rhs_ready = 0, bounds_ready = 0, ranges_ready = 0, quadratic_ready = 0;
  int header_done = 0, rows_done = 0, columns_done = 0;
  int rhs_done = 0, bounds_done = 0, ranges_done = 0, quadratic_done = 0, names_done = 0;

#pragma omp parallel num_threads(omp_get_max_threads())
  {
    std::string thread_name = "omp-parser-" + std::to_string(omp_get_thread_num());
    nvtx::name_current_thread(thread_name.c_str());

#pragma omp single
    {
      omp_event_handle_t ev_header;
#pragma omp task detach(ev_header) depend(out : header_ready)
      {
        input.registry->attach_event(mps_phase_kind::header, ev_header);
      }
      omp_event_handle_t ev_rows;
#pragma omp task detach(ev_rows) depend(out : rows_ready)
      {
        input.registry->attach_event(mps_phase_kind::rows, ev_rows);
      }
      omp_event_handle_t ev_columns;
#pragma omp task detach(ev_columns) depend(out : columns_ready)
      {
        input.registry->attach_event(mps_phase_kind::columns, ev_columns);
      }
      omp_event_handle_t ev_rhs;
#pragma omp task detach(ev_rhs) depend(out : rhs_ready)
      {
        input.registry->attach_event(mps_phase_kind::rhs, ev_rhs);
      }
      omp_event_handle_t ev_bounds;
#pragma omp task detach(ev_bounds) depend(out : bounds_ready)
      {
        input.registry->attach_event(mps_phase_kind::bounds, ev_bounds);
      }
      omp_event_handle_t ev_ranges;
#pragma omp task detach(ev_ranges) depend(out : ranges_ready)
      {
        input.registry->attach_event(mps_phase_kind::ranges, ev_ranges);
      }
      omp_event_handle_t ev_quadratic;
#pragma omp task detach(ev_quadratic) depend(out : quadratic_ready)
      {
        input.registry->attach_event(mps_phase_kind::quadratic, ev_quadratic);
      }

#pragma omp task
      {
        MPS_NVTX_RANGE(producer_task_name, nvtx::colors::io);
        try {
          stream.run_decode_tasks();
        } catch (...) {
          mark_task_error(std::current_exception());
          unblock_phase_waiters_after_error();
        }
      }

#pragma omp task depend(in : header_ready) depend(out : header_done)
      {
        run_parser_task([&] {
          MPS_NVTX_RANGE("task_header", nvtx::colors::generic);
          parse_header_range(state, input.registry->range(mps_phase_kind::header));
          phase_end("header");
        });
      }

#pragma omp task depend(in : rows_ready, header_done) depend(out : rows_done)
      {
        run_parser_task([&] {
          MPS_NVTX_RANGE("task_rows", nvtx::colors::rows);
          parse_rows_range(state, input.registry->range(mps_phase_kind::rows));
          phase_end("rows");
        });
      }

#pragma omp task depend(in : columns_ready, rows_done) depend(out : columns_done)
      {
        run_parser_task([&] {
          MPS_NVTX_RANGE("task_columns", nvtx::colors::columns);
          parse_columns_range(state, input.registry->range(mps_phase_kind::columns));
          phase_end("columns");
        });
      }

#pragma omp task depend(in : columns_done) depend(out : names_done)
      {
        run_parser_task([&] {
          MPS_NVTX_RANGE("task_materialize_names", nvtx::colors::names);
          scoped_timer_t timer("materialize_problem_names_task");
          materialize_problem_names(state);
        });
      }

#pragma omp task depend(in : rhs_ready, columns_done) depend(out : rhs_done)
      {
        run_parser_task([&] {
          MPS_NVTX_RANGE("task_rhs", nvtx::colors::rhs);
          parse_rhs_range(state, input.registry->range(mps_phase_kind::rhs));
          phase_end("rhs");
        });
      }

#pragma omp task depend(in : ranges_ready, rhs_done) depend(out : ranges_done)
      {
        run_parser_task([&] {
          MPS_NVTX_RANGE("task_ranges", nvtx::colors::ranges);
          parse_ranges_range(state, input.registry->range(mps_phase_kind::ranges), input.data);
          phase_end("ranges");
        });
      }

#pragma omp task depend(in : bounds_ready, columns_done) depend(out : bounds_done)
      {
        run_parser_task([&] {
          MPS_NVTX_RANGE("task_bounds", nvtx::colors::bounds);
          parse_bounds_range(state, input.registry->range(mps_phase_kind::bounds), input.data);
          phase_end("bounds");
        });
      }

#pragma omp task depend(in : quadratic_ready, columns_done) depend(out : quadratic_done)
      {
        run_parser_task([&] {
          MPS_NVTX_RANGE("task_quadratic", nvtx::colors::generic);
          parse_quadratic_range(
            state, input.registry->range(mps_phase_kind::quadratic), input.data);
          phase_end("quadratic");
        });
      }

#pragma omp taskwait
    }
  }

  if (first_task_error) { std::rethrow_exception(first_task_error); }

  input.size = stream.size();
  cursor.ptr = input.registry->range(mps_phase_kind::quadratic).present
                 ? input.registry->range(mps_phase_kind::quadratic).end
                 : (input.registry->range(mps_phase_kind::bounds).present
                      ? input.registry->range(mps_phase_kind::bounds).end
                      : (input.registry->range(mps_phase_kind::ranges).present
                           ? input.registry->range(mps_phase_kind::ranges).end
                           : input.registry->range(mps_phase_kind::rhs).end));
  cursor.end = input.data + input.size;
  if (!cursor.done()) { expect(cursor, "ENDATA"); }

  total_timer.reset();
  flush_timers();
  return problem;
}

template <typename i_t, typename f_t>
cuopt::linear_programming::io::mps_data_model_t<i_t, f_t> parse_mps_fast_file(
  const std::string& path, FileReadMethod read_method)
{
  FileReadMethod effective_method = effective_file_read_method(path, read_method);
  if (effective_method == FileReadMethod::Lz4) {
    Lz4InputStream stream(path);
    return parse_mps_fast_stream<Lz4InputStream, i_t, f_t>(
      stream, "parse_mps_fast_file_lz4 (total)", "task_lz4_read_decode");
  }
  if (effective_method == FileReadMethod::Read) {
    RawInputStream stream(path);
    return parse_mps_fast_stream<RawInputStream, i_t, f_t>(
      stream, "parse_mps_fast_file_raw (total)", "task_raw_read");
  }
  throw std::runtime_error("experimental fast MPS reader supports raw and LZ4 inputs only");
}

template cuopt::linear_programming::io::mps_data_model_t<int, float> parse_mps_fast_file(
  const std::string& path, FileReadMethod read_method);
template cuopt::linear_programming::io::mps_data_model_t<int, double> parse_mps_fast_file(
  const std::string& path, FileReadMethod read_method);
template cuopt::linear_programming::io::mps_data_model_t<int64_t, float> parse_mps_fast_file(
  const std::string& path, FileReadMethod read_method);
template cuopt::linear_programming::io::mps_data_model_t<int64_t, double> parse_mps_fast_file(
  const std::string& path, FileReadMethod read_method);

}  // namespace mps_fast
