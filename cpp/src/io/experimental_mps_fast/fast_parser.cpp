// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights
// reserved. SPDX-License-Identifier: Apache-2.0

#include "fast_parser.hpp"
#include "fast_parse_primitives.hpp"
#include "file_reader.hpp"
#include "hash_table_smallstr.hpp"
#include "mmap_region.hpp"
#include "mps_section_scanner.hpp"
#include "nvtx_ranges.hpp"

#include <cuda/cmath>
#if defined(MPS_FAST_PERF_COUNTERS) || defined(MPS_FAST_TIMERS)
#include <utilities/perf_counters.hpp>
#endif

#include <sys/mman.h>
#include <unistd.h>

#include <omp.h>
#include <algorithm>
#include <cassert>
#include <cctype>
#include <cerrno>
#include <charconv>
#include <climits>
#include <concepts>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include <file_to_string.hpp>

#define MPS_FAST_COMPACT_ROW_HASH
#define MPS_FAST_THP_PREFAULT

namespace cuopt::linear_programming::io::detail {

static constexpr size_t KiB = 1024;
static constexpr size_t MiB = 1024 * KiB;
static constexpr size_t GiB = 1024 * MiB;

// per-chunk row-count scratch tile for the column parsing workers
// small enough to remain warm in L1
static constexpr size_t COLUMN_ROW_COUNT_BLOCK_ROWS = 4096;
static constexpr int MPS_ROWS_THREAD_CAP            = 16;
static constexpr int MPS_COLUMNS_THREAD_CAP         = 32;
static constexpr int MPS_BOUNDS_THREAD_CAP          = 32;
static constexpr int MPS_NAMES_THREAD_CAP           = 16;
// avoid openmp setup for small bounds sections
static constexpr size_t MPS_BOUNDS_PARALLEL_MIN_BYTES = 256 * MiB;
// ordered-name fallback is cheap enough to parallelize on smaller bounds sections
static constexpr size_t MPS_BOUNDS_ORDERED_HINT_PARALLEL_MIN_BYTES = 8 * MiB;
// lower bound on columns chunk size to avoid tiny parser tasks
static constexpr size_t MPS_COLUMNS_MIN_CHUNK_BYTES = 1 * MiB;
// parser-wide thread cap switch; very small files lose to scheduling overhead
static constexpr size_t MPS_MEDIUM_FILE_THREAD_THRESHOLD_BYTES = 100ull * 1000ull * 1000ull;
// thread caps for small and large files
static constexpr int MPS_SMALL_FILE_THREAD_CAP = 16;
static constexpr int MPS_LARGE_FILE_THREAD_CAP = 32;

static int parser_thread_cap_for_size(size_t bytes)
{
  int size_cap = bytes < MPS_MEDIUM_FILE_THREAD_THRESHOLD_BYTES ? MPS_SMALL_FILE_THREAD_CAP
                                                                : MPS_LARGE_FILE_THREAD_CAP;
  return std::max(1, std::min(size_cap, omp_get_max_threads()));
}

static int phase_thread_count(int phase_cap)
{
  const int available_threads = omp_in_parallel() ? omp_get_num_threads() : omp_get_max_threads();
  return std::max(1, std::min(phase_cap, available_threads));
}

// Arena allocator for the strings (row names, column names) to avoid the dreadful overheads of
// glibc's malloc and std::vector<std::string>
class chunk_name_arena_t {
 public:
  void reserve(size_t bytes)
  {
    if (bytes > next_slab_size_) { next_slab_size_ = bytes; }
  }

  std::string_view copy(std::string_view name)
  {
    char* dst = allocate(name.size() + 1);
    std::memcpy(dst, name.data(), name.size());
    dst[name.size()] = '\0';
    return std::string_view(dst, name.size());
  }

 private:
  struct slab_t {
    std::vector<char> data;
    size_t used = 0;
  };

  char* allocate(size_t bytes)
  {
    if (slabs_.empty() || slabs_.back().used + bytes > slabs_.back().data.size()) {
      size_t capacity = std::max(bytes, next_slab_size_);
      slab_t slab;
      slab.data.resize(capacity);
      slabs_.push_back(std::move(slab));
      next_slab_size_ = std::max(next_slab_size_ * 2, capacity);
    }
    slab_t& slab = slabs_.back();
    char* ptr    = slab.data.data() + slab.used;
    slab.used += bytes;
    return ptr;
  }

  std::vector<slab_t> slabs_;
  size_t next_slab_size_ = 64 * KiB;
};

struct timer_entry_t {
  const char* name;
  double elapsed_ms;
  size_t rss_kb;
  size_t hwm_kb;
};

static std::vector<timer_entry_t>& get_timer_buffer()
{
  static std::vector<timer_entry_t> buffer;
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
#ifdef MPS_FAST_TIMERS
  std::lock_guard<std::mutex> lock(get_timer_mutex());
  auto& buffer = get_timer_buffer();
  for (const auto& entry : buffer) {
    std::fprintf(stderr,
                 "[TIMER] %s: %.3f ms rss_GB=%.3f hwm_GB=%.3f\n",
                 entry.name,
                 entry.elapsed_ms,
                 (double)entry.rss_kb / (double)(GiB / KiB),
                 (double)entry.hwm_kb / (double)(GiB / KiB));
  }
  buffer.clear();
#endif
}

enum class materialize_touch_t {
  write_2mb,
  write_4kb,
};

// instanciate a range using mmap anon pages with hugepage hints, and materialize them
// by touching each to nudge the kernel into invoking its THP mechanism
static void materialize_hugepages([[maybe_unused]] const char* label,
                                  void* data,
                                  size_t bytes,
                                  materialize_touch_t touch)
{
  if (data == nullptr || bytes == 0) return;

  constexpr size_t two_mb = 2 * MiB;
  size_t page_size        = system_page_size();
  uintptr_t start         = reinterpret_cast<uintptr_t>(data);
  uintptr_t end           = start + bytes;
  uintptr_t aligned_start = start & ~(uintptr_t)(page_size - 1);
  uintptr_t aligned_end   = (end + page_size - 1) & ~(uintptr_t)(page_size - 1);
  size_t aligned_bytes    = (size_t)(aligned_end - aligned_start);

  errno = 0;
  madvise((void*)(aligned_start), aligned_bytes, MADV_HUGEPAGE);

  size_t step        = touch == materialize_touch_t::write_2mb ? two_mb : page_size;
  volatile char* ptr = (volatile char*)(data);
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
  scoped_timer_t([[maybe_unused]] const char* name, double* accumulator = nullptr)
#ifdef MPS_FAST_TIMERS
    : name_(name),
      accumulator_(accumulator),
      nvtx_(name, nvtx::color_for_name(name)),
      start_(std::chrono::high_resolution_clock::now()){}
#else
    : accumulator_(accumulator)
  {
  }
#endif

      ~scoped_timer_t()
  {
#ifdef MPS_FAST_TIMERS
    auto end          = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start_).count();
    nvtx_.end();
    if (accumulator_) { *accumulator_ += elapsed_ms; }
    auto [rss_kb, hwm_kb] = current_process_rss_kb();
    std::lock_guard<std::mutex> lock(get_timer_mutex());
    get_timer_buffer().push_back({name_, elapsed_ms, rss_kb, hwm_kb});
#endif
  }

  scoped_timer_t(const scoped_timer_t&)            = delete;
  scoped_timer_t& operator=(const scoped_timer_t&) = delete;

 private:
#ifdef MPS_FAST_TIMERS
  const char* name_;
#endif
  double* accumulator_;
#ifdef MPS_FAST_TIMERS
  nvtx::scoped_range_t nvtx_;
  std::chrono::high_resolution_clock::time_point start_;
#endif
};

class omp_max_active_levels_guard_t {
 public:
  explicit omp_max_active_levels_guard_t(int value) : old_value_(omp_get_max_active_levels())
  {
    omp_set_max_active_levels(value);
  }

  ~omp_max_active_levels_guard_t() { omp_set_max_active_levels(old_value_); }

  omp_max_active_levels_guard_t(const omp_max_active_levels_guard_t&)            = delete;
  omp_max_active_levels_guard_t& operator=(const omp_max_active_levels_guard_t&) = delete;

 private:
  int old_value_ = 0;
};

static inline void error_unknown_row(cursor_t& cursor, const char* row_start, const char* section)
{
  const char* row_end = row_start;
  while (row_end < cursor.end && *row_end > ' ') {
    row_end++;
  }
  cursor.error("unknown row name in %s: %.*s", section, (int)(row_end - row_start), row_start);
}

// Two modes for row/column name lookup:
// - hash: arbitrary names via hash table (rows) or var_names_map (columns)
// - dense_ordered: sequential numeric suffixes like R0001/R0002 or V0/V1
enum class index_mode_t {
  hash,
  dense_ordered,
};

// Every 19-digit decimal string fits in uint64_t; 20+ digits may not and are wildly unlikely in the
// context of dense MPS rows/cols
static constexpr size_t dense_suffix_max_digits = 19;

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
  while (pos > 0 && fp64::is_digit(name[pos - 1])) {
    pos--;
  }
  if (pos == name.size()) { return false; }

  suffix_width = name.size() - pos;
  if (suffix_width > dense_suffix_max_digits) { return false; }

  uint64_t parsed = 0;
  for (size_t i = pos; i < name.size(); ++i) {
    parsed = parsed * 10 + (uint64_t)(name[i] - '0');
  }

  prefix = std::string_view(name.data(), pos);
  value  = parsed;
  return true;
}

// necessary to handle cases like R0001, ..., R2000, ...
static inline bool dense_suffix_is_zero_padded(std::string_view name, size_t suffix_width)
{
  return suffix_width > 1 && name[name.size() - suffix_width] == '0';
}

static inline size_t dense_initial_pad_width(std::string_view name, size_t suffix_width)
{
  return dense_suffix_is_zero_padded(name, suffix_width) ? suffix_width : 0;
}

static inline bool dense_suffix_width_ok(uint64_t value, size_t suffix_width, size_t pad_width)
{
  size_t digits         = decimal_digits_u64(value);
  size_t expected_width = std::max(pad_width, digits);
  return suffix_width == expected_width;
}

struct dense_name_index_t {
  std::string prefix;
  uint64_t min_id  = 0;
  uint64_t max_id  = 0;
  size_t pad_width = 0;

  void reset()
  {
    prefix.clear();
    min_id    = 0;
    max_id    = 0;
    pad_width = 0;
  }

  bool suffix_width_ok(uint64_t value, size_t suffix_width) const
  {
    return dense_suffix_width_ok(value, suffix_width, pad_width);
  }

  size_t lookup(std::string_view name) const
  {
    std::string_view parsed_prefix;
    uint64_t value      = 0;
    size_t suffix_width = 0;
    if (!parse_trailing_u64(name, parsed_prefix, value, suffix_width)) { return SIZE_MAX; }
    if (parsed_prefix != prefix || !suffix_width_ok(value, suffix_width)) { return SIZE_MAX; }
    if (value < min_id || value > max_id) { return SIZE_MAX; }
    return (size_t)(value - min_id);
  }

  void format_name(size_t idx, std::string& out) const
  {
    uint64_t value = min_id + idx;
    char digits_buf[32];
    auto [digits_end, ec] = std::to_chars(digits_buf, digits_buf + sizeof(digits_buf), value);
    if (ec != std::errc()) {
      out.assign(prefix);
      return;
    }
    size_t digits_len = (size_t)(digits_end - digits_buf);
    size_t width      = std::max(pad_width, digits_len);
    out.resize(prefix.size() + width);
    std::memcpy(out.data(), prefix.data(), prefix.size());
    char* suffix = out.data() + prefix.size();
    if (width > digits_len) {
      std::memset(suffix, '0', width - digits_len);
      suffix += width - digits_len;
    }
    std::memcpy(suffix, digits_buf, digits_len);
  }
};

struct dense_observe_state_t {
  bool candidate = true;
  dense_name_index_t index;
  size_t count = 0;
};

static inline void observe_dense_name(bool& candidate,
                                      dense_name_index_t& index,
                                      size_t& observed_count,
                                      std::string_view name,
                                      uint64_t expected_id = std::numeric_limits<uint64_t>::max())
{
  if (!candidate) { return; }

  std::string_view prefix;
  uint64_t value      = 0;
  size_t suffix_width = 0;
  if (!parse_trailing_u64(name, prefix, value, suffix_width)) {
    candidate = false;
    return;
  }

  if (observed_count == 0) {
    index.prefix.assign(prefix);
    index.min_id    = value;
    index.max_id    = value;
    index.pad_width = dense_initial_pad_width(name, suffix_width);
    observed_count  = 1;
    return;
  }

  if (prefix != index.prefix) {
    candidate = false;
    return;
  }

  if (expected_id != std::numeric_limits<uint64_t>::max() && value != expected_id) {
    candidate = false;
    return;
  }

  if (!index.suffix_width_ok(value, suffix_width)) {
    candidate = false;
    return;
  }

  index.max_id = value;
  observed_count++;
}

// Maps MPS row/column names to indices via one of two strategies, chosen per problem:
//
//   * dense_ordered - when every name in a section is a shared prefix followed by a
//     contiguous run of integers (e.g. R0001, R0002, ... or x1, x2, ...). The index is
//     then computed straight from the parsed integer (value - min_id), so no hash table
//     is built or probed. This is the common, fast case for solver-generated models.
//   * hash          - the general fallback (smallstr_hash_table_t) for arbitrary names.
//
// Each section decides its own mode while scanning: it stays a dense_ordered "candidate"
// as long as names keep matching the prefix + consecutive-integer + zero-pad-width rule
// (see observe_dense_name), and the first violation drops it to the hash path. The chosen
// mode lives in row_index_mode / col_index_mode, and every lookup branches on it
// (row_lookup / read_row_lookup vs the dense_ordered variants below). Holding this in mind
// explains most of the paired/dual code paths throughout this file.
template <typename i_t, typename f_t>
struct parse_state_t {
  mps_data_model_t<i_t, f_t>& problem;
  cursor_t& cursor;

  // backed by the input buffer
  std::vector<std::string_view> row_names_sv;
  // backed by the arena allocator
  std::vector<std::string_view> var_names_sv;
  std::vector<chunk_name_arena_t> var_name_arenas;
  std::string_view problem_name_sv;
  std::string_view objective_name_sv;
  // secondary 'N' rows in ROWS — rare; membership distinguishes them from unknown row names
  std::unordered_set<std::string_view> ignored_objective_names;

  // Column name lookup for labels like V0, V1, ...
  index_mode_t col_index_mode = index_mode_t::hash;
  dense_name_index_t col_dense;

  smallstr_hash_table_t row_hash_;

  // Row name lookup for labels like R0001, R0002, ...
  index_mode_t row_index_mode = index_mode_t::hash;
  bool row_dense_candidate    = true;
  dense_name_index_t row_dense;

  // var_names still uses STL (only used in parse_bounds, not as hot)
  std::unordered_map<std::string_view, size_t> var_names_map;

  mmap_region_t temp_A_region;
  mmap_region_t temp_A_indices_region;
  f_t* temp_A                = nullptr;
  i_t* temp_A_indices        = nullptr;
  size_t temp_csr_nnz        = 0;
  bool temp_csr_materialized = false;

  struct bounds_only_var_t {
    f_t lb    = f_t{0};
    f_t ub    = std::numeric_limits<f_t>::infinity();
    char type = 'C';
  };

  // some writers introduce zero-column variables only in BOUNDS.
  std::map<std::string_view, bounds_only_var_t> bounds_only_vars;

  struct qcmatrix_block_t {
    size_t row_idx = SIZE_MAX;
    std::string_view row_name;
    std::vector<std::tuple<i_t, i_t, f_t>> entries;
  };

  std::vector<qcmatrix_block_t> qcmatrix_blocks;

  parse_state_t(mps_data_model_t<i_t, f_t>& p, cursor_t& c) : problem(p), cursor(c) {}

  void init_row_hash_table()
  {
    if (init_row_dense_ordered_table()) { return; }
    init_row_hash_table_impl();
  }

  void observe_objective_row_name(std::string_view name)
  {
    if (objective_name_sv.empty()) {
      objective_name_sv = name;
    } else if (name != objective_name_sv) {
      ignored_objective_names.insert(name);
    }
  }

  bool init_row_dense_ordered_table()
  {
    scoped_timer_t timer("row_dense_finalize");
    size_t n_rows = row_names_sv.size();
    if (!row_dense_candidate || n_rows == 0) { return false; }
    if (row_dense.max_id < row_dense.min_id) { return false; }
    uint64_t dense_count = row_dense.max_id - row_dense.min_id + 1;
    if (dense_count != n_rows) { return false; }

    row_index_mode = index_mode_t::dense_ordered;
    return true;
  }

  // Insert all rows into the hash table. The perf-counter instrumentation is isolated in
  // these two helpers so its #ifdefs do not fragment init_row_hash_table_impl's setup flow;
  // both compile down to a bare insert loop when MPS_FAST_PERF_COUNTERS is off.
  void insert_rows_partitioned(
    int num_threads,
    const std::array<size_t, MPS_ROW_HASH_PARTITIONS + 1>& partition_offsets,
    const std::vector<size_t>& row_order,
    const std::vector<uint32_t>& row_hashes)
  {
    scoped_timer_t timer("row_hash_insert_partitioned");
#ifdef MPS_FAST_PERF_COUNTERS
    std::vector<perf_counter_snapshot_t> perf_snapshots(MPS_ROW_HASH_PARTITIONS);
#endif
#pragma omp parallel for schedule(static) num_threads(num_threads)
    for (int part_id = 0; part_id < (int)MPS_ROW_HASH_PARTITIONS; ++part_id) {
      size_t p = (size_t)part_id;
#ifdef MPS_FAST_PERF_COUNTERS
      thread_perf_counters_t perf_counters;
#endif
      for (size_t pos = partition_offsets[p]; pos < partition_offsets[p + 1]; ++pos) {
        size_t idx = row_order[pos];
        row_hash_.insert_partition(p, row_names_sv[idx], row_hashes[idx], idx);
      }
#ifdef MPS_FAST_PERF_COUNTERS
      perf_snapshots[p] = perf_counters.stop();
#endif
    }
#ifdef MPS_FAST_PERF_COUNTERS
    print_perf_totals("row_hash_insert_partitioned", perf_snapshots);
#endif
  }

  void insert_rows_serial(size_t n_rows)
  {
#ifdef MPS_FAST_PERF_COUNTERS
    thread_perf_counters_t perf_counters;
#endif
    for (size_t idx = 0; idx < n_rows; ++idx) {
      row_hash_.insert_serial(row_names_sv[idx], idx);
    }
#ifdef MPS_FAST_PERF_COUNTERS
    print_perf_totals("row_hash_insert_all", {perf_counters.stop()});
#endif
  }

  void init_row_hash_table_impl()
  {
    scoped_timer_t timer("row_hash_init_total");
    size_t n_rows              = row_names_sv.size();
    const int num_threads      = phase_thread_count(MPS_ROWS_THREAD_CAP);
    const bool use_partitioned = n_rows >= MPS_ROW_HASH_PARTITIONED_MIN_ROWS && num_threads > 1;
#ifdef MPS_FAST_COMPACT_ROW_HASH
    constexpr bool compact_row_hash = true;
#else
    constexpr bool compact_row_hash = false;
#endif
    std::vector<uint32_t> row_hashes;
    std::vector<size_t> row_order;
    std::array<size_t, MPS_ROW_HASH_PARTITIONS> partition_counts      = {};
    std::array<size_t, MPS_ROW_HASH_PARTITIONS + 1> partition_offsets = {};

    if (use_partitioned) {
      scoped_timer_t timer("row_hash_partition_metadata");
      row_hashes.resize(n_rows);
      size_t inline_rows = 0;
      for (size_t idx = 0; idx < n_rows; ++idx) {
        std::string_view name = row_names_sv[idx];
        if (UNLIKELY(name.size() > HASH_KEY_BYTES)) {
          row_hash_.note_long_name(name, idx);
          continue;
        }
        uint32_t hash   = fnv1a_hash(name.data(), name.size());
        row_hashes[idx] = hash;
        ++partition_counts[hash_partition_for(hash)];
        ++inline_rows;
      }

      for (size_t p = 0; p < MPS_ROW_HASH_PARTITIONS; ++p) {
        partition_offsets[p + 1] = partition_offsets[p] + partition_counts[p];
      }

      row_order.resize(inline_rows);
      auto next_offsets = partition_offsets;
      for (size_t idx = 0; idx < n_rows; ++idx) {
        if (UNLIKELY(row_names_sv[idx].size() > HASH_KEY_BYTES)) { continue; }
        size_t part                     = hash_partition_for(row_hashes[idx]);
        row_order[next_offsets[part]++] = idx;
      }
    }

    if (use_partitioned) {
      row_hash_.configure_partitioned_buckets(partition_counts, compact_row_hash);
    } else {
      row_hash_.configure_serial_buckets(n_rows, compact_row_hash);
    }

    {
      scoped_timer_t timer("row_hash_mmap");
      row_hash_.allocate_mmap("row hash table");
    }

#ifdef MPS_FAST_THP_PREFAULT
    {
      scoped_timer_t timer("row_hash_thp_prefault");
      materialize_hugepages("row_names_ht",
                            row_hash_.slots(),
                            row_hash_.region().size(),
                            materialize_touch_t::write_2mb);
    }
#endif

    {
      scoped_timer_t timer("row_hash_insert_all");
      row_hash_.reset_build_probe_stats();
      if (use_partitioned) {
        insert_rows_partitioned(num_threads, partition_offsets, row_order, row_hashes);
      } else {
        insert_rows_serial(n_rows);
      }
      row_hash_.print_build_probe_report(n_rows);
    }

#ifdef MPS_FAST_MADV_COLLAPSE
    {
      scoped_timer_t timer("row_hash_madv_collapse");
      row_hash_.region().advise(MADV_COLLAPSE);
    }
#endif
  }

  size_t row_lookup(std::string_view name) const
  {
    if (LIKELY(row_index_mode == index_mode_t::dense_ordered)) { return row_dense.lookup(name); }
    return row_hash_.lookup(name);
  }

  size_t read_row_lookup_dense_ordered(cursor_t& cursor) const
  {
    const char* start = cursor.ptr;
    const char* p     = start;

    size_t prefix_len = row_dense.prefix.size();
    if (prefix_len > 0) {
      if ((size_t)(cursor.end - p) < prefix_len ||
          std::memcmp(p, row_dense.prefix.data(), prefix_len) != 0) {
        cursor.read_field();
        return SIZE_MAX;
      }
      p += prefix_len;
    }

    const char* digits_start = p;
    uint64_t value           = 0;
    fp64::parse_u64_digits_advance(p, cursor.end, value);

    size_t suffix_width = (size_t)(p - digits_start);
    if (suffix_width == 0 || suffix_width > dense_suffix_max_digits || p >= cursor.end ||
        *p > ' ' || !row_dense.suffix_width_ok(value, suffix_width) || value < row_dense.min_id ||
        value > row_dense.max_id) {
      cursor.ptr = start;
      cursor.read_field();
      return SIZE_MAX;
    }

    cursor.ptr = p;
    cursor.skip_ws();
    return (size_t)(value - row_dense.min_id);
  }

  size_t read_row_lookup(cursor_t& cursor) const
  {
    if (LIKELY(row_index_mode == index_mode_t::dense_ordered)) {
      return read_row_lookup_dense_ordered(cursor);
    }

    auto row_name = cursor.read_field();
    return row_hash_.lookup(row_name);
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
  if (!state.cursor.eol()) { state.problem_name_sv = state.cursor.read_rest_of_line_trimmed(); }
  expect_eol(state.cursor);
}

template <typename i_t, typename f_t>
static void parse_objsense_section(parse_state_t<i_t, f_t>& state)
{
  scoped_timer_t timer("parse_objsense");
  if (accept(state.cursor, "OBJSENSE")) {
    if (state.cursor.eol()) { expect_eol(state.cursor); }
    auto sense = state.cursor.read_field();
    if (sense == "MIN" || sense == "MINIMIZE") {
      state.problem.maximize_ = false;
    } else if (sense == "MAX" || sense == "MAXIMIZE") {
      state.problem.maximize_ = true;
    } else {
      state.cursor.error("expected MIN/MAX or MINIMIZE/MAXIMIZE, got '%s'", sense.data());
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

struct row_chunk_boundary_t {
  const char* start;
  const char* end;
};

struct row_chunk_info_t {
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
  p = cursor_t::simd_scan<skip_whitespace>(p, end);
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
  p        = cursor_t::simd_scan<skip_whitespace>(p, end);

  const char* name_start = p;
  p                      = cursor_t::simd_scan<until_whitespace>(p, end);
  if (name_start == p) { return false; }
  row_name = std::string_view(name_start, (size_t)(p - name_start));

  // ROWS only uses fields 1-2. Fields 3-6 are ignored by the MPS spec, and
  // field 3 may start with '$' to comment the rest of the record.
  // could be SIMD'd, but in practice the newline is right after the row name
  p = rows_find_next_line(p, end);
  return true;
}

// row chunks are established based on byte count, thus boundaries can land in the middle of a row
// this cleans up chunks to have row line boundaries
static std::vector<row_chunk_boundary_t> compute_row_chunk_boundaries(const char* rows_start,
                                                                      const char* rows_end,
                                                                      int num_threads)
{
  scoped_timer_t timer("rows_compute_chunk_boundaries");

  std::vector<row_chunk_boundary_t> boundaries((size_t)num_threads);
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

// reads the row section in chunks and inserts into the worker's hash table partition
// Parallel ROWS parser: count constraints per chunk, prefix-sum, then fill the output arrays
// in parallel (with per-chunk dense-name reconciliation at the end). Must keep the same line
// grammar as its serial twin parse_rows_section_serial_impl; parse_rows_section chooses between
// them by size. Returns false if a chunk hit a malformed line (nothing committed for the fill
// pass), so the caller can reset and retry serially for clean error reporting.
template <typename i_t, typename f_t>
static bool parse_rows_section_parallel_impl(parse_state_t<i_t, f_t>& state,
                                             const char* rows_start,
                                             const char* rows_end,
                                             int num_threads)
{
  scoped_timer_t timer("parse_rows_parallel");

  auto boundaries = compute_row_chunk_boundaries(rows_start, rows_end, num_threads);
  std::vector<row_chunk_info_t> infos((size_t)num_threads);

  {
    scoped_timer_t timer("rows_count_parallel");
#pragma omp parallel for num_threads(num_threads)
    for (int t = 0; t < num_threads; ++t) {
      MPS_NVTX_RANGE(std::string("rows_count_chunk ") + std::to_string(t), nvtx::colors::rows);
      const char* p   = boundaries[(size_t)t].start;
      const char* end = boundaries[(size_t)t].end;
      row_chunk_info_t info;

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

  if (std::any_of(
        infos.begin(), infos.end(), [](const row_chunk_info_t& info) { return info.malformed; })) {
    return false;
  }

  // prefix sum to do a paralle scatter of every row entries into the global output arrays
  std::vector<size_t> offsets((size_t)num_threads + 1, 0);
  {
    scoped_timer_t timer("rows_prefix_sum");
    for (int t = 0; t < num_threads; ++t) {
      offsets[(size_t)t + 1] = offsets[(size_t)t] + infos[(size_t)t].constraints;
    }
  }

  size_t total_rows = offsets[(size_t)num_threads];
  if (UNLIKELY(total_rows > (size_t)INT_MAX)) {
    state.cursor.error("fast MPS parser requires <= INT_MAX rows, got %zu", total_rows);
  }
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
      if (name != state.objective_name_sv) { state.ignored_objective_names.insert(name); }
    }
  }

  bool dense_candidate = total_rows > 0;
  std::string_view dense_prefix;
  uint64_t dense_base_id = 0;
  size_t dense_pad_width = 0;

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
      dense_base_id   = first_value;
      dense_pad_width = dense_initial_pad_width(first_name, first_suffix_width);
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
      dense_name_index_t dense_index;
      if (local_dense_ok) {
        dense_index.prefix.assign(dense_prefix);
        dense_index.min_id    = dense_base_id;
        dense_index.max_id    = dense_base_id;
        dense_index.pad_width = dense_pad_width;
      }

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
          size_t observed_count = out;
          observe_dense_name(
            local_dense_ok, dense_index, observed_count, row_name, dense_base_id + out);
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
      state.row_dense.prefix.assign(dense_prefix);
      state.row_dense.min_id    = dense_base_id;
      state.row_dense.max_id    = dense_base_id + total_rows - 1;
      state.row_dense.pad_width = dense_pad_width;
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

    auto row_name = state.cursor.read_field();
    // ROWS fields after the row name are unused; tolerate annotations/comments there.
    state.cursor.skip_to_eol();

    // 'N' type is the objective row - store its name but don't add to constraints
    if (row_type == 'N') {
      state.observe_objective_row_name(row_name);
    } else {
      size_t row_idx = state.row_names_sv.size();
      state.row_names_sv.push_back(row_name);
      observe_dense_name(
        state.row_dense_candidate,
        state.row_dense,
        row_idx,
        row_name,
        row_idx == 0 ? std::numeric_limits<uint64_t>::max() : state.row_dense.min_id + row_idx);
      state.problem.row_types_.push_back(row_type);
    }
    expect_eol(state.cursor);
  }
  if (UNLIKELY(state.row_names_sv.size() > (size_t)INT_MAX)) {
    state.cursor.error("fast MPS parser requires <= INT_MAX rows, got %zu",
                       state.row_names_sv.size());
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
    if (rows_bytes >= 512 * MiB && num_threads > 1) {
      parsed_parallel =
        parse_rows_section_parallel_impl<i_t, f_t>(state, state.cursor.ptr, rows_end, num_threads);
      // serial fallback in case a likely malformed chunk has been encounter
      // makes error reporting much easier
      if (!parsed_parallel) {
        state.row_names_sv.clear();
        state.problem.row_types_.clear();
        state.row_dense_candidate = true;
        state.row_dense.reset();
        state.cursor.ptr = rows_start;
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

// Columns parser

// integer variable markers
struct marker_info_t {
  enum Type { INTORG, INTEND };
  Type type;
  size_t after_local_var_idx;  // SIZE_MAX means "before first variable"
};

struct row_count_block_t {
  size_t block_id       = 0;
  size_t storage_offset = 0;
};

// Each column parsing worker owns chunks of the global CSC which are parsed in parallel and then
// later scattered into the final CSR
struct chunk_result_t {
  std::vector<double> values;
  std::vector<uint32_t> row_indices;
  std::vector<size_t> col_offsets;
  std::vector<std::string_view> var_names;
  chunk_name_arena_t var_name_arena;
  std::vector<marker_info_t> markers;
  std::vector<std::pair<size_t, double>> objective_entries;  // local_col_idx -> coefficient
  // COLUMNS is parsed as chunk-local CSC. To build the global CSR, each chunk needs row counts
  // first, then row-local write cursors for scatter. Store those counts only for touched
  // 4096-row blocks instead of allocating a dense chunks*n_rows matrix
  // The same slots are rewritten as write cursors after the global CSR row offsets are known
  std::vector<int64_t> row_count_storage;
  std::vector<row_count_block_t> row_count_blocks;
  std::vector<int32_t> row_count_block_dir;
  dense_observe_state_t dense_col_stats;
};

struct chunk_boundary_t {
  const char* start;
  const char* end;
};

struct bounds_chunk_boundary_t {
  const char* start;
  const char* end;
};

// enables representing row counts per chunk as a sparse representation w/ 4096 granularity
// works well since nnzs are often clustered around the same matrix blocks
static inline int64_t& column_row_count_slot(chunk_result_t& result, size_t row_idx)
{
  size_t block_id   = row_idx / COLUMN_ROW_COUNT_BLOCK_ROWS;
  size_t local      = row_idx - block_id * COLUMN_ROW_COUNT_BLOCK_ROWS;
  int32_t block_pos = result.row_count_block_dir[block_id];
  if (UNLIKELY(block_pos < 0)) {
    block_pos                            = (int32_t)result.row_count_blocks.size();
    result.row_count_block_dir[block_id] = block_pos;
    row_count_block_t block;
    block.block_id       = block_id;
    block.storage_offset = result.row_count_storage.size();
    result.row_count_storage.resize(block.storage_offset + COLUMN_ROW_COUNT_BLOCK_ROWS, 0);
    result.row_count_blocks.push_back(std::move(block));
  }
  return result
    .row_count_storage[result.row_count_blocks[(size_t)block_pos].storage_offset + local];
}

static bool dense_col_chunk_padding_compatible(const dense_observe_state_t& stats,
                                               size_t global_pad_width)
{
  if (global_pad_width > 0) {
    return stats.index.pad_width == global_pad_width ||
           (stats.index.pad_width == 0 &&
            decimal_digits_u64(stats.index.min_id) >= global_pad_width);
  }
  return stats.index.pad_width == 0;
}

static const char* find_next_line(const char* p, const char* end)
{
  while (p < end && *p != '\n')
    p++;
  if (p < end) p++;
  return p;
}

static std::string_view peek_bounds_line_var_name(const char* line_start, const char* end)
{
  const char* p = line_start;
  for (int field = 0; field < 2; ++field) {
    while (p < end && *p <= ' ' && *p != '\n')
      p++;
    while (p < end && *p > ' ')
      p++;
  }
  while (p < end && *p <= ' ' && *p != '\n')
    p++;
  const char* var_start = p;
  while (p < end && *p > ' ')
    p++;
  return std::string_view(var_start, (size_t)(p - var_start));
}

static const char* find_line_start(const char* section_start, const char* p)
{
  while (p > section_start && p[-1] != '\n')
    --p;
  return p;
}

static std::vector<bounds_chunk_boundary_t> compute_bounds_chunk_boundaries(
  const char* section_start, const char* section_end, int num_threads)
{
  scoped_timer_t timer("bounds_compute_chunk_boundaries");

  const size_t total_size = (size_t)(section_end - section_start);
  const size_t chunk_size = total_size / (size_t)num_threads;

  std::vector<bounds_chunk_boundary_t> boundaries((size_t)num_threads);
  boundaries[0].start = section_start;
  for (int t = 0; t < num_threads; ++t) {
    if (t == num_threads - 1) {
      boundaries[(size_t)t].end = section_end;
    } else {
      const char* boundary =
        find_next_line(section_start + (size_t)(t + 1) * chunk_size, section_end);

      // Keep consecutive BOUNDS records for the same variable in one chunk.
      // Then each thread owns full LO/UP-style groups and can apply file order locally.
      while (boundary < section_end) {
        const char* prev_line = find_line_start(section_start, boundary - 1);
        const auto prev_var   = peek_bounds_line_var_name(prev_line, section_end);
        const auto next_var   = peek_bounds_line_var_name(boundary, section_end);
        if (prev_var.empty() || next_var.empty() || prev_var != next_var) { break; }
        boundary = find_next_line(boundary, section_end);
      }

      boundaries[(size_t)t].end       = boundary;
      boundaries[(size_t)t + 1].start = boundary;
    }
  }
  return boundaries;
}

static std::vector<chunk_boundary_t> compute_chunk_boundaries(const char* columns_start,
                                                              const char* columns_end,
                                                              int num_threads)
{
  scoped_timer_t timer("compute_chunk_boundaries");

  size_t total_size = (size_t)(columns_end - columns_start);
  size_t chunk_size = total_size / (size_t)num_threads;

  std::vector<chunk_boundary_t> boundaries(num_threads);

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
      std::string_view col_name = cursor_t::peek_field_at(line_start, columns_end);

      // Scan forward until column name changes (to avoid splitting a column)
      const char* boundary = line_start;
      while (boundary < columns_end) {
        const char* next_line = find_next_line(boundary, columns_end);
        if (next_line >= columns_end) break;

        std::string_view next_col = cursor_t::peek_field_at(next_line, columns_end);
        if (next_col != col_name && !next_col.empty() && next_col[0] != '\'') {
          // Found a column transition. Marker-state fixup later handles any split near markers.
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
static chunk_result_t parse_columns_chunk(const char* chunk_start,
                                          const char* chunk_end,
                                          const parse_state_t<i_t, f_t>& state)
{
  chunk_result_t result;

  if (chunk_start >= chunk_end) {
    result.col_offsets.push_back(0);
    return result;
  }

  size_t chunk_size     = (size_t)(chunk_end - chunk_start);
  size_t estimated_nnz  = chunk_size / 100;
  size_t estimated_cols = estimated_nnz / 10;
  if (UNLIKELY(state.problem.n_constraints_ > (i_t)std::numeric_limits<int32_t>::max())) {
    state.cursor.error("fast COLUMNS path requires <= INT32_MAX rows for chunk row indices");
  }
  result.values.reserve(estimated_nnz);
  result.row_indices.reserve(estimated_nnz);
  result.col_offsets.reserve(estimated_cols + 1);
  result.var_names.reserve(estimated_cols);
  result.var_name_arena.reserve(std::max<size_t>(4096, estimated_cols * 16));
  result.objective_entries.reserve(estimated_cols);
  size_t n_row_blocks =
    cuda::ceil_div((size_t)state.problem.n_constraints_, COLUMN_ROW_COUNT_BLOCK_ROWS);
  result.row_count_block_dir.resize(n_row_blocks, -1);
  size_t estimated_touched_blocks = std::min(n_row_blocks, std::max<size_t>(16, estimated_nnz));
  result.row_count_blocks.reserve(estimated_touched_blocks);
  result.row_count_storage.reserve(estimated_touched_blocks * COLUMN_ROW_COUNT_BLOCK_ROWS);

  cursor_t cursor(chunk_start, (size_t)(chunk_end - chunk_start));
  std::string_view prev_var_name = "";

  cursor.skip_ws();

  while (!cursor.done()) {
    if (UNLIKELY(*cursor.ptr == 'R')) {
      auto next = cursor.peek_field();
      // RHS section is mandatory right after COLUMNS section
      if (next == "RHS") { break; }
    }

    auto [var_name, field2] = cursor.read_two_fields();
    if (UNLIKELY(!field2.empty() && field2[0] == '$')) {
      cursor.skip_to_eol();
      expect_eol(cursor);
      continue;
    }

    // Check for integer marker
    if (UNLIKELY(field2[0] == '\'' && field2 == "'MARKER'")) {
      auto marker_type = cursor.read_field();

      marker_info_t marker;
      marker.after_local_var_idx =
        result.var_names.empty() ? SIZE_MAX : result.var_names.size() - 1;

      if (marker_type == "'INTORG'") {
        marker.type = marker_info_t::INTORG;
      } else if (marker_type == "'INTEND'") {
        marker.type = marker_info_t::INTEND;
      } else {
        cursor.error("unknown integer marker type in COLUMNS: %.*s",
                     (int)marker_type.size(),
                     marker_type.data());
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
    if (cursor.ptr + 1 < cursor.end && fp64::is_digit(cursor.ptr[0]) &&
        (cursor.ptr[1] == '\n' || cursor.ptr[1] == '\r')) {
      value = sign * (cursor.ptr[0] - '0');
      cursor.advance(1);
    } else {
      value = sign * fp64::parse_fp64_advance(cursor.ptr, cursor.end);
    }
    // usually EOL directly follows
    if (UNLIKELY(!cursor.eol())) { cursor.skip_ws(); }
    accept_comment(cursor);

    if (prev_var_name != var_name) {
      std::string_view owned_var_name = result.var_name_arena.copy(var_name);
      result.var_names.push_back(owned_var_name);
      observe_dense_name(result.dense_col_stats.candidate,
                         result.dense_col_stats.index,
                         result.dense_col_stats.count,
                         owned_var_name);
      result.col_offsets.push_back(result.values.size());
      prev_var_name = owned_var_name;
    }

    auto add_entry = [&](std::string_view rn, double val) {
      size_t row_idx = state.row_lookup(rn);
      if (LIKELY(row_idx != SIZE_MAX)) {
        assert(row_idx <= (size_t)std::numeric_limits<int32_t>::max());
        result.values.push_back(val);
        result.row_indices.push_back((uint32_t)row_idx);
        column_row_count_slot(result, row_idx)++;
      } else if (LIKELY(rn == state.objective_name_sv)) {
        result.objective_entries.push_back({result.var_names.size() - 1, val});
      } else if (state.ignored_objective_names.count(rn)) {
        return;
      } else {
        cursor.error("unknown row name in COLUMNS: %.*s", (int)rn.size(), rn.data());
      }
    };

    add_entry(row_name, value);

    // Optional second entry on same line
    if (!cursor.eol()) {
      auto row_name2 = cursor.read_field();
      if (UNLIKELY(!row_name2.empty() && row_name2[0] == '$')) {
        cursor.skip_to_eol();
        expect_eol(cursor);
        continue;
      }
      double value2 = fp64::parse_fp64_advance(cursor.ptr, cursor.end);
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
template <typename i_t>
struct column_merge_shape_t {
  int num_chunks = 0;
  i_t n_rows     = 0;
  std::vector<size_t> global_col_offset;
  size_t total_cols = 0;
  size_t total_nnz  = 0;
};

template <typename i_t>
static column_merge_shape_t<i_t> compute_column_merge_shape(
  const std::vector<chunk_result_t>& chunks, i_t n_rows)
{
  column_merge_shape_t<i_t> shape;
  shape.num_chunks = (int)chunks.size();
  shape.n_rows     = n_rows;
  shape.global_col_offset.resize((size_t)shape.num_chunks + 1);
  {
    scoped_timer_t timer("columns_global_offsets");
    for (int t = 0; t < shape.num_chunks; t++) {
      shape.global_col_offset[(size_t)t + 1] =
        shape.global_col_offset[(size_t)t] + chunks[(size_t)t].var_names.size();
      shape.total_nnz += chunks[(size_t)t].values.size();
    }
  }
  shape.total_cols = shape.global_col_offset[(size_t)shape.num_chunks];
  if constexpr (std::numeric_limits<i_t>::max() < std::numeric_limits<int64_t>::max()) {
    const size_t index_max = (size_t)std::numeric_limits<i_t>::max();
    if (shape.total_nnz > index_max) {
      mps_parser_fail(error_type_t::RuntimeError,
                      "fast MPS parser requires 64-bit indices: nnz=%zu exceeds index max=%zu",
                      shape.total_nnz,
                      index_max);
    }
    if (shape.total_cols > index_max || (size_t)n_rows > index_max) {
      mps_parser_fail(error_type_t::RuntimeError,
                      "fast MPS parser requires 64-bit indices: rows=%zu cols=%zu exceed index "
                      "max=%zu",
                      (size_t)n_rows,
                      shape.total_cols,
                      index_max);
    }
  }
  return shape;
}

template <typename i_t, typename f_t>
static void detect_dense_column_metadata(parse_state_t<i_t, f_t>& state,
                                         const std::vector<chunk_result_t>& chunks,
                                         const column_merge_shape_t<i_t>& shape)
{
  scoped_timer_t timer("columns_dense_metadata");
  bool dense_ok   = shape.total_cols > 0;
  bool have_first = false;
  std::string_view dense_prefix;
  uint64_t expected_next_id = 0;
  uint64_t dense_min_id     = 0;
  uint64_t dense_max_id     = 0;
  size_t dense_pad_width    = 0;

  for (int t = 0; t < shape.num_chunks && dense_ok; ++t) {
    const auto& stats = chunks[(size_t)t].dense_col_stats;
    if (stats.count == 0) { continue; }
    if (!stats.candidate || stats.count != chunks[(size_t)t].var_names.size()) {
      dense_ok = false;
      break;
    }
    if (!have_first) {
      have_first       = true;
      dense_prefix     = stats.index.prefix;
      expected_next_id = stats.index.min_id;
      dense_min_id     = stats.index.min_id;
      dense_pad_width  = stats.index.pad_width;
    }
    if (stats.index.prefix != dense_prefix || stats.index.min_id != expected_next_id ||
        !dense_col_chunk_padding_compatible(stats, dense_pad_width)) {
      dense_ok = false;
      break;
    }
    if (stats.index.max_id < stats.index.min_id ||
        stats.index.max_id - stats.index.min_id + 1 != stats.count) {
      dense_ok = false;
      break;
    }
    dense_max_id = stats.index.max_id;
    if (stats.index.max_id == std::numeric_limits<uint64_t>::max()) {
      dense_ok = false;
      break;
    }
    expected_next_id = stats.index.max_id + 1;
  }

  if (!have_first || dense_max_id < dense_min_id ||
      dense_max_id - dense_min_id + 1 != shape.total_cols) {
    dense_ok = false;
  }

  state.col_index_mode = dense_ok ? index_mode_t::dense_ordered : index_mode_t::hash;
  if (dense_ok) {
    state.col_dense.prefix.assign(dense_prefix);
    state.col_dense.min_id    = dense_min_id;
    state.col_dense.max_id    = dense_max_id;
    state.col_dense.pad_width = dense_pad_width;
  }
}

template <typename i_t, typename f_t>
static std::vector<i_t> build_csr_row_offsets(parse_state_t<i_t, f_t>& state,
                                              const std::vector<chunk_result_t>& chunks,
                                              const column_merge_shape_t<i_t>& shape)
{
  std::vector<i_t> global_row_counts((size_t)shape.n_rows, 0);
  {
    scoped_timer_t timer("columns_sum_row_counts");
    for (int t = 0; t < shape.num_chunks; t++) {
      for (const auto& block : chunks[(size_t)t].row_count_blocks) {
        const int64_t* block_counts =
          chunks[(size_t)t].row_count_storage.data() + block.storage_offset;
        size_t row_base    = block.block_id * COLUMN_ROW_COUNT_BLOCK_ROWS;
        size_t block_limit = std::min(COLUMN_ROW_COUNT_BLOCK_ROWS, (size_t)shape.n_rows - row_base);
        for (size_t local = 0; local < block_limit; ++local) {
          global_row_counts[row_base + local] += (i_t)block_counts[local];
        }
      }
    }
  }
  {
    scoped_timer_t timer("columns_build_row_offsets");
    state.problem.A_offsets_.resize((size_t)shape.n_rows + 1);
    state.problem.A_offsets_[0] = 0;
    for (i_t r = 0; r < shape.n_rows; r++) {
      state.problem.A_offsets_[(size_t)r + 1] =
        state.problem.A_offsets_[(size_t)r] + global_row_counts[(size_t)r];
    }
  }
  return global_row_counts;
}

template <typename i_t>
static void convert_counts_to_write_positions(std::vector<chunk_result_t>& chunks,
                                              const column_merge_shape_t<i_t>& shape,
                                              const std::vector<i_t>& row_offsets,
                                              std::vector<i_t>& global_row_counts)
{
  scoped_timer_t timer("columns_counts_to_write_positions");
  std::fill(global_row_counts.begin(), global_row_counts.end(), i_t{0});
  for (int t = 0; t < shape.num_chunks; t++) {
    for (auto& block : chunks[(size_t)t].row_count_blocks) {
      int64_t* block_counts = chunks[(size_t)t].row_count_storage.data() + block.storage_offset;
      size_t row_base       = block.block_id * COLUMN_ROW_COUNT_BLOCK_ROWS;
      size_t block_limit = std::min(COLUMN_ROW_COUNT_BLOCK_ROWS, (size_t)shape.n_rows - row_base);
      for (size_t local = 0; local < block_limit; ++local) {
        int64_t count = block_counts[local];
        if (count == 0) continue;
        size_t row          = row_base + local;
        i_t pos             = row_offsets[row] + global_row_counts[row];
        block_counts[local] = (int64_t)pos;
        global_row_counts[row] += (i_t)count;
      }
    }
  }
}

static void materialize_chunk_row_count_storage(std::vector<chunk_result_t>& chunks,
                                                int num_threads)
{
  scoped_timer_t timer("columns_row_count_storage_hugepages");
#pragma omp parallel for num_threads(num_threads)
  for (int t = 0; t < (int)chunks.size(); ++t) {
    materialize_vector_hugepages("column_row_count_storage",
                                 chunks[(size_t)t].row_count_storage,
                                 materialize_touch_t::write_2mb);
  }
}

template <typename i_t, typename f_t>
static void allocate_column_outputs(parse_state_t<i_t, f_t>& state,
                                    const column_merge_shape_t<i_t>& shape)
{
  scoped_timer_t timer("allocate_temp_csr_arrays");
  size_t values_bytes  = shape.total_nnz * sizeof(f_t);
  size_t indices_bytes = shape.total_nnz * sizeof(i_t);
  state.temp_csr_nnz   = shape.total_nnz;

#pragma omp parallel sections num_threads(4)
  {
#pragma omp section
    {
      state.temp_A_region = mmap_region_t::anonymous(
        std::max<size_t>(values_bytes, 1), PROT_READ | PROT_WRITE, MAP_PRIVATE, "temp CSR values");
      state.temp_A = (f_t*)state.temp_A_region.data();
      state.temp_A_region.advise(MADV_HUGEPAGE);
    }
#pragma omp section
    {
      state.temp_A_indices_region = mmap_region_t::anonymous(std::max<size_t>(indices_bytes, 1),
                                                             PROT_READ | PROT_WRITE,
                                                             MAP_PRIVATE,
                                                             "temp CSR column indices");
      state.temp_A_indices        = (i_t*)state.temp_A_indices_region.data();
      state.temp_A_indices_region.advise(MADV_HUGEPAGE);
    }
#pragma omp section
    {
      if (state.col_index_mode != index_mode_t::dense_ordered) {
        state.var_name_arenas.clear();
        state.var_name_arenas.resize((size_t)shape.num_chunks);
        state.var_names_sv.resize(shape.total_cols);
      }
    }
#pragma omp section
    {
      state.problem.var_types_.resize(shape.total_cols);
    }
  }
}

template <typename i_t, typename f_t>
static void scatter_column_chunks_to_csr(parse_state_t<i_t, f_t>& state,
                                         std::vector<chunk_result_t>& chunks,
                                         const column_merge_shape_t<i_t>& shape,
                                         int num_threads)
{
  scoped_timer_t timer("scatter_into_csr");
  {
    scoped_timer_t matrix_timer("scatter_matrix_entries");
#ifdef MPS_FAST_PERF_COUNTERS
    std::vector<perf_counter_snapshot_t> perf_snapshots((size_t)shape.num_chunks);
#endif
#pragma omp parallel for num_threads(num_threads)
    for (int t = 0; t < shape.num_chunks; t++) {
#ifdef MPS_FAST_PERF_COUNTERS
      thread_perf_counters_t perf_counters;
#endif
      auto& chunk = chunks[(size_t)t];
      for (size_t local_col = 0; local_col < chunk.var_names.size(); local_col++) {
        i_t global_col   = (i_t)(shape.global_col_offset[(size_t)t] + local_col);
        size_t col_start = chunk.col_offsets[local_col];
        size_t col_end   = chunk.col_offsets[local_col + 1];
        for (size_t idx = col_start; idx < col_end; idx++) {
          i_t row                    = (i_t)chunk.row_indices[idx];
          size_t row_idx             = (size_t)row;
          size_t block_id            = row_idx / COLUMN_ROW_COUNT_BLOCK_ROWS;
          size_t local               = row_idx - block_id * COLUMN_ROW_COUNT_BLOCK_ROWS;
          int32_t block_pos          = chunk.row_count_block_dir[block_id];
          row_count_block_t& block   = chunk.row_count_blocks[(size_t)block_pos];
          int64_t& write_pos         = chunk.row_count_storage[block.storage_offset + local];
          i_t dest                   = (i_t)write_pos++;
          state.temp_A[dest]         = (f_t)chunk.values[idx];
          state.temp_A_indices[dest] = global_col;
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

  if (state.col_index_mode != index_mode_t::dense_ordered) {
    scoped_timer_t names_timer("scatter_var_names");
#pragma omp parallel for num_threads(num_threads)
    for (int t = 0; t < shape.num_chunks; t++) {
      chunk_name_arena_t& arena = state.var_name_arenas[(size_t)t];
      arena.reserve(std::max<size_t>(4096, chunks[(size_t)t].var_names.size() * 16));
      for (size_t i = 0; i < chunks[(size_t)t].var_names.size(); i++) {
        state.var_names_sv[shape.global_col_offset[(size_t)t] + i] =
          arena.copy(chunks[(size_t)t].var_names[i]);
      }
    }
  } else {
    scoped_timer_t names_timer("scatter_var_names");
  }
}

struct global_marker_t {
  marker_info_t::Type type;
  size_t global_var_idx;
};

template <typename i_t, typename f_t>
static void apply_column_integer_markers(parse_state_t<i_t, f_t>& state,
                                         const std::vector<chunk_result_t>& chunks,
                                         const column_merge_shape_t<i_t>& shape)
{
  scoped_timer_t timer("columns_apply_markers");
  std::vector<global_marker_t> all_markers;
  for (int t = 0; t < shape.num_chunks; t++) {
    for (const auto& m : chunks[(size_t)t].markers) {
      global_marker_t gm;
      gm.type = m.type;
      gm.global_var_idx =
        m.after_local_var_idx == SIZE_MAX
          ? (shape.global_col_offset[(size_t)t] > 0 ? shape.global_col_offset[(size_t)t] - 1
                                                    : SIZE_MAX)
          : shape.global_col_offset[(size_t)t] + m.after_local_var_idx;
      all_markers.push_back(gm);
    }
  }

  std::stable_sort(all_markers.begin(), all_markers.end(), [](const auto& a, const auto& b) {
    if (a.global_var_idx == SIZE_MAX && b.global_var_idx != SIZE_MAX) return true;
    if (b.global_var_idx == SIZE_MAX && a.global_var_idx != SIZE_MAX) return false;
    return a.global_var_idx < b.global_var_idx;
  });

  bool is_integer   = false;
  size_t marker_idx = 0;
  for (size_t v = 0; v < shape.total_cols; v++) {
    while (marker_idx < all_markers.size() && (all_markers[marker_idx].global_var_idx == SIZE_MAX ||
                                               all_markers[marker_idx].global_var_idx < v)) {
      is_integer = all_markers[marker_idx].type == marker_info_t::INTORG;
      marker_idx++;
    }
    state.problem.var_types_[v] = is_integer ? 'I' : 'C';
  }
}

template <typename i_t, typename f_t>
static void assign_column_objective_entries(parse_state_t<i_t, f_t>& state,
                                            const std::vector<chunk_result_t>& chunks,
                                            const column_merge_shape_t<i_t>& shape)
{
  scoped_timer_t timer("columns_objective_entries");
  state.problem.c_.resize(shape.total_cols, f_t{0});
  for (int t = 0; t < shape.num_chunks; t++) {
    for (const auto& [local_col, coeff] : chunks[(size_t)t].objective_entries) {
      size_t global_col = shape.global_col_offset[(size_t)t] + local_col;
      if (global_col < shape.total_cols) { state.problem.c_[global_col] = (f_t)coeff; }
    }
  }
}

template <typename i_t, typename f_t>
static void merge_chunk_results_to_csr(parse_state_t<i_t, f_t>& state,
                                       std::vector<chunk_result_t>& chunks,
                                       int num_threads)
{
  scoped_timer_t timer("merge_chunks_to_csr");
  if (chunks.empty()) return;

  auto shape = compute_column_merge_shape<i_t>(chunks, state.problem.n_constraints_);
  detect_dense_column_metadata(state, chunks, shape);
  auto global_row_counts = build_csr_row_offsets(state, chunks, shape);
  convert_counts_to_write_positions(chunks, shape, state.problem.A_offsets_, global_row_counts);
  materialize_chunk_row_count_storage(chunks, num_threads);
  allocate_column_outputs(state, shape);
  scatter_column_chunks_to_csr(state, chunks, shape, num_threads);
  apply_column_integer_markers(state, chunks, shape);
  assign_column_objective_entries(state, chunks, shape);

  state.problem.n_vars_ = (i_t)shape.total_cols;
  state.problem.nnz_    = (i_t)shape.total_nnz;
}

template <typename i_t, typename f_t>
static void materialize_problem_csr(parse_state_t<i_t, f_t>& state)
{
  scoped_timer_t timer("materialize_problem_csr");
  size_t nnz       = state.temp_csr_nnz;
  int copy_threads = 2;
  copy_threads     = std::max(1, std::min(copy_threads, MPS_LARGE_FILE_THREAD_CAP));

  int resize_threads = copy_threads > 1 ? 2 : 1;
#pragma omp parallel sections num_threads(resize_threads)
  {
#pragma omp section
    {
      state.problem.A_.resize(nnz);
    }
#pragma omp section
    {
      state.problem.A_indices_.resize(nnz);
    }
  }

  size_t value_bytes = nnz * sizeof(f_t);
  size_t index_bytes = nnz * sizeof(i_t);
  size_t total_bytes = value_bytes + index_bytes;
  // Copy A_ and A_indices overlapping with the other phases
  // this hides the latency costs of heap alloc and default init with other parsing/IO
  // instead of making it blocking for the column parse
  // TODO: just have A_ and A_indices_ be mmap anon allocs directly in the mps_data_model_t
  // but that'd require careful work around avoiding breaking changes and the API esp cython stuff
  if (total_bytes != 0) {
#pragma omp parallel for num_threads(copy_threads) schedule(static)
    for (int t = 0; t < copy_threads; ++t) {
      size_t begin = (total_bytes * (size_t)t) / (size_t)copy_threads;
      size_t end   = (total_bytes * (size_t)(t + 1)) / (size_t)copy_threads;
      if (begin < value_bytes) {
        size_t value_end = std::min(end, value_bytes);
        if (value_end > begin) {
          std::memcpy((char*)state.problem.A_.data() + begin,
                      (const char*)state.temp_A + begin,
                      value_end - begin);
        }
      }
      if (end > value_bytes) {
        size_t index_begin = begin > value_bytes ? begin - value_bytes : 0;
        size_t index_end   = end - value_bytes;
        std::memcpy((char*)state.problem.A_indices_.data() + index_begin,
                    (const char*)state.temp_A_indices + index_begin,
                    index_end - index_begin);
      }
    }
  }

  state.temp_A                = nullptr;
  state.temp_A_indices        = nullptr;
  state.temp_csr_materialized = true;
  state.temp_A_region.reset();
  state.temp_A_indices_region.reset();
}

// COLUMNS is always parsed chunk-parallel: each chunk is counted/parsed by parse_columns_chunk
// and the per-chunk results are stitched together by merge_chunk_results_to_csr. There is no
// separate serial implementation -- a single thread just runs one chunk through the same path.
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

  auto chunk_bounds = compute_chunk_boundaries(columns_start, columns_end, num_threads);

  // Parse chunks in parallel
  std::vector<chunk_result_t> results(num_threads);

  {
    scoped_timer_t timer("parse_columns_chunk_parallel");
#ifdef MPS_FAST_PERF_COUNTERS
    std::vector<perf_counter_snapshot_t> perf_snapshots((size_t)num_threads);
#endif
    std::exception_ptr first_error = nullptr;
    std::mutex error_mutex;
    {
#pragma omp parallel for num_threads(num_threads)
      for (int t = 0; t < num_threads; t++) {
        try {
          MPS_NVTX_RANGE(std::string("columns_chunk ") + std::to_string(t), nvtx::colors::columns);
#ifdef MPS_FAST_PERF_COUNTERS
          thread_perf_counters_t perf_counters;
#endif
          results[t] =
            parse_columns_chunk<i_t, f_t>(chunk_bounds[t].start, chunk_bounds[t].end, state);
#ifdef MPS_FAST_PERF_COUNTERS
          perf_snapshots[(size_t)t] = perf_counters.stop();
#endif
        } catch (...) {
          std::lock_guard<std::mutex> lock(error_mutex);
          if (!first_error) { first_error = std::current_exception(); }
        }
      }
    }
    if (first_error) { std::rethrow_exception(first_error); }
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

  // necessary on the cold path since we directly read and lookup on the hot path
  auto reread_field_name = [](const char* start, const char* end) {
    const char* p = start;
    while (p < end && *p > ' ') {
      p++;
    }
    return std::string_view(start, (size_t)(p - start));
  };

  auto apply_rhs = [&](const char* row_start, size_t row_idx, f_t value) {
    // This is a regular non-obj row.
    if (row_idx != SIZE_MAX) {
      state.problem.b_[row_idx] = value;
      return;
    }
    // This is the objective row.
    std::string_view row_name = reread_field_name(row_start, cursor.end);
    if (row_name == state.objective_name_sv) {
      state.problem.objective_offset_ = -value;
      return;
    }
    // Other objectives, ignored currently. cold path
    if (state.ignored_objective_names.count(row_name)) { return; }
    // Unexpected!
    error_unknown_row(cursor, row_start, "RHS");
  };

  while (cursor.ptr < cursor.end) {
    [[maybe_unused]] auto rhs_name = cursor.read_field();
    if (accept_comment(cursor)) {
      expect_eol(cursor);
      continue;
    }
    const char* row_start = cursor.ptr;
    size_t row_idx        = state.read_row_lookup(cursor);
    auto value            = expect_number_fast_pm_one(cursor);
    apply_rhs(row_start, row_idx, (f_t)value);

    accept_comment(cursor);
    // Optional second entry
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

// does the job on 99% of instances, in the vast majority of cases bound names are sequential with
// occasional sparsity
static size_t find_var_after_hint(const std::vector<std::string_view>& var_names,
                                  std::string_view var_name,
                                  size_t hint_idx)
{
  const size_t n_vars = var_names.size();
  if (hint_idx + 1 < n_vars && var_names[hint_idx + 1] == var_name) { return hint_idx + 1; }
  if (hint_idx < n_vars && var_names[hint_idx] == var_name) { return hint_idx; }

  const size_t first_begin = std::min(hint_idx + 2, n_vars);
  for (size_t i = first_begin; i < n_vars; ++i) {
    if (var_names[i] == var_name) { return i; }
  }
  for (size_t i = 0; i < hint_idx && i < n_vars; ++i) {
    if (var_names[i] == var_name) { return i; }
  }
  return SIZE_MAX;
}

template <typename f_t, typename SetLb, typename SetUb, typename SetType, typename Error>
static bool apply_bound_record(std::string_view bound_type,
                               f_t value,
                               bool has_value,
                               bool first_bound_for_var,
                               SetLb&& set_lb,
                               SetUb&& set_ub,
                               SetType&& set_type,
                               Error&& error)
{
  if (bound_type == "LO") {
    set_lb(value);
  } else if (bound_type == "UP") {
    set_ub(value);
    if (first_bound_for_var && value < f_t{0}) { set_lb(-std::numeric_limits<f_t>::infinity()); }
  } else if (bound_type == "FX") {
    set_lb(value);
    set_ub(value);
  } else if (bound_type == "FR") {
    set_lb(-std::numeric_limits<f_t>::infinity());
    set_ub(std::numeric_limits<f_t>::infinity());
  } else if (bound_type == "MI") {
    set_lb(-std::numeric_limits<f_t>::infinity());
  } else if (bound_type == "PL") {
    set_ub(std::numeric_limits<f_t>::infinity());
  } else if (bound_type == "BV") {
    set_lb(f_t{0});
    set_ub(f_t{1});
    set_type('I');
  } else if (bound_type == "LI") {
    set_lb(value);
    set_type('I');
  } else if (bound_type == "UI") {
    set_ub(value);
    if (first_bound_for_var && value < f_t{0}) { set_lb(-std::numeric_limits<f_t>::infinity()); }
    set_type('I');
  } else if (bound_type == "SC") {
    if (UNLIKELY(!has_value)) {
      error("SC bound requires an upper bound value", bound_type);
      return false;
    }
    set_ub(value);
    set_type('S');
  } else {
    error("unknown bound type", bound_type);
    return false;
  }
  return true;
}

// Parallel BOUNDS parser for the common dense/ordered-name case. Returns false when the section
// is too small or not safely parallelizable, so parse_bounds_section resets and falls back to its
// serial path. Bound-type semantics (LO/UP/FX/...) are shared with the serial path through
// apply_bound_record, so the two cannot drift.
template <typename i_t, typename f_t>
static bool parse_bounds_section_parallel_dense(parse_state_t<i_t, f_t>& state,
                                                cursor_t& cursor,
                                                const char* bounds_body_start,
                                                const char* bounds_body_end,
                                                size_t n_vars)
{
  const size_t bounds_bytes   = (size_t)(bounds_body_end - bounds_body_start);
  const int num_threads       = phase_thread_count(MPS_BOUNDS_THREAD_CAP);
  const bool use_dense_lookup = state.col_index_mode == index_mode_t::dense_ordered;
  const size_t min_parallel_bytes =
    use_dense_lookup ? MPS_BOUNDS_PARALLEL_MIN_BYTES : MPS_BOUNDS_ORDERED_HINT_PARALLEL_MIN_BYTES;
  if (bounds_bytes < min_parallel_bytes || num_threads < 2) { return false; }

  MPS_NVTX_RANGE(
    use_dense_lookup ? "parse_bounds_parallel_dense" : "parse_bounds_parallel_ordered_hint",
    nvtx::colors::bounds);

  struct bounds_parallel_stats_t {
    size_t lines            = 0;
    size_t dense_hits       = 0;
    size_t dense_misses     = 0;
    size_t comments         = 0;
    size_t min_var          = SIZE_MAX;
    size_t max_var          = 0;
    size_t decreasing_order = 0;
    const char* error_ptr   = nullptr;
    char error_msg[192]     = {};
  };

  std::vector<bounds_parallel_stats_t> stats((size_t)num_threads);
  auto boundaries =
    compute_bounds_chunk_boundaries(bounds_body_start, bounds_body_end, num_threads);

  std::vector<uint8_t> bound_seen;
  {
    scoped_timer_t timer("bounds_parallel_seen_alloc");
    bound_seen.resize(n_vars, 0);
  }

  {
    scoped_timer_t timer(use_dense_lookup ? "parse_bounds_parallel_dense"
                                          : "parse_bounds_parallel_ordered_hint");
    // Repeated BOUNDS for the same variable are safe inside a group-owned chunk.
    // Parse optimistically, then accept only if chunk summaries prove no backward jumps.
#pragma omp parallel for schedule(static) num_threads(num_threads)
    for (int t = 0; t < num_threads; ++t) {
      auto& local = stats[(size_t)t];
      cursor_t cursor(boundaries[(size_t)t].start,
                      (size_t)(boundaries[(size_t)t].end - boundaries[(size_t)t].start));
      cursor.skip_ws();
      size_t prev_var = SIZE_MAX;
      size_t hint_idx = 0;
      auto lookup_var = [&](std::string_view var_name) {
        if (use_dense_lookup) { return state.col_dense.lookup(var_name); }
        // quite often variables are in order, so a cheap lookup trick is to look for the variable
        // right after this one
        return find_var_after_hint(state.var_names_sv, var_name, hint_idx);
      };
      try {
        while (cursor.ptr < cursor.end) {
          if (UNLIKELY(*cursor.ptr == '$')) {
            cursor.skip_to_eol();
            expect_eol(cursor);
            local.comments++;
            continue;
          }

          auto bound_type = cursor.read_field();
          if (UNLIKELY(bound_type.empty())) { break; }
          if (UNLIKELY(bound_type[0] == '$')) {
            cursor.skip_to_eol();
            expect_eol(cursor);
            local.comments++;
            continue;
          }

          [[maybe_unused]] auto bound_name = cursor.read_field();
          auto var_name                    = cursor.read_field();
          if (UNLIKELY(!var_name.empty() && var_name[0] == '$')) {
            cursor.skip_to_eol();
            expect_eol(cursor);
            local.comments++;
            continue;
          }

          size_t var_idx = lookup_var(var_name);
          if (UNLIKELY(var_idx == SIZE_MAX)) {
            local.dense_misses++;
            break;
          }
          hint_idx = var_idx;
          local.dense_hits++;
          local.lines++;
          local.min_var = std::min(local.min_var, var_idx);
          local.max_var = std::max(local.max_var, var_idx);
          if (prev_var != SIZE_MAX && var_idx < prev_var) { local.decreasing_order++; }
          prev_var = var_idx;

          bool first_bound_for_var = bound_seen[var_idx] == 0;
          bound_seen[var_idx]      = 1;

          f_t value      = 0;
          bool has_value = false;
          accept_comment(cursor);
          if (!cursor.eol()) {
            value     = (f_t)expect_number_fast_pm_one(cursor);
            has_value = true;
            accept_comment(cursor);
          }

          auto set_lb    = [&](f_t x) { state.problem.variable_lower_bounds_[var_idx] = x; };
          auto set_ub    = [&](f_t x) { state.problem.variable_upper_bounds_[var_idx] = x; };
          auto set_type  = [&](char t) { state.problem.var_types_[var_idx] = t; };
          auto set_error = [&](const char* msg, std::string_view type) {
            if (type.empty() || std::strcmp(msg, "unknown bound type") != 0) {
              std::snprintf(local.error_msg, sizeof(local.error_msg), "%s", msg);
            } else {
              std::snprintf(local.error_msg,
                            sizeof(local.error_msg),
                            "%s: %.*s",
                            msg,
                            (int)type.size(),
                            type.data());
            }
            local.error_ptr = cursor.ptr;
          };
          if (!apply_bound_record(bound_type,
                                  value,
                                  has_value,
                                  first_bound_for_var,
                                  set_lb,
                                  set_ub,
                                  set_type,
                                  set_error)) {
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
  size_t decreasing_order = 0;
  size_t overlap_chunks   = 0;
  size_t prev_max         = SIZE_MAX;
  for (int t = 0; t < num_threads; ++t) {
    const auto& local = stats[(size_t)t];
    if (local.error_ptr != nullptr) {
      cursor.ptr = local.error_ptr;
      cursor.error("%s", local.error_msg);
    }
    dense_misses += local.dense_misses;
    decreasing_order += local.decreasing_order;
    if (local.lines > 0) {
      if (prev_max != SIZE_MAX && local.min_var <= prev_max) { overlap_chunks++; }
      prev_max = local.max_var;
    }
  }

  const bool order_safe = dense_misses == 0 && decreasing_order == 0 && overlap_chunks == 0;

  if (!order_safe) {
    std::fprintf(stderr,
                 "[WARN] parallel BOUNDS fallback to serial: lookup_misses=%zu "
                 "decreasing_order=%zu overlap_chunks=%zu\n",
                 dense_misses,
                 decreasing_order,
                 overlap_chunks);
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
static void init_variable_bounds_defaults(parse_state_t<i_t, f_t>& state)
{
  size_t n_vars = (size_t)state.problem.n_vars_;
  {
    scoped_timer_t timer("bounds_init_defaults");
    state.problem.variable_lower_bounds_.resize(n_vars, f_t{0});
    state.problem.variable_upper_bounds_.resize(n_vars, std::numeric_limits<f_t>::infinity());
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
}

template <typename i_t, typename f_t, typename HasBound>
static void apply_unspecified_integer_bounds(parse_state_t<i_t, f_t>& state, HasBound&& has_bound)
{
  scoped_timer_t timer("bounds_integer_defaults");
  size_t n_vars = (size_t)state.problem.n_vars_;
  for (size_t i = 0; i < n_vars; ++i) {
    if (!has_bound(i) && state.problem.var_types_[i] == 'I') {
      state.problem.variable_lower_bounds_[i] = f_t{0};
      state.problem.variable_upper_bounds_[i] = f_t{1};
    }
  }
}

template <typename i_t, typename f_t>
static void init_variable_bounds_without_bounds_section(parse_state_t<i_t, f_t>& state)
{
  init_variable_bounds_defaults(state);
  apply_unspecified_integer_bounds(state, [](size_t) { return false; });
}

template <typename i_t, typename f_t>
static void parse_bounds_section(parse_state_t<i_t, f_t>& state,
                                 cursor_t& cursor,
                                 bool allow_parallel_dense = false)
{
  size_t n_vars = (size_t)state.problem.n_vars_;
  init_variable_bounds_defaults(state);

  std::vector<uint64_t> bound_seen((n_vars + 63) / 64, 0);
  auto has_bound = [&](size_t var_idx) {
    return (bound_seen[var_idx >> 6] & (uint64_t{1} << (var_idx & 63))) != 0;
  };
  auto mark_bound = [&](size_t var_idx) {
    bound_seen[var_idx >> 6] |= uint64_t{1} << (var_idx & 63);
  };

  if (!accept_section(cursor, "BOUNDS")) {
    apply_unspecified_integer_bounds(state, has_bound);
    return;
  }

  const char* bounds_body_start = cursor.ptr;
  const char* bounds_body_end   = cursor.end;
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
    while (!cursor.done()) {
      auto bound_type                  = cursor.read_field();
      [[maybe_unused]] auto bound_name = cursor.read_field();
      auto var_name                    = cursor.read_field();
      if (UNLIKELY(!var_name.empty() && var_name[0] == '$')) {
        cursor.skip_to_eol();
        expect_eol(cursor);
        continue;
      }

      // optimized lookup using hint (bounds often in same order as columns)
      size_t var_idx = SIZE_MAX;
      // handle annoying bounds-only vars that weren't declared in COLUMNS
      typename parse_state_t<i_t, f_t>::bounds_only_var_t* aux_var = nullptr;
      if (LIKELY(state.col_index_mode == index_mode_t::dense_ordered)) {
        var_idx = state.col_dense.lookup(var_name);
        if (var_idx == SIZE_MAX) { aux_var = &state.bounds_only_vars[var_name]; }
      } else {
        var_idx = find_var_after_hint(state.var_names_sv, var_name, hint_idx);
        if (var_idx == SIZE_MAX) { aux_var = &state.bounds_only_vars[var_name]; }
      }
      if (var_idx != SIZE_MAX) { hint_idx = var_idx; }
      bool first_bound_for_var = aux_var == nullptr && !has_bound(var_idx);

      f_t value      = 0;
      bool has_value = false;
      accept_comment(cursor);
      if (!cursor.eol()) {
        value     = (f_t)expect_number(cursor);
        has_value = true;
        accept_comment(cursor);
      }

      auto set_lb = [&](f_t x) {
        if (aux_var) {
          aux_var->lb = x;
        } else {
          state.problem.variable_lower_bounds_[var_idx] = x;
        }
      };
      auto set_ub = [&](f_t x) {
        if (aux_var) {
          aux_var->ub = x;
        } else {
          state.problem.variable_upper_bounds_[var_idx] = x;
        }
      };
      auto set_type = [&](char t) {
        if (aux_var) {
          aux_var->type = t;
        } else {
          state.problem.var_types_[var_idx] = t;
        }
      };

      auto set_error = [&](const char* msg, std::string_view type) {
        if (std::strcmp(msg, "unknown bound type") == 0) {
          cursor.error("%s: %.*s", msg, (int)type.size(), type.data());
        }
        cursor.error("%s", msg);
      };
      [[maybe_unused]] bool bound_applied = apply_bound_record(
        bound_type, value, has_value, first_bound_for_var, set_lb, set_ub, set_type, set_error);
      if (aux_var == nullptr) { mark_bound(var_idx); }

      expect_eol(cursor);
    }
  }
  apply_unspecified_integer_bounds(state, has_bound);
}

template <typename i_t, typename f_t>
static void init_constraint_bounds_from_rows(parse_state_t<i_t, f_t>& state)
{
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
}

template <typename i_t, typename f_t>
static void parse_ranges_section(parse_state_t<i_t, f_t>& state, cursor_t& cursor)
{
  scoped_timer_t timer("parse_ranges");
  init_constraint_bounds_from_rows(state);

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

  while (cursor.ptr < cursor.end) {
    [[maybe_unused]] auto range_name = cursor.read_field();
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
      if (UNLIKELY(!row_name2.empty() && row_name2[0] == '$')) {
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

// quadratric stuff is bare bones for now, optimize if needed

template <typename i_t, typename f_t>
static void build_var_name_map_if_needed(parse_state_t<i_t, f_t>& state)
{
  if (state.col_index_mode == index_mode_t::dense_ordered || !state.var_names_map.empty()) {
    return;
  }
  scoped_timer_t timer("quadratic_build_var_name_map");
  state.var_names_map.reserve((size_t)state.problem.n_vars_ * 2);
  for (size_t i = 0; i < state.var_names_sv.size(); ++i) {
    state.var_names_map.emplace(state.var_names_sv[i], i);
  }
}

template <typename i_t, typename f_t>
static size_t lookup_quadratic_var(parse_state_t<i_t, f_t>& state, std::string_view name)
{
  if (state.col_index_mode == index_mode_t::dense_ordered) { return state.col_dense.lookup(name); }
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

  struct expanded_entry_t {
    size_t row;
    size_t col;
    size_t seq;
    f_t value;
  };

  std::vector<expanded_entry_t> expanded;
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
static void parse_quadratic_sections(parse_state_t<i_t, f_t>& state, cursor_t& cursor)
{
  scoped_timer_t timer("parse_quadratic_sections");
  if (cursor.done()) { return; }

  build_var_name_map_if_needed(state);
  std::vector<std::tuple<i_t, i_t, f_t>> quadobj_entries;
  std::vector<std::tuple<i_t, i_t, f_t>> qmatrix_entries;
  std::vector<std::tuple<i_t, i_t, f_t>>* active_entries = nullptr;

  auto add_entry = [&](std::string_view var1, std::string_view var2, f_t value) {
    size_t var1_idx = lookup_quadratic_var(state, var1);
    if (var1_idx == SIZE_MAX) {
      cursor.error(
        "unknown variable name in quadratic section: %.*s", (int)var1.size(), var1.data());
    }
    size_t var2_idx = lookup_quadratic_var(state, var2);
    if (var2_idx == SIZE_MAX) {
      cursor.error(
        "unknown variable name in quadratic section: %.*s", (int)var2.size(), var2.data());
    }
    active_entries->emplace_back((i_t)var1_idx, (i_t)var2_idx, value);
  };

  while (cursor.ptr < cursor.end) {
    if (accept_section(cursor, "QUADOBJ")) {
      active_entries = &quadobj_entries;
      continue;
    }
    if (accept_section(cursor, "QMATRIX")) {
      active_entries = &qmatrix_entries;
      continue;
    }
    if (accept(cursor, "QCMATRIX")) {
      auto row_name = cursor.read_field();
      if (row_name.empty()) { cursor.error("QCMATRIX missing constraint row name"); }
      size_t row_idx = state.row_lookup(row_name);
      if (row_idx == SIZE_MAX) {
        cursor.error(
          "unknown constraint row name in QCMATRIX: %.*s", (int)row_name.size(), row_name.data());
      }
      char row_type = state.problem.row_types_[row_idx];
      if (row_type != 'L' && row_type != 'G') {
        cursor.error(
          "QCMATRIX row must have ROWS type L or G: %.*s", (int)row_name.size(), row_name.data());
      }
      expect_eol(cursor);
      typename parse_state_t<i_t, f_t>::qcmatrix_block_t block;
      block.row_idx  = row_idx;
      block.row_name = row_name;
      state.qcmatrix_blocks.push_back(std::move(block));
      active_entries = &state.qcmatrix_blocks.back().entries;
      continue;
    }
    if (active_entries == nullptr) { break; }

    const char* field_start = cursor.ptr;
    auto var1               = cursor.read_field();
    if (UNLIKELY(var1.empty())) { break; }
    if (UNLIKELY(var1[0] == '$' || var1[0] == '*')) {
      cursor.skip_to_eol();
      expect_eol(cursor);
      continue;
    }
    const bool starts_column_one =
      field_start == cursor.start || field_start[-1] == '\n' || field_start[-1] == '\r';
    if (UNLIKELY(starts_column_one)) {
      cursor.error("unknown quadratic section record: %.*s", (int)var1.size(), var1.data());
    }
    auto var2 = cursor.read_field();
    if (UNLIKELY(!var2.empty() && var2[0] == '$')) {
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
static void parse_bounds_range(parse_state_t<i_t, f_t>& state, mps_phase_range_t range)
{
  if (!range.present) {
    init_variable_bounds_without_bounds_section(state);
    return;
  }
  cursor_t cursor(range.begin, (size_t)(range.end - range.begin));
  parse_bounds_section(state, cursor, true);
}

template <typename i_t, typename f_t>
static void parse_ranges_range(parse_state_t<i_t, f_t>& state, mps_phase_range_t range)
{
  if (!range.present) {
    init_constraint_bounds_from_rows(state);
    return;
  }
  cursor_t cursor(range.begin, (size_t)(range.end - range.begin));
  parse_ranges_section(state, cursor);
}

template <typename i_t, typename f_t>
static void parse_quadratic_range(parse_state_t<i_t, f_t>& state, mps_phase_range_t range)
{
  if (!range.present) { return; }
  cursor_t cursor(range.begin, (size_t)(range.end - range.begin));
  parse_quadratic_sections(state, cursor);
}

template <typename i_t, typename f_t>
static void finalize_qcmatrix_constraints(parse_state_t<i_t, f_t>& state)
{
  if (state.qcmatrix_blocks.empty()) { return; }
  scoped_timer_t timer("finalize_qcmatrix_constraints");
  const size_t original_rows = (size_t)state.problem.n_constraints_;
  std::vector<uint8_t> quadratic_rows(original_rows, 0);
  std::vector<uint8_t> seen_rows(original_rows, 0);
  size_t active_blocks = 0;

  for (const auto& block : state.qcmatrix_blocks) {
    if (block.entries.empty()) { continue; }
    if (block.row_idx >= original_rows) {
      state.cursor.error("QCMATRIX row index is out of range");
    }
    if (seen_rows[block.row_idx]) {
      state.cursor.error("duplicate QCMATRIX block for constraint row: %.*s",
                         (int)block.row_name.size(),
                         block.row_name.data());
    }
    seen_rows[block.row_idx]      = 1;
    quadratic_rows[block.row_idx] = 1;
    ++active_blocks;
  }

  if (active_blocks == 0) { return; }

  // rebuild the A_ matrix. fairly ugly and brute force, could do better if we parsed the QCMATRIX
  // entries before building the CSR in COLUMNS but unclear if worth it
  for (const auto& block : state.qcmatrix_blocks) {
    if (block.entries.empty()) { continue; }

    size_t linear_begin = (size_t)state.problem.A_offsets_[block.row_idx];
    size_t linear_end   = (size_t)state.problem.A_offsets_[block.row_idx + 1];
    typename mps_data_model_t<i_t, f_t>::quadratic_constraint_t qc;
    qc.constraint_row_index = (i_t)block.row_idx;
    qc.constraint_row_name  = state.problem.row_names_[block.row_idx];
    qc.constraint_row_type  = state.problem.row_types_[block.row_idx];
    qc.rhs_value            = state.problem.b_[block.row_idx];
    qc.linear_values.assign(state.problem.A_.begin() + linear_begin,
                            state.problem.A_.begin() + linear_end);
    qc.linear_indices.assign(state.problem.A_indices_.begin() + linear_begin,
                             state.problem.A_indices_.begin() + linear_end);

    std::vector<size_t> perm(block.entries.size());
    for (size_t i = 0; i < perm.size(); ++i) {
      perm[i] = i;
    }
    std::sort(perm.begin(), perm.end(), [&](size_t a, size_t b) {
      const auto& ea = block.entries[a];
      const auto& eb = block.entries[b];
      if (std::get<0>(ea) != std::get<0>(eb)) { return std::get<0>(ea) < std::get<0>(eb); }
      return std::get<1>(ea) < std::get<1>(eb);
    });

    qc.rows.reserve(block.entries.size());
    qc.cols.reserve(block.entries.size());
    qc.vals.reserve(block.entries.size());
    for (size_t idx : perm) {
      const auto& [row, col, val] = block.entries[idx];
      qc.rows.push_back(row);
      qc.cols.push_back(col);
      qc.vals.push_back(val);
    }
    state.problem.quadratic_constraints_.push_back(std::move(qc));
  }

  std::vector<f_t> new_A;
  std::vector<i_t> new_A_indices;
  std::vector<i_t> new_A_offsets;
  std::vector<f_t> new_b;
  std::vector<f_t> new_clb;
  std::vector<f_t> new_cub;
  std::vector<std::string> new_row_names;
  std::vector<char> new_row_types;

  new_A.reserve(state.problem.A_.size());
  new_A_indices.reserve(state.problem.A_indices_.size());
  new_A_offsets.reserve(original_rows + 1 - active_blocks);
  new_b.reserve(original_rows - active_blocks);
  new_clb.reserve(original_rows - active_blocks);
  new_cub.reserve(original_rows - active_blocks);
  new_row_names.reserve(original_rows - active_blocks);
  new_row_types.reserve(original_rows - active_blocks);
  new_A_offsets.push_back(0);

  for (size_t row = 0; row < original_rows; ++row) {
    if (quadratic_rows[row]) { continue; }
    size_t begin = (size_t)state.problem.A_offsets_[row];
    size_t end   = (size_t)state.problem.A_offsets_[row + 1];
    new_A.insert(new_A.end(), state.problem.A_.begin() + begin, state.problem.A_.begin() + end);
    new_A_indices.insert(new_A_indices.end(),
                         state.problem.A_indices_.begin() + begin,
                         state.problem.A_indices_.begin() + end);
    new_A_offsets.push_back((i_t)new_A.size());
    new_b.push_back(state.problem.b_[row]);
    new_clb.push_back(state.problem.constraint_lower_bounds_[row]);
    new_cub.push_back(state.problem.constraint_upper_bounds_[row]);
    new_row_names.push_back(std::move(state.problem.row_names_[row]));
    new_row_types.push_back(state.problem.row_types_[row]);
  }

  state.problem.A_                       = std::move(new_A);
  state.problem.A_indices_               = std::move(new_A_indices);
  state.problem.A_offsets_               = std::move(new_A_offsets);
  state.problem.b_                       = std::move(new_b);
  state.problem.constraint_lower_bounds_ = std::move(new_clb);
  state.problem.constraint_upper_bounds_ = std::move(new_cub);
  state.problem.row_names_               = std::move(new_row_names);
  state.problem.row_types_               = std::move(new_row_types);
  state.problem.n_constraints_           = (i_t)state.problem.b_.size();
  state.problem.nnz_                     = (i_t)state.problem.A_.size();
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
    const bool col_dense_ordered = state.col_index_mode == index_mode_t::dense_ordered;
    size_t n = col_dense_ordered ? (size_t)state.problem.n_vars_ : state.var_names_sv.size();
    state.problem.var_names_.resize(n);
    if (col_dense_ordered && n >= 1'000'000 && num_threads > 1) {
#pragma omp parallel for schedule(static) num_threads(num_threads)
      for (size_t i = 0; i < n; ++i) {
        state.col_dense.format_name(i, state.problem.var_names_[i]);
      }
    } else if (col_dense_ordered) {
      for (size_t i = 0; i < n; ++i) {
        state.col_dense.format_name(i, state.problem.var_names_[i]);
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

template <typename i_t, typename f_t>
static void append_bounds_only_variables(parse_state_t<i_t, f_t>& state)
{
  if (state.bounds_only_vars.empty()) { return; }
  scoped_timer_t timer("append_bounds_only_variables");

  // BOUNDS-only variables have no matrix entries; append after COLUMNS vars.
  for (const auto& [name, aux] : state.bounds_only_vars) {
    state.problem.var_names_.emplace_back(name);
    state.problem.var_types_.push_back(aux.type);
    state.problem.c_.push_back(f_t{0});
    state.problem.variable_lower_bounds_.push_back(aux.lb);
    state.problem.variable_upper_bounds_.push_back(aux.ub);
  }
  state.problem.n_vars_ = (i_t)state.problem.var_names_.size();
}

template <typename i_t, typename f_t>
static std::size_t init_problem_storage(mps_data_model_t<i_t, f_t>& problem,
                                        std::size_t reserve_hint)
{
  problem.n_vars_                   = 0;
  problem.n_constraints_            = 0;
  problem.nnz_                      = 0;
  problem.maximize_                 = false;
  problem.objective_scaling_factor_ = f_t{1};
  problem.objective_offset_         = f_t{0};

  std::size_t reserve_size = std::max<std::size_t>(reserve_hint, 1 * MiB);
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
  return reserve_dim;
}

// Contract every input stream fed to parse_mps_fast_stream must satisfy.
template <typename Stream>
concept InputStream = requires(Stream stream)
{
  {stream.data()}->std::convertible_to<const char*>;
  {stream.mutable_data()}->std::convertible_to<char*>;
  {stream.size()}->std::convertible_to<std::size_t>;
  {stream.compressed_size()}->std::convertible_to<std::size_t>;
  {stream.reserve_size_hint()}->std::convertible_to<std::size_t>;
  {stream.registry()}->std::same_as<mps_phase_registry_t&>;
  {stream.view()}->std::same_as<input_stream_view_t>;
  {stream.run_decode_tasks()}->std::same_as<void>;
};

template <InputStream Stream, typename i_t, typename f_t>
static mps_data_model_t<i_t, f_t> parse_mps_fast_stream(Stream& stream,
                                                        const char* total_timer_name,
                                                        const char* producer_task_name)
{
  omp_max_active_levels_guard_t omp_active_levels(2);

  input_stream_view_t input = stream.view();
  auto total_timer          = std::make_unique<scoped_timer_t>(total_timer_name);
  mps_data_model_t<i_t, f_t> problem;
  std::size_t reserve_dim = init_problem_storage(problem, stream.reserve_size_hint());

  cursor_t cursor(input.data, 0);
  parse_state_t<i_t, f_t> state(problem, cursor);
  state.row_names_sv.reserve(reserve_dim);

  auto phase_end = [](const char*) { flush_timers(); };

  parallel_error_latch_t parser_tasks;

  auto run_parser_task = [&](auto&& fn) {
    if (parser_tasks.stopped()) { return; }
    try {
      fn();
    } catch (...) {
      parser_tasks.capture(std::current_exception());
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

  // These ints carry no data; they exist only as OpenMP task-dependency tokens. A task's
  // depend(out: X) "produces" X and depend(in: X) waits on it, so the phase ordering in the
  // task graph below (e.g. bounds after columns_done, because bounds reference variable names)
  // is expressed purely through which tokens each task depends on.
  int header_ready = 0, rows_ready = 0, columns_ready = 0;
  int rhs_ready = 0, bounds_ready = 0, ranges_ready = 0, quadratic_ready = 0;
  int header_done = 0, rows_done = 0, columns_done = 0;
  int rhs_done = 0, bounds_done = 0, ranges_done = 0, quadratic_done = 0, names_done = 0;
  int csr_done = 0;

  const std::size_t parser_size = std::max(stream.reserve_size_hint(), input.compressed_size);
  const int parser_threads      = parser_thread_cap_for_size(parser_size);

#pragma omp parallel num_threads(parser_threads)
  {
    std::string thread_name = "omp-parser-" + std::to_string(omp_get_thread_num());
    nvtx::name_current_thread(thread_name.c_str());

#pragma omp single
    {
      // Bridge between the producer and the parse tasks: each detached task below blocks
      // until run_decode_tasks() publishes that phase's byte range into the registry, then
      // completes its event and fulfills depend(out: <phase>_ready) -- releasing the matching
      // parse task. This is what lets ROWS parsing start the instant the ROWS bytes are
      // decoded, overlapping with the decode of later sections.
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

      // We intentionally keep LZ4/raw input as a stable full-buffer producer here. The
      // progressive decoded-page lifetime prototype saved RSS, but made COLUMNS/merge slower
      // and really wants a separate memory-limited parser pipeline instead of this fast path.
#pragma omp task
      {
        MPS_NVTX_RANGE(producer_task_name, nvtx::colors::io);
        try {
          stream.run_decode_tasks();
        } catch (...) {
          parser_tasks.capture(std::current_exception());
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

#pragma omp task depend(in : rows_done, columns_ready) depend(out : columns_done)
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

#pragma omp task depend(in : columns_done) depend(out : csr_done)
      {
        run_parser_task([&] {
          MPS_NVTX_RANGE("task_materialize_csr", nvtx::colors::alloc);
          materialize_problem_csr(state);
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
          parse_ranges_range(state, input.registry->range(mps_phase_kind::ranges));
          phase_end("ranges");
        });
      }

#pragma omp task depend(in : bounds_ready, columns_done) depend(out : bounds_done)
      {
        run_parser_task([&] {
          MPS_NVTX_RANGE("task_bounds", nvtx::colors::bounds);
          parse_bounds_range(state, input.registry->range(mps_phase_kind::bounds));
          phase_end("bounds");
        });
      }

#pragma omp task depend(in : quadratic_ready, columns_done) depend(out : quadratic_done)
      {
        run_parser_task([&] {
          MPS_NVTX_RANGE("task_quadratic", nvtx::colors::generic);
          parse_quadratic_range(state, input.registry->range(mps_phase_kind::quadratic));
          phase_end("quadratic");
        });
      }
    }
  }

  parser_tasks.rethrow_if_error();

  finalize_qcmatrix_constraints(state);
  append_bounds_only_variables(state);

  input.size = stream.size();
  cursor.end = input.data + input.size;
  if (!input.registry->endata_ready()) {
    cursor.ptr = input.data + input.size;
    cursor.error("input ended before ENDATA boundary was resolved");
  }
  if (input.registry->endata_present()) {
    cursor.ptr = input.registry->endata_begin();
    expect(cursor, "ENDATA");
  }

  total_timer.reset();
  flush_timers();
  return problem;
}

struct padded_memory_input_t {
  std::vector<char> buffer;
  std::size_t input_size      = 0;
  std::size_t compressed_size = 0;
};

static padded_memory_input_t read_compressed_mps_file(const std::string& path)
{
  std::vector<char> buffer = file_to_string(path);
  if (buffer.empty()) { buffer.push_back('\0'); }

  std::size_t input_size = buffer.size() - 1;
  ensure_input_buffer_padding(buffer, input_size);
  return {std::move(buffer), input_size, get_file_size(path)};
}

template <typename i_t, typename f_t>
mps_data_model_t<i_t, f_t> parse_mps_fast_file(const std::string& path, FileReadMethod read_method)
{
  FileReadMethod effective_method = effective_file_read_method(path, read_method);
  switch (effective_method) {
    case FileReadMethod::Lz4: {
      lz4_input_stream_t stream(path);
      return parse_mps_fast_stream<lz4_input_stream_t, i_t, f_t>(
        stream, "parse_mps_fast_file_lz4 (total)", "task_lz4_read_decode");
    }
    case FileReadMethod::Gzip:
    case FileReadMethod::Bzip2: {
      padded_memory_input_t input = read_compressed_mps_file(path);
      memory_input_stream_t stream(
        std::move(input.buffer), input.input_size, input.compressed_size);
      const char* timer_name = effective_method == FileReadMethod::Gzip
                                 ? "parse_mps_fast_file_gzip (total)"
                                 : "parse_mps_fast_file_bzip2 (total)";
      return parse_mps_fast_stream<memory_input_stream_t, i_t, f_t>(
        stream, timer_name, "task_memory_scan");
    }
    case FileReadMethod::Read: {
      raw_input_stream_t stream(path);
      return parse_mps_fast_stream<raw_input_stream_t, i_t, f_t>(
        stream, "parse_mps_fast_file_raw (total)", "task_raw_read");
    }
  }
  __builtin_unreachable();
}

template mps_data_model_t<int, float> parse_mps_fast_file(const std::string& path,
                                                          FileReadMethod read_method);
template mps_data_model_t<int, double> parse_mps_fast_file(const std::string& path,
                                                           FileReadMethod read_method);
template mps_data_model_t<int64_t, float> parse_mps_fast_file(const std::string& path,
                                                              FileReadMethod read_method);
template mps_data_model_t<int64_t, double> parse_mps_fast_file(const std::string& path,
                                                               FileReadMethod read_method);

}  // namespace cuopt::linear_programming::io::detail
