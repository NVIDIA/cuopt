/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cstddef>
#include <cstdlib>
#include <type_traits>

namespace cuopt::mathematical_optimization::io::detail {

struct parser_parameters_t {
  // lower bound on columns chunk size to avoid tiny parser tasks
  std::size_t columns_min_chunk_bytes       = 1ull << 20;
  std::size_t raw_input_window_bytes        = 64ull << 20;
  std::size_t raw_input_max_read_threads    = 8;
  std::size_t rows_parallel_min_bytes       = 512ull << 20;
  std::size_t row_hash_partitioned_min_rows = 64 * 1024;
  // avoid openmp setup for small bounds sections
  std::size_t bounds_parallel_min_bytes = 256ull << 20;
  // ordered-name fallback is cheap enough to parallelize on smaller bounds sections
  std::size_t bounds_ordered_hint_min_bytes = 8ull << 20;
  int copy_threads                          = 2;
  int column_alloc_threads                  = 4;
  std::size_t names_parallel_min            = 1'000'000;
  std::size_t memory_input_block_bytes      = 0;
  std::size_t lz4_window_bytes              = 64ull << 20;
  std::size_t lz4_decode_batch_bytes        = 256ull << 20;
};

#if defined(MPS_PARSER_FUZZ_HOOKS)
inline parser_parameters_t g_parser_parameters;

inline void load_parser_parameters_from_env()
{
  parser_parameters_t h;
  auto rd = [](const char* name, auto& field) {
    if (const char* v = std::getenv(name)) {
      field = static_cast<std::remove_reference_t<decltype(field)>>(std::strtoull(v, nullptr, 10));
    }
  };
  rd("MPS_HOOK_COLUMNS_MIN_CHUNK_BYTES", h.columns_min_chunk_bytes);
  rd("MPS_HOOK_RAW_INPUT_WINDOW_BYTES", h.raw_input_window_bytes);
  rd("MPS_HOOK_RAW_INPUT_MAX_READ_THREADS", h.raw_input_max_read_threads);
  rd("MPS_HOOK_ROWS_PARALLEL_MIN_BYTES", h.rows_parallel_min_bytes);
  rd("MPS_HOOK_ROW_HASH_PARTITIONED_MIN_ROWS", h.row_hash_partitioned_min_rows);
  rd("MPS_HOOK_BOUNDS_PARALLEL_MIN_BYTES", h.bounds_parallel_min_bytes);
  rd("MPS_HOOK_BOUNDS_ORDERED_HINT_MIN_BYTES", h.bounds_ordered_hint_min_bytes);
  rd("MPS_HOOK_COPY_THREADS", h.copy_threads);
  rd("MPS_HOOK_COLUMN_ALLOC_THREADS", h.column_alloc_threads);
  rd("MPS_HOOK_NAMES_PARALLEL_MIN", h.names_parallel_min);
  rd("MPS_HOOK_MEMORY_INPUT_BLOCK_BYTES", h.memory_input_block_bytes);
  rd("MPS_HOOK_LZ4_WINDOW_BYTES", h.lz4_window_bytes);
  rd("MPS_HOOK_LZ4_DECODE_BATCH_BYTES", h.lz4_decode_batch_bytes);
  g_parser_parameters = h;
}
#else
inline constexpr parser_parameters_t g_parser_parameters{};

inline void load_parser_parameters_from_env() {}
#endif

}  // namespace cuopt::mathematical_optimization::io::detail
