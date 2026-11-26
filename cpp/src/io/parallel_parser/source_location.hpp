// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// reserved. SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <simde/x86/avx2.h>

namespace cuopt::mathematical_optimization::io::detail {

static size_t count_newlines(const char* p, const char* end)
{
  const simde__m256i newline = simde_mm256_set1_epi8('\n');
  size_t count               = 0;
  for (; end - p >= 32; p += 32) {
    const simde__m256i v = simde_mm256_loadu_si256((const simde__m256i*)p);
    count += std::popcount((uint32_t)simde_mm256_movemask_epi8(simde_mm256_cmpeq_epi8(v, newline)));
  }
  return count + std::count(p, end, '\n');
}

// Blocks are retired in file order at the contiguous decode frontier. locate() only reads
// slots of already-retired blocks.
class source_location_t {
 public:
  source_location_t(const char* data, size_t capacity, size_t block_count, std::string filename)
    : data_(data),
      capacity_(capacity),
      block_end_(block_count, std::numeric_limits<size_t>::max()),
      block_newlines_(block_count, 0),
      filename_(std::move(filename))
  {
  }

  // called by the frontier owner only, under its lock
  void retire_block(size_t end_offset, size_t newlines)
  {
    assert(retired_ < block_end_.size());
    assert(end_offset <= capacity_);
    assert(retired_ == 0 || block_end_[retired_ - 1] <= end_offset);
    block_end_[retired_]      = end_offset;
    block_newlines_[retired_] = newlines;
    ++retired_;
  }

  // 1-based {line, column} of p
  std::pair<int64_t, int64_t> locate(const char* p) const
  {
    assert(data_ <= p && p <= data_ + capacity_);
    const size_t offset = (size_t)(p - data_);
    int64_t line        = 1;
    size_t b            = 0;
    while (b < block_end_.size() && block_end_[b] <= offset) {
      line += block_newlines_[b];
      ++b;
    }
    const size_t block_begin = b == 0 ? 0 : block_end_[b - 1];
    line += count_newlines(data_ + block_begin, p);

    const char* line_start = p;
    while (line_start > data_ && line_start[-1] != '\n') {
      --line_start;
    }
    return {line, (int64_t)(p - line_start) + 1};
  }

  const std::string& filename() const { return filename_; }

 private:
  const char* data_;
  size_t capacity_;
  size_t retired_ = 0;
  std::vector<size_t> block_end_;
  std::vector<size_t> block_newlines_;
  std::string filename_;
};

}  // namespace cuopt::mathematical_optimization::io::detail
