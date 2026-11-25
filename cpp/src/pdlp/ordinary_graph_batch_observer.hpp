/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cstdint>

namespace cuopt::mathematical_optimization::pdlp::detail {

struct ordinary_graph_batch_stats_t {
  uint64_t reports{0};
  uint64_t launches{0};
  uint64_t iterations{0};
  uint64_t recoveries{0};
};

// Internal observation of synchronous PDLP solves on the calling thread.
class scoped_ordinary_graph_batch_observer_t {
 public:
  explicit scoped_ordinary_graph_batch_observer_t(ordinary_graph_batch_stats_t& stats) noexcept;
  ~scoped_ordinary_graph_batch_observer_t() noexcept;

  scoped_ordinary_graph_batch_observer_t(const scoped_ordinary_graph_batch_observer_t&) = delete;
  scoped_ordinary_graph_batch_observer_t& operator=(const scoped_ordinary_graph_batch_observer_t&) =
    delete;
  scoped_ordinary_graph_batch_observer_t(scoped_ordinary_graph_batch_observer_t&&) = delete;
  scoped_ordinary_graph_batch_observer_t& operator=(scoped_ordinary_graph_batch_observer_t&&) =
    delete;

 private:
  ordinary_graph_batch_stats_t* previous_;
};

}  // namespace cuopt::mathematical_optimization::pdlp::detail
