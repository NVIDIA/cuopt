/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <pdlp/utilities/ping_pong_graph.cuh>

namespace cuopt::mathematical_optimization::pdlp {

template <typename i_t>
ping_pong_graph_t<i_t>::ping_pong_graph_t(rmm::cuda_stream_view stream_view,
                                          bool is_legacy_batch_mode)
  : stream_view_(stream_view), is_legacy_batch_mode_(is_legacy_batch_mode)
{
}

template class ping_pong_graph_t<int>;

}  // namespace cuopt::mathematical_optimization::pdlp
