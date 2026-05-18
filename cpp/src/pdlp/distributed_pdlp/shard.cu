/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <pdlp/distributed_pdlp/shard.hpp>
#include <pdlp/pdlp.cuh>
namespace cuopt::linear_programming::detail {

// This must be done in .cu file because the pdlp_solver_t is not already complete in the hpp file
template <typename i_t, typename f_t>
pdlp_shard_t<i_t, f_t>::~pdlp_shard_t() = default;




template struct pdlp_shard_t<int, double>;
//template struct pdlp_shard_t<int, float>;
}  // namespace cuopt::linear_programming::detail
