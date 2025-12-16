/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include "../../solution/solution.cuh"
#include "../guided_ejection_search.cuh"

namespace cuopt {
namespace routing {
namespace detail {

struct p_val_seq_t {
  __host__ __device__ p_val_seq_t(uint16_t p_v, uint16_t s_s) : p_val(p_v), sequence_size(s_s) {}
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  uint p_val         : 16;
  uint sequence_size : 16;
#elif __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  uint sequence_size : 16;
  uint p_val         : 16;
#endif
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
