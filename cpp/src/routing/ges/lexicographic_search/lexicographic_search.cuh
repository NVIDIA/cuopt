/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
