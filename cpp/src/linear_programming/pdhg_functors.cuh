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

#include <raft/core/math.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <utilities/copy_helpers.hpp>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
struct dual_projection_t {
  dual_projection_t(
    // i_t n_row,
    rmm::device_scalar<f_t>& scalar,
    const rmm::device_uvector<f_t>& dual,
    const rmm::device_uvector<f_t>& lower,
    const rmm::device_uvector<f_t>& upper,
    rmm::device_uvector<f_t>& out_next_dual,
    rmm::device_uvector<f_t>& out_delta_dual)
    :  // n_row_{n_row},
      scalar_{scalar.data()},
      dual_{make_span(dual)},
      lower_{make_span(lower)},
      upper_{make_span(upper)},
      out_next_dual_{make_span(out_next_dual)},
      out_delta_dual_{make_span(out_delta_dual)}
  {
  }
  __device__ __forceinline__ void operator()(i_t idx, f_t Ax_i, raft::device_span<f_t> output)
  {
    // if (idx >= n_row_) {
    //   printf("kernel idx oob\n");
    // }
    // maybe not needed
    output[idx]          = Ax_i;
    f_t next             = dual_[idx] - (*scalar_ * Ax_i);
    f_t low              = next + *scalar_ * lower_[idx];
    f_t up               = next + *scalar_ * upper_[idx];
    next                 = raft::max<f_t>(low, raft::min<f_t>(up, f_t(0)));
    out_next_dual_[idx]  = next;
    out_delta_dual_[idx] = next - dual_[idx];
  }
  // i_t n_row_;
  const f_t* scalar_;
  raft::device_span<const f_t> dual_;
  raft::device_span<const f_t> lower_;
  raft::device_span<const f_t> upper_;
  raft::device_span<f_t> out_next_dual_;
  raft::device_span<f_t> out_delta_dual_;
};

template <typename i_t, typename f_t>
struct primal_projection_t {
  primal_projection_t(const rmm::device_scalar<f_t>& step_size,
                      const rmm::device_uvector<f_t>& primal_solution,
                      const rmm::device_uvector<f_t>& obj_coeff,
                      const rmm::device_uvector<f_t>& lower,
                      const rmm::device_uvector<f_t>& upper,
                      rmm::device_uvector<f_t>& out_delta_x,
                      rmm::device_uvector<f_t>& out_tmp_primal,
                      rmm::device_uvector<f_t>& next_primal_solution)
    : step_size_{step_size.data()},
      primal_solution_{make_span(primal_solution)},
      obj_coeff_{make_span(obj_coeff)},
      lower_{make_span(lower)},
      upper_{make_span(upper)},
      out_delta_x_{make_span(out_delta_x)},
      out_tmp_primal_{make_span(out_tmp_primal)},
      next_primal_solution_{make_span(next_primal_solution)}
  {
  }

  __device__ __forceinline__ void operator()(i_t idx, f_t Aty_i, raft::device_span<f_t> output)
  {
    output[idx]                = Aty_i;
    f_t gradient               = obj_coeff_[idx] - Aty_i;
    f_t next                   = primal_solution_[idx] - (*step_size_ * gradient);
    next                       = raft::max<f_t>(raft::min<f_t>(next, upper_[idx]), lower_[idx]);
    next_primal_solution_[idx] = next;
    out_delta_x_[idx]          = next - primal_solution_[idx];
    out_tmp_primal_[idx]       = next - primal_solution_[idx] + next;
  }

  const f_t* step_size_;
  raft::device_span<const f_t> primal_solution_;
  raft::device_span<const f_t> obj_coeff_;
  raft::device_span<const f_t> lower_;
  raft::device_span<const f_t> upper_;
  raft::device_span<f_t> out_delta_x_;
  raft::device_span<f_t> out_tmp_primal_;
  raft::device_span<f_t> next_primal_solution_;
};

template <typename i_t, typename f_t>
struct next_step_size_t {
  next_step_size_t(const rmm::device_uvector<f_t>& current_AtY,
                   rmm::device_uvector<f_t>& out_tmp_primal)
    : current_AtY_(make_span(current_AtY)), out_tmp_primal_(make_span(out_tmp_primal))
  {
  }
  __device__ __forceinline__ void operator()(i_t idx, f_t Ax_i, raft::device_span<f_t> output)
  {
    output[idx]          = Ax_i;
    out_tmp_primal_[idx] = Ax_i - current_AtY_[idx];
    // printf("step_size_functor idx %d Ax_i %f current_AtY_[idx] %f out_tmp_primal_[idx] %f\n",
    // idx, Ax_i, current_AtY_[idx], out_tmp_primal_[idx]);
  }
  raft::device_span<const f_t> current_AtY_;
  raft::device_span<f_t> out_tmp_primal_;
};

}  // namespace cuopt::linear_programming::detail
