/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

// GPU alternatives that cython_types.hpp forward declares. Include only where CUDA is
// available: pulling this into cuopt_client would reintroduce its librmm dependency (#1890).

#include <cuopt/export.hpp>
#include <cuopt/mathematical_optimization/utilities/cython_types.hpp>

#include <rmm/device_buffer.hpp>

#include <memory>
#include <utility>

namespace cuopt {
namespace CUOPT_EXPORT cython {

using gpu_buffer = std::unique_ptr<rmm::device_buffer>;

struct lp_gpu_solutions_t {
  gpu_buffer primal_solution_;
  gpu_buffer dual_solution_;
  gpu_buffer reduced_cost_;
  gpu_buffer current_primal_solution_;
  gpu_buffer current_dual_solution_;
  gpu_buffer initial_primal_average_;
  gpu_buffer initial_dual_average_;
  gpu_buffer current_ATY_;
  gpu_buffer sum_primal_solutions_;
  gpu_buffer sum_dual_solutions_;
  gpu_buffer last_restart_duality_gap_primal_solution_;
  gpu_buffer last_restart_duality_gap_dual_solution_;
};

struct mip_gpu_solution_t {
  gpu_buffer solution_;
};

// Handed to the holder at construction, so a CPU-only TU calls through the pointer rather
// than referencing a symbol it cannot resolve.
// Plain new/delete: these hold host-side owners of device buffers, not device memory,
// so an RMM resource is not what belongs here.
inline void destroy_lp_gpu_solutions(lp_gpu_solutions_t* p) noexcept { delete p; }
inline void destroy_mip_gpu_solution(mip_gpu_solution_t* p) noexcept { delete p; }

inline lp_gpu_ptr make_lp_gpu_solutions(lp_gpu_solutions_t&& value)
{
  return lp_gpu_ptr{new lp_gpu_solutions_t{std::move(value)},
                    gpu_holder_deleter_t<lp_gpu_solutions_t>{&destroy_lp_gpu_solutions}};
}

inline mip_gpu_ptr make_mip_gpu_solution(gpu_buffer&& solution)
{
  return mip_gpu_ptr{new mip_gpu_solution_t{std::move(solution)},
                     gpu_holder_deleter_t<mip_gpu_solution_t>{&destroy_mip_gpu_solution}};
}

}  // namespace CUOPT_EXPORT cython
}  // namespace cuopt
