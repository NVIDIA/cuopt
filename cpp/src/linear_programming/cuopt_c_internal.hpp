/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/linear_programming/cuopt_c.h>
#include <cuopt/linear_programming/mip/solver_solution.hpp>
#include <cuopt/linear_programming/optimization_problem.hpp>
#include <cuopt/linear_programming/pdlp/solver_solution.hpp>

#include <raft/core/handle.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cuopt::linear_programming {

struct problem_and_stream_view_t {
  problem_and_stream_view_t()
    : op_problem(nullptr), stream_view(rmm::cuda_stream_per_thread), handle(stream_view)
  {
  }
  raft::handle_t* get_handle_ptr() { return &handle; }
  optimization_problem_t<cuopt_int_t, cuopt_float_t>* op_problem;
  rmm::cuda_stream_view stream_view;
  raft::handle_t handle;
};

struct solution_and_stream_view_t {
  solution_and_stream_view_t(bool solution_for_mip, rmm::cuda_stream_view stream_view)
    : is_mip(solution_for_mip),
      mip_solution_ptr(nullptr),
      lp_solution_ptr(nullptr),
      stream_view(stream_view)
  {
  }
  bool is_mip;
  mip_solution_t<cuopt_int_t, cuopt_float_t>* mip_solution_ptr;
  optimization_problem_solution_t<cuopt_int_t, cuopt_float_t>* lp_solution_ptr;
  rmm::cuda_stream_view stream_view;
};

}  // namespace cuopt::linear_programming
