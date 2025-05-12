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

#include <cuopt/linear_programming/mip/solver_settings.hpp>
#include <cuopt/linear_programming/optimization_problem.hpp>

#include <utilities/macros.cuh>

#include <mip/logger.hpp>
#include <mip/problem/problem.cuh>

#include <mip/presolve/load_balanced_bounds_presolve.cuh>
#include <mip/presolve/load_balanced_partition_helpers.cuh>
#include <raft/core/nvtx.hpp>
#include <raft/random/rng_device.cuh>
#include <raft/util/cuda_dev_essentials.cuh>
#include <raft/util/cudart_utils.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include "spmv_functors.cuh"

namespace cuopt::linear_programming::detail {

// TODO move in spmv functors
template <typename i_t, typename f_t>
struct primal_projection_t {
  primal_projection_t(const f_t* step_size,
                      const f_t* primal_solution,
                      const f_t* obj_coeff,
                      const f_t* lower,
                      const f_t* upper,
                      f_t* out_delta_x,
                      f_t* out_tmp_primal,
                      f_t* next_primal_solution)
    : step_size_(step_size),
      primal_solution_(primal_solution),
      obj_coeff_(obj_coeff),
      lower_(lower),
      upper_(upper),
      out_delta_x_(out_delta_x),
      out_tmp_primal_(out_tmp_primal),
      next_primal_solution_(next_primal_solution)
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
  const f_t* primal_solution_;
  const f_t* obj_coeff_;
  const f_t* lower_;
  const f_t* upper_;
  f_t* out_delta_x_;
  f_t* out_tmp_primal_;
  f_t* next_primal_solution_;
};

template <typename i_t, typename f_t>
struct dual_projection_t {
  dual_projection_t(const f_t* scalar,
                    const f_t* dual,
                    const f_t* lower,
                    const f_t* upper,
                    f_t* out_next_dual,
                    f_t* out_delta_dual)
    : scalar_{scalar},
      dual_{dual},
      lower_{lower},
      upper_{upper},
      out_next_dual_{out_next_dual},
      out_delta_dual_{out_delta_dual}
  {
  }
  __device__ __forceinline__ void operator()(i_t idx, f_t Ax_i, raft::device_span<f_t> output)
  {
    f_t next             = dual_[idx] - (*scalar_ * Ax_i);
    f_t low              = next + *scalar_ * lower_[idx];
    f_t up               = next + *scalar_ * upper_[idx];
    next                 = raft::max<f_t>(low, raft::min<f_t>(up, f_t(0)));
    out_next_dual_[idx]  = next;
    out_delta_dual_[idx] = next - dual_[idx];
  }
  const f_t* scalar_;
  const f_t* dual_;
  const f_t* lower_;
  const f_t* upper_;
  f_t* out_next_dual_;
  f_t* out_delta_dual_;
};

template <typename i_t, typename f_t>
struct step_size_functor {
  step_size_functor(const f_t* current_AtY, f_t* out_tmp_primal)
    : current_AtY_(current_AtY), out_tmp_primal_(out_tmp_primal)
  {
  }
  __device__ __forceinline__ void operator()(i_t idx, f_t Ax_i, raft::device_span<f_t> output)
  {
    output[idx]          = Ax_i;
    out_tmp_primal_[idx] = Ax_i - current_AtY_[idx];
  }
  const f_t* current_AtY_;
  f_t* out_tmp_primal_;
};

template <typename i_t, typename f_t>
class spmv_t {
 public:
  struct spmv_view_t {
    raft::device_span<const i_t> reorg_ids;
    raft::device_span<const i_t> offsets;
    raft::device_span<const i_t> elem;
    raft::device_span<const f_t> coeff;
    i_t nnz;
  };

  spmv_t(problem_t<i_t, f_t>& problem,
         raft::device_span<f_t> ax_input_,
         raft::device_span<f_t> ax_output_,
         raft::device_span<f_t> aty_input_,
         raft::device_span<f_t> aty_output_,
         raft::device_span<f_t> aty_next_input_,
         raft::device_span<f_t> aty_next_output_,
         const f_t* primal_step_size,
         raft::device_span<f_t> primal_solution,
         raft::device_span<f_t> next_primal_solution,
         raft::device_span<f_t> delta_primal,
         const f_t* dual_step_size,
         raft::device_span<f_t> dual_solution,
         raft::device_span<f_t> next_dual_solution,
         raft::device_span<f_t> delta_dual,
         bool debug = false);
  ~spmv_t();
  spmv_t() = delete;
  void setup_lb_problem(problem_t<i_t, f_t>& problem, bool debug = false);
  void setup_lb_meta();
  spmv_view_t get_A_view();
  spmv_view_t get_AT_view();
  template <typename functor_t = identity_functor<i_t, f_t>>
  void call_Ax_graph(raft::device_span<f_t> input,
                     raft::device_span<f_t> output,
                     bool dry_run      = false,
                     functor_t functor = identity_functor<i_t, f_t>());
  template <typename functor_t = identity_functor<i_t, f_t>>
  void call_ATy_graph(raft::device_span<f_t> input,
                      raft::device_span<f_t> output,
                      bool dry_run      = false,
                      functor_t functor = identity_functor<i_t, f_t>());

  void Ax(const raft::handle_t* h);
  void ATy(const raft::handle_t* h, raft::device_span<f_t> input, raft::device_span<f_t> output);
  void ATy_projection(const raft::handle_t* h, i_t total_pdlp_iterations);
  void Ax_projection(const raft::handle_t* h, i_t total_pdlp_iterations);

  managed_stream_pool streams;

  static constexpr i_t heavy_degree_cutoff = 16 * 1024;
  problem_t<i_t, f_t>* pb;
  const raft::handle_t* handle_ptr;

  i_t n_variables;
  i_t n_constraints;
  i_t nnz;

  // csr - cnst
  rmm::device_uvector<i_t> cnst_reorg_ids;
  rmm::device_uvector<f_t> coefficients;
  rmm::device_uvector<i_t> variables;
  rmm::device_uvector<i_t> offsets;

  // csc - vars
  rmm::device_uvector<i_t> vars_reorg_ids;
  rmm::device_uvector<f_t> reverse_coefficients;
  rmm::device_uvector<i_t> reverse_constraints;
  rmm::device_uvector<i_t> reverse_offsets;

  // lb members
  rmm::device_uvector<i_t> tmp_cnst_ids;
  rmm::device_uvector<i_t> tmp_vars_ids;

  // Number of blocks for heavy ids
  rmm::device_uvector<i_t> heavy_cnst_block_segments;
  rmm::device_uvector<i_t> heavy_vars_block_segments;
  rmm::device_uvector<i_t> heavy_cnst_vertex_ids;
  rmm::device_uvector<i_t> heavy_vars_vertex_ids;
  rmm::device_uvector<i_t> heavy_cnst_pseudo_block_ids;
  rmm::device_uvector<i_t> heavy_vars_pseudo_block_ids;

  i_t num_blocks_heavy_cnst;
  i_t num_blocks_heavy_vars;

  rmm::device_uvector<f_t> tmp_ax;
  rmm::device_uvector<f_t> tmp_aty;

  // lb sub-warp opt members
  bool is_cnst_sub_warp_single_bin;
  i_t cnst_sub_warp_count;
  rmm::device_uvector<i_t> warp_cnst_offsets;
  rmm::device_uvector<i_t> warp_cnst_id_offsets;

  bool is_vars_sub_warp_single_bin;
  i_t vars_sub_warp_count;
  rmm::device_uvector<i_t> warp_vars_offsets;
  rmm::device_uvector<i_t> warp_vars_id_offsets;

  // binning
  std::vector<i_t> cnst_bin_offsets;
  std::vector<i_t> vars_bin_offsets;

  vertex_bin_t<i_t> cnst_binner;
  vertex_bin_t<i_t> vars_binner;

  raft::device_span<f_t> ax_input;
  raft::device_span<f_t> ax_output;

  raft::device_span<f_t> aty_input;
  raft::device_span<f_t> aty_output;

  raft::device_span<f_t> aty_next_input;
  raft::device_span<f_t> aty_next_output;

  // spmv graphs
  bool ax_graph_created;
  bool aty_graph_created;
  bool aty_next_graph_created;
  bool aty_graph_proj_created;
  bool aty_graph_proj_next_created;
  bool ax_graph_proj_created;
  bool ax_graph_proj_next_created;

  cudaGraphExec_t ax_exec;
  cudaGraph_t ax_graph;

  cudaGraphExec_t aty_exec;
  cudaGraph_t aty_graph;

  cudaGraphExec_t aty_next_exec;
  cudaGraph_t aty_next_graph;

  step_size_functor<i_t, f_t> current_step_size_functor_;
  step_size_functor<i_t, f_t> next_step_size_functor_;

  cudaGraphExec_t aty_exec_proj;
  cudaGraph_t aty_graph_proj;

  cudaGraphExec_t aty_next_exec_proj;
  cudaGraph_t aty_graph_proj_next;

  primal_projection_t<i_t, f_t> current_primal_projection_functor_;
  primal_projection_t<i_t, f_t> next_primal_projection_functor_;

  cudaGraphExec_t ax_exec_proj;
  cudaGraph_t ax_graph_proj;

  cudaGraphExec_t ax_next_exec_proj;
  cudaGraph_t ax_graph_proj_next;

  dual_projection_t<i_t, f_t> current_dual_projection_functor_;
  dual_projection_t<i_t, f_t> next_dual_projection_functor_;
};

}  // namespace cuopt::linear_programming::detail
