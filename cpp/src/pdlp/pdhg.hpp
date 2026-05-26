/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once
#include <cuopt/linear_programming/pdlp/pdlp_hyper_params.cuh>
#include <mip_heuristics/problem/problem.cuh>
#include <pdlp/cusparse_view.hpp>
#include <pdlp/pdlp_climber_strategy.hpp>
#include <pdlp/saddle_point.hpp>
#include <pdlp/swap_and_resize_helper.cuh>
#include <pdlp/utilities/ping_pong_graph.cuh>

#include <raft/core/handle.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <tuple>
#include <vector>

namespace cuopt::linear_programming::detail {

// Forward-declared to avoid include cycle: multi_gpu_engine.hpp itself includes pdhg.hpp
// (engine calls per-shard pdhg compute_*). pdhg.cu does the full include.
template <typename i_t, typename f_t>
struct multi_gpu_engine_t;

template <typename i_t, typename f_t>
class pdhg_solver_t {
 public:
  pdhg_solver_t(raft::handle_t const* handle_ptr,
                problem_t<i_t, f_t>& op_problem,
                bool is_legacy_batch_mode,
                const std::vector<pdlp_climber_strategy_t>& climber_strategies,
                const pdlp_hyper_params::pdlp_hyper_params_t& hyper_params,
                const std::vector<std::tuple<i_t, i_t, f_t, f_t>>& new_bounds,
                bool enable_mixed_precision_spmv = false);

  saddle_point_state_t<i_t, f_t>& get_saddle_point_state();
  cusparse_view_t<i_t, f_t>& get_cusparse_view();
  rmm::device_uvector<f_t>& get_primal_tmp_resource();
  rmm::device_uvector<f_t>& get_dual_tmp_resource();
  rmm::device_uvector<f_t>& get_potential_next_primal_solution();
  rmm::device_uvector<f_t>& get_dual_slack();
  const rmm::device_uvector<f_t>& get_potential_next_primal_solution() const;
  rmm::device_uvector<f_t>& get_potential_next_dual_solution();
  const rmm::device_uvector<f_t>& get_potential_next_dual_solution() const;
  rmm::device_uvector<f_t>& get_reflected_dual();
  rmm::device_uvector<f_t>& get_reflected_primal();
  const rmm::device_uvector<f_t>& get_reflected_dual() const;
  const rmm::device_uvector<f_t>& get_reflected_primal() const;
  i_t get_total_pdhg_iterations();
  rmm::device_scalar<i_t>& get_d_total_pdhg_iterations();
  rmm::device_uvector<f_t>& get_primal_solution();
  rmm::device_uvector<f_t>& get_dual_solution();
  i_t get_primal_size() const;
  i_t get_dual_size() const;

  void swap_context(const thrust::universal_host_pinned_vector<swap_pair_t<i_t>>& swap_pairs);
  void resize_and_swap_new_bounds_context(
    const thrust::universal_host_pinned_vector<swap_pair_t<i_t>>& swap_pairs, i_t new_size);
  void resize_context(i_t new_size);
  ping_pong_graph_t<i_t>& get_graph_all();

  rmm::device_uvector<i_t>& get_new_bounds_climber_id() { return new_bounds_climber_id_; }
  rmm::device_uvector<i_t>& get_new_bounds_idx() { return new_bounds_idx_; }
  rmm::device_uvector<f_t>& get_new_bounds_lower() { return new_bounds_lower_; }
  rmm::device_uvector<f_t>& get_new_bounds_upper() { return new_bounds_upper_; }

  void take_step(rmm::device_uvector<f_t>& primal_step_size,
                 rmm::device_uvector<f_t>& dual_step_size,
                 const rmm::device_uvector<f_t>& bound_rescaling,  // Only used in batch mode
                 i_t iterations_since_last_restart,
                 bool last_restart_was_average,
                 i_t total_pdlp_iterations,
                 bool is_major_iteration);
  void update_solution(cusparse_view_t<i_t, f_t>& current_op_problem_evaluation_cusparse_view_);
  void refine_initial_primal_projection(const rmm::device_uvector<f_t>& bound_rescaling);

  // SpMV primitives. Public so the multi-GPU engine can drive them per-shard
  // after halo-exchanging the relevant vector. Single-GPU PDLP still calls
  // them internally via take_step / compute_next_*.
  //
  // If set_multi_gpu_engine() has been called, these dispatch to the engine
  // (halo exchange + per-shard SpMV). Otherwise they run the single-GPU
  // cusparse path on the local matrix.
  void compute_At_y();
  void compute_A_x();
  void spmvop_At_y();
  void spmvop_A_x();

  // Parameterized SpMVs used by the multi-GPU engine.
  // Both temporarily hijack a canonical input descriptor in cusparse_view_
  // (cv.dual_solution for At, cv.reflected_primal_solution for A) to point at
  // `in_buf.data()`, run the local SpMV into `out_desc`, then restore the
  // descriptor to its original buffer so other code on this shard is unaffected.
  // No multi-GPU dispatch inside — the engine is the orchestrator.
  void spmv_At_into(rmm::device_uvector<f_t>& in_buf, cusparseDnVecDescr_t out_desc);
  void spmv_A_into(rmm::device_uvector<f_t>& in_buf, cusparseDnVecDescr_t out_desc);

  // Pure cub-transform extractions. Each one is byte-identical to the inline
  // cub call it replaces — no platform dispatch inside. Callers handle the
  // single-GPU vs per-shard branching at the call site (see the
  // "if (mgpu_engine_) for shard..." blocks in compute_next_*).
  void primal_reflected_major_projection_transform(rmm::device_uvector<f_t>& primal_step_size);
  void dual_reflected_major_projection_transform(rmm::device_uvector<f_t>& dual_step_size);
  void primal_reflected_projection_transform(rmm::device_uvector<f_t>& primal_step_size);
  void dual_reflected_projection_transform(rmm::device_uvector<f_t>& dual_step_size);

  // Master PDLP wires up the engine pointer here after the engine is built.
  // Shards' pdhg_solver_ leaves this null so each shard runs single-GPU SpMV
  // on its local matrix. Also flips is_multi_gpu_ — convenience flag that any
  // pdhg participating in a distributed run (master OR shard) carries true.
  void set_multi_gpu_engine(multi_gpu_engine_t<i_t, f_t>* engine)
  {
    mgpu_engine_  = engine;
    is_multi_gpu_ = (engine != nullptr);
  }

  // Mark a shard's pdhg_solver_ as part of a distributed run without giving it
  // an engine (shards don't orchestrate; they only run local SpMV on owned
  // rows). Called from shard.cu right after sub_pdlp is constructed.
  void set_is_multi_gpu(bool v) { is_multi_gpu_ = v; }
  bool is_multi_gpu() const { return is_multi_gpu_; }
  multi_gpu_engine_t<i_t, f_t>* get_mgpu_engine() const { return mgpu_engine_; }

  i_t total_pdhg_iterations_;

 private:
  void compute_next_primal_dual_solution(rmm::device_uvector<f_t>& primal_step_size,
                                         i_t iterations_since_last_restart,
                                         bool last_restart_was_average,
                                         rmm::device_uvector<f_t>& dual_step_size,
                                         i_t total_pdlp_iterations);
  void compute_next_dual_solution(rmm::device_uvector<f_t>& dual_step_size);
  void compute_next_primal_dual_solution_reflected(
    rmm::device_uvector<f_t>& primal_step_size,
    rmm::device_uvector<f_t>& dual_step_size,
    const rmm::device_uvector<f_t>& bound_rescaling,  // Only used in batch mode
    bool should_major);

  void compute_primal_projection_with_gradient(rmm::device_uvector<f_t>& primal_step_size);
  void compute_primal_projection(rmm::device_uvector<f_t>& primal_step_size);

  bool batch_mode_{false};
  bool is_multi_gpu_{false};
  raft::handle_t const* handle_ptr_{nullptr};
  rmm::cuda_stream_view stream_view_;

  problem_t<i_t, f_t>* problem_ptr;

  i_t primal_size_h_;
  i_t dual_size_h_;

  rmm::device_uvector<f_t> tmp_primal_;
  rmm::device_uvector<f_t> tmp_dual_;

  saddle_point_state_t<i_t, f_t> current_saddle_point_state_;

  rmm::device_uvector<f_t> potential_next_primal_solution_;
  rmm::device_uvector<f_t> potential_next_dual_solution_;

  rmm::device_uvector<f_t> dual_slack_;
  rmm::device_uvector<f_t> reflected_primal_;
  rmm::device_uvector<f_t> reflected_dual_;

  // Important that vectors passed down to the cusparse_view are allocated before
  cusparse_view_t<i_t, f_t> cusparse_view_;

  const rmm::device_scalar<f_t> reusable_device_scalar_value_1_;
  const rmm::device_scalar<f_t> reusable_device_scalar_value_0_;
  const rmm::device_scalar<f_t> reusable_device_scalar_value_neg_1_;
  rmm::device_scalar<f_t> reusable_device_scalar_1_;

  // Different graphs for each case
  // Either compute the whole next primal step
  // Or skip the SpMV (most cases) if it was done at the previous iteration.
  // The reflected primal/dual path branches on `should_major`, and the two branches build
  // different graph topologies. They get separate ping-pong caches so each branch can key its
  // 2-slot cache on `total_pdlp_iterations` parity (the swap state of the primal/dual buffers
  // baked into the captured graph) without colliding with the other branch's topology.
  // graph_all serves the non-reflected path and the major reflected branch (mutually exclusive
  // at runtime); graph_all_non_major serves the non-major reflected branch.
  ping_pong_graph_t<i_t> graph_all;
  ping_pong_graph_t<i_t> graph_prim_proj_gradient_dual;

  // Needed for faster graph launch
  // Passing the host value each time would require updating the graph each time
  rmm::device_scalar<i_t> d_total_pdhg_iterations_;

  const std::vector<pdlp_climber_strategy_t>& climber_strategies_;
  const pdlp_hyper_params::pdlp_hyper_params_t& hyper_params_;
  rmm::device_uvector<i_t> new_bounds_climber_id_;
  rmm::device_uvector<i_t> new_bounds_idx_;
  rmm::device_uvector<f_t> new_bounds_lower_;
  rmm::device_uvector<f_t> new_bounds_upper_;
  cuda::fast_mod_div<size_t> batch_size_divisor_;

  // Non-owning. Set on the master pdhg_solver_ in distributed mode; null
  // (default) means single-GPU path. See compute_At_y / compute_A_x.
  multi_gpu_engine_t<i_t, f_t>* mgpu_engine_{nullptr};
};

}  // namespace cuopt::linear_programming::detail
