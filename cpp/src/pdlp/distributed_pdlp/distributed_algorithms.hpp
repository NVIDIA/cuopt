/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuopt/mathematical_optimization/pdlp/pdlp_hyper_params.cuh>

#include <rmm/device_uvector.hpp>

// Algorithm-level distributed PDLP
namespace cuopt::mathematical_optimization::pdlp {

template <typename i_t, typename f_t>
struct multi_gpu_engine_t;

template <typename i_t, typename f_t>
class pdhg_solver_t;

// Global bound/objective rescaling: allreduce the owned partial squared norms
// of the constraint bounds and (weighted) objective, then apply the identical
// scalar on every shard.
template <typename i_t, typename f_t>
void distributed_bound_objective_rescaling(multi_gpu_engine_t<i_t, f_t>& engine,
                                           f_t c_scaling_weight);

// Distributed Ruiz inf-scaling (num_iter passes). Each shard computes both its
// owned-row and owned-column inf-norms locally; a per-iteration halo broadcast
// of both cumulative scalings is the only cross-shard communication.
template <typename i_t, typename f_t>
void distributed_ruiz_inf_scaling(multi_gpu_engine_t<i_t, f_t>& engine,
                                  int num_iter,
                                  i_t n_global_vars);

// Distributed Pock-Chambolle scaling (one pass), mirroring the single-GPU
// pock_chambolle_scaling.
template <typename i_t, typename f_t>
void distributed_pock_chambolle_scaling(multi_gpu_engine_t<i_t, f_t>& engine,
                                        f_t alpha,
                                        i_t n_global_vars);

// Full distributed scaling entry point. Mirrors what scale_problem() does in
// single-GPU by orchestrating:
//   - reset per-shard scaling state
//   - Ruiz inf-scaling -> populates cumulative row/col scalings
//   - Pock-Chambolle scaling -> same
//   - per-shard apply_cummulative_scaling_to_problem() to apply the cumulative
//     scalings to A, c, variable and constraint bounds (this is scale_problem()
//     minus its shard-local bound/objective rescaling)
//   - global bound/objective rescaling via distributed_bound_objective_rescaling
template <typename i_t, typename f_t>
void distributed_scaling(multi_gpu_engine_t<i_t, f_t>& engine,
                         pdlp_hyper_params_t const& hyper_params,
                         i_t n_global_vars,
                         bool inside_mip);

// Distributed sigma_max(A) via power iteration (used to seed the initial
// step size). Returns the largest singular value of the scaled constraint
// matrix; identical on every shard.
template <typename i_t, typename f_t>
f_t distributed_max_singular_value(multi_gpu_engine_t<i_t, f_t>& engine,
                                   i_t n_global_cstrs,
                                   int max_iterations = 5000,
                                   f_t tolerance      = 1e-4);

// Gather the global potential_next primal/dual solutions and the reduced cost
// onto the master from the owned slices distributed across shards.
template <typename i_t, typename f_t>
void gather_potential_next_solutions_to_master(multi_gpu_engine_t<i_t, f_t>& engine,
                                               pdhg_solver_t<i_t, f_t>& master_pdhg,
                                               rmm::device_uvector<f_t>& master_reduced_cost);

}  // namespace cuopt::mathematical_optimization::pdlp
