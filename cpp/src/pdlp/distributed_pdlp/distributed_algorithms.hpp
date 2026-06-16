/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

// Algorithm-level distributed PDLP numerical methods.
namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
struct multi_gpu_engine_t;

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

// Distributed sigma_max(A) via power iteration (used to seed the initial
// step size). Returns the largest singular value of the scaled constraint
// matrix; identical on every shard.
template <typename i_t, typename f_t>
f_t distributed_max_singular_value(multi_gpu_engine_t<i_t, f_t>& engine,
                                   i_t n_global_cstrs,
                                   int max_iterations = 5000,
                                   f_t tolerance      = 1e-4);

}  // namespace cuopt::linear_programming::detail
