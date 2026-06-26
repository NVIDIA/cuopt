/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <dual_simplex/solution.hpp>
#include <dual_simplex/sparse_matrix.hpp>
#include <dual_simplex/types.hpp>

#include <raft/core/handle.hpp>

#include <string>

namespace cuopt::linear_programming::dual_simplex {

enum class variable_type_t : int8_t {
  CONTINUOUS = 0,
  BINARY     = 1,
  INTEGER    = 2,
};

// The objective function takes values on a lattice: k * step_size + bias
// for integer k. A step_size of 0 means no lattice structure is known.
template <typename f_t>
struct objective_step_t {
  f_t step_size{0};
  f_t bias{0};

  bool has_step() const { return step_size > 0; }
};

template <typename i_t, typename f_t>
struct user_problem_t {
  user_problem_t(raft::handle_t const* handle_ptr_)
    : handle_ptr(handle_ptr_), A(1, 1, 1), obj_constant(0.0), obj_scale(1.0)
  {
  }

  raft::handle_t const* handle_ptr;
  i_t num_rows;
  i_t num_cols;
  std::vector<f_t> objective;
  csc_matrix_t<i_t, f_t> A;
  std::vector<f_t> rhs;
  std::vector<char> row_sense;
  std::vector<f_t> lower;
  std::vector<f_t> upper;
  std::vector<i_t> range_rows;
  std::vector<f_t> range_value;
  i_t num_range_rows;
  std::string problem_name;
  std::vector<std::string> row_names;
  std::vector<std::string> col_names;
  f_t obj_constant;
  f_t obj_scale;  // positive for min, netagive for max
  bool objective_is_integral{false};
  objective_step_t<f_t> objective_step;
  std::vector<variable_type_t> var_types;
  std::vector<i_t> Q_offsets;
  std::vector<i_t> Q_indices;
  std::vector<f_t> Q_values;
  i_t cone_var_start{0};
  std::vector<i_t> second_order_cone_dims;
  // Column count before QCMATRIX->SOC expansion. When > 0, the barrier solution is in the
  // expanded layout (num_cols) and must be projected back via original_col_to_expanded_col.
  i_t original_num_cols{0};
  std::vector<i_t> original_col_to_expanded_col;
  // Linear constraint count before QCMATRIX->SOC expansion (user-facing rows).
  i_t original_num_rows{0};
  /** How each quadratic constraint was recognized during SOC conversion. */
  enum class qc_soc_recognition_path_t : int8_t {
    LORENTZ = 0,
    AFFINE  = 1,
    ROTATED = 2,
    GENERAL = 3,
  };
  /** Per-QC metadata for @ref project_barrier_qcqp_duals_to_model (barrier SOCP -> user QCQP). */
  struct qc_dual_recovery_entry_t {
    qc_soc_recognition_path_t path{qc_soc_recognition_path_t::LORENTZ};
    /** Uniform diagonal scale s in -s x_head^2 + s||tail||^2 <= 0 (Lorentz/rotated/affine). */
    f_t uniform_s{1};
    i_t cone_index{-1};
    /** Expanded-model equality row for t = -(1/s)a^T x (affine path); -1 if none. */
    i_t affine_link_row{-1};
    /** Expanded-model rows for s0 + c^T x = alpha + 1/2 and s_{r+1} + c^T x = alpha - 1/2. */
    i_t s0_link_row{-1};
    i_t sr1_link_row{-1};
    /** Rotated lift: equality rows s0 = h(x_h0+x_h1), s1 = h(x_h0-x_h1) (-1 if not rotated). */
    i_t rsoc_s0_lift_row{-1};
    i_t rsoc_s1_lift_row{-1};
    /** h = (1/sqrt(2))*sqrt(d/s) in the two-head lift; inv_sqrt(2) for constant-half affine lift.
     */
    f_t rsoc_head_lift_h{0};
    bool rsoc_head1_is_constant_half{false};
  };
  std::vector<qc_dual_recovery_entry_t> qc_dual_recovery;
};

}  // namespace cuopt::linear_programming::dual_simplex
