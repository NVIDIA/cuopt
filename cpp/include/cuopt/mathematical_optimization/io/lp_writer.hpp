/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/mathematical_optimization/io/data_model_view.hpp>
#include <cuopt/mathematical_optimization/io/mps_data_model.hpp>

#include <memory>
#include <string>

namespace cuopt::mathematical_optimization::io {

/**
 * @brief Main writer class for LP files
 *
 * Writes an optimization problem to a file in the LP format understood by
 * read_lp(). The emitted dialect is a superset of what a typical solver
 * expects and round-trips through read_lp():
 *   - Minimize / Maximize objective (with optional quadratic '[ ... ] / 2' term)
 *   - Subject To constraints (linear, plus optional quadratic '[ ... ]' term)
 *   - Bounds
 *   - Generals / Binaries / Semi-Continuous variable sections
 *
 * Notes / limitations that mirror the LP format itself:
 *   - Range rows (a linear row with two distinct finite bounds) are emitted as
 *     two constraints ('_lo' with '>=' and '_up' with '<=') because the LP
 *     format has no single-line ranged-row syntax.
 *   - The objective scaling factor is not representable in LP format and is
 *     therefore not written (identical to the MPS writer).
 *
 * @tparam i_t  data type of the indices
 * @tparam f_t  data type of the weights and variables
 */
template <typename i_t, typename f_t>
class lp_writer_t {
 public:
  /**
   * @brief Ctor. Takes a data model view as input and writes it out as an LP formatted file
   *
   * @param[in] problem Data model view to write
   */
  lp_writer_t(const data_model_view_t<i_t, f_t>& problem);

  /**
   * @brief Ctor. Takes a data model as input and writes it out as an LP formatted file
   *
   * @param[in] problem Data model to write
   */
  lp_writer_t(const mps_data_model_t<i_t, f_t>& problem);

  /**
   * @brief Writes the problem to an LP formatted file
   *
   * @param[in] lp_file_path Path to the LP file to write
   */
  void write(const std::string& lp_file_path);

 private:
  // Owned view (created when constructing from mps_data_model_t)
  std::unique_ptr<data_model_view_t<i_t, f_t>> owned_view_;
  // Reference to the view (either external or owned)
  const data_model_view_t<i_t, f_t>& problem_;

  // Helper to create view from data model
  static data_model_view_t<i_t, f_t> create_view(const mps_data_model_t<i_t, f_t>& model);
};  // class lp_writer_t

}  // namespace cuopt::mathematical_optimization::io
