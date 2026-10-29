/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/mathematical_optimization/io/data_model_view.hpp>
#include <cuopt/mathematical_optimization/io/mps_data_model.hpp>

#include <string>

namespace cuopt::mathematical_optimization::io {

/**
 * @brief Writes an optimization problem to an LP formatted file
 *
 * Writes CPLEX LP format, readable by read_lp(). Quadratic objective terms,
 * quadratic constraints and semi-continuous variables are written when present.
 *
 * Names: if any variable (constraint) is unnamed, all are written as 'C<j>'
 * ('R<k>'). Given names must be unique, valid LP identifiers and not keywords;
 * otherwise a ValidationError is thrown and no file is written.
 *
 * Limitations:
 *   - A ranged row is written as two rows, '<name>_lo' (>=) and '<name>_up' (<=).
 *   - The objective scaling factor is not written.
 *
 * @tparam i_t  data type of the indices
 * @tparam f_t  data type of the weights and variables
 *
 * @param[in] problem Data model view to write
 * @param[in] lp_file_path Path to the LP file to write
 */
template <typename i_t, typename f_t>
void write_lp(const data_model_view_t<i_t, f_t>& problem, const std::string& lp_file_path);

/**
 * @brief Writes an optimization problem to an LP formatted file
 *
 * Same as the data_model_view_t overload.
 *
 * @param[in] problem Data model to write
 * @param[in] lp_file_path Path to the LP file to write
 */
template <typename i_t, typename f_t>
void write_lp(const mps_data_model_t<i_t, f_t>& problem, const std::string& lp_file_path);

}  // namespace cuopt::mathematical_optimization::io
