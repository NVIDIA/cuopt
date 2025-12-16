/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <mps_parser/data_model_view.hpp>

// TODO: we might want to eventually rename libmps_parser to libmps_io
// (or libcuopt_io if we want to support other hypothetical formats)
namespace cuopt::mps_parser {

/**
 * @brief Writes the problem to an MPS formatted file
 *
 * Read this link http://lpsolve.sourceforge.net/5.5/mps-format.htm for more
 * details on both free and fixed MPS format.
 *
 * @param[in] problem The problem data model view to write
 * @param[in] mps_file_path Path to the MPS file to write
 */
template <typename i_t, typename f_t>
void write_mps(const data_model_view_t<i_t, f_t>& problem, const std::string& mps_file_path);

}  // namespace cuopt::mps_parser
