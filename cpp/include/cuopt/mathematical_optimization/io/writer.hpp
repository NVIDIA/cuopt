/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/mathematical_optimization/io/data_model_view.hpp>

namespace cuopt::mathematical_optimization::io {

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

}  // namespace cuopt::mathematical_optimization::io
