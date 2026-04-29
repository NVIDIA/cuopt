/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <mps_parser/mps_data_model.hpp>

#include <string>
#include <string_view>

namespace cuopt::mps_parser {

/**
 * @brief Reads the equation from an MPS or QPS file.
 *
 * The input file can be a plain text file in MPS-/QPS-format or a compressed MPS/QPS
 * file (.mps.gz or .mps.bz2).
 *
 * Read this link http://lpsolve.sourceforge.net/5.5/mps-format.htm for more
 * details on both free and fixed MPS format.
 * This function supports both standard MPS files (for linear programming) and
 * QPS files (for quadratic programming). QPS files are MPS files with additional
 * sections:
 * - QUADOBJ: Defines quadratic terms in the objective function
 * - QMATRIX: Full symmetric quadratic objective matrix (alternative to QUADOBJ)
 * - QCMATRIX: Symmetric quadratic terms for a named constraint row (QCQP)
 *
 * Note: Compressed MPS files .mps.gz, .mps.bz2 can only be read if the compression
 * libraries zlib or libbzip2 are installed, respectively.
 *
 * @param[in] mps_file_path Path to MPS/QPSfile.
 * @param[in] fixed_mps_format If MPS/QPS file should be parsed as fixed, false by default
 * @return mps_data_model_t A fully formed LP/QP problem which represents the given file
 */
template <typename i_t, typename f_t>
mps_data_model_t<i_t, f_t> parse_mps(const std::string& mps_file_path,
                                     bool fixed_mps_format = false);

/**
 * @brief Reads an MPS problem from in-memory file contents.
 *
 * This parses the same plain-text MPS format as parse_mps(), but the input is
 * already loaded in memory. Compressed .mps.gz/.mps.bz2 inputs are only supported
 * by parse_mps() because compression is detected from the file path.
 *
 * @param[in] mps_contents MPS file contents.
 * @param[in] fixed_mps_format If MPS content should be parsed as fixed, false by default.
 * @return mps_data_model_t A fully formed problem which represents the given content.
 */
template <typename i_t, typename f_t>
mps_data_model_t<i_t, f_t> parse_mps_from_string(std::string_view mps_contents,
                                                 bool fixed_mps_format = false);

}  // namespace cuopt::mps_parser
