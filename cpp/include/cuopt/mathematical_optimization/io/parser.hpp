/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/mathematical_optimization/io/mps_data_model.hpp>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>

namespace cuopt::mathematical_optimization::io {

/**
 * @brief Selects which MPS reader implementation should be used by dispatching entry points.
 *
 * The experimental fast reader is intentionally opt-in. It supports the same free-format
 * MPS/QPS scope as read_mps(): LP, MIP, QP (QUADOBJ/QMATRIX), and QCQP/SOCP (QCMATRIX).
 */
enum class mps_reader_type_t { default_reader, fast_experimental };

/**
 * @brief Reads the equation from an MPS or QPS file.
 *
 * The input file can be a plain text file in MPS-/QPS-format or a compressed MPS/QPS
 * file (.mps.gz, .mps.bz2, or .mps.lz4).
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
 * Note: Compressed MPS files .mps.gz, .mps.bz2, and .mps.lz4 can only be read if
 * zlib, libbzip2, or liblz4 are installed, respectively.
 *
 * @param[in] mps_file_path Path to MPS/QPSfile.
 * @param[in] fixed_mps_format If MPS/QPS file should be parsed as fixed, false by default
 * @return mps_data_model_t A fully formed LP/QP problem which represents the given file
 */
template <typename i_t, typename f_t>
mps_data_model_t<i_t, f_t> read_mps(const std::string& mps_file_path,
                                    bool fixed_mps_format = false);

/**
 * @brief Reads an MPS/QPS problem with the experimental SIMD-optimized reader.
 *
 * Supports the same free-format LP/MIP/QP/QCQP (SOCP-relevant QCMATRIX) scope as read_mps().
 * Fixed MPS format forcing is not supported. Accepts .mps/.qps and their .gz/.bz2/.lz4 variants
 * (compression is detected from the file path, same as read_mps()).
 *
 * @param[in] mps_file_path Path to a raw or compressed .mps or .qps file.
 * @return mps_data_model_t A fully formed LP/MIP/QP problem which represents the given file.
 */
template <typename i_t, typename f_t>
mps_data_model_t<i_t, f_t> read_mps_fast_experimental(const std::string& mps_file_path);

/**
 * @brief Reads an MPS problem from in-memory file contents.
 *
 * This parses the same plain-text MPS format as read_mps(), but the input is
 * already loaded in memory. Compressed .mps.gz/.mps.bz2 inputs are only supported
 * by read_mps() because compression is detected from the file path.
 *
 * @param[in] mps_contents MPS file contents.
 * @param[in] fixed_mps_format If MPS content should be parsed as fixed, false by default.
 * @return mps_data_model_t A fully formed problem which represents the given content.
 */
template <typename i_t, typename f_t>
mps_data_model_t<i_t, f_t> read_mps_from_string(std::string_view mps_contents,
                                                bool fixed_mps_format = false);

/**
 * @brief Reads a linear, mixed-integer, or quadratic optimization problem from
 *        a file in LP format.
 *
 * The LP format is a human-readable alternative to MPS format. This parser
 * supports the dialect in which the objective and constraints are written
 * as algebraic expressions over named variables (it does not implement the
 * alternative tableau-style LP dialect used by some open-source readers).
 *
 * Scope: LP, MIP, and QP problems are supported, plus semi-continuous
 * variables (via a Semi-Continuous section; finite upper bound required)
 * and quadratic constraints (QCQP; `<=` only).
 *
 * Quadratic terms appear inside `[ ... ]` blocks. The convention differs
 * between objective and constraints:
 *   - Objective bracket: MUST be followed by `/ 2` (the LP file states
 *     coefficients in the `0.5 x^T Q x` convention).
 *   - Constraint bracket: MUST NOT be followed by `/ 2`; coefficients are
 *     taken at face value (`x^T Q x`).
 *
 * SOS constraints, PWL objectives, general constraints, and user cuts cause
 * a ValidationError when encountered.
 *
 * Compressed inputs (.lp.gz, .lp.bz2) are supported when zlib / libbzip2
 * are installed (same dispatching as read_mps).
 *
 * @param[in] lp_file_path Path to the LP file.
 * @return mps_data_model_t A fully formed LP/MIP/QP problem representing the
 *         given file.
 */
template <typename i_t, typename f_t>
mps_data_model_t<i_t, f_t> read_lp(const std::string& lp_file_path);

/**
 * @brief Reads an LP, MIP, or QP problem from in-memory file contents.
 *
 * This parses the same plain-text LP format as read_lp(), but the input is
 * already loaded in memory. Compressed .lp.gz/.lp.bz2 inputs are only
 * supported by read_lp() because compression is detected from the file
 * path. Supports the same scope as read_lp() (LP, MIP, QP, plus
 * semi-continuous variables).
 *
 * @param[in] lp_contents LP file contents.
 * @return mps_data_model_t A fully formed LP/MIP/QP problem representing the
 *         given content.
 */
template <typename i_t, typename f_t>
mps_data_model_t<i_t, f_t> read_lp_from_string(std::string_view lp_contents);

/**
 * @brief Reads an optimization problem from a file, dispatching on the file
 *        extension. Extension matching is case-insensitive.
 *
 * Routing (case-insensitive extensions):
 *   - .mps, .mps.gz, .mps.bz2, .mps.lz4, .qps, .qps.gz, .qps.bz2, .qps.lz4
 *     → read_mps() when mps_reader == default_reader, or read_mps_fast_experimental()
 *       when mps_reader == fast_experimental (fixed_mps_format must be false)
 *   - .lp,  .lp.gz,  .lp.bz2, .lp.lz4 → read_lp()
 *   - anything else → std::logic_error
 *
 * This is the entry point of choice for user-facing tools (CLI, C API) that
 * want both formats to "just work" without an explicit format flag.
 *
 * @param[in] path Path to the input file.
 * @param[in] mps_reader Selects the MPS reader implementation for MPS/QPS inputs.
 * @param[in] fixed_mps_format If the MPS/QPS reader should use fixed format;
 *             ignored for LP inputs. False by default.
 * @return mps_data_model_t The parsed problem.
 */
template <typename i_t, typename f_t>
inline mps_data_model_t<i_t, f_t> read(const std::string& path,
                                       mps_reader_type_t mps_reader,
                                       bool fixed_mps_format = false)
{
  std::string lower(path);
  std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  for (const char* compression_suffix : {".bz2", ".gz", ".lz4"}) {
    if (lower.ends_with(compression_suffix)) {
      lower.resize(lower.size() - std::strlen(compression_suffix));
      break;
    }
  }
  if (lower.ends_with(".mps") || lower.ends_with(".qps")) {
    if (mps_reader == mps_reader_type_t::fast_experimental) {
      if (fixed_mps_format) {
        throw std::logic_error(
          "experimental fast MPS reader does not support fixed MPS format forcing");
      }
      return read_mps_fast_experimental<i_t, f_t>(path);
    }
    return read_mps<i_t, f_t>(path, fixed_mps_format);
  }
  if (lower.ends_with(".lp")) { return read_lp<i_t, f_t>(path); }
  throw std::logic_error(
    "read: unrecognized input file extension. Supported (case-insensitive): "
    ".mps, .mps.gz, .mps.bz2, .mps.lz4, .qps, .qps.gz, .qps.bz2, .qps.lz4, "
    ".lp, .lp.gz, .lp.bz2, .lp.lz4. "
    "Given path: " +
    path);
}

/**
 * @brief Reads an optimization problem from a file, dispatching on the file
 *        extension. Extension matching is case-insensitive.
 *
 * Uses the default MPS reader. See the 3-argument read() overload for routing
 * details and supported extensions.
 *
 * @param[in] path Path to the input file.
 * @param[in] fixed_mps_format If the MPS/QPS reader should use fixed format;
 *             ignored for LP inputs. False by default.
 * @return mps_data_model_t The parsed problem.
 */
template <typename i_t, typename f_t>
inline mps_data_model_t<i_t, f_t> read(const std::string& path, bool fixed_mps_format = false)
{
  return read<i_t, f_t>(path, mps_reader_type_t::default_reader, fixed_mps_format);
}

}  // namespace cuopt::mathematical_optimization::io
