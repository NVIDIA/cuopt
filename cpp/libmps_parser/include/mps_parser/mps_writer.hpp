/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <mps_parser/data_model_view.hpp>
#include <mps_parser/mps_data_model.hpp>

#include <stdarg.h>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cuopt::mps_parser {

/**
 * @brief Main writer class for MPS files
 *
 * @tparam f_t  data type of the weights and variables
 * @tparam i_t  data type of the indices
 */
template <typename i_t, typename f_t>
class mps_writer_t {
 public:
  /**
   * @brief Ctor. Takes a data model view as input and writes it out as a MPS formatted file
   *
   * @param[in] problem Data model view to write
   */
  mps_writer_t(const data_model_view_t<i_t, f_t>& problem);

  /**
   * @brief Ctor. Takes a data model as input and writes it out as a MPS formatted file
   *
   * @param[in] problem Data model to write
   */
  mps_writer_t(const mps_data_model_t<i_t, f_t>& problem);

  /**
   * @brief Writes the problem to an MPS formatted file
   *
   * @param[in] mps_file_path Path to the MPS file to write
   */
  void write(const std::string& mps_file_path);

 private:
  // Owned view (created when constructing from mps_data_model_t)
  std::unique_ptr<data_model_view_t<i_t, f_t>> owned_view_;
  // Reference to the view (either external or owned)
  const data_model_view_t<i_t, f_t>& problem_;

  // Helper to create view from data model
  static data_model_view_t<i_t, f_t> create_view(const mps_data_model_t<i_t, f_t>& model);
};  // class mps_writer_t

}  // namespace cuopt::mps_parser
