/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <dual_simplex/solution.hpp>
#include <dual_simplex/sparse_matrix.hpp>
#include <dual_simplex/types.hpp>

#include <raft/core/handle.hpp>

#include <fstream>
#include <iostream>
#include <string>

namespace cuopt::linear_programming::dual_simplex {

enum class variable_type_t : int8_t {
  CONTINUOUS = 0,
  BINARY     = 1,
  INTEGER    = 2,
};

template <typename T>
bool print_vec(const std::vector<T>& vec, const std::string& name)
{
  std::cout << name << ": " << vec.size() << std::endl;
  for (size_t i = 0; i < vec.size(); i++) {
    std::cout << vec[i] << " ";
  }
  std::cout << std::endl;
  return true;
}

template <typename i_t, typename f_t>
struct user_problem_t {
  user_problem_t(raft::handle_t const* handle_ptr_)
    : handle_ptr(handle_ptr_), A(1, 1, 1), obj_constant(0.0), obj_scale(1.0)
  {
  }

  void dump() const
  {
    // Open a binary file for output
    std::ofstream fout("user_problem_dump.bin");
    if (!fout.is_open()) {
      std::cerr << "Failed to open file for dump!" << std::endl;
      return;
    }

    // Dump A: convert to CSR first
    // For the purpose of this dump, we assume A is in CSC (Compressed Sparse Column), so we need to
    // convert it to CSR assuming A.m (#rows), A.n (#cols), A.col_start, A.row_ind, A.x

    // Step 1: Build row_start (size m+1)
    std::vector<i_t> row_start(num_rows + 1, 0);
    std::vector<i_t> csr_i;
    std::vector<f_t> csr_x;
    csr_i.reserve(A.nnz());
    csr_x.reserve(A.nnz());

    // Compute row counts
    for (i_t col = 0; col < num_cols; ++col) {
      for (i_t idx = A.col_start[col]; idx < A.col_start[col + 1]; ++idx) {
        i_t row = A.i[idx];
        row_start[row + 1]++;
      }
    }
    // Inclusive prefix sum for row_start
    for (i_t i = 0; i < num_rows; ++i)
      row_start[i + 1] += row_start[i];

    // Prepare temporary workspace to fill csr_i/x
    std::vector<i_t> current(row_start.begin(), row_start.end());
    std::vector<i_t> csr_i_(A.nnz(), -1);
    std::vector<f_t> csr_x_(A.nnz());

    for (i_t col = 0; col < num_cols; ++col) {
      for (i_t idx = A.col_start[col]; idx < A.col_start[col + 1]; ++idx) {
        i_t row      = A.i[idx];
        i_t dest     = current[row]++;
        csr_i_[dest] = col;
        csr_x_[dest] = A.x[idx];
      }
    }

    print_vec(row_start, "row_start");
    print_vec(csr_i_, "csr_i_");
    print_vec(csr_x_, "csr_x_");

    // Write out row_start, csr_i, csr_x
    // Write out num_rows and nnz to the binary file
    fout.write(reinterpret_cast<const char*>(&num_rows), sizeof(i_t));
    fout.write(reinterpret_cast<const char*>(&num_cols), sizeof(i_t));
    i_t nnz = A.nnz();
    fout.write(reinterpret_cast<const char*>(&nnz), sizeof(i_t));
    fout.write(reinterpret_cast<const char*>(row_start.data()), sizeof(i_t) * (num_rows + 1));
    fout.write(reinterpret_cast<const char*>(csr_i_.data()), sizeof(i_t) * nnz);
    fout.write(reinterpret_cast<const char*>(csr_x_.data()), sizeof(f_t) * nnz);

    std::vector<int> row_sense_int(num_rows);
    for (int i = 0; i < num_rows; i++) {
      if (row_sense[i] == 'L') {
        row_sense_int[i] = 0;
      } else if (row_sense[i] == 'G') {
        row_sense_int[i] = 1;
      } else {
        row_sense_int[i] = 2;
      }
    }
    fout.write(reinterpret_cast<const char*>(row_sense_int.data()), sizeof(int) * num_rows);

    // Dump Q_offsets, Q_indices, Q_values (assume sizes are Q_offsets.size() = num_cols+1,
    // Q_indices.size() == Q_values.size())
    i_t Q_offsets_size = Q_offsets.size();
    fout.write(reinterpret_cast<const char*>(&Q_offsets_size), sizeof(i_t));
    fout.write(reinterpret_cast<const char*>(Q_offsets.data()), sizeof(i_t) * Q_offsets_size);

    i_t Q_indices_size = Q_indices.size();
    fout.write(reinterpret_cast<const char*>(&Q_indices_size), sizeof(i_t));
    fout.write(reinterpret_cast<const char*>(Q_indices.data()), sizeof(i_t) * Q_indices_size);

    fout.write(reinterpret_cast<const char*>(Q_values.data()), sizeof(f_t) * Q_indices_size);

    print_vec<i_t>(Q_offsets, "Q_offsets");
    print_vec<i_t>(Q_indices, "Q_indices");
    print_vec<f_t>(Q_values, "Q_values");

    print_vec<f_t>(objective, "objective");
    print_vec<f_t>(rhs, "rhs");
    print_vec<f_t>(lower, "lower");
    print_vec<f_t>(upper, "upper");

    print_vec<char>(row_sense, "row_sense");

    // Dump objective as c
    fout.write(reinterpret_cast<const char*>(objective.data()), sizeof(f_t) * num_cols);

    // Dump rhs as b
    fout.write(reinterpret_cast<const char*>(rhs.data()), sizeof(f_t) * num_rows);

    // Dump lower and upper
    fout.write(reinterpret_cast<const char*>(lower.data()), sizeof(f_t) * num_cols);
    fout.write(reinterpret_cast<const char*>(upper.data()), sizeof(f_t) * num_cols);

    fout.close();
  }

  void apply_scaling()
  {
#if 0
    // dump();
    // exit(0);
    // Read LOWER_BOUND and UPPER_BOUND from environment, use if set, else defaults
    {
      const char* lower_env = std::getenv("LOWER_BOUND");
      const char* upper_env = std::getenv("UPPER_BOUND");
      f_t lower_fallback    = -100;
      f_t upper_fallback    = 100;
      f_t lower_bound       = lower_env ? static_cast<f_t>(std::atof(lower_env)) : lower_fallback;
      f_t upper_bound       = upper_env ? static_cast<f_t>(std::atof(upper_env)) : upper_fallback;
      for (int i = 0; i < num_cols; i++) {
        if (!std::isfinite(lower[i]) && lower_env) { lower[i] = lower_bound; }
        if (!std::isfinite(upper[i]) && upper_env) { upper[i] = upper_bound; }
      }
    }

#endif
    if (false && Q_offsets.size() > 0) {
      double eps = 1e-6;
      std::vector<f_t> D(num_cols, 1.0);
      for (int i = 0; i < num_cols; i++) {
        for (int jj = Q_offsets[i]; jj < Q_offsets[i + 1]; jj++) {
          int j = Q_indices[jj];
          if (i == j) {
            f_t di = std::sqrt(Q_values[jj]);
            D[i]   = 1 / (di + eps);
            break;
          }
        }

        std::cout << "D[" << i << "] = " << D[i] << std::endl;
      }

      for (int j = 0; j < num_cols; j++) {
        objective[j] *= D[j];
      }

      for (int i = 0; i < num_cols; i++) {
        for (int jj = Q_offsets[i]; jj < Q_offsets[i + 1]; jj++) {
          int j = Q_indices[jj];
          Q_values[jj] *= D[i] * D[j];
        }
      }

      for (int i = 0; i < num_cols; i++) {
        lower[i] /= D[i];
      }

      for (int i = 0; i < num_cols; i++) {
        upper[i] /= D[i];
      }

      for (int j = 0; j < num_cols; j++) {
        for (int ii = A.col_start[j]; ii < A.col_start[j + 1]; ii++) {
          A.x[ii] *= D[j];
        }
      }
    }
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
  f_t obj_scale;  // 1.0 for min, -1.0 for max
  std::vector<variable_type_t> var_types;
  std::vector<i_t> Q_offsets;
  std::vector<i_t> Q_indices;
  std::vector<f_t> Q_values;
};

}  // namespace cuopt::linear_programming::dual_simplex
