/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Local LPfeas experiment: maps owned by one native presolver instance.
#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace lpfeas_ordering {

struct native_rcm_state {
  bool active = false;
  std::vector<int> row_new_to_old, column_new_to_old;
  std::size_t excluded_rows = 0, excluded_nnz = 0, graph_components = 0;

  template <class T>
  void restore(std::vector<T>& x, std::vector<T>& y, std::vector<T>& z) const
  {
    if (!active) { return; }
    if (x.size() != column_new_to_old.size() || z.size() != column_new_to_old.size() ||
        y.size() != row_new_to_old.size()) {
      throw std::runtime_error("Native RCM inverse received a different reduced-model shape");
    }
    std::vector<T> old_x(x.size()), old_y(y.size()), old_z(z.size());
    for (std::size_t j = 0; j < column_new_to_old.size(); ++j) {
      old_x[column_new_to_old[j]] = x[j];
      old_z[column_new_to_old[j]] = z[j];
    }
    for (std::size_t i = 0; i < row_new_to_old.size(); ++i) {
      old_y[row_new_to_old[i]] = y[i];
    }
    x.swap(old_x);
    y.swap(old_y);
    z.swap(old_z);
  }
};

}  // namespace lpfeas_ordering
