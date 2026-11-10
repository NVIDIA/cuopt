/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Experimental CPU ordering of a reduced LP. No presolver state is modified.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace lpfeas_ordering {

struct options {
  bool enabled              = false;
  std::size_t hub_threshold = 1024;
};

inline options from_environment()
{
  options result;
  if (const char* mode = std::getenv("LPFEAS_REORDER")) {
    const std::string value(mode);
    if (value != "off" && value != "rcm") {
      throw std::runtime_error("LPFEAS_REORDER must be off or rcm");
    }
    result.enabled = value == "rcm";
  }
  if (const char* raw = std::getenv("LPFEAS_REORDER_HUB_THRESHOLD")) {
    const std::string value(raw);
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos) {
      throw std::runtime_error("LPFEAS_REORDER_HUB_THRESHOLD must be a positive integer");
    }
    std::size_t consumed = 0;
    const auto threshold = std::stoull(value, &consumed);
    if (consumed != value.size() || threshold == 0 ||
        threshold > std::numeric_limits<std::size_t>::max()) {
      throw std::runtime_error("Invalid ordering hub threshold");
    }
    result.hub_threshold = static_cast<std::size_t>(threshold);
  }
  return result;
}

struct problem_view {
  std::size_t m, n, nnz;
  const int* Ap;
  const int* Ai;
  const double* Ax;
  const double* c;
  const double* lhs;
  const double* rhs;
  const double* lbs;
  const double* ubs;
};

template <class Problem>
problem_view view_of(const Problem& p)
{
  return {p.m, p.n, p.nnz, p.Ap, p.Ai, p.Ax, p.c, p.lhs, p.rhs, p.lbs, p.ubs};
}

inline void validate(const problem_view& p)
{
  if (p.m > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      p.n > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      p.nnz > static_cast<std::size_t>(std::numeric_limits<int>::max()) || !p.Ap ||
      (p.nnz && (!p.Ai || !p.Ax)) || (p.m && (!p.lhs || !p.rhs)) ||
      (p.n && (!p.c || !p.lbs || !p.ubs)) || p.Ap[0] != 0 || p.Ap[p.m] != static_cast<int>(p.nnz)) {
    throw std::runtime_error("Invalid int32 CSR reduced problem for ordering");
  }
  for (std::size_t row = 0; row < p.m; ++row) {
    if (p.Ap[row] < 0 || p.Ap[row] > p.Ap[row + 1] || p.Ap[row + 1] > p.Ap[p.m]) {
      throw std::runtime_error("Nonmonotone ordering CSR offsets");
    }
    for (int k = p.Ap[row]; k < p.Ap[row + 1]; ++k) {
      if (p.Ai[k] < 0 || static_cast<std::size_t>(p.Ai[k]) >= p.n || !std::isfinite(p.Ax[k]) ||
          (k > p.Ap[row] && p.Ai[k - 1] >= p.Ai[k])) {
        throw std::runtime_error("Ordering requires finite sorted CSR without duplicates");
      }
    }
  }
}

struct reordered_problem {
  std::vector<int> Ap, Ai, row_new_to_old, column_new_to_old;
  std::vector<int> row_old_to_new, column_old_to_new;
  std::vector<double> Ax, c, lhs, rhs, lbs, ubs;
  std::size_t excluded_rows = 0, excluded_nnz = 0, graph_components = 0;

  problem_view view() const
  {
    return {lhs.size(),
            c.size(),
            Ax.size(),
            Ap.data(),
            Ai.data(),
            Ax.data(),
            c.data(),
            lhs.data(),
            rhs.data(),
            lbs.data(),
            ubs.data()};
  }

  void restore_solution(std::vector<double>& x,
                        std::vector<double>& y,
                        std::vector<double>& z) const
  {
    if (x.size() < c.size() || z.size() < c.size() || y.size() < lhs.size()) {
      throw std::runtime_error("Wrong solution sizes for inverse ordering");
    }
    std::vector<double> old_x(x), old_y(y), old_z(z);
    for (std::size_t j = 0; j < c.size(); ++j) {
      old_x[column_new_to_old[j]] = x[j];
      old_z[column_new_to_old[j]] = z[j];
    }
    for (std::size_t i = 0; i < lhs.size(); ++i) {
      old_y[row_new_to_old[i]] = y[i];
    }
    x.swap(old_x);
    y.swap(old_y);
    z.swap(old_z);
  }
};

inline reordered_problem reorder(const problem_view& p, std::size_t hub_threshold)
{
  validate(p);
  if (!hub_threshold) { throw std::runtime_error("Zero ordering hub threshold"); }
  reordered_problem result;
  const std::size_t vertices = p.m + p.n;
  std::vector<std::size_t> degree(vertices, 0), graph_offsets(vertices + 1, 0);
  for (std::size_t row = 0; row < p.m; ++row) {
    const auto count = static_cast<std::size_t>(p.Ap[row + 1] - p.Ap[row]);
    if (count > hub_threshold) {
      ++result.excluded_rows;
      result.excluded_nnz += count;
      continue;
    }
    degree[row] = count;
    for (int k = p.Ap[row]; k < p.Ap[row + 1]; ++k) {
      ++degree[p.m + p.Ai[k]];
    }
  }
  std::partial_sum(degree.begin(), degree.end(), graph_offsets.begin() + 1);
  std::vector<std::size_t> neighbors(graph_offsets.back()), cursor(graph_offsets);
  for (std::size_t row = 0; row < p.m; ++row) {
    if (static_cast<std::size_t>(p.Ap[row + 1] - p.Ap[row]) > hub_threshold) { continue; }
    for (int k = p.Ap[row]; k < p.Ap[row + 1]; ++k) {
      const std::size_t col    = p.m + p.Ai[k];
      neighbors[cursor[row]++] = col;
      neighbors[cursor[col]++] = row;
    }
  }
  // Match the replay prototype: stable minimum-degree component seeds, CSR
  // neighbor order, stable degree sorting within each BFS frontier, then reverse.
  std::vector<std::size_t> seeds(vertices), queue(vertices), fresh;
  std::iota(seeds.begin(), seeds.end(), 0);
  const auto lower_degree = [&](std::size_t a, std::size_t b) { return degree[a] < degree[b]; };
  std::stable_sort(seeds.begin(), seeds.end(), lower_degree);
  std::vector<unsigned char> visited(vertices, 0);
  std::size_t head = 0, tail = 0;
  for (const auto seed : seeds) {
    if (visited[seed]) { continue; }
    ++result.graph_components;
    visited[seed] = 1;
    queue[tail++] = seed;
    while (head < tail) {
      const auto vertex = queue[head++];
      fresh.clear();
      for (auto k = graph_offsets[vertex]; k < graph_offsets[vertex + 1]; ++k) {
        const auto adjacent = neighbors[k];
        if (!visited[adjacent]) {
          visited[adjacent] = 1;
          fresh.push_back(adjacent);
        }
      }
      std::stable_sort(fresh.begin(), fresh.end(), lower_degree);
      for (const auto adjacent : fresh) {
        queue[tail++] = adjacent;
      }
    }
  }
  if (tail != vertices) { throw std::runtime_error("Ordering omitted a vertex"); }
  result.row_new_to_old.reserve(p.m);
  result.column_new_to_old.reserve(p.n);
  for (auto it = queue.rbegin(); it != queue.rend(); ++it) {
    if (*it < p.m) {
      result.row_new_to_old.push_back(static_cast<int>(*it));
    } else {
      result.column_new_to_old.push_back(static_cast<int>(*it - p.m));
    }
  }
  result.row_old_to_new.resize(p.m);
  result.column_old_to_new.resize(p.n);
  for (std::size_t i = 0; i < p.m; ++i) {
    result.row_old_to_new[result.row_new_to_old[i]] = int(i);
  }
  for (std::size_t j = 0; j < p.n; ++j) {
    result.column_old_to_new[result.column_new_to_old[j]] = int(j);
  }
  result.Ap.resize(p.m + 1, 0);
  result.Ai.reserve(p.nnz);
  result.Ax.reserve(p.nnz);
  result.lhs.resize(p.m);
  result.rhs.resize(p.m);
  result.c.resize(p.n);
  result.lbs.resize(p.n);
  result.ubs.resize(p.n);
  struct entry {
    int column;
    int old_index;
  };
  std::vector<entry> entries;
  for (std::size_t row = 0; row < p.m; ++row) {
    const auto old_row = result.row_new_to_old[row];
    entries.clear();
    for (int k = p.Ap[old_row]; k < p.Ap[old_row + 1]; ++k) {
      entries.push_back({result.column_old_to_new[p.Ai[k]], k});
    }
    std::sort(entries.begin(), entries.end(), [](entry a, entry b) { return a.column < b.column; });
    for (const auto item : entries) {
      result.Ai.push_back(item.column);
      result.Ax.push_back(p.Ax[item.old_index]);
    }
    result.Ap[row + 1] = static_cast<int>(result.Ai.size());
    result.lhs[row]    = p.lhs[old_row];
    result.rhs[row]    = p.rhs[old_row];
  }
  for (std::size_t col = 0; col < p.n; ++col) {
    const auto old_col = result.column_new_to_old[col];
    result.c[col]      = p.c[old_col];
    result.lbs[col]    = p.lbs[old_col];
    result.ubs[col]    = p.ubs[old_col];
  }
  validate(result.view());
  return result;
}

}  // namespace lpfeas_ordering
