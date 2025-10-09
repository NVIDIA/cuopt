/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <dual_simplex/folding.hpp>

#include <dual_simplex/tic_toc.hpp>

#include <numeric>
#include <unordered_map>
#include <unordered_set>

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t>
struct color_t {
  color_t(int8_t row_or_column, int8_t active, i_t color, std::vector<i_t> vertices)
    : row_or_column(row_or_column),
      active(active),
      color(color),
      vertices(vertices.begin(), vertices.end())
  {
  }
  int8_t row_or_column;
  int8_t active;
  i_t color;
  std::unordered_set<i_t> vertices;
};

constexpr int8_t kRow      = 0;
constexpr int8_t kCol      = 1;
constexpr int8_t kActive   = 1;
constexpr int8_t kInactive = 0;

template <typename i_t>
void find_vertices_to_refine(const std::unordered_set<i_t>& refining_color_vertices,
                             const std::vector<i_t>& offset,
                             const std::vector<i_t>& vertex_list,
                             const std::vector<i_t>& color_map,
                             std::vector<i_t>& marked_vertices,
                             std::vector<i_t>& vertices_to_refine,
                             std::vector<std::vector<i_t>>& vertices_to_refine_by_color,
                             std::vector<i_t>& marked_colors,
                             std::vector<i_t>& colors_to_update)
{
  for (i_t u : refining_color_vertices) {
    const i_t start = offset[u];
    const i_t end   = offset[u + 1];
    for (i_t p = start; p < end; p++) {
      const i_t v     = vertex_list[p];
      const i_t color = color_map[v];
      if (marked_vertices[v] == 0) {
        marked_vertices[v] = 1;
        vertices_to_refine.push_back(v);
        vertices_to_refine_by_color[color].push_back(v);
      }
      if (marked_colors[color] == 0) {
        marked_colors[color] = 1;
        colors_to_update.push_back(color);
      }
    }
  }
  for (i_t v : vertices_to_refine) {
    marked_vertices[v] = 0;
  }
  for (i_t color : colors_to_update) {
    marked_colors[color] = 0;
  }
}

template <typename i_t, typename f_t>
void compute_sums_of_refined_vertices(i_t refining_color,
                                      const std::unordered_set<i_t>& refining_color_vertices,
                                      const std::vector<i_t>& vertices_to_refine,
                                      const std::vector<i_t>& offsets,
                                      const std::vector<i_t>& vertex_list,
                                      const std::vector<f_t>& weight_list,
                                      const std::vector<i_t>& color_map,
                                      std::vector<f_t>& vertex_to_sum,
                                      std::vector<f_t>& max_sum_by_color)
{
  for (i_t v : refining_color_vertices) {
    const i_t start = offsets[v];
    const i_t end   = offsets[v + 1];
    for (i_t p = start; p < end; p++) {
      const i_t u = vertex_list[p];
      vertex_to_sum[u] += weight_list[p];
    }
  }

  for (i_t v : vertices_to_refine) {
    const i_t c = color_map[v];
    if (std::isnan(max_sum_by_color[c])) {
      max_sum_by_color[c] = vertex_to_sum[v];
    } else {
      max_sum_by_color[c] = std::max(max_sum_by_color[c], vertex_to_sum[v]);
    }
  }
}

template <typename i_t, typename f_t>
void compute_sums(const csc_matrix_t<i_t, f_t>& A,
                  const csr_matrix_t<i_t, f_t>& Arow,
                  i_t num_row_colors,
                  i_t num_col_colors,
                  i_t total_colors_seen,
                  const std::vector<i_t>& row_color_map,
                  const std::vector<i_t>& col_color_map,
                  const color_t<i_t>& refining_color,
                  std::vector<i_t>& colors_to_update,
                  std::vector<i_t>& marked_colors,
                  std::vector<i_t>& vertices_to_refine,
                  std::vector<i_t>& marked_vertices,
                  std::vector<std::vector<i_t>>& vertices_to_refine_by_color,
                  std::vector<f_t>& vertex_to_sum,
                  std::vector<f_t>& max_sum_by_color)
{
  i_t num_colors = num_row_colors + num_col_colors;
  colors_to_update.clear();
  vertices_to_refine.clear();
  if (refining_color.row_or_column == kRow) {
    // The refining color is a row color
    // Find all vertices (columns) that have a neighbor in the refining color
    colors_to_update.reserve(num_col_colors);
    find_vertices_to_refine(refining_color.vertices,
                            Arow.row_start,
                            Arow.j,
                            col_color_map,
                            marked_vertices,
                            vertices_to_refine,
                            vertices_to_refine_by_color,
                            marked_colors,
                            colors_to_update);

    // Let R be the refining color, R is a subset of the rows
    // We need to compute
    // sum_{v in R} A_vw for all w such that A_vw != 0
    // Note that if A_vw != 0 and A_vw' != 0, we may
    // have that w and w' are in a different (column) color
    compute_sums_of_refined_vertices(refining_color.color,
                                     refining_color.vertices,
                                     vertices_to_refine,
                                     Arow.row_start,
                                     Arow.j,
                                     Arow.x,
                                     col_color_map,
                                     vertex_to_sum,
                                     max_sum_by_color);
  } else {
    // The refining color is a column color
    // Find all vertices (rows) that have a neighbor in the refining color
    colors_to_update.reserve(num_row_colors);
    find_vertices_to_refine(refining_color.vertices,
                            A.col_start,
                            A.i,
                            row_color_map,
                            marked_vertices,
                            vertices_to_refine,
                            vertices_to_refine_by_color,
                            marked_colors,
                            colors_to_update);

    // Let Q be the refining color, Q is a subset of the columns
    // We need to compute
    // sum_{w in Q} A_vw for all v such that A_vw != 0
    // Note that if A_vw != 0 and A_v'w != 0, we may
    // have that v and v' are in a different (row) color
    compute_sums_of_refined_vertices(refining_color.color,
                                     refining_color.vertices,
                                     vertices_to_refine,
                                     A.col_start,
                                     A.i,
                                     A.x,
                                     row_color_map,
                                     vertex_to_sum,
                                     max_sum_by_color);
  }
}

template <typename i_t, typename f_t>
void find_colors_to_split(const std::vector<i_t> colors_to_update,
                          const std::vector<color_t<i_t>>& colors,
                          const std::vector<std::vector<i_t>>& vertices_to_refine_by_color,
                          const std::vector<f_t>& vertex_to_sum,
                          std::vector<f_t>& max_sum_by_color,
                          std::vector<f_t>& min_sum_by_color,
                          std::vector<i_t>& colors_to_split)
{
  for (i_t color : colors_to_update) {
    min_sum_by_color[color] = max_sum_by_color[color];
    for (i_t v : vertices_to_refine_by_color[color]) {
      if (vertex_to_sum[v] < min_sum_by_color[color]) {
        min_sum_by_color[color] = vertex_to_sum[v];
      }
    }

    if (vertices_to_refine_by_color[color].size() != colors[color].vertices.size()) {
      // We didn't touch all the vertices in the color. These vertices must have a sum of 0.0
      if (0.0 > max_sum_by_color[color]) { max_sum_by_color[color] = 0.0; }
      if (0.0 < min_sum_by_color[color]) { min_sum_by_color[color] = 0.0; }
    } else {
      // We touched all the vertices in the color. Compute the minimum sum seen
    }
  }

  colors_to_split.clear();
  for (i_t color : colors_to_update) {
    if (max_sum_by_color[color] < min_sum_by_color[color]) {
      printf("Color %d has max sum %e < min sum %e\n",
             color,
             max_sum_by_color[color],
             min_sum_by_color[color]);
      exit(1);
    }
    if (min_sum_by_color[color] < max_sum_by_color[color]) { colors_to_split.push_back(color); }
  }
}

template <typename i_t, typename f_t>
void split_colors(i_t color,
                  i_t refining_color,
                  int8_t side_being_split,
                  std::vector<f_t>& vertex_to_sum,
                  std::unordered_map<f_t, std::vector<i_t>>& color_sums,
                  std::unordered_map<f_t, i_t>& sum_to_sizes,
                  std::vector<color_t<i_t>>& colors,
                  std::vector<i_t>& color_stack,
                  std::vector<i_t>& color_in_stack,
                  std::vector<i_t>& color_map_B,
                  std::vector<i_t>& marked_vertices,
                  std::vector<std::vector<i_t>>& vertices_to_refine_by_color,
                  std::vector<f_t>& min_sum_by_color,
                  std::vector<f_t>& max_sum_by_color,
                  i_t& num_colors,
                  i_t& num_side_colors,
                  i_t& total_colors_seen)
{
  bool in_stack = color_in_stack[color];

  sum_to_sizes.clear();
  color_sums.clear();
  for (i_t v : vertices_to_refine_by_color[color]) {
    sum_to_sizes[vertex_to_sum[v]]++;
  }
  if (vertices_to_refine_by_color[color].size() != colors[color].vertices.size()) {
    const i_t remaining_size =
      colors[color].vertices.size() - vertices_to_refine_by_color[color].size();
    if (remaining_size < 0) {
      printf("Negative remaining size %d\n", remaining_size);

      printf("Color %d vertices\n");
      for (i_t v : colors[color].vertices) {
        printf("Vertex %d\n", v);
      }
      printf("Vertices to refine by color %d\n", color);
      for (i_t v : vertices_to_refine_by_color[color]) {
        printf("Vertex %d\n", v);
      }
      exit(1);
    }
    sum_to_sizes[0.0] += remaining_size;
    color_sums[0.0] = std::vector<i_t>();
  }

  i_t max_size = -1;
  f_t max_sum  = std::numeric_limits<f_t>::quiet_NaN();
  for (auto& [sum, size] : sum_to_sizes) {
    if (size > max_size) {
      max_size = size;
      max_sum  = sum;
    }
  }

  for (i_t v : vertices_to_refine_by_color[color]) {
    color_sums[vertex_to_sum[v]].push_back(v);
  }
  bool only_one = sum_to_sizes.size() == 1;
  if (only_one) {
    printf("Color %d has only one sum. color_sums size %ld. In stack %d\n",
           color,
           color_sums.size(),
           in_stack);
    exit(1);
  }

  i_t vertices_considered = 0;
  for (auto& [sum, vertices] : color_sums) {
    i_t size = vertices.size();
    if (sum == 0.0) {
      const i_t additional_size =
        (colors[color].vertices.size() - vertices_to_refine_by_color[color].size());
      size += additional_size;
    }

    vertices_considered += size;
    if (sum == 0.0) {
      // Push the current color back onto the stack
      if (!in_stack && max_size != size && max_sum != sum) {
        color_stack.push_back(color);
        color_in_stack[color] = 1;
      }
    } else {
      // Create a new color
      colors.emplace_back(side_being_split, kActive, total_colors_seen, vertices);

      // Push the new color onto the stack
      if (in_stack || !(max_size == size && max_sum == sum)) {
        color_stack.push_back(total_colors_seen);
        color_in_stack[total_colors_seen] = 1;
      }

      for (i_t v : vertices) {
        color_map_B[v] = total_colors_seen;
      }

      total_colors_seen++;
      num_colors++;
      num_side_colors++;
    }
  }
  if (vertices_considered != colors[color].vertices.size()) {
    printf("Vertices considered %d does not match color size %ld\n",
           vertices_considered,
           colors[color].vertices.size());
    exit(1);
  }

  for (i_t v : vertices_to_refine_by_color[color]) {
    if (color_map_B[v] != color) { colors[color].vertices.erase(v); }
  }
  if (colors[color].vertices.size() == 0) {
    colors[color].active = kInactive;
    num_colors--;
    num_side_colors--;
  }

  vertices_to_refine_by_color[color].clear();
  max_sum_by_color[color] = std::numeric_limits<f_t>::quiet_NaN();
  min_sum_by_color[color] = std::numeric_limits<f_t>::quiet_NaN();
}

template <typename i_t, typename f_t>
void color_lower_bounds(const csc_matrix_t<i_t, f_t>& A,
                        const csr_matrix_t<i_t, f_t>& Arow,
                        i_t& row_lower_bound,
                        i_t& col_lower_bound)
{
  // Look at the number of unique row sums and column sum.
  // This should be a lower bound on the number of colors.
  const i_t m = A.m;
  const i_t n = A.n;
  std::vector<f_t> row_sums(m, 0.0);
  std::vector<f_t> col_sums(n, 0.0);
  for (i_t i = 0; i < m; i++) {
    const i_t row_start = Arow.row_start[i];
    const i_t row_end   = Arow.row_start[i + 1];
    f_t sum             = 0.0;
    for (i_t p = row_start; p < row_end; p++) {
      sum += Arow.x[p];
    }
    row_sums[i] = sum;
  }
  for (i_t j = 0; j < n; j++) {
    const i_t col_start = A.col_start[j];
    const i_t col_end   = A.col_start[j + 1];
    f_t sum             = 0.0;
    for (i_t p = col_start; p < col_end; p++) {
      sum += A.x[p];
    }
    col_sums[j] = sum;
  }

  std::unordered_map<f_t, i_t> unique_row_sums;
  std::unordered_map<f_t, i_t> unique_col_sums;
  for (i_t i = 0; i < m; i++) {
    unique_row_sums[row_sums[i]]++;
  }
  for (i_t j = 0; j < n; j++) {
    unique_col_sums[col_sums[j]]++;
  }

  row_lower_bound = static_cast<i_t>(unique_row_sums.size());
  col_lower_bound = static_cast<i_t>(unique_col_sums.size());
}

template <typename i_t, typename f_t>
i_t color_graph(const csc_matrix_t<i_t, f_t>& A,
                const simplex_solver_settings_t<i_t, f_t>& settings,
                std::vector<color_t<i_t>>& colors,
                i_t row_threshold,
                i_t col_threshold,
                i_t& num_row_colors,
                i_t& num_col_colors,
                i_t& num_colors,
                i_t& total_colors_seen)
{
  f_t start_time    = tic();
  f_t last_log_time = start_time;
  const i_t m       = A.m;
  const i_t n       = A.n;
  csr_matrix_t<i_t, f_t> A_row(m, n, 1);
  A.to_compressed_row(A_row);

  i_t row_lower_bound = 0;
  i_t col_lower_bound = 0;

  color_lower_bounds(A, A_row, row_lower_bound, col_lower_bound);
  settings.log.debug("Folding: Row lower bound %d / %d col lower bound %d / %d\n",
                     row_lower_bound,
                     row_threshold,
                     col_lower_bound,
                     col_threshold);
  if (row_lower_bound > row_threshold || col_lower_bound > col_threshold) {
    settings.log.debug(
      "Folding: Row lower bound %d is greater than row threshold %d or col lower bound %d is "
      "greater than col threshold %d\n",
      row_lower_bound,
      row_threshold,
      col_lower_bound,
      col_threshold);
    return -1;
  }

  std::vector<i_t> all_rows_vertices(m);
  std::iota(all_rows_vertices.begin(), all_rows_vertices.end(), 0);
  color_t<i_t> all_rows(kRow, kActive, 0, all_rows_vertices);

  std::vector<i_t> all_cols_vertices(n);
  std::iota(all_cols_vertices.begin(), all_cols_vertices.end(), 0);
  color_t<i_t> all_cols(kCol, kActive, 1, all_cols_vertices);

  std::vector<i_t> color_stack;
  color_stack.push_back(0);
  color_stack.push_back(1);

  std::vector<i_t> row_color_map(m, 0);
  std::vector<i_t> col_color_map(n, 1);

  num_row_colors    = 1;
  num_col_colors    = 1;
  num_colors        = num_row_colors + num_col_colors;
  total_colors_seen = 2;  // The total colors seen includes inactive colors

  colors.emplace_back(all_rows);
  colors.emplace_back(all_cols);

  const i_t max_vertices = std::max(m, n);
  std::vector<f_t> vertex_to_sum(max_vertices, 0.0);
  std::vector<i_t> vertices_to_refine;
  vertices_to_refine.reserve(max_vertices);
  std::vector<i_t> marked_vertices(max_vertices, 0);

  i_t max_colors = m + n;
  std::vector<i_t> color_in_stack(max_colors, 0);
  color_in_stack[0] = 1;
  color_in_stack[1] = 1;

  std::unordered_map<f_t, std::vector<i_t>> color_sums;
  std::unordered_map<f_t, i_t> sum_to_sizes;

  std::vector<std::vector<i_t>> vertices_to_refine_by_color(max_colors);
  std::vector<f_t> max_sum_by_color(max_colors, std::numeric_limits<f_t>::quiet_NaN());
  std::vector<f_t> min_sum_by_color(max_colors, std::numeric_limits<f_t>::quiet_NaN());
  std::vector<i_t> marked_colors(max_colors, 0);

  std::vector<i_t> colors_to_split;
  colors_to_split.reserve(max_vertices);
  std::vector<i_t> colors_to_update;
  colors_to_update.reserve(max_vertices);

  i_t num_refinements       = 0;
  f_t colors_per_refinement = 0.0;
  while (!color_stack.empty()) {
    num_refinements++;
    i_t refining_color_index           = color_stack.back();
    const color_t<i_t>& refining_color = colors[refining_color_index];
    color_stack.pop_back();  // Can pop since refining color is no longer needed. New colors will be
                             // added to the stack.
    color_in_stack[refining_color_index] = 0;
    colors_to_update.clear();
    compute_sums(A,
                 A_row,
                 num_row_colors,
                 num_col_colors,
                 total_colors_seen,
                 row_color_map,
                 col_color_map,
                 refining_color,
                 colors_to_update,
                 marked_colors,
                 vertices_to_refine,
                 marked_vertices,
                 vertices_to_refine_by_color,
                 vertex_to_sum,
                 max_sum_by_color);

    colors_to_split.clear();
    find_colors_to_split(colors_to_update,
                         colors,
                         vertices_to_refine_by_color,
                         vertex_to_sum,
                         max_sum_by_color,
                         min_sum_by_color,
                         colors_to_split);
    // We now need to split the current colors into new colors
    if (refining_color.row_or_column == kRow) {
      // Refining color is a row color.
      // See if we need to split the column colors
      for (i_t color : colors_to_split) {
        split_colors(color,
                     colors[refining_color_index].color,
                     kCol,
                     vertex_to_sum,
                     color_sums,
                     sum_to_sizes,
                     colors,
                     color_stack,
                     color_in_stack,
                     col_color_map,
                     marked_vertices,
                     vertices_to_refine_by_color,
                     min_sum_by_color,
                     max_sum_by_color,
                     num_colors,
                     num_col_colors,
                     total_colors_seen);
      }
    } else {
      // Refining color is a column color.
      // See if we need to split the row colors
      for (i_t color : colors_to_split) {
        split_colors(color,
                     colors[refining_color_index].color,
                     kRow,
                     vertex_to_sum,
                     color_sums,
                     sum_to_sizes,
                     colors,
                     color_stack,
                     color_in_stack,
                     row_color_map,
                     marked_vertices,
                     vertices_to_refine_by_color,
                     min_sum_by_color,
                     max_sum_by_color,
                     num_colors,
                     num_row_colors,
                     total_colors_seen);
      }
    }

    for (i_t v : vertices_to_refine) {
      vertex_to_sum[v] = 0.0;
    }

    colors_per_refinement =
      static_cast<f_t>(num_row_colors + num_col_colors) / static_cast<f_t>(num_refinements);
    i_t projected_colors =
      num_row_colors + num_col_colors +
      static_cast<i_t>(colors_per_refinement * static_cast<f_t>(color_stack.size()));

    if (total_colors_seen >= max_colors - 10) {
      settings.log.debug(
        "Folding: Increase max colors from %d to %d\n", max_colors, max_colors * 2);
      max_colors *= 2;
      color_in_stack.resize(max_colors, 0);
      vertices_to_refine_by_color.resize(max_colors);
      max_sum_by_color.resize(max_colors);
      min_sum_by_color.resize(max_colors);
      marked_colors.resize(max_colors, 0);
    }

#ifdef DEBUG
    for (i_t k = 0; k < max_vertices; k++) {
      if (vertex_to_sum[k] != 0.0) {
        settings.log.printf("Folding: Vertex %d has sum %e\n", k, vertex_to_sum[k]);
        return -2;
      }
    }
#endif

    for (i_t color : colors_to_update) {
      vertices_to_refine_by_color[color].clear();
      max_sum_by_color[color] = std::numeric_limits<f_t>::quiet_NaN();
      min_sum_by_color[color] = std::numeric_limits<f_t>::quiet_NaN();
    }

#ifdef DEBUG
    for (i_t k = 0; k < total_colors_seen; k++) {
      if (vertices_to_refine_by_color[k].size() != 0) {
        settings.log.printf("Folding: Color %d has %ld vertices to refine. Not cleared\n",
                            k,
                            vertices_to_refine_by_color[k].size());
        return -2;
      }
    }
#endif

#ifdef DEBUG
    for (i_t i = 0; i < m; i++) {
      if (row_color_map[i] >= total_colors_seen) {
        settings.log.printf("Folding: Row color %d is not in the colors vector\n",
                            row_color_map[i]);
        return -2;
      }
    }
    for (i_t j = 0; j < n; j++) {
      if (col_color_map[j] >= total_colors_seen) {
        settings.log.printf("Folding: Column color %d is not in the colors vector. %d\n",
                            col_color_map[j],
                            num_colors);
        return -2;
      }
    }
#endif

#ifdef DEBUG
    // Count the number of active colors
    i_t num_active_colors     = 0;
    i_t num_active_row_colors = 0;
    i_t num_active_col_colors = 0;
    for (color_t<i_t>& color : colors) {
      if (color.active == kActive) {
        num_active_colors++;
        if (color.row_or_column == kRow) {
          num_active_row_colors++;
          for (i_t v : color.vertices) {
            if (row_color_map[v] != color.color) {
              settings.log.printf(
                "Folding: Row color map %d does not match color %d for vertex %d\n",
                row_color_map[v],
                color.color,
                v);
              return -2;
            }
          }
        } else {
          num_active_col_colors++;
          for (i_t v : color.vertices) {
            if (col_color_map[v] != color.color) {
              settings.log.printf(
                "Folding: Column color map %d does not match color %d for vertex %d\n",
                col_color_map[v],
                color.color,
                v);
              return -2;
            }
          }
        }
      }
    }
    // printf("Number of active colors: %d\n", num_active_colors);
    if (num_active_colors != num_colors) {
      settings.log.printf("Folding: Number of active colors does not match number of colors\n");
      return -2;
    }
    // printf("Number of active row colors: %d\n", num_active_row_colors);
    if (num_active_row_colors != num_row_colors) {
      settings.log.printf(
        "Folding: Number of active row colors does not match number of row colors\n");
      return -2;
    }
    // printf("Number of active column colors: %d\n", num_active_col_colors);
    if (num_active_col_colors != num_col_colors) {
      settings.log.printf(
        "Folding: Number of active column colors does not match number of column colors\n");
      return -2;
    }
#endif

    if (toc(last_log_time) > 1.0) {
      last_log_time = tic();
      settings.log.printf("Folding: %d refinements %d colors in %.2fs",
                          num_refinements,
                          num_row_colors + num_col_colors,
                          toc(start_time));
#ifdef PRINT_INFO
      settings.log.debug(
        "Number of refinements %8d. Number of colors %d (row colors %d, col colors %d) stack size "
        "%ld colors per "
        "refinement %.2f projected colors %d in %.2f seconds\n",
        num_refinements,
        num_row_colors + num_col_colors,
        num_row_colors,
        num_col_colors,
        color_stack.size(),
        colors_per_refinement,
        projected_colors,
        toc(start_time));
#endif
    }
    if (num_row_colors >= max_vertices) {
      settings.log.printf("Folding: Too many row colors %d max %d\n", num_row_colors, max_vertices);
      return -2;
    }
    if (num_col_colors >= max_vertices) {
      settings.log.printf(
        "Folding: Too many column colors %d max %d\n", num_col_colors, max_vertices);
      return -2;
    }

    if (num_row_colors > row_threshold || num_col_colors > col_threshold) {
      settings.log.printf("Folding: Number of colors exceeds threshold");
      return -1;
    }
  }
  settings.log.printf(
    "Folding: Colors %d. Refinements: %d\n", num_row_colors + num_col_colors, num_refinements);

  return 0;
}

template <typename i_t, typename f_t>
void folding(lp_problem_t<i_t, f_t>& problem,
             const simplex_solver_settings_t<i_t, f_t>& settings,
             presolve_info_t<i_t, f_t>& presolve_info)
{
  // Handle linear programs in the form
  // minimize c^T x
  // subject to A * x = b
  //            0 <= x,
  //              x_j <= u_j,   j in U

  // These can be converted into the form
  // minimize c^T x
  // subject to A * x = b
  //            x_j + w_j = u_j, j in U
  //            0 <= x,
  //            0 <= w
  //
  // We can then construct the augmented matrix
  //
  // [ A 0   b]
  // [ I I   u]
  // [ c 0 inf]
  f_t start_time = tic();

  i_t m = problem.num_rows;
  i_t n = problem.num_cols;

  if (settings.folding == -1 && (m > 1e6 || n > 1e6)) {
    settings.log.printf("Folding: Skipping\n");
    return;
  }

  i_t nz_obj = 0;
  for (i_t j = 0; j < n; j++) {
    if (problem.objective[j] != 0.0) { nz_obj++; }
  }
  i_t nz_rhs = 0;
  for (i_t i = 0; i < m; i++) {
    if (problem.rhs[i] != 0.0) { nz_rhs++; }
  }
  i_t nz_lb = 0;
  for (i_t j = 0; j < n; j++) {
    if (problem.lower[j] != 0.0) { nz_lb++; }
  }

  std::vector<f_t> finite_upper_bounds;
  finite_upper_bounds.reserve(n);
  i_t nz_ub = 0;
  for (i_t j = 0; j < n; j++) {
    if (problem.upper[j] != inf) {
      nz_ub++;
      finite_upper_bounds.push_back(problem.upper[j]);
    }
  }

  if (nz_lb > 0) {
    settings.log.printf("Folding: Can't handle problems with nonzero lower bounds\n");
    return;
  }

  i_t m_prime      = m + 1 + nz_ub;
  i_t n_prime      = n + nz_ub + 1;
  i_t augmented_nz = problem.A.col_start[n] + nz_obj + nz_rhs + 3 * nz_ub + 1;
  settings.log.debug("Folding: Augmented matrix has %d rows, %d columns, %d nonzeros\n",
                     m_prime,
                     n_prime,
                     augmented_nz);

  csc_matrix_t<i_t, f_t> augmented(m_prime, n_prime, augmented_nz);
  i_t nnz         = 0;
  i_t upper_count = 0;
  for (i_t j = 0; j < n; j++) {
    augmented.col_start[j] = nnz;
    // A
    const i_t col_start = problem.A.col_start[j];
    const i_t col_end   = problem.A.col_start[j + 1];
    for (i_t p = col_start; p < col_end; p++) {
      augmented.i[nnz] = problem.A.i[p];
      augmented.x[nnz] = problem.A.x[p];
      nnz++;
    }
    // I
    if (problem.upper[j] != inf) {
      augmented.i[nnz] = m + upper_count;
      augmented.x[nnz] = 1.0;
      upper_count++;
      nnz++;
    }
    // c
    if (problem.objective[j] != 0.0) {
      augmented.i[nnz] = m + nz_ub;
      augmented.x[nnz] = problem.objective[j];
      nnz++;
    }
  }
  // Column [ 0; I; 0]
  for (i_t j = n; j < n + nz_ub; j++) {
    augmented.col_start[j] = nnz;
    i_t k                  = j - n;
    augmented.i[nnz]       = m + k;
    augmented.x[nnz]       = 1.0;
    nnz++;
  }
  // Final column [b; u; inf]
  augmented.col_start[n + nz_ub] = nnz;
  for (i_t i = 0; i < m; i++) {
    if (problem.rhs[i] != 0.0) {
      augmented.i[nnz] = i;
      augmented.x[nnz] = problem.rhs[i];
      nnz++;
    }
  }
  upper_count = 0;
  for (i_t j = 0; j < n; j++) {
    if (problem.upper[j] != inf) {
      augmented.i[nnz] = m + upper_count;
      augmented.x[nnz] = problem.upper[j];
      upper_count++;
      nnz++;
    }
  }
  augmented.i[nnz] = m + nz_ub;
  augmented.x[nnz] = inf;
  nnz++;
  augmented.col_start[n + nz_ub + 1] = nnz;  // Finalize the matrix
  settings.log.debug("Folding: Augmented matrix has %d nonzeros predicted %d\n", nnz, augmented_nz);

  // Ensure only 1 inf in the augmented matrice
  i_t num_inf = 0;
  for (i_t p = 0; p < augmented_nz; p++) {
    if (augmented.x[p] == inf) { num_inf++; }
  }
  settings.log.debug("Folding: Augmented matrix has %d infs\n", num_inf);
  if (num_inf != 1) {
    settings.log.printf("Folding: Augmented matrix has %d infs, expected 1\n", num_inf);
    return;
  }

#ifdef WRITE_AUGMENTED
  {
    FILE* fid;
    fid = fopen("augmented.mtx", "w");
    augmented.write_matrix_market(fid);
    fclose(fid);
  }
#endif

  std::vector<color_t<i_t>> colors;
  i_t num_row_colors;
  i_t num_col_colors;
  i_t num_colors;
  i_t total_colors_seen;
  f_t color_start_time = tic();
  f_t fold_threshold   = settings.folding == -1 ? 0.50 : 1.0;
  i_t row_threshold    = static_cast<i_t>(fold_threshold * static_cast<f_t>(m));
  i_t col_threshold    = static_cast<i_t>(fold_threshold * static_cast<f_t>(n));
  i_t status           = color_graph(augmented,
                           settings,
                           colors,
                           row_threshold,
                           col_threshold,
                           num_row_colors,
                           num_col_colors,
                           num_colors,
                           total_colors_seen);
  if (status != 0) {
    settings.log.printf("Folding: Coloring aborted in %.2f seconds\n", toc(color_start_time));
    return;
  }
  settings.log.printf("Folding: Coloring time %.2f seconds\n", toc(color_start_time));

  // Go through the active colors and ensure that the row corresponding to the objective is its own
  // color
  std::vector<f_t> full_rhs(m_prime, 0.0);
  for (i_t i = 0; i < m; i++) {
    full_rhs[i] = problem.rhs[i];
  }
  upper_count = 0;
  for (i_t j = 0; j < n; j++) {
    if (problem.upper[j] != inf) {
      full_rhs[m + upper_count] = problem.upper[j];
      upper_count++;
    }
  }
  full_rhs[m + nz_ub] = inf;

  std::vector<i_t> row_colors;
  row_colors.reserve(num_row_colors);

  bool found_objective_color = false;
  i_t objective_color        = -1;
  i_t color_count            = 0;
  for (const color_t<i_t>& color : colors) {
    if (color.active == kActive) {
      if (color.row_or_column == kRow) {
        if (color.vertices.size() == 1) {
          if (*color.vertices.begin() == m + nz_ub) {
            settings.log.debug("Folding: Row color %d is the objective color\n", color.color);
            found_objective_color = true;
            objective_color       = color_count;
          } else {
            row_colors.push_back(color_count);
          }
        } else {
          row_colors.push_back(color_count);
#ifdef ROW_RHS_CHECK
          // Check that all vertices in the same row color have the same rhs value
          auto it       = color.vertices.begin();
          f_t rhs_value = full_rhs[*it];
          for (++it; it != color.vertices.end(); ++it) {
            if (full_rhs[*it] != rhs_value) {
              settings.log.printf(
                "Folding: RHS value for vertex %d is %e, but should be %e. Difference is %e\n",
                *it,
                full_rhs[*it],
                rhs_value,
                full_rhs[*it] - rhs_value);
              return;
            }
          }
#endif
        }
      }
    }
    color_count++;
  }

  if (!found_objective_color) {
    settings.log.printf("Folding: Objective color not found\n");
    return;
  }

  // Go through the active colors and ensure that the column corresponding to the rhs is its own
  // color
  bool found_rhs_color = false;
  i_t rhs_color        = -1;
  std::vector<f_t> full_objective(n_prime, 0.0);
  for (i_t j = 0; j < n; j++) {
    full_objective[j] = problem.objective[j];
  }
  full_objective[n_prime - 1] = inf;

  std::vector<i_t> col_colors;
  col_colors.reserve(num_col_colors - 1);
  color_count = 0;
  for (const color_t<i_t>& color : colors) {
    if (color.active == kActive) {
      if (color.row_or_column == kCol) {
        if (color.vertices.size() == 1) {
          if (*color.vertices.begin() == n_prime - 1) {
            settings.log.debug("Folding: Column color %d is the rhs color\n", color.color);
            found_rhs_color = true;
            rhs_color       = color_count;
          } else {
            col_colors.push_back(color_count);
          }
        } else {
          col_colors.push_back(color_count);
#ifdef COL_OBJ_CHECK
          // Check that all vertices in the same column color have the same objective value
          auto it             = color.vertices.begin();
          f_t objective_value = full_objective[*it];
          for (; it != color.vertices.end(); ++it) {
            if (full_objective[*it] != objective_value) {
              settings.log.printf(
                "Folding: Objective value for vertex %d is %e, but should be %e. Difference is "
                "%e\n",
                *it,
                full_objective[*it],
                objective_value,
                full_objective[*it] - objective_value);
              return;
            }
          }
#endif
        }
      }
    }
    color_count++;
  }

  if (!found_rhs_color) {
    settings.log.printf("Folding: RHS color not found\n");
    return;
  }

  // The original problem is in the form
  // minimize c^T x
  // subject to A x = b
  // x >= 0
  //
  // Let A_prime = C^s A D
  // b_prime = C^s b
  // c_prime = D^T c
  //
  // where C = Pi_P
  // and D = Pi_Q
  //
  // We will construct the new problem
  //
  // minimize c_prime^T x_prime
  // subject to A_prime x_prime = b_prime
  // x_prime >= 0
  //

  i_t previous_rows = m + nz_ub;
  i_t reduced_rows  = num_row_colors - 1;
  settings.log.debug("Folding: previous_rows %d reduced_rows %d\n", previous_rows, reduced_rows);

  // Construct the matrix Pi_P
  // Pi_vP = { 1 if v in P
  //         { 0 otherwise
  settings.log.debug("Folding: Constructing Pi_P\n");
  csc_matrix_t<i_t, f_t> Pi_P(previous_rows, reduced_rows, previous_rows);
  nnz = 0;
  for (i_t k = 0; k < reduced_rows; k++) {
    Pi_P.col_start[k]         = nnz;
    const i_t color_index     = row_colors[k];
    const color_t<i_t>& color = colors[color_index];
    for (i_t v : color.vertices) {
      Pi_P.i[nnz] = v;
      Pi_P.x[nnz] = 1.0;
      nnz++;
    }
  }
  Pi_P.col_start[reduced_rows] = nnz;
  settings.log.debug("Folding: Pi_P nz %d predicted %d\n", nnz, previous_rows);
  if (nnz != previous_rows) {
    settings.log.printf("Folding: Pi_P nz %d predicted %d\n", nnz, previous_rows);
    return;
  }
#ifdef WRITE_PI_P
  FILE* fid = fopen("Pi_P.txt", "w");
  Pi_P.write_matrix_market(fid);
  fclose(fid);
#endif

  // Start by constructing the matrix C^s
  // C^s = Pi^s_P
  // C^s_tv = Pi_vt / sum_v' Pi_v't
  // We have that sum_v' Pi_v't = | T |
  // C^s_tv = Pi_vt / | T | if t corresponds to color T
  // We have that Pi_vT = {1 if v in color T, 0 otherwiseS
  // C^s_tv = { 1/|T| if v in color T
  //         { 0
  settings.log.debug("Folding: Constructing C^s row\n");
  csr_matrix_t<i_t, f_t> C_s_row(reduced_rows, previous_rows, previous_rows);
  nnz = 0;
  settings.log.debug(
    "Folding: row_colors size %ld reduced_rows %d\n", row_colors.size(), reduced_rows);
  if (row_colors.size() != reduced_rows) {
    settings.log.printf("Folding: Bad row colors\n");
    return;
  }
  for (i_t k = 0; k < reduced_rows; k++) {
    C_s_row.row_start[k]  = nnz;
    const i_t color_index = row_colors[k];
    if (color_index < 0) {
      settings.log.printf("Folding: Bad row colors\n");
      return;
    }
    const color_t<i_t>& color = colors[color_index];
    const i_t color_size      = color.vertices.size();
    for (i_t v : color.vertices) {
      C_s_row.j[nnz] = v;
      C_s_row.x[nnz] = 1.0 / static_cast<f_t>(color_size);
      nnz++;
    }
  }
  C_s_row.row_start[reduced_rows] = nnz;
  settings.log.debug("Folding: C_s nz %d predicted %d\n", nnz, previous_rows);
  settings.log.debug("Folding: Converting C^s row to compressed column\n");

  // csc_matrix_t<i_t, f_t> C_s(reduced_rows, previous_rows, 1);
  presolve_info.folding_info.C_s.resize(reduced_rows, previous_rows, 1);
  csc_matrix_t<i_t, f_t>& C_s = presolve_info.folding_info.C_s;
  C_s_row.to_compressed_col(C_s);
  settings.log.debug("Folding: Completed C^s\n");
#ifdef DEBUG
  fid = fopen("C_s.txt", "w");
  C_s.write_matrix_market(fid);
  fclose(fid);

  // Verify that C^s Pi_P = I
  csc_matrix_t<i_t, f_t> product(reduced_rows, reduced_rows, 1);
  multiply(C_s, Pi_P, product);
  csc_matrix_t<i_t, f_t> identity(reduced_rows, reduced_rows, reduced_rows);
  for (i_t i = 0; i < reduced_rows; i++) {
    identity.col_start[i] = i;
    identity.i[i]         = i;
    identity.x[i]         = 1.0;
  }
  identity.col_start[reduced_rows] = reduced_rows;
  csc_matrix_t<i_t, f_t> error(reduced_rows, reduced_rows, 1);
  add(product, identity, 1.0, -1.0, error);
  printf("|| C^s Pi_P - I ||_1 = %f\n", error.norm1());
  if (error.norm1() > 1e-6) { exit(1); }
#endif

  // Construct that matrix D
  // D = Pi_Q
  // D_vQ = { 1 if v in Q
  //        { 0 otherwise
  settings.log.debug("Folding: Constructing D\n");
  i_t previous_cols = n + nz_ub;
  i_t reduced_cols  = num_col_colors - 1;
  settings.log.debug(
    "Folding: previous columns %d reduced columns %d\n", previous_cols, reduced_cols);
  presolve_info.folding_info.D.resize(previous_cols, reduced_cols, previous_cols);
  csc_matrix_t<i_t, f_t>& D = presolve_info.folding_info.D;
  nnz                       = 0;
  for (i_t k = 0; k < reduced_cols; k++) {
    D.col_start[k]        = nnz;
    const i_t color_index = col_colors[k];
    // printf("column color %d index %d colors size %ld\n", k, color_index, colors.size());
    if (color_index < 0) {
      settings.log.printf("Folding: Bad column colors\n");
      return;
    }
    const color_t<i_t>& color = colors[color_index];
    for (const i_t v : color.vertices) {
      D.i[nnz] = v;
      D.x[nnz] = 1.0;
      nnz++;
    }
  }
  D.col_start[reduced_cols] = nnz;
  settings.log.debug("Folding: D nz %d predicted %d\n", nnz, previous_cols);
#ifdef WRITE_D
  fid = fopen("D.txt", "w");
  D.write_matrix_market(fid);
  fclose(fid);
#endif

  // Construct D^s_tv
  // D^s_Tv = D_vT / sum_v' D_v'T
  csr_matrix_t<i_t, f_t> D_s_row(reduced_cols, previous_cols, previous_cols);
  nnz = 0;
  for (i_t k = 0; k < reduced_cols; k++) {
    D_s_row.row_start[k]      = nnz;
    const i_t color_index     = col_colors[k];
    const color_t<i_t>& color = colors[color_index];
    const i_t color_size      = color.vertices.size();
    for (i_t v : color.vertices) {
      D_s_row.j[nnz] = v;
      D_s_row.x[nnz] = 1.0 / static_cast<f_t>(color_size);
      nnz++;
    }
  }
  D_s_row.row_start[reduced_cols] = nnz;
  settings.log.debug("Folding: D^s row nz %d predicted %d\n", nnz, previous_cols);
  settings.log.debug("Folding: Converting D^s row to compressed column\n");
  // csc_matrix_t<i_t, f_t> D_s(reduced_cols, previous_cols, 1);
  presolve_info.folding_info.D_s.resize(reduced_cols, previous_cols, 1);
  csc_matrix_t<i_t, f_t>& D_s = presolve_info.folding_info.D_s;
  D_s_row.to_compressed_col(D_s);
#ifdef WRITE_DS
  settings.log.printf("Folding: Writing D^s\n");
  fid = fopen("D_s.txt", "w");
  D_s.write_matrix_market(fid);
  fclose(fid);
#endif

#ifdef DEBUG
  // Verify that D^s D = I
  csc_matrix_t<i_t, f_t> D_product(reduced_cols, reduced_cols, 1);
  multiply(D_s, D, D_product);
  csc_matrix_t<i_t, f_t> D_identity(reduced_cols, reduced_cols, reduced_cols);
  for (i_t i = 0; i < reduced_cols; i++) {
    D_identity.col_start[i] = i;
    D_identity.i[i]         = i;
    D_identity.x[i]         = 1.0;
  }
  D_identity.col_start[reduced_cols] = reduced_cols;
  csc_matrix_t<i_t, f_t> D_error(reduced_cols, reduced_cols, 1);
  add(D_product, D_identity, 1.0, -1.0, D_error);
  settings.log.debug("Folding: || D^s D - I ||_1 = %f\n", D_error.norm1());
  if (D_error.norm1() > 1e-6) {
    settings.log.printf("Folding: || D^s D - I ||_1 = %f\n", D_error.norm1());
    return;
  }

  // Construct the matrix X
  // X = C C^s
  settings.log.debug("Folding: Constructing X\n");
  csc_matrix_t<i_t, f_t> X(previous_rows, previous_rows, 1);
  multiply(Pi_P, C_s, X);
  settings.log.debug("Folding: Completed X\n");
  std::vector<f_t> X_col_sums(previous_rows);
  for (i_t j = 0; j < previous_rows; j++) {
    X_col_sums[j] = 0.0;
    for (i_t p = X.col_start[j]; p < X.col_start[j + 1]; p++) {
      X_col_sums[j] += X.x[p];
    }
    if (std::abs(X_col_sums[j] - 1.0) > 1e-6) {
      settings.log.printf("Folding: X_col_sums[%d] = %f\n", j, X_col_sums[j]);
      return;
    }
  }
  csr_matrix_t<i_t, f_t> X_row(previous_rows, previous_rows, 1);
  X.to_compressed_row(X_row);
  std::vector<f_t> X_row_sums(previous_rows);
  for (i_t i = 0; i < previous_rows; i++) {
    X_row_sums[i] = 0.0;
    for (i_t p = X_row.row_start[i]; p < X_row.row_start[i + 1]; p++) {
      X_row_sums[i] += X_row.x[p];
    }
    if (std::abs(X_row_sums[i] - 1.0) > 1e-6) {
      settings.log.printf("Folding: X_row_sums[%d] = %f\n", i, X_row_sums[i]);
      return;
    }
  }
  settings.log.printf("Folding: Verified X is doubly stochastic\n");

  // Construct the matrix Y
  // Y = D D^s
  settings.log.printf("Folding: Constructing Y\n");
  csc_matrix_t<i_t, f_t> Y(previous_cols, previous_cols, 1);
  multiply(D, D_s, Y);
  settings.log.debug("Folding: Completed Y\n");

  std::vector<f_t> Y_col_sums(previous_cols);
  for (i_t j = 0; j < previous_cols; j++) {
    Y_col_sums[j] = 0.0;
    for (i_t p = Y.col_start[j]; p < Y.col_start[j + 1]; p++) {
      Y_col_sums[j] += Y.x[p];
    }
    if (std::abs(Y_col_sums[j] - 1.0) > 1e-6) {
      settings.log.printf("Folding: Y_col_sums[%d] = %f\n", j, Y_col_sums[j]);
      return;
    }
  }
  csr_matrix_t<i_t, f_t> Y_row(previous_cols, previous_cols, 1);
  Y.to_compressed_row(Y_row);
  std::vector<f_t> Y_row_sums(previous_cols);
  for (i_t i = 0; i < previous_cols; i++) {
    Y_row_sums[i] = 0.0;
    for (i_t p = Y_row.row_start[i]; p < Y_row.row_start[i + 1]; p++) {
      Y_row_sums[i] += Y_row.x[p];
    }
    if (std::abs(Y_row_sums[i] - 1.0) > 1e-6) {
      settings.log.printf("Folding: Y_row_sums[%d] = %f\n", i, Y_row_sums[i]);
      return;
    }
  }
  settings.log.printf("Folding: Verified Y is doubly stochastic\n");
#endif
  // Construct the matrix A_tilde
  settings.log.debug("Folding: Constructing A_tilde\n");
  i_t A_nnz                      = problem.A.col_start[n];
  csc_matrix_t<i_t, f_t> A_tilde = augmented;
  A_tilde.remove_row(m + nz_ub);
  A_tilde.m--;
  A_tilde.remove_column(n + nz_ub);
  A_tilde.n--;
#ifdef WRITE_A_TILDE
  {
    FILE* fid;
    fid = fopen("A_tilde.txt", "w");
    A_tilde.write_matrix_market(fid);
    fclose(fid);
  }
#endif

  csr_matrix_t<i_t, f_t> A_tilde_row(A_tilde.m, A_tilde.n, A_tilde.col_start[A_tilde.n]);
  A_tilde.to_compressed_row(A_tilde_row);

#ifdef DEBUG
  std::vector<i_t> row_to_color(A_tilde.m, -1);
  for (i_t k = 0; k < total_colors_seen; k++) {
    const color_t<i_t>& row_color = colors[k];
    if (k == objective_color) continue;
    if (row_color.active == kActive && row_color.row_or_column == kRow) {
      for (i_t u : row_color.vertices) {
        row_to_color[u] = k;
      }
    }
  }
  std::vector<i_t> col_to_color(A_tilde.n, -1);
  for (i_t k = 0; k < total_colors_seen; k++) {
    const color_t<i_t>& col_color = colors[k];
    if (k == rhs_color) continue;
    if (col_color.active == kActive && col_color.row_or_column == kCol) {
      for (i_t v : col_color.vertices) {
        col_to_color[v] = k;
        // printf("Col %d assigned to color %d =? %d\n", v, k, col_color.color);
      }
    }
  }

  // Check that the partition is equitable
  for (i_t k = 0; k < total_colors_seen; k++) {
    const color_t<i_t> col_color = colors[k];
    if (col_color.active == kActive) {
      if (col_color.row_or_column == kCol) {
        // Check sum_{w in color} Avw = sum_{w in color} Avprimew for all (v, vprime) in row color P
        for (i_t h = 0; h < total_colors_seen; h++) {
          const color_t<i_t> row_color = colors[h];
          if (row_color.active == kActive && row_color.row_or_column == kRow) {
            for (i_t u : row_color.vertices) {
              for (i_t v : row_color.vertices) {
                if (u != v) {
                  f_t sum_Av      = 0.0;
                  f_t sum_Avprime = 0.0;
                  for (i_t p = A_tilde_row.row_start[u]; p < A_tilde_row.row_start[u + 1]; p++) {
                    const i_t j = A_tilde_row.j[p];
                    if (col_to_color[j] == k) { sum_Av += A_tilde_row.x[p]; }
                  }
                  for (i_t p = A_tilde_row.row_start[v]; p < A_tilde_row.row_start[v + 1]; p++) {
                    const i_t j = A_tilde_row.j[p];
                    if (col_to_color[j] == k) { sum_Avprime += A_tilde_row.x[p]; }
                  }
                  if (std::abs(sum_Av - sum_Avprime) > 1e-12) {
                    settings.log.printf(
                      "Folding: u %d v %d row color %d sum_A%d: %f sum_A%d: = %f\n",
                      u,
                      v,
                      h,
                      u,
                      sum_Av,
                      v,
                      sum_Avprime);
                    settings.log.printf("Folding: row color %d vertices: ", h);
                    for (i_t u : row_color.vertices) {
                      settings.log.printf("%d(%d) ", u, row_to_color[u]);
                    }
                    settings.log.printf("\n");
                    settings.log.printf("col color %d vertices: ", k);
                    for (i_t v : col_color.vertices) {
                      settings.log.printf("%d(%d) ", v, col_to_color[v]);
                    }
                    settings.log.printf("\n");
                    settings.log.printf("row %d\n", u);
                    for (i_t p = A_tilde_row.row_start[u]; p < A_tilde_row.row_start[u + 1]; p++) {
                      const i_t j = A_tilde_row.j[p];
                      settings.log.printf("row %d col %d column color %d value %e\n",
                                          u,
                                          j,
                                          col_to_color[j],
                                          A_tilde_row.x[p]);
                      if (col_to_color[j] == k) { sum_Av += A_tilde_row.x[p]; }
                    }
                    settings.log.printf("row %d\n", v);
                    for (i_t p = A_tilde_row.row_start[v]; p < A_tilde_row.row_start[v + 1]; p++) {
                      const i_t j = A_tilde_row.j[p];
                      settings.log.printf("row %d col %d column color %d value %e\n",
                                          v,
                                          j,
                                          col_to_color[j],
                                          A_tilde_row.x[p]);
                      if (col_to_color[j] == k) { sum_Avprime += A_tilde_row.x[p]; }
                    }
                    settings.log.printf("total colors seen %d\n", total_colors_seen);
                    return;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  settings.log.printf("Folding: Verified that the column partition is equitable\n");

  for (i_t k = 0; k < num_colors; k++) {
    const color_t<i_t>& row_color = colors[k];
    if (row_color.active == kActive) {
      if (row_color.row_or_column == kRow) {
        for (i_t h = 0; h < num_colors; h++) {
          const color_t<i_t>& col_color = colors[h];
          if (col_color.active == kActive && col_color.row_or_column == kCol) {
            for (i_t u : col_color.vertices) {
              for (i_t v : col_color.vertices) {
                if (u != v) {
                  f_t sum_A_u = 0.0;
                  f_t sum_A_v = 0.0;
                  for (i_t p = A_tilde.col_start[u]; p < A_tilde.col_start[u + 1]; p++) {
                    const i_t i = A_tilde.i[p];
                    if (row_to_color[i] == k) { sum_A_u += A_tilde.x[p]; }
                  }
                  for (i_t p = A_tilde.col_start[v]; p < A_tilde.col_start[v + 1]; p++) {
                    const i_t i = A_tilde.i[p];
                    if (row_to_color[i] == k) { sum_A_v += A_tilde.x[p]; }
                  }
                  if (std::abs(sum_A_u - sum_A_v) > 1e-12) {
                    settings.log.printf(
                      "Folding: u %d v %d row color %d sum_A%d: %f sum_A%d: = %f\n",
                      u,
                      v,
                      k,
                      u,
                      sum_A_u,
                      v,
                      sum_A_v);
                    return;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  settings.log.printf("Folding: Verified that the row partition is equitable\n");

  fid = fopen("X.txt", "w");
  X.write_matrix_market(fid);
  fclose(fid);
  fid = fopen("Y.txt", "w");
  Y.write_matrix_market(fid);
  fclose(fid);
#endif

  if (A_tilde.m != previous_rows || A_tilde.n != previous_cols) {
    settings.log.printf("Folding: A_tilde has %d rows and %d cols, expected %d and %d\n",
                        A_tilde.m,
                        A_tilde.n,
                        previous_rows,
                        previous_cols);
    return;
  }

  settings.log.debug("Folding: partial A_tilde nz %d predicted %d\n", nnz, A_nnz + 2 * nz_ub);

#ifdef DEBUG
  // Verify that XA = AY
  csc_matrix_t<i_t, f_t> XA(previous_rows, previous_cols, 1);
  multiply(X, A_tilde, XA);
  fid = fopen("XA.txt", "w");
  XA.write_matrix_market(fid);
  fclose(fid);
  csc_matrix_t<i_t, f_t> AY(previous_rows, previous_cols, 1);
  multiply(A_tilde, Y, AY);
  fid = fopen("AY.txt", "w");
  AY.write_matrix_market(fid);
  fclose(fid);
  csc_matrix_t<i_t, f_t> XA_AY_error(previous_rows, previous_cols, 1);
  add(XA, AY, 1.0, -1.0, XA_AY_error);
  printf("|| XA - AY ||_1 = %f\n", XA_AY_error.norm1());
#endif
  // Construct the matrix A_prime
  settings.log.debug("Folding: Constructing A_prime\n");
  csc_matrix_t<i_t, f_t> A_prime(reduced_rows, reduced_cols, 1);
  csc_matrix_t<i_t, f_t> AD(previous_rows, reduced_cols, 1);
  settings.log.debug("Folding: Multiplying A_tilde and D\n");
  multiply(A_tilde, D, AD);
  settings.log.debug("Folding: Multiplying C_s and AD\n");
  multiply(C_s, AD, A_prime);

  // Construct the vector b_prime
  settings.log.debug("Folding: Constructing b_prime\n");
  std::vector<f_t> b_tilde(previous_rows);
  for (i_t i = 0; i < m; i++) {
    b_tilde[i] = problem.rhs[i];
  }
  for (i_t i = m; i < m + nz_ub; i++) {
    b_tilde[i] = finite_upper_bounds[i - m];
  }
  std::vector<f_t> b_prime(reduced_rows);
  matrix_vector_multiply(C_s, 1.0, b_tilde, 0.0, b_prime);

  // Construct the vector c_prime
  settings.log.debug("Folding: Constructing c_prime\n");
  std::vector<f_t> c_tilde(previous_cols);
  for (i_t j = 0; j < n; j++) {
    c_tilde[j] = problem.objective[j];
  }
  for (i_t j = n; j < n + nz_ub; j++) {
    c_tilde[j] = 0.0;
  }
  std::vector<f_t> c_prime(reduced_cols);
  matrix_transpose_vector_multiply(D, 1.0, c_tilde, 0.0, c_prime);

  if (reduced_rows > reduced_cols) {
    settings.log.printf("Folding: Reduced rows %d > reduced cols %d\n", reduced_rows, reduced_cols);
    return;
  }

  // Construct a new problem
  settings.log.printf(
    "Folding: Constructing reduced problem: %d constraints %d variables and %d nonzeros\n",
    reduced_rows,
    reduced_cols,
    A_prime.col_start[reduced_cols]);

#ifdef SOLVE_REDUCED_PROBLEM
  user_problem_t<i_t, f_t> reduced_problem(problem.handle_ptr);
  reduced_problem.num_rows       = reduced_rows;
  reduced_problem.num_cols       = reduced_cols;
  reduced_problem.A              = A_prime;
  reduced_problem.objective      = c_prime;
  reduced_problem.rhs            = b_prime;
  reduced_problem.lower          = std::vector<f_t>(reduced_cols, 0.0);
  reduced_problem.upper          = std::vector<f_t>(reduced_cols, inf);
  reduced_problem.obj_constant   = 0.0;
  reduced_problem.obj_scale      = 1.0;
  reduced_problem.num_range_rows = 0;
  reduced_problem.row_sense      = std::vector<char>(reduced_rows, 'E');
  reduced_problem.var_types =
    std::vector<variable_type_t>(reduced_cols, variable_type_t::CONTINUOUS);
#endif

  problem.num_rows  = reduced_rows;
  problem.num_cols  = reduced_cols;
  problem.objective = c_prime;
  problem.A         = A_prime;
  problem.rhs       = b_prime;
  problem.lower     = std::vector<f_t>(reduced_cols, 0.0);
  problem.upper     = std::vector<f_t>(reduced_cols, inf);

  presolve_info.folding_info.c_tilde          = c_tilde;
  presolve_info.folding_info.A_tilde          = A_tilde;
  presolve_info.folding_info.is_folded        = true;
  presolve_info.folding_info.num_upper_bounds = nz_ub;

  settings.log.printf("Folding: time %.2f seconds\n", toc(start_time));

#ifdef SOLVE_REDUCED_PROBLEM
  // Solve the reduced problem
  lp_solution_t<i_t, f_t> reduced_solution(reduced_rows, reduced_cols);
  simplex_solver_settings_t<i_t, f_t> reduced_settings;
  reduced_settings.folding          = false;
  reduced_settings.barrier          = true;
  reduced_settings.barrier_presolve = true;
  reduced_settings.log.log          = true;

  solve_linear_program_with_barrier(reduced_problem, reduced_settings, reduced_solution);

  std::vector<f_t>& x_prime = reduced_solution.x;
  std::vector<f_t>& y_prime = reduced_solution.y;
  std::vector<f_t>& z_prime = reduced_solution.z;
  settings.log.printf("Folding: Reduced objective = %e\n", reduced_solution.objective);

  std::vector<f_t> x(previous_cols);
  std::vector<f_t> y(previous_rows);
  std::vector<f_t> z(previous_cols);
  matrix_vector_multiply(D, 1.0, x_prime, 0.0, x);
  matrix_transpose_vector_multiply(C_s, 1.0, y_prime, 0.0, y);
  matrix_transpose_vector_multiply(D_s, 1.0, z_prime, 0.0, z);

  settings.log.printf("Folding: Original primal objective = %e\n", dot<i_t, f_t>(c_tilde, x));
  settings.log.printf("Folding: Original dual objective   = %e\n", dot<i_t, f_t>(b_tilde, y));

  settings.log.printf("Folding: || y ||_2 = %e\n", vector_norm2<i_t, f_t>(y));
  printf("|| z ||_2 = %e\n", vector_norm2<i_t, f_t>(z));

  std::vector<f_t> dual_residual(previous_cols);
  for (i_t j = 0; j < previous_cols; j++) {
    dual_residual[j] = z[j] - c_tilde[j];
  }
  matrix_transpose_vector_multiply(A_tilde, 1.0, y, 1.0, dual_residual);
  settings.log.printf("Folding: Original dual residual = %e\n",
                      vector_norm_inf<i_t, f_t>(dual_residual));
#endif
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE

template void folding<int, double>(lp_problem_t<int, double>& problem,
                                   const simplex_solver_settings_t<int, double>& settings,
                                   presolve_info_t<int, double>& presolve_info);
#endif

}  // namespace cuopt::linear_programming::dual_simplex
