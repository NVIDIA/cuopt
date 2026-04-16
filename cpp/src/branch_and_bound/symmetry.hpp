/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <dual_simplex/presolve.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>
#include <dual_simplex/tic_toc.hpp>
#include <dual_simplex/types.hpp>
#include <dual_simplex/user_problem.hpp>
#include <branch_and_bound/mip_node.hpp>


#include "dejavu.h"

#include <memory>
#include <numeric>
#include <sstream>

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t, typename f_t>
struct mip_symmetry_t {
  mip_symmetry_t(std::unique_ptr<dejavu::groups::random_schreier> schreier_,
                 i_t domain_size_,
                 i_t num_original_vars_,
                 const std::vector<variable_type_t>& var_types,
                 const std::vector<f_t>& lower,
                 const std::vector<f_t>& upper)
    : schreier(std::move(schreier_)),
      domain_size(domain_size_),
      num_original_vars(num_original_vars_),
      is_binary(num_original_vars_, 0)
  {
    for (i_t j = 0; j < num_original_vars_; j++) {
      if (var_types[j] == variable_type_t::CONTINUOUS) {
        continuous_variables.push_back(j);
      } else if (lower[j] == 0.0 && upper[j] == 1.0) {
        is_binary[j] = 1;
      } else {
        general_integer_variables.push_back(j);
      }
    }
  }

  std::unique_ptr<dejavu::groups::random_schreier> schreier;
  i_t domain_size;
  i_t num_original_vars;
  std::vector<i_t> continuous_variables;
  std::vector<i_t> general_integer_variables;
  std::vector<i_t> is_binary;
};

template <typename i_t, typename f_t>
class orbital_fixing_t {
 public:
  explicit orbital_fixing_t(mip_symmetry_t<i_t, f_t>& root)
    : domain_size_(root.domain_size),
      num_original_vars_(root.num_original_vars),
      marked_variables_(root.num_original_vars, 0),
      orbit_has_b1_(root.domain_size, 0),
      orbit_has_b0_(root.domain_size, 0),
      orbit_has_continuous_(root.domain_size, 0),
      orbit_has_f0_(root.domain_size, 0),
      orbit_has_f1_(root.domain_size, 0),
      marked_b0_(root.num_original_vars, 0),
      marked_b1_(root.num_original_vars, 0),
      marked_f0_(root.num_original_vars, 0),
      marked_f1_(root.num_original_vars, 0)
  {
    f0_.reserve(root.num_original_vars);
    f1_.reserve(root.num_original_vars);

    // Clone the root Schreier structure into a per-worker copy. The root structure
    // is shared and read-only, but set_base/get_stabilizer_orbit mutate internal state,
    // so each worker needs its own copy. Since random_schreier is not copyable, we
    // reconstruct it by creating a fresh structure with the same base and sifting all
    // generators from the root into it.
    schreier_ = std::make_unique<dejavu::groups::random_schreier>(root.domain_size);
    std::vector<int> base(root.num_original_vars);
    std::iota(base.begin(), base.end(), 0);
    schreier_->set_base(base);

    int num_gens = root.schreier->get_number_of_generators();
    dejavu::groups::automorphism_workspace ws(root.domain_size);
    for (int i = 0; i < num_gens; i++) {
      root.schreier->get_generator(i, ws);
      schreier_->sift(ws);
    }
  }

  bool disabled () const { return disabled_; }
  void disable() { disabled_ = true; }
  void enable() { disabled_ = false; }

  void orbital_fixing(mip_symmetry_t<i_t, f_t>* symmetry,
                      const simplex_solver_settings_t<i_t, f_t>& settings,
                      const std::vector<variable_type_t>& var_types,
                      mip_node_t<i_t, f_t>* node_ptr,
                      lp_problem_t<i_t, f_t>& problem)
  {
    // Collect binary branchings only; skip general integer branchings
    std::vector<i_t> branched_zero;
    std::vector<i_t> branched_one;
    branched_zero.reserve(node_ptr->depth);
    branched_one.reserve(node_ptr->depth);
    mip_node_t<i_t, f_t>* node = node_ptr;
    while (node != nullptr && node->branch_var >= 0) {
      i_t v = node->branch_var;
      bool is_binary = (symmetry->is_binary[v] == 1);
      if (is_binary) {
        if (node->branch_var_upper == 0.0) {
          branched_zero.push_back(v);
          marked_b0_[v] = 1;
        } else if (node->branch_var_lower == 1.0) {
          branched_one.push_back(v);
          marked_b1_[v] = 1;
        }
      }
      node = node->parent;
    }

    for (i_t j = 0; j < num_original_vars_; j++) {
      if (var_types[j] == variable_type_t::CONTINUOUS) continue;
      if (marked_b1_[j] == 0 && problem.lower[j] == 1.0) {
        f1_.push_back(j);
        marked_f1_[j] = 1;
      }
      if (marked_b0_[j] == 0 && problem.upper[j] == 0.0) {
        f0_.push_back(j);
        marked_f0_[j] = 1;
      }
    }

    // Compute Stab(G', B1) where G' = Stab(G, general_integer_variables)
    // Base order: [general_int_vars, B1, remaining_vars]
    const auto& gen_int = symmetry->general_integer_variables;
    std::vector<i_t> new_base;
    new_base.reserve(num_original_vars_);
    for (i_t j : gen_int) {
      new_base.push_back(j);
      marked_variables_[j] = 1;
    }
    for (i_t j : branched_one) {
      new_base.push_back(j);
      marked_variables_[j] = 1;
    }
    for (i_t j = 0; j < num_original_vars_; j++) {
      if (marked_variables_[j] == 0) { new_base.push_back(j); }
    }
    for (i_t j : gen_int) {
      marked_variables_[j] = 0;
    }
    for (i_t j : branched_one) {
      marked_variables_[j] = 0;
    }

    schreier_->set_base(new_base);

    dejavu::groups::orbit orb;
    orb.initialize(domain_size_);
    i_t stabilizer_level = static_cast<i_t>(gen_int.size() + branched_one.size());
    schreier_->get_stabilizer_orbit(static_cast<int>(stabilizer_level), orb);

    // Check if the stabilizer is trivial (all orbits are singletons)
    bool trivial_stabilizer = true;
    for (i_t j = 0; j < num_original_vars_; j++) {
      if (orb.orbit_size(orb.find_orbit(j)) > 1) {
        trivial_stabilizer = false;
        break;
      }
    }

    if (trivial_stabilizer) {
      disabled_ = true;
      for (i_t v : branched_one) {
        marked_b1_[v] = 0;
      }
      for (i_t v : branched_zero) {
        marked_b0_[v] = 0;
      }
      for (i_t v : f0_) {
        marked_f0_[v] = 0;
      }
      for (i_t v : f1_) {
        marked_f1_[v] = 0;
      }
      f0_.clear();
      f1_.clear();
    } else {
      for (i_t v : branched_one) {
        orbit_has_b1_[orb.find_orbit(v)] = 1;
      }

      for (i_t v : branched_zero) {
        orbit_has_b0_[orb.find_orbit(v)] = 1;
      }

      for (i_t v : symmetry->continuous_variables) {
        orbit_has_continuous_[orb.find_orbit(v)] = 1;
      }

      for (i_t v : f0_) {
        orbit_has_f0_[orb.find_orbit(v)] = 1;
      }

      for (i_t v : f1_) {
        orbit_has_f1_[orb.find_orbit(v)] = 1;
      }

      std::vector<i_t> fix_zero;  // The set L0 of variables that can be fixed to 0
      std::vector<i_t> fix_one;   // The set L1 of variables that can be fixed to 1

      for (i_t j = 0; j < num_original_vars_; j++) {
        i_t o = orb.find_orbit(j);
        if (orb.orbit_size(o) < 2) continue;

        if (orbit_has_b1_[o] == 1 || orbit_has_continuous_[o] == 1) {
          // The orbit contains variables in B1 or continuous variables
          // So we can't fix any variables in this orbit
          continue;
        }

        if (orbit_has_b0_[o] == 1 || orbit_has_f0_[o] == 1) {
          // The orbit of this variable contains variables in B0 or F0
          // So we can fix this variable to zero (provided its not already in B0 or F0)
          if (marked_b0_[j] == 0 && marked_f0_[j] == 0) { fix_zero.push_back(j); }
        }

        if (orbit_has_f1_[o] == 1) {
          // The orbit of this variable contains variables in F1
          // So we can fix this variable to one (provided its not already in F1)
          if (marked_f1_[j] == 0) { fix_one.push_back(j); }
        }
      }

      // Restore the work arrays
      for (i_t v : branched_one) {
        orbit_has_b1_[orb.find_orbit(v)] = 0;
        marked_b1_[v]                    = 0;
      }

      for (i_t v : branched_zero) {
        orbit_has_b0_[orb.find_orbit(v)] = 0;
        marked_b0_[v]                    = 0;
      }

      for (i_t v : symmetry->continuous_variables) {
        orbit_has_continuous_[orb.find_orbit(v)] = 0;
      }

      for (i_t v : f0_) {
        orbit_has_f0_[orb.find_orbit(v)] = 0;
        marked_f0_[v]                    = 0;
      }

      for (i_t v : f1_) {
        orbit_has_f1_[orb.find_orbit(v)] = 0;
        marked_f1_[v]                    = 0;
      }

      f0_.clear();
      f1_.clear();

      if (fix_zero.size() > 0 || fix_one.size() > 0) {
        settings.log.printf("Orbital fixing at node %d (depth %d): fixing %d to 0 and %d to 1\n",
                            node_ptr->node_id,
                            node_ptr->depth,
                            (int)fix_zero.size(),
                            (int)fix_one.size());
      }

      // Finally fix the variables in L0 and L1
      for (i_t v : fix_zero) {
        problem.lower[v] = 0.0;
        problem.upper[v] = 0.0;
      }
      for (i_t v : fix_one) {
        problem.lower[v] = 1.0;
        problem.upper[v] = 1.0;
      }
    }
  }

 private:
  std::unique_ptr<dejavu::groups::random_schreier> schreier_;
  i_t domain_size_;
  i_t num_original_vars_;
  bool disabled_ = false;

  std::vector<i_t> marked_variables_;
  std::vector<i_t> orbit_has_b1_;     // orbit_has_b1_[o] = 1 if the orbit o contains variables in B1
  std::vector<i_t> orbit_has_b0_;     // orbit_has_b0_[o] = 1 if the orbit o contains variables in B0
  std::vector<i_t> orbit_has_continuous_; // orbit_has_continuous_[o] = 1 if the orbit o contains continuous variables
  std::vector<i_t> orbit_has_f0_;     // orbit_has_f0_[o] = 1 if the orbit o contains variables in F0
  std::vector<i_t> orbit_has_f1_;     // orbit_has_f1_[o] = 1 if the orbit o contains variables in F1
  std::vector<i_t> marked_b0_;        // marked_b0_[v] = 1 if the variable v has been branched down
  std::vector<i_t> marked_b1_;        // marked_b1_[v] = 1 if the variable v has been branched up
  std::vector<i_t> f0_;               // set of variables that can be fixed to 0
  std::vector<i_t> f1_;               // set of variables that can be fixed to 1
  std::vector<i_t> marked_f0_;        // marked_f0_[v] = 1 if the variable v has been branched down
  std::vector<i_t> marked_f1_;        // marked_f1_[v] = 1 if the variable v has been branched up
};

template <typename i_t, typename f_t>
std::unique_ptr<mip_symmetry_t<i_t, f_t>> detect_symmetry(
  const user_problem_t<i_t, f_t>& user_problem,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  bool& has_symmetry)
{
  has_symmetry = false;

  f_t start_time = tic();
  lp_problem_t<i_t, f_t> problem(user_problem.handle_ptr, 1, 1, 1);
  std::vector<i_t> new_slacks;
  dualize_info_t<i_t, f_t> dualize_info;
  convert_user_problem(user_problem, settings, problem, new_slacks, dualize_info);
  std::vector<variable_type_t> var_types = user_problem.var_types;
  if (problem.num_cols > user_problem.num_cols) {
    var_types.resize(problem.num_cols);
    for (i_t k = user_problem.num_cols; k < problem.num_cols; k++) {
      var_types[k] = variable_type_t::CONTINUOUS;
    }
  }

  // We now have the problem in the form:
  // minimize   c^T x
  // subject to A * x = b,
  //            l <= x <= u
  //            x_j in Z  for all j such that var_types[j] == variable_type_t::INTEGER

  // Construct a graph G(V, W, R, E)
  // where V is the set of nodes corresponding to the variables
  // R is the set of nodes corresponding to the the constraints
  // W is the set of nodes corresponding to the nonzero coefficients in the A matrix
  // E is the set of edges
  //
  // Associated with each node in this graph is a color: c_v, c_w, c_r.

  const i_t V_size = problem.num_cols;
  const i_t R_size = problem.num_rows;

  const f_t tol = 1e-10;

  // Compute the colors for the variables
  std::vector<i_t> obj_perm(problem.num_cols);
  std::iota(obj_perm.begin(), obj_perm.end(), 0);
  std::sort(obj_perm.begin(), obj_perm.end(), [&](i_t a, i_t b) {
    if (problem.objective[a] != problem.objective[b])
      return problem.objective[a] < problem.objective[b];
    if (problem.lower[a] != problem.lower[b]) return problem.lower[a] < problem.lower[b];
    if (problem.upper[a] != problem.upper[b]) return problem.upper[a] < problem.upper[b];
    return var_types[a] < var_types[b];
  });
  std::vector<i_t> var_colors(problem.num_cols, -1);
  i_t var_color             = 0;
  f_t last_obj              = problem.objective[obj_perm[0]];
  f_t last_lower            = problem.lower[obj_perm[0]];
  f_t last_upper            = problem.upper[obj_perm[0]];
  variable_type_t last_type = var_types[obj_perm[0]];
  var_colors[obj_perm[0]]   = var_color;
  for (i_t k = 1; k < problem.num_cols; k++) {
    const i_t j   = obj_perm[k];
    const f_t obj = problem.objective[j];
    if (obj - last_obj > tol || problem.lower[j] != last_lower || problem.upper[j] != last_upper ||
        var_types[j] != last_type) {
      var_color++;
      last_obj   = obj;
      last_lower = problem.lower[j];
      last_upper = problem.upper[j];
      last_type  = var_types[j];
    }
    var_colors[j] = var_color;
  }

  // Compute the colors for the constraints
  std::vector<i_t> rhs_perm(problem.num_rows);
  std::iota(rhs_perm.begin(), rhs_perm.end(), 0);
  std::sort(rhs_perm.begin(), rhs_perm.end(), [&](i_t a, i_t b) {
    return problem.rhs[a] < problem.rhs[b];
  });
  std::vector<i_t> rhs_colors(problem.num_rows, -1);
  i_t rhs_color           = var_color + 1;
  f_t last_rhs            = problem.rhs[rhs_perm[0]];
  rhs_colors[rhs_perm[0]] = rhs_color;
  for (i_t k = 1; k < problem.num_rows; k++) {
    const i_t i   = rhs_perm[k];
    const f_t rhs = problem.rhs[i];
    if (rhs - last_rhs > tol) {
      rhs_color++;
      last_rhs = rhs;
    }
    rhs_colors[i] = rhs_color;
  }

  // Calculate the number of colors needed for each nonzero coefficient in the A matrix
  const i_t nnz = problem.A.col_start[problem.num_cols];
  // Construct the graph
  // We begin by creating the vertex set := V union R union W

  std::vector<i_t> vertices;
  vertices.reserve(V_size + R_size + nnz);
  std::vector<i_t> vertex_colors;
  vertex_colors.reserve(V_size + R_size + nnz);
  for (i_t j = 0; j < V_size; j++) {
    vertices.push_back(j);
    vertex_colors.push_back(var_colors[j]);
  }
  for (i_t i = 0; i < R_size; i++) {
    vertices.push_back(V_size + i);
    vertex_colors.push_back(rhs_colors[i]);
  }

  std::vector<i_t> edge_in;
  std::vector<i_t> edge_out;
  edge_in.reserve(2 * nnz);
  edge_out.reserve(2 * nnz);

#ifdef FULL_GRAPH
  // Every nonzero should have an edge between (v, r) with v in V and r in R
  // To handle the edge color we create a new node w and an edge (v, w) and (w, r)
  // where w is colored according to the edge color.

  std::vector<f_t> nonzeros = problem.A.x;
  std::vector<i_t> nonzero_perm(nnz);
  std::iota(nonzero_perm.begin(), nonzero_perm.end(), 0);
  std::sort(nonzero_perm.begin(), nonzero_perm.end(), [&](i_t a, i_t b) {
    return nonzeros[a] < nonzeros[b];
  });
  std::vector<i_t> nonzero_colors(nnz, -1);
  i_t edge_color                  = rhs_color + 1;
  f_t last_nz                     = nonzeros[nonzero_perm[0]];
  nonzero_colors[nonzero_perm[0]] = edge_color;
  for (i_t q = 1; q < nnz; q++) {
    const i_t p   = nonzero_perm[q];
    const f_t val = nonzeros[p];
    if (val - last_nz > tol) {
      edge_color++;
      last_nz = val;
    }
    nonzero_colors[p] = edge_color;
  }

  for (i_t j = 0; j < problem.num_cols; j++) {
    const i_t col_start = problem.A.col_start[j];
    const i_t col_end   = problem.A.col_start[j + 1];
    for (i_t p = col_start; p < col_end; p++) {
      const i_t i = problem.A.i[p];
      vertices.push_back(V_size + R_size + p);
      vertex_colors.push_back(nonzero_colors[p]);
      edge_in.push_back(j);
      edge_out.push_back(V_size + R_size + p);
      edge_in.push_back(V_size + R_size + p);
      edge_out.push_back(V_size + i);
    }
  }
#else

  // Let r_i be the vertex associated with the row i
  // Let V_i,c be the set of variables in row i with the same color c
  // We create a new vertex w_i,c and edges (v_j, w_i,c) and (w_i,c, r_i) for all v_j in V_i,c
  csr_matrix_t<i_t, f_t> A_row(problem.num_rows, problem.num_cols, 0);
  problem.A.to_compressed_row(A_row);

  std::vector<f_t> nonzeros = A_row.x;
  std::vector<i_t> nonzero_perm(nnz);
  std::iota(nonzero_perm.begin(), nonzero_perm.end(), 0);
  std::sort(nonzero_perm.begin(), nonzero_perm.end(), [&](i_t a, i_t b) {
    return nonzeros[a] < nonzeros[b];
  });
  std::vector<i_t> nonzero_colors(nnz, -1);
  i_t edge_color                  = 0;
  f_t last_nz                     = nonzeros[nonzero_perm[0]];
  nonzero_colors[nonzero_perm[0]] = edge_color;
  for (i_t q = 1; q < nnz; q++) {
    const i_t p   = nonzero_perm[q];
    const f_t val = nonzeros[p];
    if (val - last_nz > tol) {
      edge_color++;
      last_nz = val;
    }
    nonzero_colors[p] = edge_color;
  }

  i_t num_edge_colors = edge_color + 1;
  std::vector<i_t> edge_color_map(num_edge_colors, 0);
  i_t max_nzs_per_row = 0;
  for (i_t i = 0; i < problem.num_rows; i++) {
    const i_t row_start = A_row.row_start[i];
    const i_t row_end   = A_row.row_start[i + 1];
    max_nzs_per_row     = std::max(max_nzs_per_row, row_end - row_start);
  }
  std::vector<i_t> row_colors;
  std::vector<i_t> sorted_nonzeros_in_row;
  row_colors.reserve(max_nzs_per_row);
  sorted_nonzeros_in_row.reserve(max_nzs_per_row);

  i_t current_vertex = V_size + R_size;
  for (i_t i = 0; i < problem.num_rows; i++) {
    const i_t row_start = A_row.row_start[i];
    const i_t row_end   = A_row.row_start[i + 1];
    const i_t row_nz    = row_end - row_start;
    row_colors.clear();
    sorted_nonzeros_in_row.resize(row_nz);

    // Pass 1: Count the number of occurences of each color and the number of unique colors in the
    // current row
    for (i_t p = row_start; p < row_end; p++) {
      const i_t edge_color = nonzero_colors[p];
      edge_color_map[edge_color]++;
      if (edge_color_map[edge_color] == 1) { row_colors.push_back(edge_color); }
    }

    // Pass 2: Compute the prefix sum directly in edge_color_map
    //         Note we only touch the colors that are present in the current row
    i_t cumulative_sum = 0;
    for (i_t k = 0; k < static_cast<i_t>(row_colors.size()); k++) {
      const i_t edge_color       = row_colors[k];
      const i_t count            = edge_color_map[edge_color];
      edge_color_map[edge_color] = cumulative_sum;
      cumulative_sum += count;
    }

    // Pass 3: Place the nonzeros in sorted order
    for (i_t p = row_start; p < row_end; p++) {
      const i_t edge_color                                 = nonzero_colors[p];
      sorted_nonzeros_in_row[edge_color_map[edge_color]++] = p;
    }

    // Clear the edge_color_map for the colors we used
    for (i_t edge_color : row_colors) {
      edge_color_map[edge_color] = 0;
    }

    // Pass 4: iterate over the sorted nonzeros, create new vertices and edges
    i_t last_color = -1;
    for (i_t k = 0; k < row_nz; k++) {
      const i_t p          = sorted_nonzeros_in_row[k];
      const i_t j          = A_row.j[p];
      const i_t edge_color = nonzero_colors[p];
      if (edge_color == last_color) {
        // We don't need to create a new vertex
        // Add the edge (v_j, w_i_c)
        edge_in.push_back(j);
        edge_out.push_back(current_vertex - 1);
      } else {
        last_color = edge_color;
        // Create a new vertex w_i_c
        vertices.push_back(current_vertex);
        vertex_colors.push_back(rhs_color + 1 + edge_color);
        // Add the edge (v_j, w_i_c)
        edge_in.push_back(j);
        edge_out.push_back(current_vertex);
        // Add the edge (w_i_c, r_i)
        edge_in.push_back(current_vertex);
        edge_out.push_back(V_size + i);
        current_vertex++;
      }
    }
  }

#endif

  settings.log.printf("Graph construction time %f\n", toc(start_time));
  f_t dejavu_start_time = tic();

  // The graph should now be described by:
  // vertices, edge_in, edge_out, vertex_colors

  // Dejavu needs the degree of each vertex
  std::vector<i_t> degrees(vertices.size(), 0);
  const i_t num_edges = edge_in.size();
  for (i_t i = 0; i < num_edges; i++) {
    degrees[edge_in[i]]++;
    degrees[edge_out[i]]++;
  }

  dejavu::static_graph g;
  g.initialize_graph(vertices.size(), edge_in.size());

  const i_t num_vertices = vertices.size();
  for (i_t i = 0; i < num_vertices; i++) {
    g.add_vertex(vertex_colors[i], degrees[i]);
  }
  for (i_t i = 0; i < num_edges; i++) {
    const i_t u = edge_in[i];
    const i_t v = edge_out[i];
    g.add_edge(std::min(u, v), std::max(u, v));
  }

  auto schreier = std::make_unique<dejavu::groups::random_schreier>(num_vertices);
  std::vector<int> base;
  base.reserve(user_problem.num_cols);
  for (i_t j = 0; j < user_problem.num_cols; j++) {
    base.push_back(j);
  }
  schreier->set_base(base);

  dejavu::hooks::schreier_hook s_hook(*schreier);

  i_t num_generators          = 0;
  const i_t num_original_vars = user_problem.num_cols;
  dejavu_hook counting_hook = [&num_generators, num_original_vars](int, const int*, int nsupp, const int* supp) {
    for (int s = 0; s < nsupp; s++) {
      if (supp[s] < num_original_vars) {
        num_generators++;
        return;
      }
    }
  };

  dejavu::hooks::multi_hook combined;
  combined.add_hook(s_hook.get_hook());
  combined.add_hook(&counting_hook);

  dejavu::solver d;
  d.automorphisms(&g, combined);

  std::ostringstream grp_size_str;
  grp_size_str << d.get_automorphism_group_size();
  settings.log.printf(
    "Automorphism group size %s, %d generators\n", grp_size_str.str().c_str(), num_generators);
  settings.log.printf("Dejavu time %f\n", toc(dejavu_start_time));

  has_symmetry = false;
  if (num_generators > 0) {
    dejavu::groups::orbit orb;
    orb.initialize(num_vertices);
    schreier->get_stabilizer_orbit(0, orb);

    i_t num_nontrivial_orbits = 0;
    i_t max_orbit_size = 0;
    i_t total_vars_in_orbits = 0;
    std::vector<i_t> orbit_histogram(num_original_vars + 1, 0);
    for (i_t j = 0; j < num_original_vars; j++) {
      if (orb.represents_orbit(j)) {
        i_t sz = orb.orbit_size(j);
        if (sz >= 2) {
          num_nontrivial_orbits++;
          max_orbit_size = std::max(max_orbit_size, sz);
          total_vars_in_orbits += sz;
          orbit_histogram[sz]++;
        }
      }
    }
    settings.log.printf("Orbits: %d non-trivial, max size %d, %d/%d (%.1f%%) variables in orbits\n",
                        num_nontrivial_orbits, max_orbit_size, total_vars_in_orbits, num_original_vars,
                        100.0 * total_vars_in_orbits / num_original_vars);
    settings.log.printf("Orbit histogram (orbit size: number of orbits):");
    for (i_t sz = 2; sz <= max_orbit_size; sz++) {
      if (orbit_histogram[sz] > 0) {
        settings.log.printf(" %d:%d", sz, orbit_histogram[sz]);
      }
    }
    settings.log.printf("\n");

    has_symmetry = (max_orbit_size >= 4) ||
                    (num_nontrivial_orbits >= 3 && max_orbit_size >= 2) ||
                    (total_vars_in_orbits >= 10);
  }

  settings.log.printf("Total symmetry detection time %f\n", toc(start_time));

  if (!has_symmetry) { return nullptr; }

  return std::make_unique<mip_symmetry_t<i_t, f_t>>(
    std::move(schreier), num_vertices, user_problem.num_cols, var_types,
    user_problem.lower, user_problem.upper);
}

}  // namespace cuopt::linear_programming::dual_simplex
