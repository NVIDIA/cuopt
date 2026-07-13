/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <branch_and_bound/mip_node.hpp>
#include <dual_simplex/presolve.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>
#include <dual_simplex/user_problem.hpp>
#include <math_optimization/tic_toc.hpp>
#include <math_optimization/types.hpp>

#include "dejavu.h"

#include <memory>
#include <numeric>
#include <sstream>

namespace cuopt::mathematical_optimization::mip {

// permutation_t stores a dense permutation plus its support (non-identity entries).
template <typename i_t>
class permutation_t {
 public:
  // Takes vectors by value so callers can std::move or copy as needed.
  permutation_t(std::vector<i_t> p) : n_(p.size()), p_(std::move(p))
  {
    for (i_t k = 0; k < n_; k++) {
      if (p_[k] != k) { support_.push_back(k); }
    }
  }
  permutation_t(std::vector<i_t> p, std::vector<i_t> support)
    : n_(p.size()), p_(std::move(p)), support_(std::move(support))
  {
  }

  permutation_t(int n, const int* p, int nsupp, const int* supp) : n_(n)
  {
    p_.resize(n);
    std::iota(p_.begin(), p_.end(), 0);
    for (int k = 0; k < nsupp; k++) {
      const int i = supp[k];
      p_[i]       = p[i];
      support_.push_back(i);
    }
  }

  i_t size() const { return n_; }
  const std::vector<i_t>& dense_permutation() const { return p_; }
  const std::vector<i_t>& support() const { return support_; }

  permutation_t<i_t> inverse() const
  {
    std::vector<i_t> pinv(n_);
    for (i_t k = 0; k < n_; k++) {
      pinv[p_[k]] = k;
    }
    return permutation_t<i_t>(std::move(pinv), support_);
  }

 private:
  i_t n_;
  std::vector<i_t> p_;
  std::vector<i_t> support_;
};

// group_generators_t stores a list of permutations. Can be constructed from a sparse
// representation.
template <typename i_t>
class group_generators_t {
 public:
  group_generators_t() : n_(-1) {}
  void add_generator(int n, const int* p, int nsupp, const int* supp)
  {
    if (n_ == -1) { n_ = n; }
    if (n != n_) {
      return;
    } else {
      generators_.emplace_back(n, p, nsupp, supp);
    }
  }
  void add_generator(const permutation_t<i_t>& p)
  {
    if (n_ == -1) { n_ = p.size(); }
    if (p.size() != n_) { return; }
    generators_.emplace_back(p);
  }
  void add_generator(std::vector<i_t> p, std::vector<i_t> support)
  {
    const i_t n = static_cast<i_t>(p.size());
    if (n_ == -1) { n_ = n; }
    if (n != n_) { return; }
    generators_.emplace_back(std::move(p), std::move(support));
  }
  i_t size() const { return n_; }
  size_t num_generators() const { return generators_.size(); }

  const permutation_t<i_t>& get_generator(i_t i) const { return generators_[i]; }

  // Remove generators that don't preserve the given bounds.
  // A generator is invalid if it maps variable j to p[j] where
  // lower[j] != lower[p[j]] or upper[j] != upper[p[j]].
  template <typename f_t>
  i_t prune_by_bounds(const std::vector<f_t>& lower, const std::vector<f_t>& upper)
  {
    i_t num_removed = 0;
    size_t k        = 0;
    while (k < generators_.size()) {
      const auto& perm = generators_[k];
      const auto& p    = perm.dense_permutation();
      bool valid       = true;
      for (i_t j : perm.support()) {
        i_t pj = p[j];
        if (lower[j] != lower[pj] || upper[j] != upper[pj]) {
          valid = false;
          break;
        }
      }
      if (!valid) {
        std::swap(generators_[k], generators_.back());
        generators_.pop_back();
        num_removed++;
      } else {
        k++;
      }
    }
    return num_removed;
  }

 private:
  i_t n_;
  std::vector<permutation_t<i_t>> generators_;
};

// orbits_t computes the orbits of a set of generators using the union-find algorithm.
template <typename i_t>
class orbits_t {
 public:
  orbits_t(i_t n) : n_(n), parent_(n), size_(n, 1), dirty_(n, 0)
  {
    std::iota(parent_.begin(), parent_.end(), 0);
    dirty_list_.reserve(n);
  }

  void compute_orbits(const std::vector<i_t>& indices, const group_generators_t<i_t>& generators)
  {
    for (i_t i : indices) {
      const permutation_t<i_t>& perm = generators.get_generator(i);
      const std::vector<i_t>& p      = perm.dense_permutation();
      for (i_t k : perm.support()) {
        union_sets(k, p[k]);
      }
    }
  }

  // Incrementally update orbits with a single mapping u -> v.
  void add_mapping(i_t u, i_t v) { union_sets(u, v); }

  void compute_orbits(const group_generators_t<i_t>& generators)
  {
    std::vector<i_t> indices(generators.num_generators());
    std::iota(indices.begin(), indices.end(), 0);
    compute_orbits(indices, generators);
  }

  i_t find_orbit(i_t v) { return find(v); }

  bool represents_orbit(i_t v) { return find(v) == v; }

  i_t orbit_size(i_t v) { return size_[find(v)]; }

  // Reset only the given indices back to identity (parent[j] = j, size[j] = 1).
  void reset()
  {
    for (i_t j : dirty_list_) {
      parent_[j] = j;
      size_[j]   = 1;
      dirty_[j]  = 0;
    }
    dirty_list_.clear();
  }

 private:
  void union_sets(i_t u, i_t v)
  {
    i_t root_u = find(u);
    i_t root_v = find(v);
    if (root_u == root_v) return;  // Already in the same set
    mark_dirty(root_u);
    mark_dirty(root_v);
    if (size_[root_u] < size_[root_v]) {
      parent_[root_u] = root_v;
      size_[root_v] += size_[root_u];
    } else {
      parent_[root_v] = root_u;
      size_[root_u] += size_[root_v];
    }
  }

  i_t find(i_t v)
  {
    i_t root = v;
    while (parent_[root] != root) {
      root = parent_[root];
    }
    // Path compression
    while (parent_[v] != root) {
      i_t next   = parent_[v];
      parent_[v] = root;
      mark_dirty(v);
      v = next;
    }
    return root;
  }

  void mark_dirty(i_t v)
  {
    if (dirty_[v] == 0) {
      dirty_list_.push_back(v);
      dirty_[v] = 1;
    }
  }

  i_t n_;
  std::vector<i_t> parent_;
  std::vector<i_t> size_;
  std::vector<i_t> dirty_;
  std::vector<i_t> dirty_list_;
};

template <typename i_t, typename f_t>
struct mip_symmetry_t {
  group_generators_t<i_t> generators;
  i_t num_original_vars;
  int num_generators = 0;
  std::vector<i_t> binary_variables;
  std::vector<i_t> general_integer_variables;
  std::vector<i_t> is_binary;

  // Precomputed orbit representative for each original variable under the projected group.
  // orbit_rep[j] = orbit representative of variable j (for j < num_original_vars).
  std::vector<i_t> orbit_rep;

  group_generators_t<i_t> inverse_generators;
};

template <typename i_t, typename f_t>
class orbital_fixing_t {
 public:
  explicit orbital_fixing_t(mip_symmetry_t<i_t, f_t>& root)
    : num_original_vars_(root.num_original_vars),
      max_generators_(root.num_generators),
      start_plunge_(true),
      orb_(root.num_original_vars),
      orbit_has_b1_(root.num_original_vars, 0),
      orbit_has_b0_(root.num_original_vars, 0),
      orbit_has_f0_(root.num_original_vars, 0),
      orbit_has_f1_(root.num_original_vars, 0),
      marked_b0_(root.num_original_vars, 0),
      marked_b1_(root.num_original_vars, 0),
      marked_f0_(root.num_original_vars, 0),
      marked_f1_(root.num_original_vars, 0)
  {
    branched_zero_.reserve(root.num_original_vars);
    branched_one_.reserve(root.num_original_vars);
    f0_.reserve(root.num_original_vars);
    f1_.reserve(root.num_original_vars);
    fix_zero_.reserve(root.num_original_vars);
    fix_one_.reserve(root.num_original_vars);

    surviving_generators_.resize(max_generators_);
    std::iota(surviving_generators_.begin(), surviving_generators_.end(), 0);
  }

  bool disabled() const { return surviving_generators_.empty(); }
  void disable() { surviving_generators_.clear(); }

  // Store the current cumulative orbital fixings into the node without
  // running the full orbital fixing computation.  Called when orbital
  // fixing is disabled (trivial stabilizer) so that children starting
  // a new plunge still inherit ancestor fixings.
  void propagate_cumulative_fixings(mip_node_t<i_t, f_t>* node_ptr)
  {
    node_ptr->orbital_fix_zero = cumulative_fix_zero_;
    node_ptr->orbital_fix_one  = cumulative_fix_one_;
  }

  void reset(mip_symmetry_t<i_t, f_t>* symmetry, mip_node_t<i_t, f_t>* node_ptr)
  {
    for (i_t j : branched_zero_) {
      marked_b0_[j]                     = 0;
      orbit_has_b0_[orb_.find_orbit(j)] = 0;
    }
    for (i_t j : branched_one_) {
      marked_b1_[j]                     = 0;
      orbit_has_b1_[orb_.find_orbit(j)] = 0;
    }
    branched_zero_.clear();
    branched_one_.clear();
    orb_.reset();

    mip_node_t<i_t, f_t>* node = node_ptr->parent;
    while (node != nullptr && node->branch_var >= 0) {
      i_t v          = node->branch_var;
      bool is_binary = (symmetry->is_binary[v] == 1);
      if (is_binary) {
        if (node->branch_var_upper == 0.0) {
          branched_zero_.push_back(v);
          marked_b0_[v] = 1;
        } else if (node->branch_var_lower == 1.0) {
          branched_one_.push_back(v);
          marked_b1_[v] = 1;
        }
      }
      node = node->parent;
    }

    surviving_generators_.resize(max_generators_);
    std::iota(surviving_generators_.begin(), surviving_generators_.end(), 0);

    // Seed cumulative fixings from the parent's stored orbital fixings.
    // These were accumulated during the parent's plunge and represent all
    // orbital fixing bound changes from root to parent.
    cumulative_fix_zero_.clear();
    cumulative_fix_one_.clear();
    if (node_ptr->parent != nullptr) {
      cumulative_fix_zero_ = node_ptr->parent->orbital_fix_zero;
      cumulative_fix_one_  = node_ptr->parent->orbital_fix_one;
    }

    start_plunge_ = true;
  }

  // Returns the number of free variables in conflicting orbits (orbits with
  // both zero and one sources).
  i_t orbital_fixing(mip_symmetry_t<i_t, f_t>* symmetry,
                     const simplex::simplex_solver_settings_t<i_t, f_t>& settings,
                     mip_node_t<i_t, f_t>* node_ptr,
                     simplex::lp_problem_t<i_t, f_t>& problem,
                     const std::vector<f_t>& start_lower,
                     const std::vector<f_t>& start_upper)
  {
    // At the start of a new plunge, restore the parent's cumulative orbital
    // fixings into the problem.  These bound changes were derived during the
    // parent's plunge and are valid for the entire subtree.  Without this,
    // orbital fixings from ancestors with larger stabilizers would be lost,
    // causing the LP bound to regress and the global lower bound to become
    // non-monotonic.
    if (start_plunge_) {
      for (i_t v : cumulative_fix_zero_) {
        problem.lower[v] = 0.0;
        problem.upper[v] = 0.0;
      }
      for (i_t v : cumulative_fix_one_) {
        problem.lower[v] = 1.0;
        problem.upper[v] = 1.0;
      }
    }

    // Collect binary branchings only; skip general integer branchings
    i_t v                 = node_ptr->branch_var;
    bool is_binary        = (symmetry->is_binary[v] == 1);
    bool should_recompute = start_plunge_;
    if (is_binary) {
      if (node_ptr->branch_var_upper == 0.0) {
        branched_zero_.push_back(v);
        marked_b0_[v] = 1;
        // Need to recompute: propagation at this node may have fixed new variables
        // to 1, invalidating generators that map B0 variables to those newly-fixed vars.
        should_recompute = true;
      } else if (node_ptr->branch_var_lower == 1.0) {
        branched_one_.push_back(v);
        marked_b1_[v]    = 1;
        should_recompute = true;
      }
    }

    // Collect F0/F1: variables fixed by node-level propagation only.
    // Exclude root-level fixings (from reduced cost fixing / bound strengthening
    // after cuts) because these can be asymmetric w.r.t. the symmetry group.
    for (i_t j : symmetry->binary_variables) {
      if (marked_b1_[j] == 0 && problem.lower[j] == 1.0 && start_lower[j] < 1.0) {
        f1_.push_back(j);
        marked_f1_[j] = 1;
      }
      if (marked_b0_[j] == 0 && problem.upper[j] == 0.0 && start_upper[j] > 0.0) {
        f0_.push_back(j);
        marked_f0_[j] = 1;
      }
    }

    // In true orbital fixing we would compute the group G' = stabilizer(G, B1)
    // Instead we compute a subgroup H of G' using generator filtering, additionally
    // checking that for each variable in B0, its image is not fixed to 1.
    // This is necessary because propagation using asymmetric cuts can fix variables
    // to 1 that the symmetry group maps to 0-branched variables, creating false conflicts.
    // See van Doornmalen & Hojny (2022), "A Unified Framework for Symmetry Handling".
    //
    // We always recompute because propagation at each node can invalidate generators.
    if (should_recompute) {
      for (size_t k = 0; k < surviving_generators_.size(); k++) {
        const i_t h                    = surviving_generators_[k];
        const permutation_t<i_t>& perm = symmetry->generators.get_generator(h);
        const std::vector<i_t>& p      = perm.dense_permutation();
        bool stabilizes                = true;

        if (start_plunge_) {
          // At the start of a plunge, surviving_generators_ was reset to all generators.
          // We need to check each generator against all of B1 and B0.
          for (i_t j : branched_one_) {
            if (marked_b1_[p[j]] == 0) {
              stabilizes = false;
              break;
            }
          }
          // Check B0: for each variable branched to 0, its image must not be fixed to 1.
          if (stabilizes) {
            for (i_t j : branched_zero_) {
              if (problem.lower[p[j]] >= 1.0) {
                stabilizes = false;
                break;
              }
            }
          }
        } else {
          // Incremental check. The surviving generators already stabilize the parent's B1.
          // Check the new branching variable.
          if (is_binary && node_ptr->branch_var_lower == 1.0) {
            // Branched to 1: check g(v) is also in B1
            if (marked_b1_[p[v]] == 0) { stabilizes = false; }
          }
          // Check B0: propagation at this node may have newly fixed variables to 1.
          // A generator that was valid at the parent may now map a B0 variable to
          // something that propagation fixed to 1.
          if (stabilizes) {
            for (i_t j : branched_zero_) {
              if (problem.lower[p[j]] >= 1.0) {
                stabilizes = false;
                break;
              }
            }
          }
        }

        if (!stabilizes) {
          std::swap(surviving_generators_[k], surviving_generators_.back());
          surviving_generators_.pop_back();
          k--;
        }
      }

      // Clear old orbit_has values before orbits change
      for (i_t v : branched_one_) {
        orbit_has_b1_[orb_.find_orbit(v)] = 0;
      }
      for (i_t v : branched_zero_) {
        orbit_has_b0_[orb_.find_orbit(v)] = 0;
      }

      orb_.reset();
      orb_.compute_orbits(surviving_generators_, symmetry->generators);

      for (i_t v : branched_one_) {
        orbit_has_b1_[orb_.find_orbit(v)] = 1;
      }
      for (i_t v : branched_zero_) {
        orbit_has_b0_[orb_.find_orbit(v)] = 1;
      }
    } else if (!is_binary) {
      // Non-binary variable branching: no symmetry recomputation needed.
      // (Non-binary variables are excluded from orbital fixing.)
    }
    start_plunge_ = false;

    for (i_t v : f0_) {
      orbit_has_f0_[orb_.find_orbit(v)] = 1;
    }

    for (i_t v : f1_) {
      orbit_has_f1_[orb_.find_orbit(v)] = 1;
    }

    fix_zero_.clear();
    fix_one_.clear();
    i_t num_conflicts          = 0;
    bool stabilizer_nontrivial = false;
    for (i_t j : symmetry->binary_variables) {
      i_t o = orb_.find_orbit(j);
      if (orb_.orbit_size(o) < 2) continue;

      if (orbit_has_b1_[o] == 1) {
        // The orbit contains variables in B1
        // So we can't fix any variables in this orbit
        continue;
      }

      // A non-B1 orbit of size >= 2 exists, so the stabilizer may produce fixings
      // in this node or a descendant.
      stabilizer_nontrivial = true;

      // Only fix free variables (not already fixed by branching or propagation)
      bool is_free =
        (marked_b0_[j] == 0 && marked_b1_[j] == 0 && marked_f0_[j] == 0 && marked_f1_[j] == 0);
      if (!is_free) continue;

      bool has_zero_source = (orbit_has_b0_[o] == 1 || orbit_has_f0_[o] == 1);
      bool has_one_source  = (orbit_has_f1_[o] == 1);

      // An orbit with both zero and one sources is conflicting.
      // Skip the orbit — no valid additional fixing can be derived from it.
      if (has_zero_source && has_one_source) {
        num_conflicts++;
        continue;
      }

      if (has_zero_source) {
        // The orbit of this variable contains variables in B0 or F0
        // So we can fix this variable to zero
        fix_zero_.push_back(j);
      }

      if (has_one_source) {
        // The orbit of this variable contains variables in F1
        // So we can fix this variable to one
        fix_one_.push_back(j);
      }
    }

    if (!stabilizer_nontrivial) { surviving_generators_.clear(); }

    // Restore the work arrays
    for (i_t v : f0_) {
      orbit_has_f0_[orb_.find_orbit(v)] = 0;
      marked_f0_[v]                     = 0;
    }
    for (i_t v : f1_) {
      orbit_has_f1_[orb_.find_orbit(v)] = 0;
      marked_f1_[v]                     = 0;
    }
    f0_.clear();
    f1_.clear();

    // Apply the fixings from non-conflicting orbits
    for (i_t v : fix_zero_) {
      problem.lower[v] = 0.0;
      problem.upper[v] = 0.0;
    }
    for (i_t v : fix_one_) {
      problem.lower[v] = 1.0;
      problem.upper[v] = 1.0;
    }

    // Accumulate this node's fixings and store in the node so that
    // children starting a new plunge can restore them.
    cumulative_fix_zero_.insert(cumulative_fix_zero_.end(), fix_zero_.begin(), fix_zero_.end());
    cumulative_fix_one_.insert(cumulative_fix_one_.end(), fix_one_.begin(), fix_one_.end());
    node_ptr->orbital_fix_zero = cumulative_fix_zero_;
    node_ptr->orbital_fix_one  = cumulative_fix_one_;

    return num_conflicts;
  }

 private:
  i_t num_original_vars_;
  i_t max_generators_;
  bool start_plunge_;
  orbits_t<i_t> orb_;

  std::vector<i_t> branched_zero_;
  std::vector<i_t> branched_one_;

  std::vector<i_t> surviving_generators_;

  std::vector<i_t> orbit_has_b1_;  // orbit_has_b1_[o] = 1 if orbit o contains variables in B1
  std::vector<i_t> orbit_has_b0_;  // orbit_has_b0_[o] = 1 if orbit o contains variables in B0
  std::vector<i_t> orbit_has_f0_;  // orbit_has_f0_[o] = 1 if orbit o contains variables in F0
  std::vector<i_t> orbit_has_f1_;  // orbit_has_f1_[o] = 1 if orbit o contains variables in F1
  std::vector<i_t> marked_b0_;     // marked_b0_[v] = 1 if variable v has been branched down
  std::vector<i_t> marked_b1_;     // marked_b1_[v] = 1 if variable v has been branched up
  std::vector<i_t> f0_;            // set of variables fixed to 0 by bound propagation
  std::vector<i_t> f1_;            // set of variables fixed to 1 by bound propagation
  std::vector<i_t> marked_f0_;     // marked_f0_[v] = 1 if variable v is in F0
  std::vector<i_t> marked_f1_;     // marked_f1_[v] = 1 if variable v is in F1
  std::vector<i_t> fix_zero_;      // set of variables fixed to 0 by orbital fixing
  std::vector<i_t> fix_one_;       // set of variables fixed to 1 by orbital fixing

  // Running accumulation of orbital fixings across the current plunge.
  // Stored into node_ptr after each successful orbital_fixing() call,
  // so children starting a new plunge can restore the parent's fixings.
  std::vector<i_t> cumulative_fix_zero_;
  std::vector<i_t> cumulative_fix_one_;
};

template <typename i_t, typename f_t>
class lexical_reduction_t {
 public:
  lexical_reduction_t(i_t num_original_vars)
  {
    reverse_branched_variables_.reserve(num_original_vars);
  }
  // Return -1 to prune the node, otherwise return the number of fixings applied.
  i_t lexical_reduce(mip_symmetry_t<i_t, f_t>* symmetry,
                     mip_node_t<i_t, f_t>* node_ptr,
                     simplex::lp_problem_t<i_t, f_t>& problem)
  {
    reverse_branched_variables_.clear();
    mip_node_t<i_t, f_t>* node = node_ptr;
    while (node != nullptr && node->branch_var >= 0) {
      i_t v = node->branch_var;
      if (symmetry->is_binary[v] == 1) { reverse_branched_variables_.push_back(v); }
      node = node->parent;
    }

    bool prune      = false;
    i_t num_fixings = 0;
    for (size_t k = 0; k < symmetry->inverse_generators.num_generators(); k++) {
      const permutation_t<i_t>& perm = symmetry->inverse_generators.get_generator(k);
      const std::vector<i_t>& p      = perm.dense_permutation();
      const size_t reverse_size      = reverse_branched_variables_.size();
      for (size_t h = reverse_size; h > 0; --h) {
        i_t j = reverse_branched_variables_[h - 1];  // This orders the variables from the root down
                                                     // to the current node
        const i_t p_j = p[j];
        if (p_j == j) continue;
        // clang-format off
        // Compare x[j] with x[p[j]
        // x[j] = 1, x[p[j]] = 1, continue to next variable
        // x[j] = 1, x[p[j]] = 0, strict greater. stop (continue to next generator), constraint is satisfied
        // x[j] = 1, x[p[j]] = free, stop (continue to next generator)
        // x[j] = 0, x[p[j]] = 0, continue to next variable
        // x[j] = 0, x[p[j]] = 1, violated. Prune the node
        // x[j] = 0, x[p[j]] = free, fix x[p[j]] to 0, continue to the next variable
        // x[j] = free, x[p[j]] = any, stop (continue to next generator)
        // clang-format on
        i_t val_j = -1;
        if (problem.lower[j] == problem.upper[j]) { val_j = static_cast<i_t>(problem.lower[j]); }
        i_t val_p_j = -1;
        if (problem.lower[p_j] == problem.upper[p_j]) {
          val_p_j = static_cast<i_t>(problem.lower[p_j]);
        }
        if (val_j == -1) {  // free. continue to next generator
          break;
        }
        if (val_j == 1 && val_p_j == 1) {
          continue;  // continue to next variable
        }
        if (val_j == 1 && val_p_j == 0) {
          break;  // stop. continue to next generator. Lex constraint is satisfied
        }
        if (val_j == 1 && val_p_j == -1) {
          break;  // stop. continue to the next generator.
        }
        if (val_j == 0 && val_p_j == 0) {
          continue;  // continue to next variable
        }
        if (val_j == 0 && val_p_j == 1) {
          prune = true;  // violated. Prune the node
          break;
        }
        if (val_j == 0 && val_p_j == -1) {
          problem.lower[p_j] = 0.0;
          problem.upper[p_j] = 0.0;
          num_fixings++;
          continue;  // continue to the next pair
        }
      }
      if (prune) break;
    }

    return prune ? -1 : num_fixings;
  }

 private:
  std::vector<i_t> reverse_branched_variables_;
};

template <typename i_t, typename f_t>
std::unique_ptr<mip_symmetry_t<i_t, f_t>> detect_symmetry(
  const simplex::user_problem_t<i_t, f_t>& user_problem,
  const simplex::simplex_solver_settings_t<i_t, f_t>& settings,
  bool& has_symmetry)
{
  settings.log.printf("\nRunning symmetry detection...\n");

  has_symmetry = false;

  f_t start_time = tic();
  simplex::lp_problem_t<i_t, f_t> problem(user_problem.handle_ptr, 1, 1, 1);
  std::vector<i_t> new_slacks;
  simplex::dualize_info_t<i_t, f_t> dualize_info;
  convert_user_problem(user_problem, settings, problem, new_slacks, dualize_info);
  std::vector<simplex::variable_type_t> var_types = user_problem.var_types;
  if (problem.num_cols > user_problem.num_cols) {
    var_types.resize(problem.num_cols);
    for (i_t k = user_problem.num_cols; k < problem.num_cols; k++) {
      var_types[k] = simplex::variable_type_t::CONTINUOUS;
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
  i_t var_color                      = 0;
  f_t last_obj                       = problem.objective[obj_perm[0]];
  f_t last_lower                     = problem.lower[obj_perm[0]];
  f_t last_upper                     = problem.upper[obj_perm[0]];
  simplex::variable_type_t last_type = var_types[obj_perm[0]];
  var_colors[obj_perm[0]]            = var_color;
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

  const i_t num_original_vars = user_problem.num_cols;

  auto result               = std::make_unique<mip_symmetry_t<i_t, f_t>>();
  result->num_original_vars = num_original_vars;
  result->is_binary.resize(num_original_vars, 0);
  for (i_t j = 0; j < num_original_vars; j++) {
    if (var_types[j] != simplex::variable_type_t::CONTINUOUS) {
      if (user_problem.lower[j] == 0.0 && user_problem.upper[j] == 1.0) {
        result->is_binary[j] = 1;
        result->binary_variables.push_back(j);
      } else {
        result->general_integer_variables.push_back(j);
      }
    }
  }

  // Project generators incrementally inside the dejavu callback.
  // This avoids storing full-graph dense vectors (size num_vertices per generator).
  const size_t max_generators =
    std::max(size_t{1}, static_cast<size_t>(64000000 / num_original_vars));
  orbits_t<i_t> orb(num_original_vars);
  int num_dejavu_generators = 0;
  int projected_count       = 0;
  int skipped_non_binary    = 0;
  std::vector<i_t> projected_p(num_original_vars);
  std::iota(projected_p.begin(), projected_p.end(), 0);
  std::vector<i_t> var_support;

  dejavu_hook generator_hook = [&num_original_vars,
                                &num_dejavu_generators,
                                &projected_p,
                                &var_support,
                                &orb,
                                &result,
                                &projected_count,
                                &skipped_non_binary,
                                &max_generators](int n, const int* p, int nsupp, const int* supp) {
    // Check if any support element is an original variable
    bool moves_variable = false;
    for (int s = 0; s < nsupp; s++) {
      if (supp[s] < num_original_vars) {
        moves_variable = true;
        break;
      }
    }
    if (!moves_variable) { return; }
    num_dejavu_generators++;

    // Project onto binary variables only.
    // Skip generators that move any non-binary variable (continuous or general integer).
    // Only touch support entries; reset them back to identity afterward.
    // dejavu guarantees supp is exactly the non-identity entries of p.
    bool moves_non_binary = false;
    var_support.clear();
    for (int s = 0; s < nsupp; s++) {
      const i_t j = supp[s];
      if (j >= num_original_vars) { continue; }
      if (result->is_binary[j] == 0) {
        moves_non_binary = true;
        break;
      }
      projected_p[j] = p[j];
      var_support.push_back(j);
    }

    if (!moves_non_binary && !var_support.empty()) {
      for (i_t j : var_support) {
        orb.add_mapping(j, projected_p[j]);
      }
      if (result->generators.num_generators() < max_generators) {
        result->generators.add_generator(projected_p, var_support);
      }
      projected_count++;
    } else if (moves_non_binary) {
      skipped_non_binary++;
    }

    // Reset modified entries back to identity
    for (i_t j : var_support) {
      projected_p[j] = j;
    }
  };

  dejavu::hooks::multi_hook combined;
  combined.add_hook(&generator_hook);

  dejavu::solver d;
  d.set_print(false);
  d.automorphisms(&g, combined);

  std::ostringstream grp_size_str;
  grp_size_str << d.get_automorphism_group_size();
  settings.log.printf("Automorphism group size %s, %d dejavu generators (%d move variables)\n",
                      grp_size_str.str().c_str(),
                      num_dejavu_generators,
                      projected_count);
  settings.log.printf("Dejavu time %f\n", toc(dejavu_start_time));

  result->num_generators = result->generators.num_generators();
  if (projected_count > static_cast<int>(result->num_generators)) {
    settings.log.printf("Generator limit: kept %d/%d projected generators (limit %d)\n",
                        result->num_generators,
                        projected_count,
                        (int)max_generators);
  }
  settings.log.printf(
    "Projected %d generators onto %d binary variables (%d skipped non-binary), "
    "%d stored\n",
    projected_count,
    (int)num_original_vars,
    skipped_non_binary,
    result->num_generators);

  // Compute orbit statistics from the incrementally built orbits.
  // All non-trivial orbits contain only binary variables (non-binary generators were excluded).
  has_symmetry              = false;
  i_t num_nontrivial_orbits = 0;
  i_t max_orbit_size        = 0;
  i_t total_vars_in_orbits  = 0;
  std::vector<i_t> orbit_histogram(num_original_vars + 1, 0);
  for (i_t j : result->binary_variables) {
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

  if (projected_count > 0) {
    settings.log.printf(
      "Binary orbits: %d non-trivial, max size %d, %d/%d (%.1f%%) binary variables in orbits\n",
      num_nontrivial_orbits,
      max_orbit_size,
      total_vars_in_orbits,
      num_original_vars,
      100.0 * total_vars_in_orbits / num_original_vars);
    settings.log.printf("Orbit histogram (size: count):");
    for (i_t sz = 2; sz <= max_orbit_size; sz++) {
      if (orbit_histogram[sz] > 0) { settings.log.printf(" %d:%d", sz, orbit_histogram[sz]); }
    }
    settings.log.printf("\n");

    has_symmetry = (max_orbit_size >= 4) || (num_nontrivial_orbits >= 3 && max_orbit_size >= 2) ||
                   (total_vars_in_orbits >= 10);
  }

  settings.log.printf("Total symmetry detection time %f\n", toc(start_time));

  if (!has_symmetry) {
    settings.log.printf(
      "No exploitable symmetry found (%d generators, %d non-trivial orbits, max orbit size %d)\n",
      projected_count,
      num_nontrivial_orbits,
      max_orbit_size);
    return nullptr;
  }

  // Precompute orbit representatives from the projected group's orbits.
  result->orbit_rep.resize(num_original_vars);
  for (i_t j = 0; j < num_original_vars; j++) {
    result->orbit_rep[j] = orb.find_orbit(j);
  }

  for (size_t i = 0; i < result->generators.num_generators(); i++) {
    result->inverse_generators.add_generator(result->generators.get_generator(i).inverse());
  }

  return result;
}

}  // namespace cuopt::mathematical_optimization::mip
