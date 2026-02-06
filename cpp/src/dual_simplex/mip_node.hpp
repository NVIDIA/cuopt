/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <dual_simplex/initial_basis.hpp>
#include <dual_simplex/types.hpp>
#include <utilities/omp_helpers.hpp>

#include <cmath>
#include <list>
#include <memory>
#include <vector>

namespace cuopt::linear_programming::dual_simplex {

enum class node_status_t : int {
  PENDING          = 0,  // Node is still in the tree, waiting to be solved
  INTEGER_FEASIBLE = 1,  // Node has an integer feasible solution
  INFEASIBLE       = 2,  // Node is infeasible
  FATHOMED         = 3,  // Node objective is greater than the upper bound
  HAS_CHILDREN     = 4,  // Node has children to explore
  NUMERICAL        = 5   // Encountered numerical issue when solving the LP relaxation
};

enum class rounding_direction_t : int8_t { NONE = -1, DOWN = 0, UP = 1 };

bool inactive_status(node_status_t status);

template <typename i_t, typename f_t>
class mip_node_t {
 public:
  mip_node_t()
    : status(node_status_t::PENDING),
      lower_bound(-std::numeric_limits<f_t>::infinity()),
      depth(0),
      parent(nullptr),
      node_id(0),
      branch_var(-1),
      branch_dir(rounding_direction_t::NONE),
      branch_var_lower(-std::numeric_limits<f_t>::infinity()),
      branch_var_upper(std::numeric_limits<f_t>::infinity()),
      fractional_val(std::numeric_limits<f_t>::infinity()),
      objective_estimate(std::numeric_limits<f_t>::infinity()),
      vstatus(0)
  {
    children[0] = nullptr;
    children[1] = nullptr;
  }

  mip_node_t(f_t root_lower_bound, const std::vector<variable_status_t>& basis)
    : status(node_status_t::PENDING),
      lower_bound(root_lower_bound),
      depth(0),
      parent(nullptr),
      node_id(0),
      branch_var(-1),
      branch_dir(rounding_direction_t::NONE),
      integer_infeasible(-1),
      objective_estimate(std::numeric_limits<f_t>::infinity()),
      vstatus(basis)
  {
    children[0] = nullptr;
    children[1] = nullptr;
  }

  mip_node_t(const lp_problem_t<i_t, f_t>& problem,
             mip_node_t* parent_node,
             i_t node_num,
             i_t branch_variable,
             rounding_direction_t branch_direction,
             f_t branch_var_value,
             i_t integer_inf,
             const std::vector<variable_status_t>& basis)
    : status(node_status_t::PENDING),
      lower_bound(parent_node->lower_bound),
      depth(parent_node->depth + 1),
      parent(parent_node),
      node_id(node_num),
      branch_var(branch_variable),
      branch_dir(branch_direction),
      fractional_val(branch_var_value),
      integer_infeasible(integer_inf),
      objective_estimate(parent_node->objective_estimate),
      vstatus(basis)
  {
    branch_var_lower = branch_direction == rounding_direction_t::DOWN ? problem.lower[branch_var]
                                                                      : std::ceil(branch_var_value);
    branch_var_upper = branch_direction == rounding_direction_t::DOWN ? std::floor(branch_var_value)
                                                                      : problem.upper[branch_var];
    children[0]      = nullptr;
    children[1]      = nullptr;
  }

  void get_variable_bounds(std::vector<f_t>& lower,
                           std::vector<f_t>& upper,
                           std::vector<bool>& bounds_changed) const
  {
    update_branched_variable_bounds(lower, upper, bounds_changed);

    mip_node_t<i_t, f_t>* parent_ptr = parent;
    while (parent_ptr != nullptr && parent_ptr->node_id != 0) {
      parent_ptr->update_branched_variable_bounds(lower, upper, bounds_changed);
      parent_ptr = parent_ptr->parent;
    }
  }

  // Here we assume that we are traversing from the deepest node to the
  // root of the tree
  void update_branched_variable_bounds(std::vector<f_t>& lower,
                                       std::vector<f_t>& upper,
                                       std::vector<bool>& bounds_changed) const
  {
    assert(branch_var >= 0);
    assert(lower.size() > branch_var);
    assert(upper.size() > branch_var);
    assert(bounds_changed.size() > branch_var);

    // If the bounds have already been updated on another node,
    // skip this node as it contains looser bounds, since we
    // are traversing up the tree toward the root
    if (bounds_changed[branch_var]) { return; }

    // Apply the bounds at the current node
    lower[branch_var]          = branch_var_lower;
    upper[branch_var]          = branch_var_upper;
    bounds_changed[branch_var] = true;
  }

  mip_node_t* get_down_child() const { return children[0].get(); }

  mip_node_t* get_up_child() const { return children[1].get(); }

  void add_children(std::unique_ptr<mip_node_t>&& down_child,
                    std::unique_ptr<mip_node_t>&& up_child)
  {
    children[0] = std::move(down_child);
    children[1] = std::move(up_child);
    // When we add children we no longer need to store our basis
    vstatus.clear();
  }

  bool is_inactive() const
  {
    if (inactive_status(status)) { return true; }
    if ((children[0] != nullptr && inactive_status(children[0]->status)) &&
        (children[1] != nullptr && inactive_status(children[1]->status))) {
      return true;
    }
    if (children[0] == nullptr && inactive_status(children[1]->status)) { return true; }
    if (children[1] == nullptr && inactive_status(children[0]->status)) { return true; }
    return false;
  }

  void update_bound()
  {
    if (children[0] != nullptr && children[1] != nullptr) {
      if (inactive_status(children[0]->status) && inactive_status(children[1]->status)) {
        lower_bound = std::min(children[0]->lower_bound, children[1]->lower_bound);
      }
    }
    if (children[0] != nullptr && children[1] == nullptr) {
      if (inactive_status(children[0]->status)) { lower_bound = children[0]->lower_bound; }
    }
    if (children[1] != nullptr && children[0] == nullptr) {
      if (inactive_status(children[1]->status)) { lower_bound = children[1]->lower_bound; }
    }
  }

  // outputs a stack containing inactive nodes in the tree that can be freed
  void set_status(node_status_t node_status, std::vector<mip_node_t*>& stack)
  {
    status = node_status;
    if (inactive_status(status)) {
      update_bound();
      stack.push_back(this);
      // Propagate to parent
      mip_node_t* parent_ptr = parent;
      while (parent_ptr != nullptr) {
        if (parent_ptr->is_inactive()) {
          parent_ptr->status = node_status_t::FATHOMED;
          parent_ptr->update_bound();
          stack.push_back(parent_ptr);
        } else {
          break;
        }
        parent_ptr = parent_ptr->parent;
      }
    }
  }

  // Only used for debugging
  void traverse_children()
  {
    std::list<mip_node_t<i_t, f_t>*> to_visit;
    to_visit.push_back(this);
    while (to_visit.size() > 0) {
      mip_node_t<i_t, f_t>* current_node = to_visit.front();
      to_visit.pop_front();
      if (current_node->children[0] != nullptr) {
        to_visit.push_front(current_node->children[0].get());
      }
      if (current_node->children[1] != nullptr) {
        to_visit.push_front(current_node->children[1].get());
      }

      if (current_node->children[0] == nullptr && current_node->children[1] == nullptr &&
          current_node->depth < 10) {
        printf("Node %d with no children at depth %d lower bound %e. status %d\n",
               current_node->node_id,
               current_node->depth,
               current_node->lower_bound,
               current_node->status);
        if (current_node->parent != nullptr) {
          printf("Parent status %d. Sibiling status %d\n",
                 current_node->parent->status,
                 current_node->parent->children[0].get() != this
                   ? current_node->parent->children[0]->status
                   : current_node->parent->children[1]->status);
        }
      }
    }
  }

  // This method creates a copy of the current node
  // with its parent set to `nullptr` and `depth = 0`.
  // This detaches the node from the tree.
  mip_node_t<i_t, f_t> detach_copy() const
  {
    mip_node_t<i_t, f_t> copy(lower_bound, vstatus);
    copy.branch_var         = branch_var;
    copy.branch_dir         = branch_dir;
    copy.branch_var_lower   = branch_var_lower;
    copy.branch_var_upper   = branch_var_upper;
    copy.fractional_val     = fractional_val;
    copy.objective_estimate = objective_estimate;
    copy.node_id            = node_id;
    return copy;
  }

  node_status_t status;
  f_t lower_bound;
  f_t objective_estimate;
  i_t depth;
  i_t node_id;
  i_t branch_var;
  rounding_direction_t branch_dir;
  f_t branch_var_lower;
  f_t branch_var_upper;
  f_t fractional_val;
  i_t integer_infeasible;

  mip_node_t<i_t, f_t>* parent;
  std::unique_ptr<mip_node_t> children[2];

  std::vector<variable_status_t> vstatus;
};

template <typename i_t, typename f_t>
void remove_fathomed_nodes(std::vector<mip_node_t<i_t, f_t>*>& stack)
{
  for (int i = 0; i < stack.size(); ++i) {
    for (int child = 0; child < 2; ++child) {
      if (stack[i]->children[child] != nullptr) { stack[i]->children[child].reset(); }
    }
  }
}

template <typename i_t, typename f_t>
class search_tree_t {
 public:
  search_tree_t() : num_nodes(0) {}

  search_tree_t(mip_node_t<i_t, f_t>&& node) : root(std::move(node)), num_nodes(0) {}

  void update(mip_node_t<i_t, f_t>* node_ptr, node_status_t status)
  {
    std::lock_guard<omp_mutex_t> lock(mutex);
    std::vector<mip_node_t<i_t, f_t>*> stack;
    node_ptr->set_status(status, stack);
    remove_fathomed_nodes(stack);
  }

  void branch(mip_node_t<i_t, f_t>* parent_node,
              const i_t branch_var,
              const f_t fractional_val,
              const i_t integer_infeasible,
              const std::vector<variable_status_t>& parent_vstatus,
              const lp_problem_t<i_t, f_t>& original_lp,
              logger_t& log)
  {
    i_t id = num_nodes.fetch_add(2);

    auto down_child = std::make_unique<mip_node_t<i_t, f_t>>(original_lp,
                                                             parent_node,
                                                             ++id,
                                                             branch_var,
                                                             rounding_direction_t::DOWN,
                                                             fractional_val,
                                                             integer_infeasible,
                                                             parent_vstatus);
    graphviz_edge(log,
                  parent_node,
                  down_child.get(),
                  branch_var,
                  rounding_direction_t::DOWN,
                  std::floor(fractional_val));

    auto up_child = std::make_unique<mip_node_t<i_t, f_t>>(original_lp,
                                                           parent_node,
                                                           ++id,
                                                           branch_var,
                                                           rounding_direction_t::UP,
                                                           fractional_val,
                                                           integer_infeasible,
                                                           parent_vstatus);

    graphviz_edge(log,
                  parent_node,
                  up_child.get(),
                  branch_var,
                  rounding_direction_t::UP,
                  std::ceil(fractional_val));

    assert(parent_vstatus.size() == original_lp.num_cols);
    parent_node->add_children(std::move(down_child),
                              std::move(up_child));  // child pointers moved into the tree
  }

  void graphviz_node(logger_t& log,
                     const mip_node_t<i_t, f_t>* node_ptr,
                     const std::string label,
                     const f_t val)
  {
    if (write_graphviz) {
      log.printf("Node%d [label=\"%s %.16e\"]\n", node_ptr->node_id, label.c_str(), val);
    }
  }

  void graphviz_edge(logger_t& log,
                     const mip_node_t<i_t, f_t>* origin_ptr,
                     const mip_node_t<i_t, f_t>* dest_ptr,
                     const i_t branch_var,
                     rounding_direction_t branch_dir,
                     const f_t bound)
  {
    if (write_graphviz) {
      log.printf("Node%d -> Node%d [label=\"x%d %s %e\"]\n",
                 origin_ptr->node_id,
                 dest_ptr->node_id,
                 branch_var,
                 branch_dir == rounding_direction_t::DOWN ? "<=" : ">=",
                 bound);
    }
  }

  mip_node_t<i_t, f_t> root;
  omp_mutex_t mutex;
  omp_atomic_t<i_t> num_nodes;

  static constexpr bool write_graphviz = false;
};

}  // namespace cuopt::linear_programming::dual_simplex
