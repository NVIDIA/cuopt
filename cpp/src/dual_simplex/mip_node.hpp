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
  ACTIVE           = 0,  // Node still in the tree
  INTEGER_FEASIBLE = 1,  // Node has an integer feasible solution
  INFEASIBLE       = 2,  // Node is infeasible
  FATHOMED         = 3,  // Node objective is greater than the upper bound
  HAS_CHILDREN     = 4,  // Node has children to explore
  NUMERICAL        = 5,  // Encountered numerical issue when solving the LP relaxation
  TIME_LIMIT       = 6   // Time out during the LP relaxation
};

bool inactive_status(node_status_t status);

template <typename i_t, typename f_t>
class mip_node_t {
 public:
  mip_node_t(f_t root_lower_bound, const std::vector<variable_status_t>& basis)
    : status(node_status_t::ACTIVE),
      lower_bound(root_lower_bound),
      depth(0),
      parent(nullptr),
      node_id(0),
      branch_var(-1),
      branch_dir(-1),
      vstatus(basis)
  {
    children[0] = nullptr;
    children[1] = nullptr;
  }

  mip_node_t(const lp_problem_t<i_t, f_t>& problem,
             mip_node_t* parent_node,
             i_t node_num,
             i_t branch_variable,
             i_t branch_direction,
             f_t branch_var_value,
             const std::vector<variable_status_t>& basis)
    : status(node_status_t::ACTIVE),
      lower_bound(parent_node->lower_bound),
      depth(parent_node->depth + 1),
      parent(parent_node),
      node_id(node_num),
      branch_var(branch_variable),
      branch_dir(branch_direction),
      fractional_val(branch_var_value),
      vstatus(basis)

  {
    branch_var_lower =
      branch_direction == 0 ? problem.lower[branch_var] : std::ceil(branch_var_value);
    branch_var_upper =
      branch_direction == 0 ? std::floor(branch_var_value) : problem.upper[branch_var];
    children[0] = nullptr;
    children[1] = nullptr;
  }

  void get_variable_bounds(std::vector<f_t>& lower,
                           std::vector<f_t>& upper,
                           std::vector<bool>& bounds_changed) const
  {
    std::fill(bounds_changed.begin(), bounds_changed.end(), false);
    // Apply the bounds at the current node
    assert(lower.size() > branch_var);
    assert(upper.size() > branch_var);
    lower[branch_var]                = branch_var_lower;
    upper[branch_var]                = branch_var_upper;
    bounds_changed[branch_var]       = true;
    mip_node_t<i_t, f_t>* parent_ptr = parent;
    while (parent_ptr != nullptr) {
      if (parent_ptr->node_id == 0) { break; }
      assert(parent_ptr->branch_var >= 0);
      assert(lower.size() > parent_ptr->branch_var);
      assert(upper.size() > parent_ptr->branch_var);
      lower[parent_ptr->branch_var]          = parent_ptr->branch_var_lower;
      upper[parent_ptr->branch_var]          = parent_ptr->branch_var_upper;
      bounds_changed[parent_ptr->branch_var] = true;
      parent_ptr                             = parent_ptr->parent;
    }
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
  // with its parent set to `nullptr`, `node_id = 0`
  // and `depth = 0` such that it is the root
  // of a separated tree.
  mip_node_t<i_t, f_t> detach_copy() const
  {
    mip_node_t<i_t, f_t> copy(lower_bound, vstatus);
    copy.branch_var       = branch_var;
    copy.branch_dir       = branch_dir;
    copy.branch_var_lower = branch_var_lower;
    copy.branch_var_upper = branch_var_upper;
    copy.fractional_val   = fractional_val;
    copy.node_id          = node_id;
    return copy;
  }

  node_status_t status;
  f_t lower_bound;
  i_t depth;
  i_t node_id;
  i_t branch_var;
  i_t branch_dir;
  f_t branch_var_lower;
  f_t branch_var_upper;
  f_t fractional_val;

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
class node_compare_t {
 public:
  bool operator()(const mip_node_t<i_t, f_t>& a, const mip_node_t<i_t, f_t>& b) const
  {
    return a.lower_bound >
           b.lower_bound;  // True if a comes before b, elements that come before are output last
  }

  bool operator()(const mip_node_t<i_t, f_t>* a, const mip_node_t<i_t, f_t>* b) const
  {
    return a->lower_bound >
           b->lower_bound;  // True if a comes before b, elements that come before are output last
  }
};

template <typename i_t, typename f_t>
class search_tree_t {
 public:
  search_tree_t(f_t root_lower_bound, const std::vector<variable_status_t>& basis)
    : root(root_lower_bound, basis), num_nodes(0)
  {
  }

  search_tree_t(mip_node_t<i_t, f_t>&& node) : root(std::move(node)), num_nodes(0) {}

  void update_tree(mip_node_t<i_t, f_t>* node_ptr, node_status_t status)
  {
    mutex.lock();
    std::vector<mip_node_t<i_t, f_t>*> stack;
    node_ptr->set_status(status, stack);
    remove_fathomed_nodes(stack);
    mutex.unlock();
  }

  void branch(mip_node_t<i_t, f_t>* parent_node,
              const i_t branch_var,
              const f_t fractional_val,
              const std::vector<variable_status_t>& parent_vstatus,
              const lp_problem_t<i_t, f_t>& original_lp,
              logger_t& log)
  {
    i_t id = num_nodes.fetch_add(2);

    // down child
    auto down_child = std::make_unique<mip_node_t<i_t, f_t>>(
      original_lp, parent_node, ++id, branch_var, 0, fractional_val, parent_vstatus);

    graphviz_edge(log, parent_node, down_child.get(), branch_var, 0, std::floor(fractional_val));

    // up child
    auto up_child = std::make_unique<mip_node_t<i_t, f_t>>(
      original_lp, parent_node, ++id, branch_var, 1, fractional_val, parent_vstatus);

    graphviz_edge(log, parent_node, up_child.get(), branch_var, 1, std::ceil(fractional_val));

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
                     const i_t branch_dir,
                     const f_t bound)
  {
    if (write_graphviz) {
      log.printf("Node%d -> Node%d [label=\"x%d %s %e\"]\n",
                 origin_ptr->node_id,
                 dest_ptr->node_id,
                 branch_var,
                 branch_dir == 0 ? "<=" : ">=",
                 bound);
    }
  }

  mip_node_t<i_t, f_t> root;
  omp_mutex_t mutex;
  omp_atomic_t<i_t> num_nodes;

  static constexpr bool write_graphviz = false;
};

}  // namespace cuopt::linear_programming::dual_simplex
