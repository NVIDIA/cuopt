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

#include <cmath>
#include <list>
#include <memory>
#include <vector>

namespace cuopt::linear_programming::dual_simplex {

enum class node_status_t : int {
  ACTIVE           = 0,  // Node still in the tree
  IN_PROGRESS      = 1,  // Node is currently being solved
  INTEGER_FEASIBLE = 2,  // Node has an integer feasible solution
  INFEASIBLE       = 3,  // Node is infeasible
  FATHOMED         = 4,  // Node objective is greater than the upper bound
};

bool inactive_status(node_status_t status);

template <typename i_t, typename f_t>
class mip_node_t {
 public:
  mip_node_t(f_t root_lower_bound, std::vector<variable_status_t>& basis)
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
             std::vector<variable_status_t>& basis)
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
  bool operator()(mip_node_t<i_t, f_t>& a, mip_node_t<i_t, f_t>& b)
  {
    return a.lower_bound >
           b.lower_bound;  // True if a comes before b, elements that come before are output last
  }
};

}  // namespace cuopt::linear_programming::dual_simplex
