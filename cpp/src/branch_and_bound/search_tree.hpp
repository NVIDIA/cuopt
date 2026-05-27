/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <branch_and_bound/mip_node.hpp>

namespace cuopt::linear_programming::dual_simplex {

template <typename i_t, typename f_t>
class search_tree_t {
 public:
  search_tree_t() : num_nodes(0) {}

  search_tree_t(mip_node_t<i_t, f_t>&& node) : root(std::move(node)), num_nodes(0) {}

  ~search_tree_t() { clean(); }

  void update(mip_node_t<i_t, f_t>* node_ptr, node_status_t status)
  {
    std::lock_guard lock(mutex);

    --num_open_nodes;
    if (status == node_status_t::HAS_CHILDREN) {
      ++num_inner_nodes;
    } else {
      ++num_final_nodes;
      progress += std::pow(2, -node_ptr->depth);
    }

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
                                                             branch_direction_t::DOWN,
                                                             fractional_val,
                                                             integer_infeasible,
                                                             parent_vstatus);
    graphviz_edge(log,
                  parent_node,
                  down_child.get(),
                  branch_var,
                  branch_direction_t::DOWN,
                  std::floor(fractional_val));

    auto up_child = std::make_unique<mip_node_t<i_t, f_t>>(original_lp,
                                                           parent_node,
                                                           ++id,
                                                           branch_var,
                                                           branch_direction_t::UP,
                                                           fractional_val,
                                                           integer_infeasible,
                                                           parent_vstatus);

    graphviz_edge(log,
                  parent_node,
                  up_child.get(),
                  branch_var,
                  branch_direction_t::UP,
                  std::ceil(fractional_val));

    assert(parent_vstatus.size() == original_lp.num_cols);
    parent_node->add_children(std::move(down_child),
                              std::move(up_child));  // child pointers moved into the tree
    num_open_nodes += 2;
  }

  static void graphviz_node(logger_t& log,
                            const mip_node_t<i_t, f_t>* node_ptr,
                            const std::string label,
                            const f_t val)
  {
    if (write_graphviz) {
      log.printf("Node%d [label=\"%s %.16e\"]\n", node_ptr->node_id, label.c_str(), val);
    }
  }

  static void graphviz_edge(logger_t& log,
                            const mip_node_t<i_t, f_t>* origin_ptr,
                            const mip_node_t<i_t, f_t>* dest_ptr,
                            const i_t branch_var,
                            branch_direction_t branch_dir,
                            const f_t bound)
  {
    if (write_graphviz) {
      log.printf("Node%d -> Node%d [label=\"x%d %s %e\"]\n",
                 origin_ptr->node_id,
                 dest_ptr->node_id,
                 branch_var,
                 branch_dir == branch_direction_t::DOWN ? "<=" : ">=",
                 bound);
    }
  }

  // Clean the tree using a depth first scheme
  void clean()
  {
    std::vector<std::unique_ptr<mip_node_t<i_t, f_t>>> stack;

    if (root.children[0]) stack.push_back(std::move(root.children[0]));
    if (root.children[1]) stack.push_back(std::move(root.children[1]));

    while (!stack.empty()) {
      auto node = std::move(stack.back());
      stack.pop_back();
      if (node->children[0]) stack.push_back(std::move(node->children[0]));
      if (node->children[1]) stack.push_back(std::move(node->children[1]));
      // Implicitly call destructor for `node`
    }
  }

  mip_node_t<i_t, f_t> root;
  omp_mutex_t mutex;
  omp_atomic_t<uint64_t> num_nodes;

  // Number of nodes that still needs to be explored
  omp_atomic_t<uint64_t> num_open_nodes;

  // Number of integer feasible, infeasible or fathomed nodes
  omp_atomic_t<uint64_t> num_final_nodes;

  // Number of inner nodes
  omp_atomic_t<uint64_t> num_inner_nodes;

  // Track the solver progress based on how much the tree was explored
  omp_atomic_t<f_t> progress = 0.0;

  static constexpr bool write_graphviz = false;
};

}  // namespace cuopt::linear_programming::dual_simplex
