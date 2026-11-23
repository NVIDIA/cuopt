/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <branch_and_bound/mip_node.hpp>

#include <gtest/gtest.h>

#include <memory>
#include <utility>
#include <vector>

namespace cuopt::mathematical_optimization::test {
namespace {
using node_t = mip::mip_node_t<int, double>;

std::unique_ptr<node_t> make_deep_tree(int depth)
{
  auto root  = std::make_unique<node_t>();
  auto* node = root.get();
  for (int i = 0; i < depth; ++i) {
    const int side               = i % 2;
    node->children[side]         = std::make_unique<node_t>();
    node->children[side]->parent = node;
    node                         = node->children[side].get();
    node->packed_vstatus.resize(8, i % 256);
  }
  return root;
}
}  // namespace

TEST(MipNodeTest, DestroysDeepTreeWithoutRecursiveStackGrowth)
{
  auto root = make_deep_tree(100000);
  root.reset();
  EXPECT_EQ(root, nullptr);
}

TEST(MipNodeTest, DestroysMovedTreeWithoutFollowingStaleParents)
{
  auto original = make_deep_tree(100000);
  auto moved    = std::make_unique<node_t>(std::move(*original));
  original.reset();
  auto assigned = make_deep_tree(1000);
  *assigned     = std::move(*moved);
  moved.reset();
  assigned.reset();
  EXPECT_EQ(assigned, nullptr);
}

TEST(MipNodeTest, DestroyingSubtreePreservesItsParentAndSibling)
{
  node_t root;
  root.children[0]          = make_deep_tree(100000);
  root.children[1]          = make_deep_tree(1000);
  root.children[0]->parent  = &root;
  root.children[1]->parent  = &root;
  root.node_id              = 42;
  root.children[1]->node_id = 17;
  root.children[0].reset();
  EXPECT_EQ(root.node_id, 42);
  ASSERT_NE(root.children[1], nullptr);
  EXPECT_EQ(root.children[1]->node_id, 17);
  EXPECT_EQ(root.children[1]->parent, &root);
}

TEST(MipNodeTest, DestroysBalancedTreeWithBothChildren)
{
  auto root = std::make_unique<node_t>();
  std::vector<node_t*> nodes{root.get()};
  for (int i = 0; i < 10000; ++i) {
    auto* node = nodes[i];
    for (auto& child : node->children) {
      child         = std::make_unique<node_t>();
      child->parent = node;
      child->packed_vstatus.resize(8, 3);
      nodes.push_back(child.get());
    }
  }
  root.reset();
  EXPECT_EQ(root, nullptr);
}

}  // namespace cuopt::mathematical_optimization::test
