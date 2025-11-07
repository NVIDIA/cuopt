/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "cycle_finder_kernels.cuh"

#include <raft/core/nvtx.hpp>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <cub/cub.cuh>

#include <deque>
#include <unordered_set>

namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t, typename f_t, size_t max_routes>
bool ExactCycleFinder<i_t, f_t, max_routes>::call_init(graph_t<i_t, f_t>& graph)
{
  raft::common::nvtx::range fun_scope("call_init");
  auto n_vertices = graph.get_num_vertices();
  auto n_threads  = 128;
  auto n_blocks   = n_vertices;
  int level       = 0;
  size_t sh_size  = max_graph_nodes_per_row * (sizeof(double) + sizeof(i_t));
  bool is_set     = set_shmem_of_kernel(init_kernel<i_t, f_t, max_routes>, sh_size);
  if (!is_set) { return false; }

  init_kernel<i_t, f_t, max_routes><<<n_blocks, n_threads, sh_size, handle_ptr->get_stream()>>>(
    graph.view(), d_valid_paths.subspan(level));
  RAFT_CHECK_CUDA(handle_ptr->get_stream());
  // we have a safe-guard in the kernel for the global array stores
  // do the safe guard here for the occupied size
  clamp_occupied<max_routes><<<1, 1, 0, handle_ptr->get_stream()>>>(d_valid_paths.subspan(level));
  return true;
}

template <typename i_t, typename f_t, size_t max_routes>
void ExactCycleFinder<i_t, f_t, max_routes>::sort_cycle_costs_by_key(int n_items)
{
  raft::common::nvtx::range fun_scope("sort_cycle_costs_by_key");
  // sort cycle candidates by weight
  // resize only once if the problem size don't change
  if (sorted_key_indices.size() < (size_t)n_items) {
    sorted_key_indices.resize(n_items, handle_ptr->get_stream());
    copy_indices.resize(n_items, handle_ptr->get_stream());
  }
  if (copy_cost.size() < cycle_candidates.costs.size()) {
    copy_cost.resize(cycle_candidates.costs.size(), handle_ptr->get_stream());
  }
  thrust::sequence(
    handle_ptr->get_thrust_policy(), sorted_key_indices.begin(), sorted_key_indices.end());

  raft::copy(copy_indices.data(),
             sorted_key_indices.data(),
             sorted_key_indices.size(),
             handle_ptr->get_stream());
  raft::copy(copy_cost.data(),
             cycle_candidates.costs.data(),
             cycle_candidates.costs.size(),
             handle_ptr->get_stream());

  std::size_t temp_storage_bytes = 0;
  int begin_bit                  = 0;
  int end_bit                    = sizeof(double) * 8;
  cub::DeviceRadixSort::SortPairs(static_cast<void*>(nullptr),
                                  temp_storage_bytes,
                                  copy_cost.data(),
                                  cycle_candidates.costs.data(),
                                  copy_indices.data(),
                                  sorted_key_indices.data(),
                                  n_items,
                                  begin_bit,
                                  end_bit,
                                  handle_ptr->get_stream());

  // Allocate temporary storage
  if (d_cub_storage_bytes.size() < temp_storage_bytes) {
    d_cub_storage_bytes.resize(temp_storage_bytes, handle_ptr->get_stream());
  }
  // Run sorting operation
  cub::DeviceRadixSort::SortPairs(d_cub_storage_bytes.data(),
                                  temp_storage_bytes,
                                  copy_cost.data(),
                                  cycle_candidates.costs.data(),
                                  copy_indices.data(),
                                  sorted_key_indices.data(),
                                  n_items,
                                  begin_bit,
                                  end_bit,
                                  handle_ptr->get_stream());
}

template <typename i_t, typename f_t, size_t max_routes>
bool ExactCycleFinder<i_t, f_t, max_routes>::call_find(graph_t<i_t, f_t>& graph, i_t level)
{
  raft::common::nvtx::range fun_scope("call_find");
  auto n_threads = 128;
  auto n_blocks  = graph.get_num_vertices();
  size_t sh_size =
    2 * raft::WarpSize * sizeof(double) + (sizeof(double) + sizeof(i_t)) * max_graph_nodes_per_row;

  // cycle_candidates.reset(n_blocks, handle_ptr);
  bool last_level = level == (max_level - 1);
  if (last_level) {
    if (!set_shmem_of_kernel(find_kernel<i_t, f_t, max_routes, true>, sh_size)) { return false; }
    find_kernel<i_t, f_t, max_routes, true>
      <<<n_blocks, n_threads, sh_size, handle_ptr->get_stream()>>>(
        level,
        graph.view(),
        d_valid_paths.subspan(level - 1),
        d_valid_paths.subspan(level),
        cycle_candidates.level_view(level),
        depot_included);
  } else {
    if (!set_shmem_of_kernel(find_kernel<i_t, f_t, max_routes, false>, sh_size)) { return false; }
    find_kernel<i_t, f_t, max_routes, false>
      <<<n_blocks, n_threads, sh_size, handle_ptr->get_stream()>>>(
        level,
        graph.view(),
        d_valid_paths.subspan(level - 1),
        d_valid_paths.subspan(level),
        cycle_candidates.level_view(level),
        depot_included);
  }

  RAFT_CHECK_CUDA(handle_ptr->get_stream());
  return true;
}

template <typename map_key_t, typename value_t>
void detail::device_map_t<map_key_t, value_t>::clear(rmm::cuda_stream_view stream)
{
  auto max_vals  = max_level * max_available;
  auto n_threads = 256;
  auto n_blocks  = std::min((max_vals + n_threads - 1) / n_threads, max_blocks);
  clear_map<map_key_t, value_t><<<n_blocks, n_threads, 0, stream>>>(this->view());
  RAFT_CHECK_CUDA(stream);
}

template <size_t max_routes>
bool test_empty(typename detail::device_map_t<key_t<max_routes>, double>::view_t const map_view,
                rmm::cuda_stream_view stream)
{
  auto max_vals  = map_view.max_available;
  auto n_threads = 256;
  auto n_blocks  = (max_vals + n_threads - 1) / n_threads;
  test_empty<key_t<max_routes>, double><<<n_blocks, n_threads, 0, stream>>>(map_view);
  RAFT_CHECK_CUDA(stream);
  return true;
}

template <typename i_t, typename f_t, size_t max_routes>
bool ExactCycleFinder<i_t, f_t, max_routes>::find_cycle(graph_t<i_t, f_t>& graph)
{
  raft::common::nvtx::range fun_scope("find_cycle");
  d_valid_paths.clear(handle_ptr->get_stream());
  cuopt_assert(test_empty<max_routes>(d_valid_paths.subspan(0), handle_ptr->get_stream()), "");
  if (!call_init(graph)) { return false; }
  for (int i = 1; i < max_level; ++i) {
    cuopt_assert(test_empty<max_routes>(d_valid_paths.subspan(i), handle_ptr->get_stream()), "");
    int curr_level_occupied = d_valid_paths.get_size(i - 1, handle_ptr->get_stream());
    if (!curr_level_occupied) break;
    sort_occupied(i - 1, graph, curr_level_occupied);
    if (!call_find(graph, i)) { return false; }
  }
  return true;
}

template <typename i_t, typename f_t, size_t max_routes>
void ExactCycleFinder<i_t, f_t, max_routes>::get_cycle(graph_t<i_t, f_t>& graph,
                                                       ret_cycles_t<i_t, f_t>& d_ret)
{
  raft::common::nvtx::range fun_scope("get_cycle");
  auto n_threads = 256;
  auto n_blocks  = std::min((d_valid_paths.max_available + n_threads - 1) / n_threads, max_blocks);

  const i_t n_cycles = best_cycles.n_cycles.value(handle_ptr->get_stream());
  auto level_vec     = host_copy(best_cycles.level_ptr.data(), n_cycles, handle_ptr->get_stream());
  cuopt_func_call(d_ret.total_cycle_cost = 0.);
  for (i_t cycle_id = 0; cycle_id < n_cycles; ++cycle_id) {
    init_cycle<i_t, f_t, max_routes>
      <<<1, 1, 0, handle_ptr->get_stream()>>>(d_ret.view(), best_cycles.subspan(cycle_id));
    RAFT_CHECK_CUDA(handle_ptr->get_stream());
    i_t level = level_vec[cycle_id];

    for (int i = level; i > 0; --i) {
      extend_cycle<i_t, f_t, max_routes>
        <<<n_blocks, n_threads, 0, handle_ptr->get_stream()>>>(graph.view(),
                                                               d_valid_paths.subspan(i),
                                                               best_cycles.subspan(cycle_id),
                                                               d_ret.view(),
                                                               i,
                                                               (level + 1) - i);
      RAFT_CHECK_CUDA(handle_ptr->get_stream());
    }
    close_cycle<i_t, f_t, max_routes><<<1, 1, 0, handle_ptr->get_stream()>>>(
      d_ret.view(), best_cycles.subspan(cycle_id), level + 1);
    cuopt_func_call(d_ret.total_cycle_cost +=
                    best_cycles.cost_ptr.element(cycle_id, handle_ptr->get_stream()));
    RAFT_CHECK_CUDA(handle_ptr->get_stream());
  }
}

template <typename i_t, typename f_t, size_t max_routes>
bool ExactCycleFinder<i_t, f_t, max_routes>::check_cycle(graph_t<i_t, f_t>& graph,
                                                         ret_cycles_t<i_t, f_t>& ret)
{
  auto h_graph      = graph.to_host();
  auto h_cycles     = ret.to_host();
  bool cost_matches = true;
  std::unordered_set<i_t> changed_route_ids;
  for (i_t cycle = 0; cycle < h_cycles.n_cycles; ++cycle) {
    auto best_cost = best_cycles.cost_ptr.element(cycle, handle_ptr->get_stream());

    double cost = 0.;
    auto start  = h_cycles.offsets[cycle];
    auto end    = h_cycles.offsets[cycle + 1];
    std::deque tmp_cycle_path(h_cycles.paths.data() + start, h_cycles.paths.data() + end);
    tmp_cycle_path.push_front(tmp_cycle_path.back());

    for (size_t i = tmp_cycle_path.size() - 1; i > 0; --i) {
      auto node                      = tmp_cycle_path[i];
      [[maybe_unused]] bool is_cycle = false;
      auto row_size                  = h_graph.row_sizes[node];
      auto offset                    = node * max_graph_nodes_per_row;
      for (int col = offset; col < offset + row_size; ++col) {
        int dst       = h_graph.indices[col];
        double weight = h_graph.weights[col];
        if (dst == tmp_cycle_path[i - 1]) {
          cost += weight;
          is_cycle = true;
        }
      }
      cuopt_assert(is_cycle, "Is not a cycle");
    }
    if (std::abs(best_cost - cost) > EPSILON) {
      cost_matches = false;
      printf("best cost: %f cost: %f\n", best_cost, cost);
    }
  }
  return cost_matches;
}

template <typename i_t, typename f_t, size_t max_routes>
bool ExactCycleFinder<i_t, f_t, max_routes>::check_occupied_head(int level,
                                                                 graph_t<i_t, f_t>& graph)
{
  auto curr_map            = d_valid_paths.subspan(level);
  auto curr_level_occupied = d_valid_paths.get_size(level, handle_ptr->get_stream());
  auto prev_level_occupied =
    d_valid_paths.get_size(std::max(0, level - 1), handle_ptr->get_stream());

  auto host_occupied = host_copy(curr_map.occupied_indices, handle_ptr->get_stream());
  auto size_per_head = host_copy(curr_map.size_per_head, handle_ptr->get_stream());
  handle_ptr->sync_stream();
  cuopt_assert(host_occupied[curr_level_occupied - 1].x != -1, "last item is not valid");
  cuopt_assert(
    host_occupied[curr_level_occupied].x == -1 || curr_level_occupied >= curr_map.max_size,
    "the next item is valid index");
  for (size_t i = 0; i < curr_level_occupied; ++i) {
    size_per_head[host_occupied[i].y]--;
  }
  for (size_t i = 0; i < graph.get_num_vertices(); ++i) {
    cuopt_assert(size_per_head[i] == 0, "remaining_size is incorrect");
  }
  // check the head
  thrust::for_each(handle_ptr->get_thrust_policy(),
                   curr_map.occupied_indices.data(),
                   curr_map.occupied_indices.data() + curr_level_occupied,
                   [curr_map, curr_level_occupied, prev_level_occupied] __device__(int2 a) -> void {
                     cuopt_assert(a.y == curr_map.keys[a.x].head,
                                  "occupied indices and map heads don't match");
                   });
  return true;
}

// sort the occupied indices according to the heads, to group them together
template <typename i_t, typename f_t, size_t max_routes>
void ExactCycleFinder<i_t, f_t, max_routes>::sort_occupied(int level,
                                                           graph_t<i_t, f_t>& graph,
                                                           int curr_level_occupied)
{
  raft::common::nvtx::range fun_scope("sort_occupied");
  auto curr_map = d_valid_paths.subspan(level);
  cuopt_assert(check_occupied_head(level, graph), "");

  size_t temp_storage_bytes = 0;
  cub::DeviceMergeSort::SortKeys(
    static_cast<void*>(nullptr),
    temp_storage_bytes,
    curr_map.occupied_indices.data(),
    curr_level_occupied,
    [] __device__(int2 a, int2 b) -> bool { return a.y < b.y; },
    handle_ptr->get_stream());
  // Allocate temporary storage
  if (d_cub_storage_bytes.size() < temp_storage_bytes) {
    d_cub_storage_bytes.resize(temp_storage_bytes, handle_ptr->get_stream());
  }
  // Run sorting operation
  cub::DeviceMergeSort::SortKeys(
    d_cub_storage_bytes.data(),
    temp_storage_bytes,
    curr_map.occupied_indices.data(),
    curr_level_occupied,
    [] __device__(int2 a, int2 b) -> bool { return a.y < b.y; },
    handle_ptr->get_stream());

  // do an exclusive scan for the offsets of heads, this will be used in kernels
  temp_storage_bytes = 0;
  cub::DeviceScan::ExclusiveSum(static_cast<void*>(nullptr),
                                temp_storage_bytes,
                                curr_map.size_per_head.data(),
                                curr_map.size_per_head.data(),
                                graph.get_num_vertices() + 1,
                                handle_ptr->get_stream());
  // Allocate temporary storage
  if (d_cub_storage_bytes.size() < temp_storage_bytes) {
    d_cub_storage_bytes.resize(temp_storage_bytes, handle_ptr->get_stream());
  }
  // Run exclusive prefix sum
  cub::DeviceScan::ExclusiveSum(d_cub_storage_bytes.data(),
                                temp_storage_bytes,
                                curr_map.size_per_head.data(),
                                curr_map.size_per_head.data(),
                                graph.get_num_vertices() + 1,
                                handle_ptr->get_stream());
}

template <typename i_t, typename f_t, size_t max_routes>
void ExactCycleFinder<i_t, f_t, max_routes>::find_best_cycles(
  graph_t<i_t, f_t>& graph,
  ret_cycles_t<i_t, f_t>& ret,
  const solution_handle_t<i_t, f_t>* sol_handle)
{
  raft::common::nvtx::range fun_scope("find_best_cycles");
  handle_ptr = const_cast<solution_handle_t<i_t, f_t>*>(sol_handle);
  best_cycles.reset(handle_ptr->get_stream());
  cycle_candidates.reset(graph.get_num_vertices(), handle_ptr);
  if (!find_cycle(graph)) { return; }
  sort_cycle_costs_by_key(cycle_candidates.size * cycle_candidates.n_paths);
  // record best cycles
  record_best_cycles<i_t, f_t, max_routes>
    <<<1, 1, 0, handle_ptr->get_stream()>>>(cycle_candidates.size * cycle_candidates.n_paths,
                                            graph.view(),
                                            cycle_candidates.view(),
                                            best_cycles.view(),
                                            sorted_key_indices.data());
  get_cycle(graph, ret);
  cuopt_assert(check_cycle(graph, ret), "Recomputed cost mismatch");
}

template void ExactCycleFinder<int, float, 128>::find_best_cycles(
  graph_t<int, float>&, ret_cycles_t<int, float>&, solution_handle_t<int, float> const* sol_handle);
template void ExactCycleFinder<int, float, 1024>::find_best_cycles(
  graph_t<int, float>&, ret_cycles_t<int, float>&, solution_handle_t<int, float> const* sol_handle);

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
