/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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

#include <cuopt/error.hpp>
#include <utilities/macros.cuh>

#include <rmm/cuda_stream_view.hpp>

#pragma once

namespace cuopt {
namespace routing {
namespace detail {

// This is not a thread-safe class, be careful on multi-threading
struct cuda_graph_t {
  void start_capture(rmm::cuda_stream_view stream)
  {
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    capture_started = true;
  }

  void end_capture(rmm::cuda_stream_view stream)
  {
    cuopt_assert(capture_started, "start_capture was not called before end_capture!");
    cuopt_expects(capture_started, error_type_t::RuntimeError, "A runtime error occurred!");
    cudaStreamEndCapture(stream, &graph);
    capture_started = false;
    if (graph_created) {
      // If the graph fails to update, errorNode will be set to the
      // node causing the failure and updateResult will be set to a
      // reason code.
      cudaGraphExecUpdate(instance, graph, &errorNode, &updateResult);
    }
    // Instantiate during the first iteration or whenever the update
    // fails for any reason
    if (!graph_created || updateResult != cudaGraphExecUpdateSuccess) {
      // If a previous update failed, destroy the cudaGraphExec_t
      // before re-instantiating it
      if (graph_created) { cudaGraphExecDestroy(instance); }
      // Instantiate graphExec from graph. The error node and
      // error message parameters are unused here.
      cudaGraphInstantiate(&instance, graph);
      graph_created = true;
    }
    cudaGraphDestroy(graph);
  }

  void launch_graph(rmm::cuda_stream_view stream) { cudaGraphLaunch(instance, stream); }

  bool graph_created   = false;
  bool capture_started = false;
  cudaGraph_t graph;
  cudaGraphExec_t instance;
  cudaGraphExecUpdateResult updateResult;
  cudaGraphNode_t errorNode;
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
