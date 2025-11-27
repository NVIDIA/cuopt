/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <raft/util/cudart_utils.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <mutex>

namespace cuopt {
namespace routing {
namespace detail {

/**
 * @brief A resource handle class that is inspired by raft/handle
 * @note In the future if the synchronization becomes an issue, maintain a per solution async memory
 * pool
 */
template <typename i_t, typename f_t>
class solution_handle_t {
 public:
  solution_handle_t(const solution_handle_t&)            = delete;
  solution_handle_t& operator=(const solution_handle_t&) = delete;
  solution_handle_t(solution_handle_t&&)                 = delete;
  solution_handle_t& operator=(solution_handle_t&&)      = delete;

  solution_handle_t(rmm::cuda_stream_view stream)
    : dev_id_([]() -> i_t {
        i_t cur_dev = -1;
        RAFT_CUDA_TRY(cudaGetDevice(&cur_dev));
        return cur_dev;
      }()),
      stream_view_(stream)
  {
    create_resources();
  }

  rmm::exec_policy& get_thrust_policy() const noexcept { return *thrust_policy_; }
  rmm::cuda_stream_view get_stream() const noexcept { return stream_view_; }
  i_t get_device() const { return dev_id_; }
  void sync_stream() const { stream_view_.synchronize(); };

  const cudaDeviceProp& get_device_properties() const
  {
    std::lock_guard<std::mutex> _(mutex_);
    if (!device_prop_initialized_) {
      RAFT_CUDA_TRY_NO_THROW(cudaGetDeviceProperties(&prop_, dev_id_));
      device_prop_initialized_ = true;
    }
    return prop_;
  }

  int get_num_sms() const
  {
    int sm_count;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev_id_);
    return sm_count;
  }

 private:
  void create_resources() { thrust_policy_ = std::make_shared<rmm::exec_policy>(stream_view_); }

  const i_t dev_id_{0};
  mutable cudaDeviceProp prop_;
  mutable std::mutex mutex_;
  mutable bool device_prop_initialized_{false};

  mutable bool shared_attr_initialized_{false};
  rmm::cuda_stream_view stream_view_{};
  // this is a shared pointer to be able to copy construct and keep a copy of a solution
  std::shared_ptr<rmm::exec_policy> thrust_policy_{nullptr};
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
