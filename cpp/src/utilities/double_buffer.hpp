/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <rmm/device_uvector.hpp>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t, typename T>
struct double_buffer_t {
  double_buffer_t(i_t size, const raft::handle_t* handle_ptr)
    : bufs{rmm::device_uvector<T>(size, handle_ptr->get_stream()),
           rmm::device_uvector<T>(size, handle_ptr->get_stream())}
  {
  }

  void resize(i_t size, const raft::handle_t* handle_ptr)
  {
    for (i_t i = 0; i < 2; i++) {
      bufs[i].resize(size, handle_ptr->get_stream());
    }
  }

  void flip() { flip_bit = 1 - flip_bit; }

  T* active() { return bufs[flip_bit].data(); }
  const T* active() const { return bufs[flip_bit].data(); }

  T* standby() { return bufs[1 - flip_bit].data(); }
  const T* stanby() const { return bufs[1 - flip_bit].data(); }

  size_t size() const { return bufs[0].size(); }

  i_t flip_bit{0};
  rmm::device_uvector<T> bufs[2];
};

}  // namespace cuopt::linear_programming::detail
