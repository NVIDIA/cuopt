/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <pdlp/cusparse_view.hpp>
#include <utilities/macros.cuh>

#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/core/cusparse_macros.hpp>
#include <raft/core/handle.hpp>

#include <rmm/device_scalar.hpp>

#include <cusparse_v2.h>

#include <memory>
#include <type_traits>

namespace cuopt::mathematical_optimization::barrier {

struct cusparse_spgemm_deleter_t {
  void operator()(cusparseSpGEMMDescr_t descr) const noexcept
  {
    if (descr) { CUOPT_CUSPARSE_TRY_NO_THROW(cusparseSpGEMM_destroyDescr(descr)); }
  }
};

using cusparse_spgemm_uptr =
  std::unique_ptr<std::remove_pointer_t<cusparseSpGEMMDescr_t>, cusparse_spgemm_deleter_t>;

template <typename i_t, typename f_t>
struct cusparse_info_t {
  cusparse_info_t(raft::handle_t const* handle)
    : alpha(handle->get_stream()),
      beta(handle->get_stream()),
      buffer_size(0, handle->get_stream()),
      buffer_size_2(0, handle->get_stream()),
      buffer_size_3(0, handle->get_stream()),
      buffer_size_4(0, handle->get_stream()),
      buffer_size_5(0, handle->get_stream())
  {
    f_t v{1};
    alpha.set_value_async(v, handle->get_stream());
    beta.set_value_async(v, handle->get_stream());
  }

  pdlp::cusparse_sp_mat_uptr matA_descr;
  pdlp::cusparse_sp_mat_uptr matDAT_descr;
  pdlp::cusparse_sp_mat_uptr matADAT_descr;
  cusparse_spgemm_uptr spgemm_descr;
  rmm::device_scalar<f_t> alpha;
  rmm::device_scalar<f_t> beta;
  rmm::device_uvector<uint8_t> buffer_size;
  rmm::device_uvector<uint8_t> buffer_size_2;
  rmm::device_uvector<uint8_t> buffer_size_3;
  rmm::device_uvector<uint8_t> buffer_size_4;
  rmm::device_uvector<uint8_t> buffer_size_5;
  size_t buffer_size_size;
  size_t buffer_size_2_size;
  size_t buffer_size_3_size;
  size_t buffer_size_4_size;
  size_t buffer_size_5_size;
};

}  // namespace cuopt::mathematical_optimization::barrier
