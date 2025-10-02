/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
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

#include <raft/sparse/detail/cusparse_macros.h>
#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/core/handle.hpp>

#include <rmm/device_scalar.hpp>

#include <cusparse_v2.h>

namespace cuopt::linear_programming::dual_simplex {

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

  ~cusparse_info_t()
  {
    //    RAFT_CUSPARSE_TRY(cusparseSpGEMM_destroyDescr(spgemm_descr));
    //    RAFT_CUSPARSE_TRY(cusparseDestroySpMat(matA_descr));
    //   RAFT_CUSPARSE_TRY(cusparseDestroySpMat(matDAT_descr));
    //   RAFT_CUSPARSE_TRY(cusparseDestroySpMat(matADAT_descr));
  }

  cusparseSpMatDescr_t matA_descr;
  cusparseSpMatDescr_t matDAT_descr;
  cusparseSpMatDescr_t matADAT_descr;
  cusparseSpGEMMDescr_t spgemm_descr;
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

}  // namespace cuopt::linear_programming::dual_simplex
