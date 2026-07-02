/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/linear_programming/utilities/lp_solve_session.hpp>

#include <barrier/barrier_symbolic_cache.hpp>

#include <optional>
#include <utility>

namespace cuopt::cython {

struct lp_solve_session_t::impl {
  std::unique_ptr<rmm::cuda_stream> stream;
  std::unique_ptr<raft::handle_t> handle;
  std::optional<linear_programming::dual_simplex::barrier_symbolic_cache_t<int, double>>
    symbolic_cache;
};

lp_solve_session_t::lp_solve_session_t(std::unique_ptr<rmm::cuda_stream> stream,
                                       std::unique_ptr<raft::handle_t> handle)
  : impl_(std::make_unique<impl>(impl{std::move(stream), std::move(handle), std::nullopt}))
{
}

lp_solve_session_t::~lp_solve_session_t() = default;

lp_solve_session_t::lp_solve_session_t(lp_solve_session_t&&) noexcept            = default;
lp_solve_session_t& lp_solve_session_t::operator=(lp_solve_session_t&&) noexcept = default;

std::unique_ptr<lp_solve_session_t> lp_solve_session_t::create(unsigned stream_flags)
{
  auto stream = std::make_unique<rmm::cuda_stream>(static_cast<rmm::cuda_stream::flags>(stream_flags));
  auto handle = std::make_unique<raft::handle_t>(*stream);
  return std::unique_ptr<lp_solve_session_t>(
    new lp_solve_session_t(std::move(stream), std::move(handle)));
}

raft::handle_t* lp_solve_session_t::handle_ptr()
{
  return impl_->handle.get();
}

raft::handle_t const* lp_solve_session_t::handle_ptr() const
{
  return impl_->handle.get();
}

rmm::cuda_stream_view lp_solve_session_t::stream_view() const
{
  return impl_->stream->view();
}

linear_programming::dual_simplex::barrier_symbolic_cache_t<int, double>*
lp_solve_session_t::symbolic_cache_for_reuse(raft::handle_t const* handle)
{
  if (handle == nullptr || !impl_->symbolic_cache.has_value() || !impl_->symbolic_cache->valid ||
      impl_->symbolic_cache->handle_ptr != handle) {
    return nullptr;
  }
  return &(*impl_->symbolic_cache);
}

void lp_solve_session_t::clear_symbolic_cache()
{
  impl_->symbolic_cache.reset();
}

void lp_solve_session_t::store_symbolic_cache(
  linear_programming::dual_simplex::iteration_data_t<int, double>& data)
{
  if (!impl_->symbolic_cache.has_value()) {
    impl_->symbolic_cache.emplace(impl_->handle->get_stream());
  }
  linear_programming::dual_simplex::barrier_store_symbolic_cache_from_iteration_data(
    data, *impl_->symbolic_cache);
}

}  // namespace cuopt::cython
