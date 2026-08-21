/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/error.hpp>
#include <cuopt/mathematical_optimization/utilities/barrier_cache.hpp>

#include <barrier/barrier_symbolic_cache.hpp>
#include <pdlp/utilities/barrier_front_end_cache.hpp>

#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cuopt::cython {

using barrier_iteration_data_t =
  mathematical_optimization::barrier::iteration_data_t<int, double>;
using barrier_iteration_data_ptr =
  std::unique_ptr<barrier_iteration_data_t, void (*)(barrier_iteration_data_t*)>;

struct barrier_cache_t::impl {
  impl(std::unique_ptr<rmm::cuda_stream> stream_in, std::unique_ptr<raft::handle_t> handle_in)
    : stream(std::move(stream_in)),
      handle(std::move(handle_in)),
      iteration_data(nullptr, &mathematical_optimization::barrier::destroy_iteration_data)
  {
  }

  std::unique_ptr<rmm::cuda_stream> stream;
  std::unique_ptr<raft::handle_t> handle;
  std::optional<mathematical_optimization::barrier::barrier_symbolic_cache_t<int, double>>
    symbolic_cache;
  barrier_iteration_data_ptr iteration_data;
  std::unique_ptr<barrier_front_end_cache_t> front_end;
};

barrier_cache_t::barrier_cache_t(std::unique_ptr<rmm::cuda_stream> stream,
                                       std::unique_ptr<raft::handle_t> handle)
  : impl_(std::make_unique<impl>(std::move(stream), std::move(handle)))
{
}

barrier_cache_t::~barrier_cache_t() = default;

barrier_cache_t::barrier_cache_t(barrier_cache_t&&) noexcept            = default;
barrier_cache_t& barrier_cache_t::operator=(barrier_cache_t&&) noexcept = default;

std::unique_ptr<barrier_cache_t> barrier_cache_t::create(unsigned stream_flags)
{
  auto stream = std::make_unique<rmm::cuda_stream>(static_cast<rmm::cuda_stream::flags>(stream_flags));
  auto handle = std::make_unique<raft::handle_t>(*stream);
  return std::unique_ptr<barrier_cache_t>(
    new barrier_cache_t(std::move(stream), std::move(handle)));
}

raft::handle_t* barrier_cache_t::handle_ptr()
{
  return impl_->handle.get();
}

raft::handle_t const* barrier_cache_t::handle_ptr() const
{
  return impl_->handle.get();
}

rmm::cuda_stream_view barrier_cache_t::stream_view() const
{
  return impl_->stream->view();
}

mathematical_optimization::barrier::barrier_symbolic_cache_t<int, double>*
barrier_cache_t::symbolic_cache_for_reuse(raft::handle_t const* handle)
{
  if (handle == nullptr || !impl_->symbolic_cache.has_value() || !impl_->symbolic_cache->valid ||
      impl_->symbolic_cache->handle_ptr != handle) {
    return nullptr;
  }
  return &(*impl_->symbolic_cache);
}

void barrier_cache_t::clear_symbolic_cache()
{
  impl_->symbolic_cache.reset();
  clear_iteration_data();
  clear_front_end_cache();
}

void barrier_cache_t::store_symbolic_cache(
  mathematical_optimization::barrier::iteration_data_t<int, double>& data)
{
  if (!impl_->symbolic_cache.has_value()) {
    impl_->symbolic_cache.emplace(impl_->handle->get_stream());
  }
  mathematical_optimization::barrier::barrier_store_symbolic_cache_from_iteration_data(
    data, *impl_->symbolic_cache);
}

void barrier_cache_t::store_iteration_data(barrier_iteration_data_t* data)
{
  impl_->iteration_data.reset(data);
}

barrier_iteration_data_t* barrier_cache_t::release_iteration_data()
{
  return impl_->iteration_data.release();
}

barrier_iteration_data_t* barrier_cache_t::iteration_data()
{
  return impl_->iteration_data.get();
}

void barrier_cache_t::clear_iteration_data() { impl_->iteration_data.reset(); }

void barrier_cache_t::store_front_end_cache(std::unique_ptr<barrier_front_end_cache_t> cache)
{
  impl_->front_end = std::move(cache);
}

barrier_front_end_cache_t* barrier_cache_t::front_end_cache() { return impl_->front_end.get(); }

barrier_front_end_cache_t const* barrier_cache_t::front_end_cache() const
{
  return impl_->front_end.get();
}

void barrier_cache_t::clear_front_end_cache() { impl_->front_end.reset(); }

void barrier_cache_t::set_c_dirty(bool dirty)
{
  if (impl_->front_end) { impl_->front_end->c_dirty = dirty; }
}

bool barrier_cache_t::c_dirty() const
{
  return impl_->front_end != nullptr && impl_->front_end->c_dirty;
}

bool barrier_cache_t::has_front_end_cache() const { return impl_->front_end != nullptr; }

void barrier_cache_t::update_linear_objective(double const* c, int n)
{
  cuopt_expects(impl_->front_end != nullptr,
                error_type_t::ValidationError,
                "update_q: no front-end cache; Solve with sequence_solve first.");
  cuopt_expects(impl_->iteration_data.get() != nullptr,
                error_type_t::ValidationError,
                "update_q: no cached iteration_data; Solve a QP to Optimal first.");
  std::vector<double> crushed;
  try {
    crushed = crush_user_linear_objective(*impl_->front_end, c, n);
  } catch (std::invalid_argument const& e) {
    cuopt_expects(false, error_type_t::ValidationError, "%s", e.what());
  }
  if (impl_->front_end->linear_obj_shift.size() == crushed.size()) {
    for (std::size_t j = 0; j < crushed.size(); ++j) {
      crushed[j] += impl_->front_end->linear_obj_shift[j];
    }
  }
  try {
    mathematical_optimization::barrier::apply_barrier_linear_objective(
      *impl_->iteration_data, crushed.data(), static_cast<int>(crushed.size()));
  } catch (std::invalid_argument const& e) {
    cuopt_expects(false, error_type_t::ValidationError, "%s", e.what());
  }
  impl_->front_end->c_dirty = true;
}

}  // namespace cuopt::cython
