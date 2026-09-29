/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/error.hpp>
#include <cuopt/mathematical_optimization/utilities/barrier_cache.hpp>

#include <barrier/barrier_transform.hpp>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cuopt::mathematical_optimization {

using barrier_iteration_data_t = barrier::iteration_data_t<int, double>;
using barrier_iteration_data_ptr =
  std::unique_ptr<barrier_iteration_data_t, void (*)(barrier_iteration_data_t*)>;

struct barrier_cache_t::impl {
  impl(std::unique_ptr<rmm::cuda_stream> stream_in, std::unique_ptr<raft::handle_t> handle_in)
    : stream(std::move(stream_in)),
      handle(std::move(handle_in)),
      iteration_data(nullptr, &barrier::destroy_iteration_data)
  {
  }

  std::unique_ptr<rmm::cuda_stream> stream;
  std::unique_ptr<raft::handle_t> handle;
  // Destroy iteration_data before transform: it may const-ref A/Q stored on the transform.
  std::unique_ptr<barrier_transform_t> transform;
  barrier_iteration_data_ptr iteration_data;
  bool c_dirty{false};
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
  auto stream =
    std::make_unique<rmm::cuda_stream>(static_cast<rmm::cuda_stream::flags>(stream_flags));
  auto handle = std::make_unique<raft::handle_t>(*stream);
  return std::unique_ptr<barrier_cache_t>(
    new barrier_cache_t(std::move(stream), std::move(handle)));
}

raft::handle_t* barrier_cache_t::handle_ptr() { return impl_->handle.get(); }

raft::handle_t const* barrier_cache_t::handle_ptr() const { return impl_->handle.get(); }

void barrier_cache_t::clear()
{
  impl_->iteration_data.reset();
  impl_->transform.reset();
  impl_->c_dirty = false;
}

void barrier_cache_t::store_iteration_data(barrier_iteration_data_t* data)
{
  impl_->iteration_data.reset(data);
}

barrier_iteration_data_t* barrier_cache_t::release_iteration_data()
{
  return impl_->iteration_data.release();
}

void barrier_cache_t::store_transform(std::unique_ptr<barrier_transform_t> transform)
{
  impl_->transform = std::move(transform);
}

barrier_transform_t* barrier_cache_t::transform() { return impl_->transform.get(); }

barrier_transform_t const* barrier_cache_t::transform() const { return impl_->transform.get(); }

void barrier_cache_t::set_c_dirty(bool dirty) { impl_->c_dirty = dirty; }

bool barrier_cache_t::c_dirty() const
{
  return impl_->c_dirty && impl_->transform != nullptr && impl_->iteration_data.get() != nullptr;
}

void barrier_cache_t::update_linear_objective(double const* c, int n)
{
  cuopt_expects(impl_->transform != nullptr,
                error_type_t::ValidationError,
                "update_linear_objective: no barrier transform; Solve with CUOPT_SEQUENCE_SOLVE "
                "enabled first.");
  cuopt_expects(impl_->iteration_data.get() != nullptr,
                error_type_t::ValidationError,
                "update_linear_objective: no cached iteration_data; Solve a QP to Optimal first.");
  // Cached Q and c are in minimization space.
  std::vector<double> user_objective;
  if (impl_->transform->maximize && c != nullptr && n > 0) {
    user_objective.assign(c, c + n);
    for (double& value : user_objective) {
      value = -value;
    }
    c = user_objective.data();
  }
  std::vector<double> crushed;
  try {
    crushed = crush_user_linear_objective(*impl_->transform, c, n);
  } catch (std::invalid_argument const& e) {
    cuopt_expects(false, error_type_t::ValidationError, "%s", e.what());
  }
  // x = x' + l folded c^T l into obj_constant. crushed is still crush(user c); previous
  // crushed c is barrier objective minus linear_obj_shift. column_scales undo crush.
  auto const& linear_obj_shift = impl_->transform->linear_obj_shift;
  auto const& column_scales    = impl_->transform->column_scales;
  auto const& translated_lower = impl_->transform->presolve_info.removed_lower_bounds;
  auto& barrier_lp             = *impl_->transform->barrier_lp;
  if (!translated_lower.empty() && linear_obj_shift.size() == crushed.size() &&
      column_scales.size() == crushed.size() && barrier_lp.objective.size() == crushed.size()) {
    double obj_constant_delta = 0.0;
    std::size_t const n_lower = std::min(translated_lower.size(), crushed.size());
    for (std::size_t j = 0; j < n_lower; ++j) {
      double const crushed_before = barrier_lp.objective[j] - linear_obj_shift[j];
      obj_constant_delta += (crushed[j] - crushed_before) * column_scales[j] * translated_lower[j];
    }
    barrier_lp.obj_constant += obj_constant_delta;
  }
  if (linear_obj_shift.size() == crushed.size()) {
    for (std::size_t j = 0; j < crushed.size(); ++j) {
      crushed[j] += linear_obj_shift[j];
    }
  }
  // The next solve builds its solver from barrier_lp, so keep its objective and the cached
  // iteration workspace on the same c.
  auto& barrier_objective = barrier_lp.objective;
  cuopt_expects(barrier_objective.size() == crushed.size(),
                error_type_t::ValidationError,
                "update_linear_objective: crushed objective size does not match the cached "
                "barrier LP.");
  barrier_objective = crushed;
  barrier::apply_barrier_linear_objective(
    *impl_->iteration_data, crushed.data(), static_cast<int>(crushed.size()));
  impl_->c_dirty = true;
}

}  // namespace cuopt::mathematical_optimization
