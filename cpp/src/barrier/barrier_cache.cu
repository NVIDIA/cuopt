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
#include <string>
#include <utility>
#include <vector>

namespace cuopt::mathematical_optimization {

template <typename i_t, typename f_t>
static void require_cache(barrier_transform_t<i_t, f_t> const* transform,
                          barrier::iteration_data_t<i_t, f_t> const* data,
                          char const* api)
{
  cuopt_expects(transform != nullptr,
                error_type_t::ValidationError,
                "%s: no barrier transform; Solve with CUOPT_SEQUENCE_SOLVE enabled first.",
                api);
  cuopt_expects(data != nullptr,
                error_type_t::ValidationError,
                "%s: no cached iteration_data; Solve a QP to Optimal first.",
                api);
}

// Re-adds the first solve's barrier-minus-crush shift so the update lands in the presolved model.
template <typename f_t>
static void add_shift(std::vector<f_t>& crushed, std::vector<f_t> const& shift)
{
  cuopt_expects(shift.size() == crushed.size(),
                error_type_t::ValidationError,
                "barrier cache shift length does not match the crushed vector.");
  for (std::size_t i = 0; i < crushed.size(); ++i) {
    crushed[i] += shift[i];
  }
}

template <typename i_t, typename f_t>
struct barrier_cache_t<i_t, f_t>::impl {
  using iteration_data_t   = barrier::iteration_data_t<i_t, f_t>;
  using iteration_data_ptr = std::unique_ptr<iteration_data_t, void (*)(iteration_data_t*)>;

  impl(std::unique_ptr<rmm::cuda_stream> stream_in, std::unique_ptr<raft::handle_t> handle_in)
    : stream(std::move(stream_in)),
      handle(std::move(handle_in)),
      iteration_data(nullptr, &barrier::destroy_iteration_data)
  {
  }

  std::unique_ptr<rmm::cuda_stream> stream;
  std::unique_ptr<raft::handle_t> handle;
  // Destroy iteration_data before transform: it may const-ref A/Q stored on the transform.
  std::unique_ptr<barrier_transform_t<i_t, f_t>> transform;
  iteration_data_ptr iteration_data;
  bool linear_objective_dirty{false};
  bool rhs_dirty{false};
  bool rhs_infeasible{false};
};

template <typename i_t, typename f_t>
barrier_cache_t<i_t, f_t>::barrier_cache_t(std::unique_ptr<rmm::cuda_stream> stream,
                                           std::unique_ptr<raft::handle_t> handle)
  : impl_(std::make_unique<impl>(std::move(stream), std::move(handle)))
{
}

template <typename i_t, typename f_t>
barrier_cache_t<i_t, f_t>::~barrier_cache_t() = default;

template <typename i_t, typename f_t>
barrier_cache_t<i_t, f_t>::barrier_cache_t(barrier_cache_t&&) noexcept = default;

template <typename i_t, typename f_t>
barrier_cache_t<i_t, f_t>& barrier_cache_t<i_t, f_t>::operator=(barrier_cache_t&&) noexcept =
  default;

template <typename i_t, typename f_t>
std::unique_ptr<barrier_cache_t<i_t, f_t>> barrier_cache_t<i_t, f_t>::create(unsigned stream_flags)
{
  auto stream =
    std::make_unique<rmm::cuda_stream>(static_cast<rmm::cuda_stream::flags>(stream_flags));
  auto handle = std::make_unique<raft::handle_t>(*stream);
  return std::unique_ptr<barrier_cache_t<i_t, f_t>>(
    new barrier_cache_t(std::move(stream), std::move(handle)));
}

template <typename i_t, typename f_t>
raft::handle_t* barrier_cache_t<i_t, f_t>::handle_ptr()
{
  return impl_->handle.get();
}

template <typename i_t, typename f_t>
raft::handle_t const* barrier_cache_t<i_t, f_t>::handle_ptr() const
{
  return impl_->handle.get();
}

template <typename i_t, typename f_t>
void barrier_cache_t<i_t, f_t>::clear()
{
  impl_->iteration_data.reset();
  impl_->transform.reset();
  mark_clean();
}

template <typename i_t, typename f_t>
void barrier_cache_t<i_t, f_t>::store_iteration_data(barrier::iteration_data_t<i_t, f_t>* data)
{
  impl_->iteration_data.reset(data);
}

template <typename i_t, typename f_t>
barrier::iteration_data_t<i_t, f_t>* barrier_cache_t<i_t, f_t>::release_iteration_data()
{
  return impl_->iteration_data.release();
}

template <typename i_t, typename f_t>
void barrier_cache_t<i_t, f_t>::store_transform(
  std::unique_ptr<barrier_transform_t<i_t, f_t>> transform)
{
  impl_->transform = std::move(transform);
}

template <typename i_t, typename f_t>
barrier_transform_t<i_t, f_t>* barrier_cache_t<i_t, f_t>::transform()
{
  return impl_->transform.get();
}

template <typename i_t, typename f_t>
barrier_transform_t<i_t, f_t> const* barrier_cache_t<i_t, f_t>::transform() const
{
  return impl_->transform.get();
}

template <typename i_t, typename f_t>
bool barrier_cache_t<i_t, f_t>::dirty() const
{
  return (impl_->linear_objective_dirty || impl_->rhs_dirty) && impl_->transform != nullptr &&
         impl_->iteration_data.get() != nullptr;
}

template <typename i_t, typename f_t>
void barrier_cache_t<i_t, f_t>::mark_clean()
{
  impl_->linear_objective_dirty = false;
  impl_->rhs_dirty              = false;
  impl_->rhs_infeasible         = false;
}

template <typename i_t, typename f_t>
bool barrier_cache_t<i_t, f_t>::rhs_infeasible() const
{
  return impl_->rhs_infeasible;
}

template <typename i_t, typename f_t>
void barrier_cache_t<i_t, f_t>::update_linear_objective(f_t const* c, i_t n)
{
  require_cache(impl_->transform.get(), impl_->iteration_data.get(), "update_linear_objective");
  // Cached Q and c are in minimization space.
  std::vector<f_t> user_objective;
  if (impl_->transform->maximize && c != nullptr && n > 0) {
    user_objective.assign(c, c + n);
    for (f_t& value : user_objective) {
      value = -value;
    }
    c = user_objective.data();
  }
  std::vector<f_t> crushed;
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
  simplex::lp_problem_t<i_t, f_t>& barrier_lp = *impl_->transform->barrier_lp;
  if (!translated_lower.empty() && linear_obj_shift.size() == crushed.size() &&
      column_scales.size() == crushed.size() && barrier_lp.objective.size() == crushed.size()) {
    f_t obj_constant_delta    = 0.0;
    std::size_t const n_lower = std::min(translated_lower.size(), crushed.size());
    for (std::size_t j = 0; j < n_lower; ++j) {
      f_t const crushed_before = barrier_lp.objective[j] - linear_obj_shift[j];
      obj_constant_delta += (crushed[j] - crushed_before) * column_scales[j] * translated_lower[j];
    }
    barrier_lp.obj_constant += obj_constant_delta;
  }
  add_shift(crushed, linear_obj_shift);
  // The next solve builds its solver from barrier_lp, so keep its objective and the cached
  // iteration workspace on the same c.
  std::vector<f_t>& barrier_objective = barrier_lp.objective;
  cuopt_expects(barrier_objective.size() == crushed.size(),
                error_type_t::ValidationError,
                "update_linear_objective: crushed objective size does not match the cached "
                "barrier LP.");
  barrier_objective = crushed;
  barrier::apply_barrier_linear_objective(
    *impl_->iteration_data, crushed.data(), static_cast<i_t>(crushed.size()));
  impl_->linear_objective_dirty = true;
}

template <typename i_t, typename f_t>
void barrier_cache_t<i_t, f_t>::update_rhs(f_t const* b, i_t m)
{
  require_cache(impl_->transform.get(), impl_->iteration_data.get(), "update_rhs");
  std::vector<f_t> crushed;
  std::string error;
  crush_rhs_status_t const status = crush_user_rhs(*impl_->transform, b, m, crushed, error);
  if (status == crush_rhs_status_t::infeasible) {
    // Cache stays usable for a later feasible update; the next solve reports INFEASIBLE from
    // this flag without running barrier.
    impl_->rhs_infeasible = true;
    impl_->rhs_dirty      = true;
    return;
  }
  cuopt_expects(
    status == crush_rhs_status_t::success, error_type_t::ValidationError, "%s", error.c_str());
  impl_->rhs_infeasible = false;
  add_shift(crushed, impl_->transform->rhs_shift);
  // barrier_lp->rhs also seeds the next solve's Mehrotra start, so keep it and the cached
  // workspace on the same b.
  std::vector<f_t>& barrier_rhs = impl_->transform->barrier_lp->rhs;
  cuopt_expects(barrier_rhs.size() == crushed.size(),
                error_type_t::ValidationError,
                "update_rhs: crushed RHS size does not match the cached barrier LP.");
  barrier_rhs = crushed;
  barrier::apply_barrier_rhs(
    *impl_->iteration_data, crushed.data(), static_cast<i_t>(crushed.size()));
  impl_->rhs_dirty = true;
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE
template class barrier_cache_t<int, double>;
#endif

}  // namespace cuopt::mathematical_optimization
