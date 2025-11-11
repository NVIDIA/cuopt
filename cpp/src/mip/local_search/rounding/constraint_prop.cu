/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <mip/mip_constants.hpp>
#include <mip/relaxed_lp/relaxed_lp.cuh>
#include <mip/utils.cuh>
#include <utilities/copy_helpers.hpp>
#include <utilities/seed_generator.cuh>
#include "constraint_prop.cuh"
#include "simple_rounding.cuh"

#include <thrust/copy.h>
#include <thrust/gather.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/partition.h>
#include <thrust/sort.h>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
repair_stats_t constraint_prop_t<i_t, f_t>::repair_stats;

template <typename i_t, typename f_t>
constraint_prop_t<i_t, f_t>::constraint_prop_t(mip_solver_context_t<i_t, f_t>& context_)
  : context(context_),
    temp_problem(*context.problem_ptr),
    temp_sol(*context.problem_ptr),
    bounds_update(context),
    multi_probe(context),
    bounds_repair(*context.problem_ptr, bounds_update),
    conditional_bounds_update(*context.problem_ptr),
    set_vars(context.problem_ptr->n_variables, context.problem_ptr->handle_ptr->get_stream()),
    unset_vars(context.problem_ptr->n_variables, context.problem_ptr->handle_ptr->get_stream()),
    lb_restore(context.problem_ptr->n_variables, context.problem_ptr->handle_ptr->get_stream()),
    ub_restore(context.problem_ptr->n_variables, context.problem_ptr->handle_ptr->get_stream()),
    assignment_restore(context.problem_ptr->n_variables,
                       context.problem_ptr->handle_ptr->get_stream()),
    rng(cuopt::seed_generator::get_seed(), 0, 0)
{
}

constexpr int n_subsections          = 3 * 7;
constexpr size_t size_of_subsections = n_subsections + 1;

template <typename i_t, typename f_t>
__device__ void assign_offsets(
  raft::device_span<i_t> offsets, i_t category, i_t idx, f_t frac_1, f_t frac_2)
{
  if (frac_1 <= 0.02 && frac_2 > 0.02) {
    offsets[category * 7 + 1] = idx + 1;
  } else if (frac_1 <= 0.05 && frac_2 > 0.05) {
    offsets[category * 7 + 2] = idx + 1;
  } else if (frac_1 <= 0.1 && frac_2 > 0.1) {
    offsets[category * 7 + 3] = idx + 1;
  } else if (frac_1 <= 0.2 && frac_2 > 0.2) {
    offsets[category * 7 + 4] = idx + 1;
  } else if (frac_1 <= 0.3 && frac_2 > 0.3) {
    offsets[category * 7 + 5] = idx + 1;
  } else if (frac_1 <= 0.4 && frac_2 > 0.4) {
    offsets[category * 7 + 6] = idx + 1;
  }
}

template <typename i_t, typename f_t>
void sort_subsections(raft::device_span<i_t> vars,
                      rmm::device_uvector<f_t>& random_vector,
                      rmm::device_uvector<i_t>& offsets,
                      const raft::handle_t* handle_ptr)
{
  size_t temp_storage_bytes = 0;
  rmm::device_uvector<std::byte> d_temp_storage(0, handle_ptr->get_stream());
  rmm::device_uvector<f_t> input_random_vec(random_vector, handle_ptr->get_stream());
  rmm::device_uvector<i_t> input_vars(vars.size(), handle_ptr->get_stream());
  raft::copy(input_vars.data(), vars.data(), vars.size(), handle_ptr->get_stream());
  cub::DeviceSegmentedSort::SortPairs(d_temp_storage.data(),
                                      temp_storage_bytes,
                                      input_random_vec.data(),
                                      random_vector.data(),
                                      input_vars.data(),
                                      vars.data(),
                                      vars.size(),
                                      n_subsections,
                                      offsets.data(),
                                      offsets.data() + 1,
                                      handle_ptr->get_stream());

  // Allocate temporary storage
  d_temp_storage.resize(temp_storage_bytes, handle_ptr->get_stream());

  // Run sorting operation
  cub::DeviceSegmentedSort::SortPairs(d_temp_storage.data(),
                                      temp_storage_bytes,
                                      input_random_vec.data(),
                                      random_vector.data(),
                                      input_vars.data(),
                                      vars.data(),
                                      vars.size(),
                                      n_subsections,
                                      offsets.data(),
                                      offsets.data() + 1,
                                      handle_ptr->get_stream());
  handle_ptr->sync_stream();
}

template <typename i_t, typename f_t>
__global__ void compute_implied_slack_consumption_per_var(
  typename problem_t<i_t, f_t>::view_t pb,
  raft::device_span<i_t> var_indices,
  raft::device_span<f_t> min_activity,
  raft::device_span<f_t> max_activity,
  raft::device_span<f_t> implied_var_slack_consumption,
  bool is_problem_ii,
  typename mip_solver_settings_t<i_t, f_t>::tolerances_t tols)
{
  i_t var_idx = var_indices[blockIdx.x];
  cuopt_assert(pb.is_integer_var(var_idx), "Variable must be integer!");
  i_t var_offset                       = pb.reverse_offsets[var_idx];
  i_t var_degree                       = pb.reverse_offsets[var_idx + 1] - var_offset;
  f_t th_var_implied_slack_consumption = 0.;
  auto var_bnd                         = pb.variable_bounds[var_idx];
  f_t lb                               = get_lower(var_bnd);
  f_t ub                               = get_upper(var_bnd);
  for (i_t i = threadIdx.x; i < var_degree; i += blockDim.x) {
    auto a        = pb.reverse_coefficients[var_offset + i];
    auto cnst_idx = pb.reverse_constraints[var_offset + i];
    auto min_a    = min_activity[cnst_idx];
    auto max_a    = max_activity[cnst_idx];
    auto cnstr_ub = pb.constraint_upper_bounds[cnst_idx];
    auto cnstr_lb = pb.constraint_lower_bounds[cnst_idx];
    // don't consider constraints that are infeasible
    if ((min_a >= cnstr_ub + tols.absolute_tolerance) ||
        (max_a <= cnstr_lb - tols.absolute_tolerance)) {
      continue;
    }

    auto slack_min_act = cnstr_ub - min_a;
    auto slack_max_act = cnstr_lb - max_a;
#pragma unroll
    for (auto act : {slack_min_act, slack_max_act}) {
      f_t slack_consumption_ratio;
      if (is_problem_ii && abs(act) < tols.absolute_tolerance) {
        slack_consumption_ratio = 1000.;
      } else {
        slack_consumption_ratio = (a / act) * (a / act);
      }
      th_var_implied_slack_consumption += slack_consumption_ratio;
    }
  }
  __shared__ f_t shmem[raft::WarpSize];
  f_t block_var_implied_slack_consumption =
    raft::blockReduce(th_var_implied_slack_consumption, (char*)shmem);
  if (threadIdx.x == 0) {
    implied_var_slack_consumption[blockIdx.x] = block_var_implied_slack_consumption;
  }
}

// sort by the implied percent of slack consumption
// across all constraints, sum the square roots of implied slack consumption percent
template <typename i_t, typename f_t>
void constraint_prop_t<i_t, f_t>::sort_by_implied_slack_consumption(solution_t<i_t, f_t>& sol,
                                                                    raft::device_span<i_t> vars,
                                                                    bool problem_ii)
{
  CUOPT_LOG_TRACE("Sorting vars by importance");
  rmm::device_uvector<f_t> implied_slack_consumption_per_var(vars.size(),
                                                             sol.handle_ptr->get_stream());
  const i_t block_dim = 128;
  auto min_activity   = selected_update ? make_span(multi_probe.upd_1.min_activity)
                                        : make_span(multi_probe.upd_0.min_activity);
  auto max_activity   = selected_update ? make_span(multi_probe.upd_1.max_activity)
                                        : make_span(multi_probe.upd_0.max_activity);
  compute_implied_slack_consumption_per_var<i_t, f_t>
    <<<vars.size(), block_dim, 0, sol.handle_ptr->get_stream()>>>(
      sol.problem_ptr->view(),
      vars,
      min_activity,
      max_activity,
      make_span(implied_slack_consumption_per_var),
      problem_ii,
      context.settings.get_tolerances());
  thrust::sort_by_key(sol.handle_ptr->get_thrust_policy(),
                      implied_slack_consumption_per_var.begin(),
                      implied_slack_consumption_per_var.end(),
                      vars.data(),
                      thrust::greater<f_t>{});
  sol.handle_ptr->sync_stream();
}

template <typename i_t, typename f_t>
void constraint_prop_t<i_t, f_t>::sort_by_interval_and_frac(solution_t<i_t, f_t>& sol,
                                                            raft::device_span<i_t> vars,
                                                            std::mt19937 rng)
{
  // we can't call this function when the problem is ii. it causes false offset computations
  // TODO add assert that the problem is not ii
  auto assgn = make_span(sol.assignment);
  thrust::stable_sort(
    sol.handle_ptr->get_thrust_policy(),
    vars.begin(),
    vars.end(),
    [bnds = sol.problem_ptr->variable_bounds.data(), assgn] __device__(i_t v_idx_1, i_t v_idx_2) {
      auto bnd_1            = bnds[v_idx_1];
      auto bnd_2            = bnds[v_idx_2];
      f_t bounds_interval_1 = get_upper(bnd_1) - get_lower(bnd_1);
      f_t bounds_interval_2 = get_upper(bnd_2) - get_lower(bnd_2);
      // if bounds interval are equal (binary and ternary) check fraction
      // if both bounds intervals are greater than 2. then do fraction
      if ((bounds_interval_1 == bounds_interval_2) ||
          (bounds_interval_1 > 2 && bounds_interval_2 > 2)) {
        f_t frac_1 = get_fractionality_of_val(assgn[v_idx_1]);
        f_t frac_2 = get_fractionality_of_val(assgn[v_idx_2]);
        return frac_1 < frac_2;
      } else {
        return bounds_interval_1 < bounds_interval_2;
      }
    });
  // now do the suffling, for that we need to assign some random values to rnd array
  // we will sort this rnd array and the vars in subsections, so that each subsection will be
  // shuffled in total we will have 3(binary, ternary and rest) x 7 intervals = 21 subsections.
  // first extract these subsections from the data
  rmm::device_uvector<i_t> subsection_offsets(size_of_subsections, sol.handle_ptr->get_stream());
  thrust::fill(
    sol.handle_ptr->get_thrust_policy(), subsection_offsets.begin(), subsection_offsets.end(), -1);
  subsection_offsets.set_element(0, 0, sol.handle_ptr->get_stream());
  subsection_offsets.set_element(n_subsections, vars.size(), sol.handle_ptr->get_stream());
  thrust::for_each(sol.handle_ptr->get_thrust_policy(),
                   thrust::make_counting_iterator(0),
                   thrust::make_counting_iterator((i_t)vars.size() - 1),
                   [bnds    = make_span(sol.problem_ptr->variable_bounds),
                    offsets = make_span(subsection_offsets),
                    vars,
                    assgn] __device__(i_t idx) {
                     i_t var_1             = vars[idx];
                     i_t var_2             = vars[idx + 1];
                     auto bnd_1            = bnds[var_1];
                     auto bnd_2            = bnds[var_2];
                     f_t bounds_interval_1 = get_upper(bnd_1) - get_lower(bnd_1);
                     f_t bounds_interval_2 = get_upper(bnd_2) - get_lower(bnd_2);
                     f_t frac_1            = get_fractionality_of_val(assgn[var_1]);
                     f_t frac_2            = get_fractionality_of_val(assgn[var_2]);
                     if (bounds_interval_1 == 1 && bounds_interval_2 == 1) {
                       i_t category = 0;
                       assign_offsets<i_t, f_t>(offsets, category, idx, frac_1, frac_2);
                     } else if (bounds_interval_1 == 1 && bounds_interval_2 == 2) {
                       offsets[7] = idx + 1;
                     } else if (bounds_interval_1 == 2 && bounds_interval_2 == 2) {
                       i_t category = 1;
                       assign_offsets<i_t, f_t>(offsets, category, idx, frac_1, frac_2);
                     } else if (bounds_interval_1 == 2 && bounds_interval_2 > 2) {
                       offsets[14] = idx + 1;
                     } else {
                       i_t category = 2;
                       assign_offsets<i_t, f_t>(offsets, category, idx, frac_1, frac_2);
                     }
                   });
  // if there are any empty sections fill their offsets as the previous offset
  thrust::for_each(sol.handle_ptr->get_thrust_policy(),
                   thrust::make_counting_iterator(0),
                   thrust::make_counting_iterator(1),
                   [offsets = subsection_offsets.data()] __device__(i_t idx) {
                     i_t last_existing_offset = 0;
                     for (i_t i = n_subsections; i > 0; --i) {
                       if (offsets[i] == -1) {
                         offsets[i] = last_existing_offset;
                       } else {
                         last_existing_offset = offsets[i];
                       }
                     }
                   });
  auto random_vector = get_random_uniform_vector<i_t, f_t>((i_t)vars.size(), rng);
  rmm::device_uvector<f_t> device_random_vector(random_vector.size(), sol.handle_ptr->get_stream());
  raft::copy(device_random_vector.data(),
             random_vector.data(),
             random_vector.size(),
             sol.handle_ptr->get_stream());
  sort_subsections<i_t, f_t>(vars, device_random_vector, subsection_offsets, sol.handle_ptr);
}

template <typename i_t, typename f_t>
void constraint_prop_t<i_t, f_t>::sort_by_frac(solution_t<i_t, f_t>& sol,
                                               raft::device_span<i_t> vars)
{
  auto assgn = make_span(sol.assignment);
  thrust::sort(sol.handle_ptr->get_thrust_policy(),
               vars.begin(),
               vars.end(),
               [assgn] __device__(i_t v_idx_1, i_t v_idx_2) {
                 f_t frac_1 = get_fractionality_of_val(assgn[v_idx_1]);
                 f_t frac_2 = get_fractionality_of_val(assgn[v_idx_2]);
                 return frac_1 < frac_2;
               });
}

template <typename i_t, typename f_t>
struct find_set_int_t {
  // This functor should be called only on integer variables
  f_t eps;
  raft::device_span<f_t> var_lb;
  raft::device_span<f_t> var_ub;
  raft::device_span<f_t> assignment;
  find_set_int_t(f_t eps_,
                 raft::device_span<f_t> lb_,
                 raft::device_span<f_t> ub_,
                 raft::device_span<f_t> assignment_)
    : eps(eps_), var_lb(lb_), var_ub(ub_), assignment(assignment_)
  {
  }

  HDI bool operator()(i_t idx)
  {
    auto var_val = assignment[idx];
    bool is_set  = is_integer<f_t>(var_val);
    return is_set;
  }
};

template <typename i_t, typename f_t>
struct find_unset_int_t {
  // This functor should be called only on integer variables
  f_t eps;
  raft::device_span<f_t> assignment;
  find_unset_int_t(f_t eps_, raft::device_span<f_t> assignment_)
    : eps(eps_), assignment(assignment_)
  {
  }

  HDI bool operator()(i_t idx)
  {
    auto var_val = assignment[idx];
    bool is_set  = is_integer<f_t>(var_val, eps);
    return !is_set;
  }
};

// TODO verify this logic
template <typename i_t, typename f_t>
__device__ bool round_val_on_singleton_and_crossing(
  f_t& assign, f_t v_lb, f_t v_ub, f_t o_lb, f_t o_ub)
{
  if (v_lb == v_ub) {
    assign = floor(v_lb + 0.5);
    return true;
  } else if (v_ub <= o_lb && v_lb <= o_ub) {
    assign = floor(v_lb + 0.5);
    return true;
  } else if (v_ub <= o_lb && v_lb >= o_ub) {
    if (!isfinite(o_lb)) {
      assign = ceil(o_ub - 0.5);
    } else if (!isfinite(o_ub)) {
      assign = floor(o_lb + 0.5);
    } else {
      assign = round((o_lb + o_ub) / 2);
    }
    return true;
  } else if (v_lb >= o_ub && v_ub >= o_lb) {
    assign = ceil(v_ub - 0.5);
    return true;
  }
  // if all cases fail
  else if (v_lb > v_ub) {
    if (!isfinite(o_lb)) {
      assign = ceil(o_ub - 0.5);
    } else if (!isfinite(o_ub)) {
      assign = floor(o_lb + 0.5);
    } else {
      assign = round((o_lb + o_ub) / 2);
    }
    return true;
  }
  return false;
}

template <typename i_t, typename f_t>
void constraint_prop_t<i_t, f_t>::collapse_crossing_bounds(problem_t<i_t, f_t>& problem,
                                                           problem_t<i_t, f_t>& orig_problem,
                                                           const raft::handle_t* handle_ptr)
{
  auto v_bnds          = make_span(problem.variable_bounds);
  auto original_v_bnds = make_span(orig_problem.variable_bounds);
  thrust::for_each(
    handle_ptr->get_thrust_policy(),
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator((i_t)v_bnds.size()),
    [v_bnds,
     original_v_bnds,
     variable_types = make_span(problem.variable_types),
     int_tol        = problem.tolerances.integrality_tolerance] __device__(i_t idx) {
      auto v_bnd  = v_bnds[idx];
      auto ov_bnd = original_v_bnds[idx];
      auto v_lb   = get_lower(v_bnd);
      auto v_ub   = get_upper(v_bnd);
      auto o_lb   = get_lower(ov_bnd);
      auto o_ub   = get_upper(ov_bnd);
      if (v_lb > v_ub) {
        f_t val_to_collapse;
        if (variable_types[idx] == var_t::INTEGER) {
          round_val_on_singleton_and_crossing<i_t, f_t>(val_to_collapse, v_lb, v_ub, o_lb, o_ub);
        } else {
          if (isfinite(o_lb) && isfinite(o_ub)) {
            val_to_collapse = (o_lb + o_ub) / 2;
          } else {
            val_to_collapse = isfinite(o_lb) ? o_lb : o_ub;
          }
        }

        cuopt_assert(o_lb - int_tol <= val_to_collapse && val_to_collapse <= o_ub + int_tol,
                     "Out of original bounds!");
        v_bnds[idx] = typename type_2<f_t>::type{val_to_collapse, val_to_collapse};
      }
    });
}

template <typename i_t, typename f_t>
void constraint_prop_t<i_t, f_t>::set_bounds_on_fixed_vars(solution_t<i_t, f_t>& sol)
{
  auto assgn      = make_span(sol.assignment);
  auto var_bounds = make_span(sol.problem_ptr->variable_bounds);
  thrust::for_each(sol.handle_ptr->get_thrust_policy(),
                   sol.problem_ptr->integer_indices.begin(),
                   sol.problem_ptr->integer_indices.end(),
                   [pb = sol.problem_ptr->view(), assgn, var_bounds] __device__(i_t idx) {
                     auto var_val = assgn[idx];
                     if (pb.is_integer(var_val)) {
                       var_bounds[idx] = typename type_2<f_t>::type{var_val, var_val};
                     }
                   });
}

template <typename i_t, typename f_t, typename f_t2>
struct is_bound_fixed_t {
  // This functor should be called only on integer variables
  f_t eps;
  raft::device_span<typename type_2<f_t>::type> bnd;
  raft::device_span<typename type_2<f_t>::type> original_bnd;
  raft::device_span<f_t> assignment;
  is_bound_fixed_t(f_t eps_,
                   raft::device_span<f_t2> bnd_,
                   raft::device_span<f_t2> original_bnd_,
                   raft::device_span<f_t> assignment_)
    : eps(eps_), bnd(bnd_), original_bnd(original_bnd_), assignment(assignment_)
  {
  }

  HDI bool operator()(i_t idx)
  {
    auto v_bnd  = bnd[idx];
    auto v_lb   = get_lower(v_bnd);
    auto v_ub   = get_upper(v_bnd);
    auto ov_bnd = original_bnd[idx];
    auto o_lb   = get_lower(ov_bnd);
    auto o_ub   = get_upper(ov_bnd);
    bool is_singleton =
      round_val_on_singleton_and_crossing<i_t, f_t>(assignment[idx], v_lb, v_ub, o_lb, o_ub);
    return is_singleton;
  }
};

template <typename i_t, typename f_t>
struct fix_bounds_t {
  f_t eps;
  raft::device_span<f_t> lb;
  raft::device_span<f_t> ub;
  raft::device_span<f_t> assign;

  fix_bounds_t(f_t eps_,
               raft::device_span<f_t> lb_,
               raft::device_span<f_t> ub_,
               raft::device_span<f_t> assign_)
    : eps(eps_), lb(lb_), ub(ub_), assign(assign_)
  {
  }

  HDI void operator()(i_t idx)
  {
    auto val = assign[idx];
    lb[idx]  = round(val) - eps;
    ub[idx]  = round(val) + eps;
  }
};

template <typename i_t, typename f_t>
struct greater_than_threshold_t {
  f_t threshold;
  raft::device_span<f_t> assignment;

  greater_than_threshold_t(f_t t, raft::device_span<f_t> assignment_)
    : threshold(t), assignment(assignment_)
  {
  }

  __host__ __device__ bool operator()(const i_t& x) const { return assignment[x] > threshold; }
};

template <typename i_t, typename f_t>
void constraint_prop_t<i_t, f_t>::copy_bounds(
  rmm::device_uvector<typename type_2<f_t>::type>& output_bounds,
  const rmm::device_uvector<f_t>& input_lb,
  const rmm::device_uvector<f_t>& input_ub,
  const raft::handle_t* handle_ptr)
{
  thrust::transform(
    handle_ptr->get_thrust_policy(),
    thrust::make_zip_iterator(thrust::make_tuple(input_lb.begin(), input_ub.begin())),
    thrust::make_zip_iterator(thrust::make_tuple(input_lb.end(), input_ub.end())),
    output_bounds.begin(),
    [] __device__(auto bounds) {
      return typename type_2<f_t>::type{thrust::get<0>(bounds), thrust::get<1>(bounds)};
    });
}

template <typename i_t, typename f_t>
void constraint_prop_t<i_t, f_t>::copy_bounds(
  rmm::device_uvector<f_t>& output_lb,
  rmm::device_uvector<f_t>& output_ub,
  const rmm::device_uvector<typename type_2<f_t>::type>& input_bounds,
  const raft::handle_t* handle_ptr)
{
  thrust::transform(
    handle_ptr->get_thrust_policy(),
    input_bounds.begin(),
    input_bounds.end(),
    thrust::make_zip_iterator(thrust::make_tuple(output_lb.begin(), output_ub.begin())),
    [] __device__(auto bounds) {
      return thrust::make_tuple(get_lower(bounds), get_upper(bounds));
    });
}

template <typename i_t, typename f_t>
void constraint_prop_t<i_t, f_t>::copy_bounds(rmm::device_uvector<f_t>& output_lb,
                                              rmm::device_uvector<f_t>& output_ub,
                                              const rmm::device_uvector<f_t>& input_lb,
                                              const rmm::device_uvector<f_t>& input_ub,
                                              const raft::handle_t* handle_ptr)
{
  raft::copy(output_lb.data(), input_lb.data(), input_lb.size(), handle_ptr->get_stream());
  raft::copy(output_ub.data(), input_ub.data(), input_ub.size(), handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
void constraint_prop_t<i_t, f_t>::copy_bounds(rmm::device_uvector<f_t>& output_lb,
                                              rmm::device_uvector<f_t>& output_ub,
                                              rmm::device_uvector<f_t>& output_assignment,
                                              const rmm::device_uvector<f_t>& input_lb,
                                              const rmm::device_uvector<f_t>& input_ub,
                                              const rmm::device_uvector<f_t>& input_assignment,
                                              const raft::handle_t* handle_ptr)
{
  copy_bounds(output_lb, output_ub, input_lb, input_ub, handle_ptr);
  raft::copy(output_assignment.data(),
             input_assignment.data(),
             input_assignment.size(),
             handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
void constraint_prop_t<i_t, f_t>::save_bounds(solution_t<i_t, f_t>& sol)
{
  copy_bounds(lb_restore, ub_restore, sol.problem_ptr->variable_bounds, sol.handle_ptr);
  raft::copy(assignment_restore.data(),
             sol.assignment.data(),
             sol.assignment.size(),
             sol.handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
void constraint_prop_t<i_t, f_t>::restore_bounds(solution_t<i_t, f_t>& sol)
{
  copy_bounds(sol.problem_ptr->variable_bounds, lb_restore, ub_restore, sol.handle_ptr);
  raft::copy(sol.assignment.data(),
             assignment_restore.data(),
             assignment_restore.size(),
             sol.handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
void constraint_prop_t<i_t, f_t>::restore_original_bounds(solution_t<i_t, f_t>& sol,
                                                          solution_t<i_t, f_t>& orig_sol)
{
  raft::copy(sol.problem_ptr->variable_bounds.data(),
             orig_sol.problem_ptr->variable_bounds.data(),
             orig_sol.problem_ptr->variable_bounds.size(),
             orig_sol.handle_ptr->get_stream());
  raft::copy(sol.assignment.data(),
             orig_sol.assignment.data(),
             orig_sol.assignment.size(),
             orig_sol.handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
thrust::pair<f_t, f_t> constraint_prop_t<i_t, f_t>::generate_double_probing_pair(
  const solution_t<i_t, f_t>& sol,
  const solution_t<i_t, f_t>& orig_sol,
  i_t unset_var_idx,
  const std::optional<std::reference_wrapper<probing_config_t<i_t, f_t>>> probing_config,
  bool bulk_rounding)
{
  f_t first_probe, second_probe;
  if (probing_config.has_value()) {
    // for now get the first one
    auto [from_first, from_second] = probing_config.value().get().probing_values[unset_var_idx];
    std::mt19937 rng(cuopt::seed_generator::get_seed());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    f_t random_value  = dist(rng);
    f_t average_value = (from_first + from_second) / 2;
    if (random_value > 0.5) {
      average_value = ceil(average_value - orig_sol.problem_ptr->tolerances.integrality_tolerance);
    } else {
      average_value = floor(average_value + orig_sol.problem_ptr->tolerances.integrality_tolerance);
    }
    // if the first set of values are more represented use the second value
    if (probing_config.value().get().n_of_fixed_from_first >
        probing_config.value().get().n_of_fixed_from_second) {
      // if it is the same as less represented
      if (average_value == from_second) {
        first_probe  = from_second;
        second_probe = from_first;
      } else {
        first_probe  = average_value;
        second_probe = from_second;
      }
    } else {
      // if it is the same as less represented
      if (average_value == from_first) {
        first_probe  = from_first;
        second_probe = from_second;
      } else {
        first_probe  = average_value;
        second_probe = from_first;
      }
    }
  } else {
    std::tie(first_probe, std::ignore, second_probe) = probing_values(sol, orig_sol, unset_var_idx);
    // do another draw in bulk rounding, so the second vector is randomly drawn
    if (bulk_rounding) {
      std::tie(second_probe, std::ignore, std::ignore) =
        probing_values(sol, orig_sol, unset_var_idx);
    }
  }
  return thrust::make_pair(first_probe, second_probe);
}

template <typename i_t, typename f_t>
bool test_var_out_of_bounds(const solution_t<i_t, f_t>& orig_sol,
                            i_t unset_var_idx,
                            f_t probe,
                            f_t int_tol,
                            const raft::handle_t* handle_ptr)
{
  auto var_bnd =
    orig_sol.problem_ptr->variable_bounds.element(unset_var_idx, handle_ptr->get_stream());
  return (get_lower(var_bnd) <= probe + int_tol) && (probe - int_tol <= get_upper(var_bnd));
}

template <typename i_t, typename f_t>
std::tuple<std::vector<i_t>, std::vector<f_t>, std::vector<f_t>>
constraint_prop_t<i_t, f_t>::generate_bulk_rounding_vector(
  const solution_t<i_t, f_t>& sol,
  const solution_t<i_t, f_t>& orig_sol,
  const std::vector<i_t>& host_vars_to_set,
  const std::optional<std::reference_wrapper<probing_config_t<i_t, f_t>>> probing_config)
{
  const f_t int_tol = orig_sol.problem_ptr->tolerances.integrality_tolerance;
  std::string log_str{"Setting var:\t"};
  std::tuple<std::vector<i_t>, std::vector<f_t>, std::vector<f_t>> var_probe_vals;
  std::get<0>(var_probe_vals).resize(host_vars_to_set.size());
  std::get<1>(var_probe_vals).resize(host_vars_to_set.size());
  std::get<2>(var_probe_vals).resize(host_vars_to_set.size());
  for (i_t i = 0; i < (i_t)host_vars_to_set.size(); ++i) {
    auto unset_var_idx = host_vars_to_set[i];
    f_t first_probe, second_probe;
    // if it is a bulk rounding do
    if (host_vars_to_set.size() > 1) {
      cuda::std::tie(first_probe, second_probe) =
        generate_double_probing_pair(sol, orig_sol, unset_var_idx, probing_config, true);
    } else {
      cuda::std::tie(first_probe, second_probe) =
        generate_double_probing_pair(sol, orig_sol, unset_var_idx, probing_config, false);
    }
    cuopt_assert(
      test_var_out_of_bounds(orig_sol, unset_var_idx, first_probe, int_tol, sol.handle_ptr),
      "Variable out of original bounds!");
    cuopt_assert(
      test_var_out_of_bounds(orig_sol, unset_var_idx, second_probe, int_tol, sol.handle_ptr),
      "Variable out of original bounds!");
    cuopt_assert(orig_sol.problem_ptr->is_integer(first_probe), "Probing value must be an integer");
    cuopt_assert(orig_sol.problem_ptr->is_integer(second_probe),
                 "Probing value must be an integer");
    f_t val_to_round = first_probe;
    // check probing cache if some implied bounds exists
    if (use_probing_cache &&
        bounds_update.probing_cache.contains(*sol.problem_ptr, unset_var_idx)) {
      // check if there are any conflicting bounds
      val_to_round = bounds_update.probing_cache.get_least_conflicting_rounding(*sol.problem_ptr,
                                                                                multi_probe.host_lb,
                                                                                multi_probe.host_ub,
                                                                                unset_var_idx,
                                                                                first_probe,
                                                                                second_probe,
                                                                                int_tol);
      if (val_to_round == second_probe) { second_probe = first_probe; }
    }
    cuopt_assert(
      test_var_out_of_bounds(orig_sol, unset_var_idx, val_to_round, int_tol, sol.handle_ptr),
      "Variable out of original bounds!");
    cuopt_assert(
      test_var_out_of_bounds(orig_sol, unset_var_idx, second_probe, int_tol, sol.handle_ptr),
      "Variable out of original bounds!");
    std::get<0>(var_probe_vals)[i] = unset_var_idx;
    std::get<1>(var_probe_vals)[i] = val_to_round;
    std::get<2>(var_probe_vals)[i] = second_probe;
    log_str.append(std::to_string(unset_var_idx) + ", ");
  }
  CUOPT_LOG_TRACE("%s", log_str.c_str());
  return var_probe_vals;
}

template <typename i_t, typename f_t>
void constraint_prop_t<i_t, f_t>::update_host_assignment(const solution_t<i_t, f_t>& sol)
{
  raft::copy(curr_host_assignment.data(),
             sol.assignment.data(),
             sol.problem_ptr->n_variables,
             sol.handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
void constraint_prop_t<i_t, f_t>::set_host_bounds(const solution_t<i_t, f_t>& sol)
{
  std::tie(multi_probe.host_lb, multi_probe.host_ub) =
    extract_host_bounds<f_t>(sol.problem_ptr->variable_bounds, sol.handle_ptr);
}

template <typename i_t, typename f_t>
void constraint_prop_t<i_t, f_t>::restore_original_bounds_on_unfixed(
  problem_t<i_t, f_t>& problem,
  problem_t<i_t, f_t>& original_problem,
  const raft::handle_t* handle_ptr)
{
  thrust::for_each(handle_ptr->get_thrust_policy(),
                   thrust::make_counting_iterator(0),
                   thrust::make_counting_iterator(problem.n_variables),
                   [p_v = problem.view(), op_v = original_problem.view()] __device__(i_t var_idx) {
                     auto p_v_var_bnd = p_v.variable_bounds[var_idx];
                     if (!p_v.integer_equal(get_lower(p_v_var_bnd), get_upper(p_v_var_bnd)) ||
                         !p_v.is_integer_var(var_idx)) {
                       p_v.variable_bounds[var_idx] = op_v.variable_bounds[var_idx];
                     }
                   });
}

template <typename i_t, typename f_t>
bool constraint_prop_t<i_t, f_t>::run_repair_procedure(problem_t<i_t, f_t>& problem,
                                                       problem_t<i_t, f_t>& original_problem,
                                                       timer_t& timer,
                                                       const raft::handle_t* handle_ptr)
{
  // select the first probing value
  i_t select = 0;
  multi_probe.set_updated_bounds(problem, select, handle_ptr);
  bounds_update.copy_input_bounds(problem);
  repair_stats.repair_attempts++;
  f_t repair_start_time                = timer.remaining_time();
  i_t n_of_repairs_needed_for_feasible = 0;
  do {
    n_of_repairs_needed_for_feasible++;
    if (timer.check_time_limit()) {
      CUOPT_LOG_DEBUG("Time limit is reached in repair loop!");
      f_t repair_end_time = timer.remaining_time();
      repair_stats.total_time_spent_on_repair += repair_start_time - repair_end_time;
      return false;
    }
    repair_stats.total_repair_loops++;
    collapse_crossing_bounds(problem, original_problem, handle_ptr);
    bool bounds_repaired =
      bounds_repair.repair_problem(problem, original_problem, timer, handle_ptr);
    if (bounds_repaired) {
      repair_stats.intermediate_repair_success++;
      CUOPT_LOG_DEBUG("Bounds repair success, running bounds prop to verify feasibility!");
    }
    f_t bounds_prop_start_time = timer.remaining_time();
    // restore all bounds to the original bounds and run bounds prop on
    // it. note that the number of fixed vars will still be the same, as repair only shifts the vars
    restore_original_bounds_on_unfixed(problem, original_problem, handle_ptr);
    bounds_update.settings.iteration_limit = 100;
    bounds_update.settings.time_limit      = timer.remaining_time();
    auto term_crit                         = bounds_update.solve(problem);
    bounds_update.settings.iteration_limit = 50;
    if (timer.check_time_limit()) {
      CUOPT_LOG_DEBUG("Time limit is reached in repair loop!");
      f_t repair_end_time = timer.remaining_time();
      repair_stats.total_time_spent_on_repair += repair_start_time - repair_end_time;
      return false;
    }
    if (termination_criterion_t::NO_UPDATE != term_crit) {
      bounds_update.set_updated_bounds(problem);
    }

    f_t bounds_prop_end_time = timer.remaining_time();
    repair_stats.total_time_spent_bounds_prop_after_repair +=
      bounds_prop_start_time - bounds_prop_end_time;
  } while (bounds_update.infeas_constraints_count > 0);
  repair_stats.repair_success++;
  CUOPT_LOG_DEBUG("Repair success: n_of_repair_calls needed: %d", n_of_repairs_needed_for_feasible);
  f_t repair_end_time = timer.remaining_time();
  repair_stats.total_time_spent_on_repair += repair_start_time - repair_end_time;
  // test that bounds are really repaired and no ii cstr is present
  cuopt_assert(!is_problem_ii(problem), "Problem must not be ii after repair success");
  return true;
}

template <typename i_t, typename f_t>
void constraint_prop_t<i_t, f_t>::find_unset_integer_vars(solution_t<i_t, f_t>& sol,
                                                          rmm::device_uvector<i_t>& unset_vars)
{
  unset_vars.resize(sol.problem_ptr->n_integer_vars, sol.handle_ptr->get_stream());
  auto iter =
    thrust::copy_if(sol.handle_ptr->get_thrust_policy(),
                    sol.problem_ptr->integer_indices.begin(),
                    sol.problem_ptr->integer_indices.end(),
                    unset_vars.begin(),
                    find_unset_int_t<i_t, f_t>{sol.problem_ptr->tolerances.integrality_tolerance,
                                               make_span(sol.assignment)});
  unset_vars.resize(iter - unset_vars.begin(), sol.handle_ptr->get_stream());
}

template <typename i_t, typename f_t>
bool constraint_prop_t<i_t, f_t>::is_problem_ii(problem_t<i_t, f_t>& problem)
{
  bounds_update.calculate_activity_on_problem_bounds(problem);
  bounds_update.calculate_infeasible_redundant_constraints(problem);
  bool problem_ii = bounds_update.infeas_constraints_count > 0;
  return problem_ii;
}

template <typename i_t, typename f_t>
bool constraint_prop_t<i_t, f_t>::find_integer(
  solution_t<i_t, f_t>& sol,
  solution_t<i_t, f_t>& orig_sol,
  f_t lp_run_time_after_feasible,
  timer_t& timer,
  std::optional<std::reference_wrapper<probing_config_t<i_t, f_t>>> probing_config)
{
  using crit_t             = termination_criterion_t;
  auto& unset_integer_vars = unset_vars;
  std::mt19937 rng(cuopt::seed_generator::get_seed());
  lb_restore.resize(sol.problem_ptr->n_variables, sol.handle_ptr->get_stream());
  ub_restore.resize(sol.problem_ptr->n_variables, sol.handle_ptr->get_stream());
  assignment_restore.resize(sol.problem_ptr->n_variables, sol.handle_ptr->get_stream());
  unset_integer_vars.resize(sol.problem_ptr->n_integer_vars, sol.handle_ptr->get_stream());
  curr_host_assignment.resize(sol.problem_ptr->n_variables);
  // round vals that are close enough
  bounds_update.settings.time_limit      = max_timer.remaining_time();
  bounds_update.settings.iteration_limit = 50;
  bounds_update.resize(*sol.problem_ptr);
  multi_probe.settings.iteration_limit = 50;
  multi_probe.settings.time_limit      = max_timer.remaining_time();
  multi_probe.resize(*sol.problem_ptr);
  if (max_timer.check_time_limit()) {
    CUOPT_LOG_DEBUG("Time limit is reached before bounds prop rounding!");
    sol.round_nearest();
    expand_device_copy(orig_sol.assignment, sol.assignment, sol.handle_ptr->get_stream());
    cuopt_func_call(orig_sol.test_variable_bounds());
    return orig_sol.compute_feasibility();
  }
  if (round_all_vars) {
    raft::copy(unset_integer_vars.data(),
               sol.problem_ptr->integer_indices.data(),
               sol.problem_ptr->n_integer_vars,
               sol.handle_ptr->get_stream());
  } else {
    find_unset_integer_vars(sol, unset_integer_vars);
    sort_by_frac(sol, make_span(unset_integer_vars));
    // round first unset_integer_vars.size() - 50, leave last 50 to be rounded by the algo
    i_t n_to_round = std::max(unset_integer_vars.size() - 50, 0lu);
    if (n_to_round > 0) {
      thrust::for_each(
        sol.handle_ptr->get_thrust_policy(),
        unset_integer_vars.begin(),
        unset_integer_vars.begin() + n_to_round,
        [sol = sol.view(), seed = cuopt::seed_generator::get_seed()] __device__(i_t var_idx) {
          raft::random::PCGenerator rng(seed, var_idx, 0);
          auto var_bnd            = sol.problem.variable_bounds[var_idx];
          sol.assignment[var_idx] = round_nearest(sol.assignment[var_idx],
                                                  get_lower(var_bnd),
                                                  get_upper(var_bnd),
                                                  sol.problem.tolerances.integrality_tolerance,
                                                  rng);
        });
      find_unset_integer_vars(sol, unset_integer_vars);
    }
    set_bounds_on_fixed_vars(sol);
  }

  CUOPT_LOG_DEBUG("Bounds propagation rounding: unset vars %lu", unset_integer_vars.size());
  if (unset_integer_vars.size() == 0) {
    CUOPT_LOG_DEBUG("No integer variables provided in the bounds prop rounding");
    expand_device_copy(orig_sol.assignment, sol.assignment, sol.handle_ptr->get_stream());
    cuopt_func_call(orig_sol.test_variable_bounds());
    return orig_sol.compute_feasibility();
  }
  // this is needed for the sort inside of the loop
  bool problem_ii = is_problem_ii(*sol.problem_ptr);
  // if the problem is ii, run the bounds prop in the beginning
  if (problem_ii) {
    bool bounds_repaired =
      bounds_repair.repair_problem(*sol.problem_ptr, *orig_sol.problem_ptr, timer, sol.handle_ptr);
    if (bounds_repaired) {
      CUOPT_LOG_DEBUG("Initial ii is repaired by bounds repair!");
    } else {
      auto term_crit = bounds_update.solve(*sol.problem_ptr);
      if (termination_criterion_t::NO_UPDATE != term_crit) {
        bounds_update.set_updated_bounds(*sol.problem_ptr);
      }
      rounding_ii = true;
    }
  }
  // do the sort if the problem is not ii. crossing bounds might cause some issues on the sort order
  else {
    // this is a sort to have initial shuffling, so that stable sort within will keep the order and
    // some randomness will be achieved
    sort_by_interval_and_frac(sol, make_span(unset_integer_vars), rng);
  }
  set_host_bounds(sol);
  size_t set_count               = 0;
  bool timeout_happened          = false;
  i_t n_failed_repair_iterations = 0;
  while (set_count < unset_integer_vars.size()) {
    CUOPT_LOG_TRACE("n_set_vars %d vars to set %lu", set_count, unset_integer_vars.size());
    update_host_assignment(sol);
    if (max_timer.check_time_limit()) {
      CUOPT_LOG_DEBUG("Second time limit is reached returning nearest rounding!");
      sol.round_nearest();
      timeout_happened = true;
      break;
    }
    if (!rounding_ii && timer.check_time_limit()) {
      CUOPT_LOG_DEBUG("First time limit is reached! Continuing without backtracking and repair!");
      rounding_ii = true;
      // this is to not try the repair procedure again
      n_failed_repair_iterations = std::numeric_limits<i_t>::max();
    }
    const i_t n_curr_unset = unset_integer_vars.size() - set_count;
    if (single_rounding_only) {
      bounds_prop_interval = 1;
    } else if (!recovery_mode || rounding_ii) {
      if (n_curr_unset > 36) {
        bounds_prop_interval = sqrt(n_curr_unset);
      } else {
        bounds_prop_interval = 1;
      }
    }
    i_t n_vars_to_set = recovery_mode ? 1 : bounds_prop_interval;
    // if we are not at the last stage or if we are in recovery mode, don't sort
    if (n_vars_to_set != 1) {
      sort_by_implied_slack_consumption(
        sol, make_span(unset_integer_vars, set_count, unset_integer_vars.size()), problem_ii);
    }
    std::vector<i_t> host_vars_to_set(n_vars_to_set);
    raft::copy(host_vars_to_set.data(),
               unset_integer_vars.data() + set_count,
               n_vars_to_set,
               sol.handle_ptr->get_stream());
    auto var_probe_vals =
      generate_bulk_rounding_vector(sol, orig_sol, host_vars_to_set, probing_config);
    probe(
      sol, orig_sol.problem_ptr, var_probe_vals, &set_count, unset_integer_vars, probing_config);
    if (!(n_failed_repair_iterations >= max_n_failed_repair_iterations) && rounding_ii &&
        !timeout_happened) {
      timer_t repair_timer{std::min(timer.remaining_time() / 5, timer.elapsed_time() / 3)};
      save_bounds(sol);
      // update bounds and run repair procedure
      bool bounds_repaired =
        run_repair_procedure(*sol.problem_ptr, *orig_sol.problem_ptr, repair_timer, sol.handle_ptr);
      if (!bounds_repaired) {
        restore_bounds(sol);
        n_failed_repair_iterations++;
      } else {
        CUOPT_LOG_DEBUG(
          "Bounds are repaired! Deactivating recovery mode in bounds prop. n_curr_unset %d  "
          "bounds_prop_interval %d",
          n_curr_unset,
          bounds_prop_interval);
        recovery_mode      = false;
        rounding_ii        = false;
        n_iter_in_recovery = 0;
        // during repair procedure some variables might be collapsed
        auto iter =
          thrust::stable_partition(sol.handle_ptr->get_thrust_policy(),
                                   unset_vars.begin() + set_count,
                                   unset_vars.end(),
                                   is_bound_fixed_t<i_t, f_t, typename type_2<f_t>::type>{
                                     orig_sol.problem_ptr->tolerances.integrality_tolerance,
                                     make_span(sol.problem_ptr->variable_bounds),
                                     make_span(orig_sol.problem_ptr->variable_bounds),
                                     make_span(sol.assignment)});
        i_t n_fixed_vars = (iter - (unset_vars.begin() + set_count));
        CUOPT_LOG_TRACE("After repair procedure, number of additional fixed vars %d", n_fixed_vars);
        set_count += n_fixed_vars;
      }
    }
    // we can keep normal bounds_update here because this is only activated after the  repair
    if (recovery_mode && multi_probe.infeas_constraints_count_0 > 0 &&
        multi_probe.infeas_constraints_count_1 > 0) {
      CUOPT_LOG_DEBUG("Problem is ii in constraint prop. n_curr_unset %d  bounds_prop_interval %d",
                      n_curr_unset,
                      bounds_prop_interval);
      rounding_ii   = true;
      recovery_mode = false;
    }
    if (recovery_mode && (++n_iter_in_recovery == bounds_prop_interval)) {
      CUOPT_LOG_DEBUG(
        "Deactivating recovery mode in bounds prop. n_curr_unset %d  bounds_prop_interval %d",
        n_curr_unset,
        bounds_prop_interval);
      recovery_mode      = false;
      n_iter_in_recovery = 0;
    }
    // we use this to utilize the caching
    // we update from the problem bounds and not the final bounds of bounds update
    // because we might be in a recovery mode where we want to continue with the bounds before bulk
    // which is the unchanged problem bounds
    multi_probe.update_host_bounds(sol.handle_ptr, make_span(sol.problem_ptr->variable_bounds));
  }
  CUOPT_LOG_DEBUG(
    "Bounds propagation rounding end: ii constraint count first buffer %d, second buffer %d",
    multi_probe.infeas_constraints_count_0,
    multi_probe.infeas_constraints_count_1);
  cuopt_assert(sol.test_number_all_integer(), "All integers must be rounded");
  expand_device_copy(orig_sol.assignment, sol.assignment, sol.handle_ptr->get_stream());
  cuopt_func_call(orig_sol.test_variable_bounds());
  // if the constraint is not ii, run LP
  if ((multi_probe.infeas_constraints_count_0 == 0 ||
       multi_probe.infeas_constraints_count_1 == 0) &&
      !timeout_happened && lp_run_time_after_feasible > 0) {
    relaxed_lp_settings_t lp_settings;
    lp_settings.time_limit            = lp_run_time_after_feasible;
    lp_settings.tolerance             = orig_sol.problem_ptr->tolerances.absolute_tolerance;
    lp_settings.save_state            = false;
    lp_settings.return_first_feasible = true;
    run_lp_with_vars_fixed(*orig_sol.problem_ptr,
                           orig_sol,
                           orig_sol.problem_ptr->integer_indices,
                           lp_settings,
                           static_cast<bound_presolve_t<i_t, f_t>*>(nullptr));
  }
  bool res_feasible = orig_sol.compute_feasibility();
  orig_sol.handle_ptr->sync_stream();
  return res_feasible;
}

template <typename i_t, typename f_t>
bool constraint_prop_t<i_t, f_t>::apply_round(
  solution_t<i_t, f_t>& sol,
  f_t lp_run_time_after_feasible,
  timer_t& timer,
  std::optional<std::reference_wrapper<probing_config_t<i_t, f_t>>> probing_config)
{
  raft::common::nvtx::range fun_scope("constraint prop round");
  max_timer = timer_t{max_time_for_bounds_prop};
  if (check_brute_force_rounding(sol)) { return true; }
  recovery_mode      = false;
  rounding_ii        = false;
  n_iter_in_recovery = 0;
  sol.compute_constraints();
  problem_t<i_t, f_t> p(*sol.problem_ptr);
  temp_sol.resize_copy(sol);
  temp_sol.problem_ptr       = &p;
  f_t bounds_prop_start_time = max_timer.remaining_time();
  cuopt_func_call(temp_sol.test_variable_bounds(false));
  bool sol_found = find_integer(temp_sol, sol, lp_run_time_after_feasible, timer, probing_config);
  f_t bounds_prop_end_time = max_timer.remaining_time();
  repair_stats.total_time_spent_on_bounds_prop += bounds_prop_start_time - bounds_prop_end_time;

  CUOPT_LOG_DEBUG(
    "repair_success %lu repair_attempts %lu intermediate_repair_success %lu total_repair_loops %lu "
    "total_time_spent_on_repair %f total_time_spent_bounds_prop_after_repair %f "
    "total_time_spent_on_bounds_prop %f",
    repair_stats.repair_success,
    repair_stats.repair_attempts,
    repair_stats.intermediate_repair_success,
    repair_stats.total_repair_loops,
    repair_stats.total_time_spent_on_repair,
    repair_stats.total_time_spent_bounds_prop_after_repair,
    repair_stats.total_time_spent_on_bounds_prop);
  if (!sol_found) {
    sol.compute_feasibility();
    return false;
  }
  return sol.compute_feasibility();
}

template <typename i_t, typename f_t>
std::tuple<f_t, f_t, f_t> constraint_prop_t<i_t, f_t>::probing_values(
  const solution_t<i_t, f_t>& sol, const solution_t<i_t, f_t>& orig_sol, i_t idx)
{
  auto v_lb    = multi_probe.host_lb[idx];
  auto v_ub    = multi_probe.host_ub[idx];
  auto var_val = curr_host_assignment[idx];

  const f_t int_tol  = sol.problem_ptr->tolerances.integrality_tolerance;
  auto eps           = int_tol;
  auto within_bounds = (v_lb - eps <= var_val) && (var_val <= v_ub + eps);
  // if it is a collapsed var, return immediately one of the bounds
  if (orig_sol.problem_ptr->integer_equal(v_lb, v_ub)) {
    return std::make_tuple(v_lb, var_val, v_lb);
  }
  if (within_bounds) {
    // the value might have been brought within the bounds when it was out of bounds
    f_t first_round_val = round_nearest(var_val, v_lb, v_ub, int_tol, rng);
    f_t second_round_val;
    auto v_f = std::floor(first_round_val - int_tol);
    auto v_c = std::ceil(first_round_val + int_tol);
    if (first_round_val - var_val >= 0) {
      second_round_val = v_f;
      bool floor_within_bounds =
        (v_lb - eps <= second_round_val) && (second_round_val <= v_ub + eps);
      if (!floor_within_bounds) { second_round_val = v_c; }
    } else {
      second_round_val = v_c;
      bool ceil_within_bounds =
        (v_lb - eps <= second_round_val) && (second_round_val <= v_ub + eps);
      if (!ceil_within_bounds) { second_round_val = v_f; }
    }

    cuopt_assert(v_lb <= first_round_val && first_round_val <= v_ub, "probing value out of bounds");
    cuopt_assert(v_lb <= second_round_val && second_round_val <= v_ub,
                 "probing value out of bounds");
    return std::make_tuple(first_round_val, var_val, second_round_val);
  } else {
    auto orig_v_bnd =
      orig_sol.problem_ptr->variable_bounds.element(idx, sol.handle_ptr->get_stream());
    auto orig_v_lb = get_lower(orig_v_bnd);
    auto orig_v_ub = get_upper(orig_v_bnd);
    cuopt_assert(v_lb >= orig_v_lb, "Current lb should be greater than original lb");
    cuopt_assert(v_ub <= orig_v_ub, "Current ub should be smaller than original ub");
    v_lb = std::max(v_lb, orig_v_lb);
    v_ub = std::min(v_ub, orig_v_ub);
    // the bounds might cross, so correct them here
    if (v_lb > v_ub) {
      v_lb = orig_v_lb;
      v_ub = orig_v_lb;
    }
    auto v_f = std::floor(var_val);
    auto v_c = std::ceil(var_val);
    if (std::ceil(v_lb) == std::floor(v_ub)) {
      return std::make_tuple(std::ceil(v_lb), var_val, std::floor(v_ub));
    } else if (v_f < std::ceil(v_lb)) {
      v_f = std::ceil(v_lb);
      v_c = v_f + 1;
    } else if (v_c > std::floor(v_ub)) {
      v_c = std::floor(v_ub);
      v_f = v_c - 1;
    }
    cuopt_assert(orig_v_lb <= v_f && v_f <= orig_v_ub, "probing value out of bounds");
    cuopt_assert(orig_v_lb <= v_c && v_c <= orig_v_ub, "probing value out of bounds");
    return std::make_tuple(v_f, var_val, v_c);
  }
}

template <typename i_t, typename f_t>
bool constraint_prop_t<i_t, f_t>::handle_fixed_vars(
  solution_t<i_t, f_t>& sol,
  problem_t<i_t, f_t>* original_problem,
  const std::tuple<std::vector<i_t>, std::vector<f_t>, std::vector<f_t>>& var_probe_vals,
  size_t* set_count_ptr,
  rmm::device_uvector<i_t>& unset_vars)
{
  auto set_count    = *set_count_ptr;
  const f_t int_tol = sol.problem_ptr->tolerances.integrality_tolerance;
  // which other variables were affected?
  auto iter        = thrust::stable_partition(sol.handle_ptr->get_thrust_policy(),
                                       unset_vars.begin() + set_count,
                                       unset_vars.end(),
                                       is_bound_fixed_t<i_t, f_t, typename type_2<f_t>::type>{
                                         int_tol,
                                         make_span(sol.problem_ptr->variable_bounds),
                                         make_span(original_problem->variable_bounds),
                                         make_span(sol.assignment)});
  i_t n_fixed_vars = (iter - (unset_vars.begin() + set_count));
  cuopt_assert(n_fixed_vars >= std::get<0>(var_probe_vals).size(),
               "Error in number of vars fixed!");
  set_count += n_fixed_vars;
  CUOPT_LOG_TRACE("Set var count increased from %d to %d", *set_count_ptr, set_count);
  *set_count_ptr = set_count;
  return multi_probe.infeas_constraints_count_0 == 0 || multi_probe.infeas_constraints_count_1 == 0;
}

template <typename i_t, typename f_t>
bool constraint_prop_t<i_t, f_t>::probe(
  solution_t<i_t, f_t>& sol,
  problem_t<i_t, f_t>* original_problem,
  const std::tuple<std::vector<i_t>, std::vector<f_t>, std::vector<f_t>>& var_probe_vals,
  size_t* set_count_ptr,
  rmm::device_uvector<i_t>& unset_vars,
  std::optional<std::reference_wrapper<probing_config_t<i_t, f_t>>> probing_config)
{
  const bool use_host_bounds      = true;
  multi_probe.settings.time_limit = max_timer.remaining_time();
  multi_probe.solve(*sol.problem_ptr, var_probe_vals, use_host_bounds);
  // if we are ii at this point, backtrack the number of variables we have set in this given
  // interval then start setting one by one
  // if we determined that the rounding is ii then don't do any recovery and finish ronuding
  // quickly
  bool first_bounds_update_ii  = multi_probe.infeas_constraints_count_0 > 0;
  bool second_bounds_update_ii = multi_probe.infeas_constraints_count_1 > 0;
  // if we are on single rounding mode, direcly mark as ii so we can run repair
  if (!rounding_ii && single_rounding_only && (first_bounds_update_ii && second_bounds_update_ii)) {
    rounding_ii = true;
    return false;
  }
  if (!recovery_mode && !rounding_ii && (first_bounds_update_ii && second_bounds_update_ii) &&
      bounds_prop_interval != 1) {
    CUOPT_LOG_DEBUG(
      "Activating recovery mode in bounds prop: bounds_prop_interval %d n_ii cstr %d %d",
      bounds_prop_interval,
      multi_probe.infeas_constraints_count_0,
      multi_probe.infeas_constraints_count_1);
    // do backtracking
    recovery_mode                          = true;
    multi_probe.infeas_constraints_count_0 = 0;
    multi_probe.infeas_constraints_count_1 = 0;
    n_iter_in_recovery                     = 0;
    return false;
  }
  selected_update = 0;
  if (first_bounds_update_ii) { selected_update = 1; }
  // if we are doing single rounding
  if (probing_config.has_value() && probing_config.value().get().use_balanced_probing) {
    cuopt_assert(std::get<0>(var_probe_vals).size() == 1,
                 "Balanced probing must be used with single rounding");
    i_t var_idx = std::get<0>(var_probe_vals)[0];
    f_t value_chosen =
      selected_update ? std::get<1>(var_probe_vals)[0] : std::get<2>(var_probe_vals)[0];
    if (value_chosen == probing_config.value().get().probing_values[var_idx].first) {
      probing_config.value().get().n_of_fixed_from_first++;
    } else {
      probing_config.value().get().n_of_fixed_from_second++;
    }
    CUOPT_LOG_TRACE("Balanced probing: n_of_fixed_from_first %d n_of_fixed_from_second %d",
                    probing_config.value().get().n_of_fixed_from_first,
                    probing_config.value().get().n_of_fixed_from_second);
  }
  multi_probe.set_updated_bounds(*sol.problem_ptr, selected_update, sol.handle_ptr);
  return handle_fixed_vars(sol, original_problem, var_probe_vals, set_count_ptr, unset_vars);
}

#if MIP_INSTANTIATE_FLOAT
template class constraint_prop_t<int, float>;
#endif

#if MIP_INSTANTIATE_DOUBLE
template class constraint_prop_t<int, double>;
#endif

}  // namespace cuopt::linear_programming::detail
