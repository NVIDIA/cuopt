/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "../state.hpp"
namespace cuopt::mathematical_optimization::mip {

// An eliminated coordinate, recorded in original variable indices. Records are lifted in reverse.
template <typename i_t, typename f_t>
struct fj_equality_substitution_t {
  i_t variable;
  f_t constant;
  std::vector<std::pair<i_t, f_t>> terms;
};

template <typename i_t, typename f_t>
void detect_implied_integers(fj_cpu_climber_t<i_t, f_t>&, fj_cpu_problem_t<i_t, f_t>&);
template <typename i_t, typename f_t>
void detect_free_equality_singletons(fj_cpu_climber_t<i_t, f_t>&);
template <typename i_t, typename f_t>
void precompute_problem_features(fj_cpu_climber_t<i_t, f_t>&, fj_cpu_problem_t<i_t, f_t>&);
template <typename i_t, typename f_t>
void build_cardinality_index(fj_cpu_climber_t<i_t, f_t>&, fj_cpu_problem_t<i_t, f_t>&);
template <typename i_t, typename f_t>
void certify_epigraph_variables(fj_cpu_climber_t<i_t, f_t>&, i_t);
template <typename i_t, typename f_t>
void build_one_sided_rows(fj_cpu_climber_t<i_t, f_t>&);
template <typename i_t, typename f_t>
std::unique_ptr<fj_cpu_climber_t<i_t, f_t>> make_equality_reduced_climber(
  fj_cpu_climber_t<i_t, f_t>&,
  double,
  std::vector<fj_equality_substitution_t<i_t, f_t>>&,
  std::vector<i_t>&);
}  // namespace cuopt::mathematical_optimization::mip
