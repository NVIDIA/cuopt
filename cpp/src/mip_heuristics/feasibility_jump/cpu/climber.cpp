/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include "climber.hpp"
#include "internal.hpp"
#include "problem.hpp"
#include "search/api.hpp"
#include "setup/lp.hpp"
#include "setup/structure.hpp"

namespace cuopt::mathematical_optimization::mip {

template <typename i_t, typename f_t>
void init_fj_cpu_from_template(fj_cpu_climber_t<i_t, f_t>& fj_cpu,
                               const fj_cpu_climber_t<i_t, f_t>& tmpl,
                               const std::vector<f_t>& left_weights,
                               const std::vector<f_t>& right_weights,
                               f_t objective_weight)
{
  const i_t n_variables   = (i_t)tmpl.problem->reverse_offsets.size() - 1;
  const i_t n_constraints = (i_t)tmpl.problem->offsets.size() - 1;
  const i_t nnz           = (i_t)tmpl.problem->coefficients.size();

  cuopt_assert(n_variables == tmpl.problem->n_variables, "template variable count mismatch");
  cuopt_assert(n_constraints == tmpl.problem->n_constraints, "template constraint count mismatch");
  cuopt_assert(nnz == tmpl.problem->nnz, "template nnz mismatch");
  cuopt_assert(left_weights.size() == static_cast<size_t>(n_constraints),
               "left weight size mismatch");
  cuopt_assert(right_weights.size() == static_cast<size_t>(n_constraints),
               "right weight size mismatch");

  // Shared, not copied: read-only for the whole solve.
  fj_cpu.problem = tmpl.problem;
  fj_cpu.hp      = tmpl.hp;

  fj_cpu.h_initial_left_weights  = left_weights;
  fj_cpu.h_initial_right_weights = right_weights;
  fj_cpu.max_weight              = 1.0;
  fj_cpu.h_objective_weight      = objective_weight;
  fj_cpu.h_assignment            = tmpl.h_assignment;
  fj_cpu.h_best_assignment       = tmpl.h_assignment;
  fj_cpu.h_var_bounds            = tmpl.h_var_bounds;
  fj_cpu.h_is_binary_variable    = tmpl.h_is_binary_variable;
  fj_cpu.h_binary_indices        = tmpl.h_binary_indices;
  fj_cpu.n_binary_vars           = tmpl.n_binary_vars;
  fj_cpu.n_integer_vars          = tmpl.n_integer_vars;
  fj_cpu.h_tabu_nodec_until.resize(n_variables, 0);
  fj_cpu.h_tabu_noinc_until.resize(n_variables, 0);
  fj_cpu.h_tabu_lastdec.resize(n_variables, 0);
  fj_cpu.h_tabu_lastinc.resize(n_variables, 0);
  fj_cpu.iterations = 0;

  finalize_fj_cpu_host_initialization_from_template(
    fj_cpu, tmpl, n_variables, n_constraints, tmpl.n_integer_vars, nnz, tmpl.problem->tolerances);
}

template <typename i_t, typename f_t>
void set_host_data_view(fj_cpu_climber_t<i_t, f_t>& fj_cpu,
                        i_t n_variables,
                        i_t n_constraints,
                        i_t n_integer_vars,
                        i_t nnz,
                        const typename mip_solver_settings_t<i_t, f_t>::tolerances_t& tolerances)
{
  cuopt_assert(fj_cpu.problem->n_variables == n_variables, "problem variable count mismatch");
  cuopt_assert(fj_cpu.problem->n_constraints == n_constraints, "problem constraint count mismatch");
  cuopt_assert(fj_cpu.problem->nnz == nnz, "problem nonzero count mismatch");
  fj_cpu.row_tolerance  = tolerances.absolute_tolerance * (f_t)0.9;
  fj_cpu.n_integer_vars = n_integer_vars;
}

template <typename i_t, typename f_t>
void wire_fj_cpu_host_views(
  fj_cpu_climber_t<i_t, f_t>& fj_cpu,
  i_t n_variables,
  i_t n_constraints,
  i_t n_integer_vars,
  i_t nnz,
  const typename mip_solver_settings_t<i_t, f_t>::tolerances_t& tolerances)
{
  cuopt_assert(n_variables >= 0, "invalid variable count");
  cuopt_assert(n_constraints >= 0, "invalid constraint count");
  cuopt_assert(fj_cpu.problem->offsets.size() == static_cast<size_t>(n_constraints + 1),
               "invalid CSR offsets");
  cuopt_assert(fj_cpu.problem->reverse_offsets.size() == static_cast<size_t>(n_variables + 1),
               "invalid reverse offsets");
  cuopt_assert(fj_cpu.h_assignment.size() == static_cast<size_t>(n_variables),
               "start assignment size mismatch");

  set_host_data_view(fj_cpu, n_variables, n_constraints, n_integer_vars, nnz, tolerances);

  fj_cpu.h_best_objective = +std::numeric_limits<f_t>::infinity();

  // cached_mtm_moves, cached_mtm_moves_version and h_cstr_version are indexed by search row and
  // search nonzero, so build_one_sided_rows sizes them; nothing reads them before it runs.

  fj_cpu.flip_move_stamp.assign(n_variables, 0);
  fj_cpu.flip_move_epoch = 1;

  certify_epigraph_variables<i_t, f_t>(fj_cpu, n_variables);
}

template <typename i_t, typename f_t>
void finalize_fj_cpu_host_initialization(
  fj_cpu_climber_t<i_t, f_t>& fj_cpu,
  fj_cpu_problem_t<i_t, f_t>& problem,
  i_t n_variables,
  i_t n_constraints,
  i_t n_integer_vars,
  i_t nnz,
  const typename mip_solver_settings_t<i_t, f_t>::tolerances_t& tolerances)
{
  raft::common::nvtx::range scope("finalize_fj_cpu_host_initialization");
  cuopt_assert(fj_cpu.problem.get() == &problem, "mutable problem builder does not match climber");

  detect_implied_integers(fj_cpu, problem);
  wire_fj_cpu_host_views(fj_cpu, n_variables, n_constraints, n_integer_vars, nnz, tolerances);

  problem.h_objective_vars.resize(n_variables);
  auto end = std::copy_if(
    thrust::counting_iterator<i_t>(0),
    thrust::counting_iterator<i_t>(n_variables),
    problem.h_objective_vars.begin(),
    [&problem](i_t idx) { return !problem.integer_equal(problem.h_obj_coeffs[idx], (f_t)0); });
  problem.h_objective_vars.resize(end - problem.h_objective_vars.begin());
  // get_breakthrough_move divides by the coefficient of every variable in here.
  for ([[maybe_unused]] auto var_idx : problem.h_objective_vars) {
    cuopt_assert(problem.h_obj_coeffs[var_idx] != f_t{0}, "null coefficient in the objective vars");
    cuopt_assert(std::isfinite((f_t)problem.h_obj_coeffs[var_idx]),
                 "non-finite objective coefficient");
  }

  f_t abs_obj_sum = 0;
  for (auto var_idx : problem.h_objective_vars) {
    const f_t coeff = problem.h_obj_coeffs[var_idx];
    abs_obj_sum += coeff < 0 ? -coeff : coeff;
  }
  problem.obj_magnitude = abs_obj_sum > 0 ? abs_obj_sum / problem.h_objective_vars.size() : f_t{1};
  cuopt_assert(std::isfinite(problem.obj_magnitude) && problem.obj_magnitude > 0,
               "objective magnitude unit must be finite and positive");

  // Must precede recompute_lhs, which is what first populates them.
  fj_cpu.violated_constraints.resize(n_constraints);
  fj_cpu.satisfied_constraints.resize(n_constraints);

  {
    phase_timer_t timer(fj_cpu.t_init_lhs);
    recompute_lhs(fj_cpu);
  }
}

template <typename i_t, typename f_t>
static void initialize_climber_state(
  fj_cpu_climber_t<i_t, f_t>& fj_cpu,
  fj_cpu_problem_t<i_t, f_t>& problem,
  std::vector<f_t> assignment,
  i_t n_integer_vars,
  const typename mip_solver_settings_t<i_t, f_t>::tolerances_t& tolerances)
{
  const i_t n_variables   = problem.n_variables;
  const i_t n_constraints = problem.n_constraints;
  fj_cpu.h_initial_left_weights.resize(n_constraints, f_t{1});
  fj_cpu.h_initial_right_weights.resize(n_constraints, f_t{1});
  fj_cpu.max_weight         = f_t{1};
  fj_cpu.h_objective_weight = f_t{0};
  fj_cpu.h_assignment       = assignment;
  fj_cpu.h_best_assignment  = std::move(assignment);
  fj_cpu.h_lhs.resize(n_constraints);
  fj_cpu.h_lhs_sumcomp.resize(n_constraints, f_t{0});
  fj_cpu.h_tabu_nodec_until.resize(n_variables, 0);
  fj_cpu.h_tabu_noinc_until.resize(n_variables, 0);
  fj_cpu.h_tabu_lastdec.resize(n_variables, 0);
  fj_cpu.h_tabu_lastinc.resize(n_variables, 0);
  fj_cpu.iterations = 0;

  finalize_fj_cpu_host_initialization(
    fj_cpu, problem, n_variables, n_constraints, n_integer_vars, problem.nnz, tolerances);
}

template <typename i_t, typename f_t>
void finalize_fj_cpu_host_initialization_from_template(
  fj_cpu_climber_t<i_t, f_t>& fj_cpu,
  const fj_cpu_climber_t<i_t, f_t>& tmpl,
  i_t n_variables,
  i_t n_constraints,
  i_t n_integer_vars,
  i_t nnz,
  const typename mip_solver_settings_t<i_t, f_t>::tolerances_t& tolerances)
{
  raft::common::nvtx::range scope("finalize_fj_cpu_host_initialization_from_template");

  cuopt_assert(tmpl.h_lhs.size() == static_cast<size_t>(n_constraints), "template lhs mismatch");
  cuopt_assert(tmpl.violated_constraints.max_size() == n_constraints,
               "template violated set mismatch");
  cuopt_assert(tmpl.satisfied_constraints.max_size() == n_constraints,
               "template satisfied set mismatch");

  fj_cpu.h_lhs                    = tmpl.h_lhs;
  fj_cpu.h_lhs_sumcomp            = tmpl.h_lhs_sumcomp;
  fj_cpu.violated_constraints     = tmpl.violated_constraints;
  fj_cpu.satisfied_constraints    = tmpl.satisfied_constraints;
  fj_cpu.total_violations         = tmpl.total_violations;
  fj_cpu.total_violations_sumcomp = tmpl.total_violations_sumcomp;
  fj_cpu.h_incumbent_objective    = tmpl.h_incumbent_objective;
  fj_cpu.h_objective_sumcomp      = tmpl.h_objective_sumcomp;

  fj_cpu.bin_eliminated_rows = tmpl.bin_eliminated_rows;
  fj_cpu.bin_singletons      = tmpl.bin_singletons;
  fj_cpu.bin_ignore_row      = tmpl.bin_ignore_row;
  fj_cpu.bin_ignore_var      = tmpl.bin_ignore_var;
  fj_cpu.has_bin_elimination = tmpl.has_bin_elimination;

  wire_fj_cpu_host_views(fj_cpu, n_variables, n_constraints, n_integer_vars, nnz, tolerances);
}

template <typename i_t, typename f_t>
std::unique_ptr<fj_cpu_climber_t<i_t, f_t>> init_fj_cpu_from_host_lp(
  const lp_problem_t<i_t, f_t>& problem,
  const std::vector<variable_type_t>& variable_types,
  i_t n_structural,
  const std::vector<f_t>& start_assignment,
  const simplex_solver_settings_t<i_t, f_t>& settings,
  std::atomic<bool>& preemption_flag,
  int64_t seed)
{
  using f_t2 = typename type_2<f_t>::type;

  cuopt_assert(variable_types.size() >= static_cast<size_t>(problem.num_cols),
               "variable type size mismatch");

  typename mip_solver_settings_t<i_t, f_t>::tolerances_t tolerances{};
  tolerances.absolute_tolerance    = settings.primal_tol;
  tolerances.relative_tolerance    = settings.zero_tol;
  tolerances.integrality_tolerance = settings.integer_tol;
  tolerances.absolute_mip_gap      = settings.absolute_mip_gap_tol;
  tolerances.relative_mip_gap      = settings.relative_mip_gap_tol;

  const i_t n_constraints = problem.num_rows;

  csr_matrix_t<i_t, f_t> csr_A(problem.num_rows, problem.num_cols, problem.A.nnz());
  problem.A.to_compressed_row(csr_A);

  std::vector<f_t> constraint_lower_bounds;
  std::vector<f_t> constraint_upper_bounds;
  i_t n_variables;
  if (n_structural > 0 && n_structural < problem.num_cols) {
    eliminate_slacks(
      problem, n_structural, csr_A, constraint_lower_bounds, constraint_upper_bounds);
    n_variables = n_structural;
  } else {
    n_variables = problem.num_cols;
    // Standard form: every row is an equality.
    constraint_lower_bounds = problem.rhs;
    constraint_upper_bounds = problem.rhs;
  }

  std::vector<f_t> coefficients = csr_A.x;
  std::vector<i_t> variables    = csr_A.j;
  std::vector<i_t> offsets      = csr_A.row_start;
  std::vector<f_t2> variable_bounds(n_variables);
  std::vector<var_t> cpufj_variable_types(n_variables);
  std::vector<i_t> is_binary_variable(n_variables, 0);
  i_t n_integer_vars = 0;

  for (i_t j = 0; j < n_variables; ++j) {
    variable_bounds[j]  = f_t2{problem.lower[j], problem.upper[j]};
    const auto var_type = variable_types[j];
    cpufj_variable_types[j] =
      var_type == variable_type_t::CONTINUOUS ? var_t::CONTINUOUS : var_t::INTEGER;

    const bool is_integer = cpufj_variable_types[j] == var_t::INTEGER;
    const bool is_binary  = is_integer &&
                           std::abs(problem.lower[j] - f_t{0}) <= settings.integer_tol &&
                           std::abs(problem.upper[j] - f_t{1}) <= settings.integer_tol;
    if (is_integer) { ++n_integer_vars; }
    if (is_binary) { is_binary_variable[j] = 1; }
  }

  const i_t nnz = static_cast<i_t>(variables.size());
  csc_matrix_t<i_t, f_t> reverse_csc(n_constraints, n_variables, nnz);
  csr_A.to_compressed_col(reverse_csc);
  std::vector<f_t> reverse_coefficients = std::move(reverse_csc.x);
  std::vector<i_t> reverse_constraints  = std::move(reverse_csc.i);
  std::vector<i_t> reverse_offsets      = std::move(reverse_csc.col_start);

  std::vector<f_t> projected_start(n_variables, f_t{0});
  for (i_t j = 0; j < n_variables; ++j) {
    f_t value = j < static_cast<i_t>(start_assignment.size()) ? start_assignment[j] : f_t{0};
    value     = std::clamp(value, problem.lower[j], problem.upper[j]);
    if (variable_types[j] != variable_type_t::CONTINUOUS) {
      value = std::clamp(std::round(value), problem.lower[j], problem.upper[j]);
    }
    projected_start[j] = value;
  }

  fj_settings_t fj_settings;
  fj_settings.mode                   = fj_mode_t::EXIT_NON_IMPROVING;
  fj_settings.n_of_minimums_for_exit = std::numeric_limits<int>::max();
  fj_settings.time_limit             = std::numeric_limits<f_t>::infinity();
  fj_settings.iteration_limit        = std::numeric_limits<int>::max();
  fj_settings.update_weights         = true;
  fj_settings.feasibility_run        = false;
  fj_settings.seed                   = seed >= 0 ? seed : cuopt::seed_generator::get_seed();

  auto fj_cpu                 = std::make_unique<fj_cpu_climber_t<i_t, f_t>>(preemption_flag);
  fj_cpu->settings            = fj_settings;
  auto problem_data           = std::make_shared<fj_cpu_problem_t<i_t, f_t>>();
  fj_cpu->problem             = problem_data;
  problem_data->tolerances    = tolerances;
  problem_data->n_variables   = n_variables;
  problem_data->n_constraints = n_constraints;
  problem_data->nnz           = nnz;
  problem_data->objective_scaling_factor = problem.obj_scale;
  problem_data->objective_offset         = problem.obj_constant;

  problem_data->reverse_coefficients = std::move(reverse_coefficients);
  problem_data->reverse_constraints  = std::move(reverse_constraints);
  problem_data->reverse_offsets      = std::move(reverse_offsets);
  problem_data->coefficients         = std::move(coefficients);
  problem_data->offsets              = std::move(offsets);
  problem_data->variables            = std::move(variables);
  problem_data->h_obj_coeffs =
    std::vector<f_t>(problem.objective.begin(), problem.objective.begin() + n_variables);
  fj_cpu->h_var_bounds         = std::move(variable_bounds);
  problem_data->cstr_lb        = std::move(constraint_lower_bounds);
  problem_data->cstr_ub        = std::move(constraint_upper_bounds);
  problem_data->h_var_types    = std::move(cpufj_variable_types);
  fj_cpu->h_is_binary_variable = std::move(is_binary_variable);

  initialize_climber_state(
    *fj_cpu, *problem_data, std::move(projected_start), n_integer_vars, tolerances);
  return fj_cpu;
}

template <typename i_t, typename f_t>
std::unique_ptr<fj_cpu_climber_t<i_t, f_t>> init_fj_cpu_from_host_model(
  i_t n_variables,
  i_t n_constraints,
  i_t nnz,
  bool maximize,
  f_t objective_scaling_factor,
  f_t objective_offset,
  std::vector<f_t> coefficients,
  std::vector<i_t> variables,
  std::vector<i_t> offsets,
  std::vector<f_t> objective_coefficients,
  std::vector<f_t> variable_lower_bounds,
  std::vector<f_t> variable_upper_bounds,
  std::vector<f_t> constraint_lower_bounds,
  std::vector<f_t> constraint_upper_bounds,
  std::vector<f_t> constraint_bounds,
  std::vector<char> row_types,
  std::vector<var_t> variable_types,
  const typename mip_solver_settings_t<i_t, f_t>::tolerances_t& tolerances,
  std::atomic<bool>& preemption_flag,
  fj_settings_t settings)
{
  using f_t2 = typename type_2<f_t>::type;

  raft::common::nvtx::range scope("init_fj_cpu_from_host_model");

  cuopt_assert(coefficients.size() == (size_t)nnz, "coefficient size mismatch");
  cuopt_assert(variables.size() == (size_t)nnz, "variable index size mismatch");
  cuopt_assert(offsets.size() == (size_t)(n_constraints + 1), "constraint offset size mismatch");
  cuopt_assert(!offsets.empty() && offsets.front() == 0, "invalid first constraint offset");
  cuopt_assert(offsets.back() == nnz, "invalid final constraint offset");
  cuopt_assert(std::is_sorted(offsets.begin(), offsets.end()), "unsorted constraint offsets");
  cuopt_assert(
    std::all_of(variables.begin(),
                variables.end(),
                [n_variables](i_t variable) { return variable >= 0 && variable < n_variables; }),
    "variable index out of range");
  cuopt_assert(objective_coefficients.size() == (size_t)n_variables, "objective size mismatch");
  cuopt_assert(variable_lower_bounds.empty() || variable_lower_bounds.size() == (size_t)n_variables,
               "variable lower bound size mismatch");
  cuopt_assert(variable_upper_bounds.empty() || variable_upper_bounds.size() == (size_t)n_variables,
               "variable upper bound size mismatch");

  if (constraint_lower_bounds.empty() && constraint_upper_bounds.empty()) {
    cuopt_assert(row_types.size() == (size_t)n_constraints, "row type size mismatch");
    cuopt_assert(constraint_bounds.size() == (size_t)n_constraints,
                 "constraint bound size mismatch");
    constraint_lower_bounds.resize(n_constraints);
    constraint_upper_bounds.resize(n_constraints);
    for (i_t row = 0; row < n_constraints; ++row) {
      const f_t bound = constraint_bounds[row];
      if (row_types[row] == 'E') {
        constraint_lower_bounds[row] = bound;
        constraint_upper_bounds[row] = bound;
      } else if (row_types[row] == 'G') {
        constraint_lower_bounds[row] = bound;
        constraint_upper_bounds[row] = std::numeric_limits<f_t>::infinity();
      } else {
        cuopt_assert(row_types[row] == 'L', "invalid row type");
        constraint_lower_bounds[row] = -std::numeric_limits<f_t>::infinity();
        constraint_upper_bounds[row] = bound;
      }
    }
  } else {
    cuopt_assert(constraint_lower_bounds.size() == (size_t)n_constraints,
                 "constraint lower bound size mismatch");
    cuopt_assert(constraint_upper_bounds.size() == (size_t)n_constraints,
                 "constraint upper bound size mismatch");
  }

  if (variable_lower_bounds.empty()) { variable_lower_bounds.assign(n_variables, f_t{0}); }
  if (variable_upper_bounds.empty()) {
    variable_upper_bounds.assign(n_variables, std::numeric_limits<f_t>::infinity());
  }
  if (variable_types.empty()) { variable_types.assign(n_variables, var_t::CONTINUOUS); }
  cuopt_assert(variable_types.size() == (size_t)n_variables, "variable type size mismatch");

  if (maximize) {
    std::transform(objective_coefficients.begin(),
                   objective_coefficients.end(),
                   objective_coefficients.begin(),
                   std::negate<f_t>{});
  }

  std::vector<f_t2> variable_bounds(n_variables);
  std::vector<i_t> is_binary_variable(n_variables, 0);
  std::vector<i_t> binary_indices;
  binary_indices.reserve(n_variables);
  i_t n_integer_vars = 0;
  for (i_t variable = 0; variable < n_variables; ++variable) {
    f_t lower             = variable_lower_bounds[variable];
    f_t upper             = variable_upper_bounds[variable];
    const bool is_integer = variable_types[variable] == var_t::INTEGER;
    if (is_integer) {
      lower = std::ceil(lower);
      upper = std::floor(upper);
      ++n_integer_vars;
    }
    cuopt_assert(lower <= upper, "crossing variable bounds");
    variable_bounds[variable] = f_t2{lower, upper};
    if (is_integer && lower == f_t{0} && upper == f_t{1}) {
      is_binary_variable[variable] = 1;
      binary_indices.push_back(variable);
    }
  }

  csr_matrix_t<i_t, f_t> csr(n_constraints, n_variables, nnz);
  csr.x         = coefficients;
  csr.j         = variables;
  csr.row_start = offsets;
  csc_matrix_t<i_t, f_t> csc(n_constraints, n_variables, nnz);
  csr.to_compressed_col(csc);

  std::vector<f_t> assignment(n_variables, f_t{0});
  for (i_t variable = 0; variable < n_variables; ++variable) {
    f_t value = std::clamp(
      f_t{0}, get_lower(variable_bounds[variable]), get_upper(variable_bounds[variable]));
    if (variable_types[variable] == var_t::INTEGER) { value = std::round(value); }
    assignment[variable] = value;
  }

  auto fj_cpu                 = std::make_unique<fj_cpu_climber_t<i_t, f_t>>(preemption_flag);
  fj_cpu->settings            = settings;
  auto problem_data           = std::make_shared<fj_cpu_problem_t<i_t, f_t>>();
  fj_cpu->problem             = problem_data;
  problem_data->tolerances    = tolerances;
  problem_data->n_variables   = n_variables;
  problem_data->n_constraints = n_constraints;
  problem_data->nnz           = nnz;
  problem_data->objective_scaling_factor =
    maximize ? -objective_scaling_factor : objective_scaling_factor;
  problem_data->objective_offset = maximize ? -objective_offset : objective_offset;

  problem_data->reverse_coefficients = std::move(csc.x);
  problem_data->reverse_constraints  = std::move(csc.i);
  problem_data->reverse_offsets      = std::move(csc.col_start);
  problem_data->coefficients         = std::move(coefficients);
  problem_data->offsets              = std::move(offsets);
  problem_data->variables            = std::move(variables);
  problem_data->h_obj_coeffs         = std::move(objective_coefficients);
  fj_cpu->h_var_bounds               = std::move(variable_bounds);
  problem_data->cstr_lb              = std::move(constraint_lower_bounds);
  problem_data->cstr_ub              = std::move(constraint_upper_bounds);
  problem_data->h_var_types          = std::move(variable_types);
  fj_cpu->h_is_binary_variable       = std::move(is_binary_variable);
  fj_cpu->h_binary_indices           = std::move(binary_indices);

  initialize_climber_state(
    *fj_cpu, *problem_data, std::move(assignment), n_integer_vars, tolerances);
  return fj_cpu;
}

template <typename i_t, typename f_t>
std::unique_ptr<fj_cpu_climber_t<i_t, f_t>> init_fj_cpu_clone(
  const fj_cpu_climber_t<i_t, f_t>& tmpl,
  std::atomic<bool>& preemption_flag,
  fj_settings_t settings)
{
  raft::common::nvtx::range scope("init_fj_cpu_clone");

  auto fj_cpu = std::make_unique<fj_cpu_climber_t<i_t, f_t>>(preemption_flag);

  std::vector<f_t> default_weights(tmpl.problem->n_constraints, 1.0);
  init_fj_cpu_from_template(*fj_cpu, tmpl, default_weights, default_weights, f_t{0});
  fj_cpu->settings = settings;

  return fj_cpu;
}

#if MIP_INSTANTIATE_FLOAT
template std::unique_ptr<fj_cpu_climber_t<int, float>> init_fj_cpu_from_host_lp(
  const lp_problem_t<int, float>&,
  const std::vector<variable_type_t>&,
  int,
  const std::vector<float>&,
  const simplex_solver_settings_t<int, float>&,
  std::atomic<bool>&,
  int64_t);
template std::unique_ptr<fj_cpu_climber_t<int, float>> init_fj_cpu_from_host_model(
  int,
  int,
  int,
  bool,
  float,
  float,
  std::vector<float>,
  std::vector<int>,
  std::vector<int>,
  std::vector<float>,
  std::vector<float>,
  std::vector<float>,
  std::vector<float>,
  std::vector<float>,
  std::vector<float>,
  std::vector<char>,
  std::vector<var_t>,
  const typename mip_solver_settings_t<int, float>::tolerances_t&,
  std::atomic<bool>&,
  fj_settings_t);
template void finalize_fj_cpu_host_initialization(
  fj_cpu_climber_t<int, float>&,
  fj_cpu_problem_t<int, float>&,
  int,
  int,
  int,
  int,
  const typename mip_solver_settings_t<int, float>::tolerances_t&);
template std::unique_ptr<fj_cpu_climber_t<int, float>> init_fj_cpu_clone(
  const fj_cpu_climber_t<int, float>&, std::atomic<bool>&, fj_settings_t);
#endif

#if MIP_INSTANTIATE_DOUBLE
template std::unique_ptr<fj_cpu_climber_t<int, double>> init_fj_cpu_from_host_lp(
  const lp_problem_t<int, double>&,
  const std::vector<variable_type_t>&,
  int,
  const std::vector<double>&,
  const simplex_solver_settings_t<int, double>&,
  std::atomic<bool>&,
  int64_t);
template std::unique_ptr<fj_cpu_climber_t<int, double>> init_fj_cpu_from_host_model(
  int,
  int,
  int,
  bool,
  double,
  double,
  std::vector<double>,
  std::vector<int>,
  std::vector<int>,
  std::vector<double>,
  std::vector<double>,
  std::vector<double>,
  std::vector<double>,
  std::vector<double>,
  std::vector<double>,
  std::vector<char>,
  std::vector<var_t>,
  const typename mip_solver_settings_t<int, double>::tolerances_t&,
  std::atomic<bool>&,
  fj_settings_t);
template void finalize_fj_cpu_host_initialization(
  fj_cpu_climber_t<int, double>&,
  fj_cpu_problem_t<int, double>&,
  int,
  int,
  int,
  int,
  const typename mip_solver_settings_t<int, double>::tolerances_t&);
template std::unique_ptr<fj_cpu_climber_t<int, double>> init_fj_cpu_clone(
  const fj_cpu_climber_t<int, double>&, std::atomic<bool>&, fj_settings_t);
#endif

}  // namespace cuopt::mathematical_optimization::mip
