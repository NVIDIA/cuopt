# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved. # noqa
# SPDX-License-Identifier: Apache-2.0


# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from libcpp cimport bool
from libcpp.pair cimport pair
from libcpp.string cimport string
from libcpp.vector cimport vector

from pylibraft.common.handle cimport *
from rmm.librmm.device_buffer cimport device_buffer

from cuopt.linear_programming.data_model.data_model cimport data_model_view_t
from cuopt.linear_programming.solver_settings.solver_settings cimport (
    method_t,
    pdlp_solver_mode_t,
    solver_settings_t,
)


cdef extern from "cuopt/mathematical_optimization/optimization_problem.hpp" namespace "cuopt::mathematical_optimization": # noqa
    ctypedef enum problem_category_t "cuopt::mathematical_optimization::problem_category_t": # noqa
        LP "cuopt::mathematical_optimization::problem_category_t::LP"
        MIP "cuopt::mathematical_optimization::problem_category_t::MIP"
        IP "cuopt::mathematical_optimization::problem_category_t::IP"

cdef extern from "cuopt/error.hpp" namespace "cuopt": # noqa
    ctypedef enum error_type_t "cuopt::error_type_t": # noqa
        Success "cuopt::error_type_t::Success" # noqa
        ValidationError "cuopt::error_type_t::ValidationError" # noqa
        OutOfMemoryError "cuopt::error_type_t::OutOfMemoryError" # noqa
        RuntimeError "cuopt::error_type_t::RuntimeError" # noqa

cdef extern from "cuopt/mathematical_optimization/mip/solver_solution.hpp" namespace "cuopt::mathematical_optimization": # noqa
    ctypedef enum mip_termination_status_t "cuopt::mathematical_optimization::mip_termination_status_t": # noqa
        NoTermination "cuopt::mathematical_optimization::mip_termination_status_t::NoTermination" # noqa
        Optimal "cuopt::mathematical_optimization::mip_termination_status_t::Optimal"
        FeasibleFound "cuopt::mathematical_optimization::mip_termination_status_t::FeasibleFound" # noqa
        Infeasible "cuopt::mathematical_optimization::mip_termination_status_t::Infeasible" # noqa
        Unbounded "cuopt::mathematical_optimization::mip_termination_status_t::Unbounded" # noqa
        TimeLimit "cuopt::mathematical_optimization::mip_termination_status_t::TimeLimit" # noqa
        WorkLimit "cuopt::mathematical_optimization::mip_termination_status_t::WorkLimit" # noqa
        UnboundedOrInfeasible "cuopt::mathematical_optimization::mip_termination_status_t::UnboundedOrInfeasible" # noqa


cdef extern from "cuopt/mathematical_optimization/pdlp/solver_solution.hpp" namespace "cuopt::mathematical_optimization": # noqa
    ctypedef enum pdlp_termination_status_t "cuopt::mathematical_optimization::pdlp_termination_status_t": # noqa
        NoTermination "cuopt::mathematical_optimization::pdlp_termination_status_t::NoTermination" # noqa
        NumericalError "cuopt::mathematical_optimization::pdlp_termination_status_t::NumericalError" # noqa
        Optimal "cuopt::mathematical_optimization::pdlp_termination_status_t::Optimal" # noqa
        PrimalInfeasible "cuopt::mathematical_optimization::pdlp_termination_status_t::PrimalInfeasible" # noqa
        DualInfeasible "cuopt::mathematical_optimization::pdlp_termination_status_t::DualInfeasible" # noqa
        IterationLimit "cuopt::mathematical_optimization::pdlp_termination_status_t::IterationLimit" # noqa
        TimeLimit "cuopt::mathematical_optimization::pdlp_termination_status_t::TimeLimit" # noqa
        ConcurrentLimit "cuopt::mathematical_optimization::pdlp_termination_status_t::ConcurrentLimit" # noqa
        PrimalFeasible "cuopt::mathematical_optimization::pdlp_termination_status_t::PrimalFeasible" # noqa
        UnboundedOrInfeasible "cuopt::mathematical_optimization::pdlp_termination_status_t::UnboundedOrInfeasible" # noqa


cdef extern from "cuopt/mathematical_optimization/utilities/cython_types.hpp" namespace "cuopt::cython": # noqa
    # Inner struct types for LP solution vectors (GPU backend)
    cdef cppclass lp_gpu_solutions_t "cuopt::cython::linear_programming_ret_t::gpu_solutions_t": # noqa
        unique_ptr[device_buffer] primal_solution_
        unique_ptr[device_buffer] dual_solution_
        unique_ptr[device_buffer] reduced_cost_
        unique_ptr[device_buffer] current_primal_solution_
        unique_ptr[device_buffer] current_dual_solution_
        unique_ptr[device_buffer] initial_primal_average_
        unique_ptr[device_buffer] initial_dual_average_
        unique_ptr[device_buffer] current_ATY_
        unique_ptr[device_buffer] sum_primal_solutions_
        unique_ptr[device_buffer] sum_dual_solutions_
        unique_ptr[device_buffer] last_restart_duality_gap_primal_solution_
        unique_ptr[device_buffer] last_restart_duality_gap_dual_solution_

    # Inner struct types for LP solution vectors (CPU backend)
    cdef cppclass lp_cpu_solutions_t "cuopt::cython::linear_programming_ret_t::cpu_solutions_t": # noqa
        vector[double] primal_solution_
        vector[double] dual_solution_
        vector[double] reduced_cost_
        vector[double] current_primal_solution_
        vector[double] current_dual_solution_
        vector[double] initial_primal_average_
        vector[double] initial_dual_average_
        vector[double] current_ATY_
        vector[double] sum_primal_solutions_
        vector[double] sum_dual_solutions_
        vector[double] last_restart_duality_gap_primal_solution_
        vector[double] last_restart_duality_gap_dual_solution_

cdef extern from "cuopt/mathematical_optimization/utilities/cython_solve.hpp" namespace "cuopt::cython": # noqa
    # Unified LP solution struct — solutions_ variant accessed via helpers
    cdef cppclass linear_programming_ret_t:
        # PDLP warm start scalars
        double initial_primal_weight_
        double initial_step_size_
        int total_pdlp_iterations_
        int total_pdhg_iterations_
        double last_candidate_kkt_score_
        double last_restart_kkt_score_
        double sum_solution_weight_
        int iterations_since_last_restart_
        # Termination metadata
        pdlp_termination_status_t termination_status_
        error_type_t error_status_
        string error_message_
        double l2_primal_residual_
        double l2_dual_residual_
        double primal_objective_
        double dual_objective_
        double gap_
        int nb_iterations_
        double solve_time_
        method_t solved_by_
        bool is_gpu()

    # Unified MIP solution struct — solution_ variant accessed via helpers
    cdef cppclass mip_ret_t:
        mip_termination_status_t termination_status_
        error_type_t error_status_
        string error_message_
        double objective_
        double mip_gap_
        double solution_bound_
        double total_solve_time_
        double presolve_time_
        double max_constraint_violation_
        double max_int_violation_
        double max_variable_bound_violation_
        int nodes_
        int simplex_iterations_
        bool is_gpu()

    cdef cppclass solver_ret_t:
        problem_category_t problem_type
        linear_programming_ret_t lp_ret
        mip_ret_t mip_ret

    cdef unique_ptr[solver_ret_t] call_solve(
        data_model_view_t[int, double]* data_model,
        solver_settings_t[int, double]* solver_settings,
    ) except + nogil

    cdef pair[vector[unique_ptr[solver_ret_t]], double] call_batch_solve( # noqa
        vector[data_model_view_t[int, double] *] data_models,
        solver_settings_t[int, double]* solver_settings,
    ) except + nogil

# Variant helper functions — Cython can't call std::get directly, so we use
# inline C++ helpers to extract the GPU/CPU alternatives from inner variants.
cdef extern from *:
    """
    #include <variant>
    #include <cuopt/mathematical_optimization/utilities/cython_solve.hpp>

    // MIP: extract GPU (unique_ptr<device_buffer>) or CPU (vector<double>) solution
    inline std::unique_ptr<rmm::device_buffer>& get_gpu_mip_solution(cuopt::cython::mip_ret_t& m) {
        return std::get<cuopt::cython::gpu_buffer>(m.solution_);
    }
    inline std::vector<double>& get_cpu_mip_solution(cuopt::cython::mip_ret_t& m) {
        return std::get<cuopt::cython::cpu_buffer>(m.solution_);
    }

    // LP: extract GPU (gpu_solutions_t) or CPU (cpu_solutions_t) solution struct
    inline cuopt::cython::linear_programming_ret_t::gpu_solutions_t&
    get_gpu_lp_solutions(cuopt::cython::linear_programming_ret_t& lp) {
        return std::get<cuopt::cython::linear_programming_ret_t::gpu_solutions_t>(lp.solutions_);
    }
    inline cuopt::cython::linear_programming_ret_t::cpu_solutions_t&
    get_cpu_lp_solutions(cuopt::cython::linear_programming_ret_t& lp) {
        return std::get<cuopt::cython::linear_programming_ret_t::cpu_solutions_t>(lp.solutions_);
    }
    """
    cdef unique_ptr[device_buffer]& get_gpu_mip_solution(mip_ret_t& m)
    cdef vector[double]& get_cpu_mip_solution(mip_ret_t& m)
    cdef lp_gpu_solutions_t& get_gpu_lp_solutions(linear_programming_ret_t& lp)
    cdef lp_cpu_solutions_t& get_cpu_lp_solutions(linear_programming_ret_t& lp)
