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


cdef extern from "cuopt/linear_programming/utilities/internals.hpp" namespace "cuopt::internals": # noqa
    cdef cppclass base_solution_callback_t

cdef extern from "cuopt/linear_programming/pdlp/solver_settings.hpp" namespace "cuopt::linear_programming": # noqa
    ctypedef enum pdlp_solver_mode_t "cuopt::linear_programming::pdlp_solver_mode_t": # noqa
        Stable1 "cuopt::linear_programming::pdlp_solver_mode_t::Stable1" # noqa
        Stable2 "cuopt::linear_programming::pdlp_solver_mode_t::Stable2" # noqa
        Methodical1 "cuopt::linear_programming::pdlp_solver_mode_t::Methodical1" # noqa
        Fast1 "cuopt::linear_programming::pdlp_solver_mode_t::Fast1" # noqa
        Stable3 "cuopt::linear_programming::pdlp_solver_mode_t::Stable3" # noqa

cdef extern from "cuopt/linear_programming/solver_settings.hpp" namespace "cuopt::linear_programming": # noqa

    cdef cppclass solver_settings_t[i_t, f_t]:
        solver_settings_t() except +

        void set_pdlp_warm_start_data(
            const f_t* current_primal_solution,
            const f_t* current_dual_solution,
            const f_t* initial_primal_average,
            const f_t* initial_dual_average,
            const f_t* current_ATY,
            const f_t* sum_primal_solutions,
            const f_t* sum_dual_solutions,
            const f_t* last_restart_duality_gap_primal_solution,
            const f_t* last_restart_duality_gap_dual_solution,
            i_t primal_size,
            i_t dual_size,
            f_t initial_primal_weight_,
            f_t initial_step_size_,
            i_t total_pdlp_iterations_,
            i_t total_pdhg_iterations_,
            f_t last_candidate_kkt_score_,
            f_t last_restart_kkt_score_,
            f_t sum_solution_weight_,
            i_t iterations_since_last_restart_) except +

        void set_parameter_from_string(
            const string& name,
            const string& value
        ) except +

        string get_parameter_as_string(const string& name) except +

        vector[string] get_parameter_names() except +

        # LP settings
        void set_initial_pdlp_primal_solution(
            const f_t* initial_primal_solution,
            i_t size
        ) except +
        void set_initial_pdlp_dual_solution(
            const f_t* initial_dual_solution,
            i_t size
        ) except +

        # MIP settings
        void add_initial_mip_solution(
            const f_t* initial_solution,
            i_t size
        ) except +
        void set_mip_callback(
            base_solution_callback_t* callback,
            void* user_data
        ) except +


cdef extern from "cuopt/linear_programming/optimization_problem.hpp" namespace "cuopt::linear_programming": # noqa
    ctypedef enum problem_category_t "cuopt::linear_programming::problem_category_t": # noqa
        LP "cuopt::linear_programming::problem_category_t::LP"
        MIP "cuopt::linear_programming::problem_category_t::MIP"
        IP "cuopt::linear_programming::problem_category_t::IP"

cdef extern from "cuopt/error.hpp" namespace "cuopt": # noqa
    ctypedef enum error_type_t "cuopt::error_type_t": # noqa
        Success "cuopt::error_type_t::Success" # noqa
        ValidationError "cuopt::error_type_t::ValidationError" # noqa
        OutOfMemoryError "cuopt::error_type_t::OutOfMemoryError" # noqa
        RuntimeError "cuopt::error_type_t::RuntimeError" # noqa

cdef extern from "cuopt/linear_programming/mip/solver_solution.hpp" namespace "cuopt::linear_programming": # noqa
    ctypedef enum mip_termination_status_t "cuopt::linear_programming::mip_termination_status_t": # noqa
        NoTermination "cuopt::linear_programming::mip_termination_status_t::NoTermination" # noqa
        Optimal "cuopt::linear_programming::mip_termination_status_t::Optimal"
        FeasibleFound "cuopt::linear_programming::mip_termination_status_t::FeasibleFound" # noqa
        Infeasible "cuopt::linear_programming::mip_termination_status_t::Infeasible" # noqa
        Unbounded "cuopt::linear_programming::mip_termination_status_t::Unbounded" # noqa
        TimeLimit "cuopt::linear_programming::mip_termination_status_t::TimeLimit" # noqa


cdef extern from "cuopt/linear_programming/pdlp/solver_solution.hpp" namespace "cuopt::linear_programming": # noqa
    ctypedef enum pdlp_termination_status_t "cuopt::linear_programming::pdlp_termination_status_t": # noqa
        NoTermination "cuopt::linear_programming::pdlp_termination_status_t::NoTermination" # noqa
        NumericalError "cuopt::linear_programming::pdlp_termination_status_t::NumericalError" # noqa
        Optimal "cuopt::linear_programming::pdlp_termination_status_t::Optimal" # noqa
        PrimalInfeasible "cuopt::linear_programming::pdlp_termination_status_t::PrimalInfeasible" # noqa
        DualInfeasible "cuopt::linear_programming::pdlp_termination_status_t::DualInfeasible" # noqa
        IterationLimit "cuopt::linear_programming::pdlp_termination_status_t::IterationLimit" # noqa
        TimeLimit "cuopt::linear_programming::pdlp_termination_status_t::TimeLimit" # noqa
        ConcurrentLimit "cuopt::linear_programming::pdlp_termination_status_t::ConcurrentLimit" # noqa
        PrimalFeasible "cuopt::linear_programming::pdlp_termination_status_t::PrimalFeasible" # noqa


cdef extern from "<variant>" namespace "std":
    # Declare std::variant support
    cdef cppclass variant[T1, T2]:
        variant() except +
        variant(T1&) except +
        variant(T2&) except +

cdef extern from "cuopt/linear_programming/utilities/cython_solve.hpp" namespace "cuopt::cython": # noqa
    # GPU-backed LP solution struct (device memory)
    cdef cppclass linear_programming_ret_t:
        unique_ptr[device_buffer] primal_solution_
        unique_ptr[device_buffer] dual_solution_
        unique_ptr[device_buffer] reduced_cost_
        # PDLP warm start data
        unique_ptr[device_buffer] current_primal_solution_
        unique_ptr[device_buffer] current_dual_solution_
        unique_ptr[device_buffer] initial_primal_average_
        unique_ptr[device_buffer] initial_dual_average_
        unique_ptr[device_buffer] current_ATY_
        unique_ptr[device_buffer] sum_primal_solutions_
        unique_ptr[device_buffer] sum_dual_solutions_
        unique_ptr[device_buffer] last_restart_duality_gap_primal_solution_
        unique_ptr[device_buffer] last_restart_duality_gap_dual_solution_
        double initial_primal_weight_
        double initial_step_size_
        int total_pdlp_iterations_
        int total_pdhg_iterations_
        double last_candidate_kkt_score_
        double last_restart_kkt_score_
        double sum_solution_weight_
        int iterations_since_last_restart_
        # /PDLP warm start data
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
        bool solved_by_pdlp_

    # CPU-backed LP solution struct (host memory)
    cdef cppclass cpu_linear_programming_ret_t:
        vector[double] primal_solution_
        vector[double] dual_solution_
        vector[double] reduced_cost_
        # PDLP warm start data
        vector[double] current_primal_solution_
        vector[double] current_dual_solution_
        vector[double] initial_primal_average_
        vector[double] initial_dual_average_
        vector[double] current_ATY_
        vector[double] sum_primal_solutions_
        vector[double] sum_dual_solutions_
        vector[double] last_restart_duality_gap_primal_solution_
        vector[double] last_restart_duality_gap_dual_solution_
        double initial_primal_weight_
        double initial_step_size_
        int total_pdlp_iterations_
        int total_pdhg_iterations_
        double last_candidate_kkt_score_
        double last_restart_kkt_score_
        double sum_solution_weight_
        int iterations_since_last_restart_
        # /PDLP warm start data
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
        bool solved_by_pdlp_

    # GPU-backed MIP solution struct (device memory)
    cdef cppclass mip_ret_t:
        unique_ptr[device_buffer] solution_
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

    # CPU-backed MIP solution struct (host memory)
    cdef cppclass cpu_mip_ret_t:
        vector[double] solution_
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

    # Main return struct using variants
    cdef cppclass solver_ret_t:
        problem_category_t problem_type
        variant[linear_programming_ret_t, cpu_linear_programming_ret_t] lp_ret
        variant[mip_ret_t, cpu_mip_ret_t] mip_ret

    cdef unique_ptr[solver_ret_t] call_solve(
        data_model_view_t[int, double]* data_model,
        solver_settings_t[int, double]* solver_settings,
    ) except +

    cdef pair[vector[unique_ptr[solver_ret_t]], double] call_batch_solve( # noqa
        vector[data_model_view_t[int, double] *] data_models,
        solver_settings_t[int, double]* solver_settings,
    ) except +

# Variant helper functions - Cython doesn't directly support variant access
# so we need C++ helper functions
cdef extern from *:
    """
    #include <variant>
    #include <cuopt/linear_programming/utilities/cython_solve.hpp>

    // Check which alternative is active
    inline bool holds_linear_programming_ret_t(const std::variant<cuopt::cython::linear_programming_ret_t, cuopt::cython::cpu_linear_programming_ret_t>& v) {
        return std::holds_alternative<cuopt::cython::linear_programming_ret_t>(v);
    }
    inline bool holds_mip_ret_t(const std::variant<cuopt::cython::mip_ret_t, cuopt::cython::cpu_mip_ret_t>& v) {
        return std::holds_alternative<cuopt::cython::mip_ret_t>(v);
    }

    // Get references to the active alternative
    inline cuopt::cython::linear_programming_ret_t& get_linear_programming_ret_t(std::variant<cuopt::cython::linear_programming_ret_t, cuopt::cython::cpu_linear_programming_ret_t>& v) {
        return std::get<cuopt::cython::linear_programming_ret_t>(v);
    }
    inline cuopt::cython::cpu_linear_programming_ret_t& get_cpu_linear_programming_ret_t(std::variant<cuopt::cython::linear_programming_ret_t, cuopt::cython::cpu_linear_programming_ret_t>& v) {
        return std::get<cuopt::cython::cpu_linear_programming_ret_t>(v);
    }
    inline cuopt::cython::mip_ret_t& get_mip_ret_t(std::variant<cuopt::cython::mip_ret_t, cuopt::cython::cpu_mip_ret_t>& v) {
        return std::get<cuopt::cython::mip_ret_t>(v);
    }
    inline cuopt::cython::cpu_mip_ret_t& get_cpu_mip_ret_t(std::variant<cuopt::cython::mip_ret_t, cuopt::cython::cpu_mip_ret_t>& v) {
        return std::get<cuopt::cython::cpu_mip_ret_t>(v);
    }
    """
    # Declare helper functions for Cython to use
    cdef bool holds_linear_programming_ret_t(variant[linear_programming_ret_t, cpu_linear_programming_ret_t]& v)
    cdef bool holds_mip_ret_t(variant[mip_ret_t, cpu_mip_ret_t]& v)
    cdef linear_programming_ret_t& get_linear_programming_ret_t(variant[linear_programming_ret_t, cpu_linear_programming_ret_t]& v)
    cdef cpu_linear_programming_ret_t& get_cpu_linear_programming_ret_t(variant[linear_programming_ret_t, cpu_linear_programming_ret_t]& v)
    cdef mip_ret_t& get_mip_ret_t(variant[mip_ret_t, cpu_mip_ret_t]& v)
    cdef cpu_mip_ret_t& get_cpu_mip_ret_t(variant[mip_ret_t, cpu_mip_ret_t]& v)
