# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved. # noqa
# SPDX-License-Identifier: Apache-2.0


# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from libcpp.memory cimport unique_ptr
from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "cuopt/mathematical_optimization/utilities/internals.hpp" namespace "cuopt::internals": # noqa
    cdef cppclass base_solution_callback_t

cdef extern from "cuopt/mathematical_optimization/pdlp/solver_settings.hpp" namespace "cuopt::mathematical_optimization": # noqa
    ctypedef enum pdlp_solver_mode_t "cuopt::mathematical_optimization::pdlp_solver_mode_t": # noqa
        Stable1 "cuopt::mathematical_optimization::pdlp_solver_mode_t::Stable1" # noqa
        Stable2 "cuopt::mathematical_optimization::pdlp_solver_mode_t::Stable2" # noqa
        Methodical1 "cuopt::mathematical_optimization::pdlp_solver_mode_t::Methodical1" # noqa
        Fast1 "cuopt::mathematical_optimization::pdlp_solver_mode_t::Fast1" # noqa
        Stable3 "cuopt::mathematical_optimization::pdlp_solver_mode_t::Stable3" # noqa

    ctypedef enum method_t "cuopt::mathematical_optimization::method_t": # noqa
        Concurrent "cuopt::mathematical_optimization::method_t::Concurrent" # noqa
        PDLP "cuopt::mathematical_optimization::method_t::PDLP" # noqa
        DualSimplex "cuopt::mathematical_optimization::method_t::DualSimplex" # noqa
        Barrier "cuopt::mathematical_optimization::method_t::Barrier" # noqa
        Unset "cuopt::mathematical_optimization::method_t::Unset" # noqa

cdef extern from "cuopt/mathematical_optimization/solver_settings.hpp" namespace "cuopt::mathematical_optimization": # noqa

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

        void set_initial_pdlp_primal_solution(
            const f_t* initial_primal_solution,
            i_t size
        ) except +
        void set_initial_pdlp_dual_solution(
            const f_t* initial_dual_solution,
            i_t size
        ) except +

        void add_initial_mip_solution(
            const f_t* initial_solution,
            i_t size
        ) except +
        void set_mip_callback(
            base_solution_callback_t* callback,
            void* user_data
        ) except +

        bool dump_parameters_to_file(
            const string& path,
            bool hyperparameters_only,
        ) except +

        void load_parameters_from_file(const string& path) except +


cdef class SolverSettings:
    cdef unique_ptr[solver_settings_t[int, double]] c_solver_settings
    cdef public dict settings_dict
    cdef public object pdlp_warm_start_data
    cdef public list mip_callbacks
