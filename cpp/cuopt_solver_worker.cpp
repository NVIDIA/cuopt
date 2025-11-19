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

/**
 * @file cuopt_solver_worker.cpp
 * @brief Solver worker process for async job queue
 *
 * This worker:
 * - Reads solve jobs from a shared memory queue
 * - Solves problems using GPU
 * - Writes results back to result queue
 * - Runs in a separate process for isolation
 */

#include <cuopt_remote.pb.h>
#include <cuopt/linear_programming/solve.hpp>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <csignal>
#include <cstring>
#include <iostream>
#include <string>

// Shared memory queue structure
struct JobQueueEntry {
  char job_id[64];
  uint32_t problem_type;  // 0 = LP, 1 = MIP
  uint32_t data_size;
  uint8_t data[1024 * 1024];  // 1MB buffer
  bool ready;
  bool processed;
};

struct ResultQueueEntry {
  char job_id[64];
  uint32_t status;  // 0 = success, 1 = error
  uint32_t data_size;
  uint8_t data[2 * 1024 * 1024];  // 2MB buffer for results
  bool ready;
  bool retrieved;
};

const size_t MAX_JOBS    = 100;
const size_t MAX_RESULTS = 100;

// Global flag for graceful shutdown
volatile sig_atomic_t keep_running = 1;

void signal_handler(int signal)
{
  if (signal == SIGINT || signal == SIGTERM) {
    std::cout << "[Worker] Received shutdown signal\n";
    keep_running = 0;
  }
}

// Convert protobuf OptimizationProblem to optimization_problem_t
template <typename i_t, typename f_t>
static cuopt::linear_programming::optimization_problem_t<i_t, f_t> protobuf_to_problem(
  const cuopt::remote::OptimizationProblem& pb_problem)
{
  cuopt::linear_programming::optimization_problem_t<i_t, f_t> problem;

  problem.set_maximize(pb_problem.maximize());
  problem.set_objective_scaling_factor(static_cast<f_t>(pb_problem.objective_scaling_factor()));
  problem.set_objective_offset(static_cast<f_t>(pb_problem.objective_offset()));

  std::vector<f_t> matrix_values;
  std::vector<i_t> matrix_indices;
  std::vector<i_t> matrix_offsets;

  for (int i = 0; i < pb_problem.constraint_matrix_values_size(); ++i) {
    matrix_values.push_back(static_cast<f_t>(pb_problem.constraint_matrix_values(i)));
  }
  for (int i = 0; i < pb_problem.constraint_matrix_indices_size(); ++i) {
    matrix_indices.push_back(static_cast<i_t>(pb_problem.constraint_matrix_indices(i)));
  }
  for (int i = 0; i < pb_problem.constraint_matrix_offsets_size(); ++i) {
    matrix_offsets.push_back(static_cast<i_t>(pb_problem.constraint_matrix_offsets(i)));
  }

  problem.set_csr_constraint_matrix(matrix_values.data(),
                                    matrix_values.size(),
                                    matrix_indices.data(),
                                    matrix_indices.size(),
                                    matrix_offsets.data(),
                                    matrix_offsets.size());

  std::vector<f_t> obj_coeffs, constraint_bounds, var_lower, var_upper;

  for (int i = 0; i < pb_problem.objective_coefficients_size(); ++i) {
    obj_coeffs.push_back(static_cast<f_t>(pb_problem.objective_coefficients(i)));
  }
  for (int i = 0; i < pb_problem.constraint_bounds_size(); ++i) {
    constraint_bounds.push_back(static_cast<f_t>(pb_problem.constraint_bounds(i)));
  }
  for (int i = 0; i < pb_problem.variable_lower_bounds_size(); ++i) {
    var_lower.push_back(static_cast<f_t>(pb_problem.variable_lower_bounds(i)));
  }
  for (int i = 0; i < pb_problem.variable_upper_bounds_size(); ++i) {
    var_upper.push_back(static_cast<f_t>(pb_problem.variable_upper_bounds(i)));
  }

  problem.set_objective_coefficients(obj_coeffs.data(), obj_coeffs.size());
  problem.set_constraint_bounds(constraint_bounds.data(), constraint_bounds.size());
  problem.set_variable_lower_bounds(var_lower.data(), var_lower.size());
  problem.set_variable_upper_bounds(var_upper.data(), var_upper.size());

  if (pb_problem.constraint_lower_bounds_size() > 0) {
    std::vector<f_t> constraint_lower;
    for (int i = 0; i < pb_problem.constraint_lower_bounds_size(); ++i) {
      constraint_lower.push_back(static_cast<f_t>(pb_problem.constraint_lower_bounds(i)));
    }
    problem.set_constraint_lower_bounds(constraint_lower.data(), constraint_lower.size());
  }

  if (pb_problem.constraint_upper_bounds_size() > 0) {
    std::vector<f_t> constraint_upper;
    for (int i = 0; i < pb_problem.constraint_upper_bounds_size(); ++i) {
      constraint_upper.push_back(static_cast<f_t>(pb_problem.constraint_upper_bounds(i)));
    }
    problem.set_constraint_upper_bounds(constraint_upper.data(), constraint_upper.size());
  }

  if (!pb_problem.row_types().empty()) {
    const std::string& rt = pb_problem.row_types();
    problem.set_row_types(rt.data(), rt.size());
  }

  return problem;
}

// Convert LP solution to protobuf
template <typename i_t, typename f_t>
static void lp_solution_to_protobuf(
  cuopt::linear_programming::optimization_problem_solution_t<i_t, f_t>& solution,
  cuopt::remote::LPSolution* pb_solution)
{
  for (const auto& val : solution.get_primal_solution()) {
    pb_solution->add_primal_solution(static_cast<double>(val));
  }
  for (const auto& val : solution.get_dual_solution()) {
    pb_solution->add_dual_solution(static_cast<double>(val));
  }
  for (const auto& val : solution.get_reduced_cost()) {
    pb_solution->add_reduced_cost(static_cast<double>(val));
  }

  pb_solution->set_termination_status(
    static_cast<cuopt::remote::PDLPTerminationStatus>(solution.get_termination_status()));

  const auto& stats = solution.get_additional_termination_information();
  pb_solution->set_primal_objective(stats.primal_objective);
  pb_solution->set_dual_objective(stats.dual_objective);
  pb_solution->set_solve_time(stats.solve_time);
  pb_solution->set_l2_primal_residual(stats.l2_primal_residual);
  pb_solution->set_l2_dual_residual(stats.l2_dual_residual);
  pb_solution->set_gap(stats.gap);
  pb_solution->set_nb_iterations(stats.number_of_steps_taken);
  pb_solution->set_solved_by_pdlp(stats.solved_by_pdlp);
}

// Convert MIP solution to protobuf
template <typename i_t, typename f_t>
static void mip_solution_to_protobuf(cuopt::linear_programming::mip_solution_t<i_t, f_t>& solution,
                                     cuopt::remote::MIPSolution* pb_solution)
{
  for (const auto& val : solution.get_solution()) {
    pb_solution->add_solution(static_cast<double>(val));
  }

  pb_solution->set_termination_status(
    static_cast<cuopt::remote::MIPTerminationStatus>(solution.get_termination_status()));
  pb_solution->set_objective(solution.get_objective_value());
  pb_solution->set_solution_bound(solution.get_solution_bound());
  pb_solution->set_total_solve_time(solution.get_total_solve_time());
  pb_solution->set_presolve_time(solution.get_presolve_time());
  pb_solution->set_mip_gap(solution.get_mip_gap());
  pb_solution->set_max_constraint_violation(solution.get_max_constraint_violation());
  pb_solution->set_max_int_violation(solution.get_max_int_violation());
  pb_solution->set_max_variable_bound_violation(solution.get_max_variable_bound_violation());
  pb_solution->set_nodes(solution.get_num_nodes());
  pb_solution->set_simplex_iterations(solution.get_num_simplex_iterations());
}

int main(int argc, char* argv[])
{
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  std::cout << "==========================================================\n";
  std::cout << "cuOpt Solver Worker Process\n";
  std::cout << "==========================================================\n";

  signal(SIGINT, signal_handler);
  signal(SIGTERM, signal_handler);

  // Open shared memory for job queue
  int job_shm_fd = shm_open("/cuopt_job_queue", O_RDWR, 0666);
  if (job_shm_fd == -1) {
    std::cerr << "[Worker] Error: Failed to open job queue shared memory\n";
    return 1;
  }

  JobQueueEntry* job_queue = static_cast<JobQueueEntry*>(mmap(
    nullptr, sizeof(JobQueueEntry) * MAX_JOBS, PROT_READ | PROT_WRITE, MAP_SHARED, job_shm_fd, 0));

  if (job_queue == MAP_FAILED) {
    std::cerr << "[Worker] Error: Failed to map job queue\n";
    close(job_shm_fd);
    return 1;
  }

  // Open shared memory for result queue
  int result_shm_fd = shm_open("/cuopt_result_queue", O_RDWR, 0666);
  if (result_shm_fd == -1) {
    std::cerr << "[Worker] Error: Failed to open result queue shared memory\n";
    munmap(job_queue, sizeof(JobQueueEntry) * MAX_JOBS);
    close(job_shm_fd);
    return 1;
  }

  ResultQueueEntry* result_queue =
    static_cast<ResultQueueEntry*>(mmap(nullptr,
                                        sizeof(ResultQueueEntry) * MAX_RESULTS,
                                        PROT_READ | PROT_WRITE,
                                        MAP_SHARED,
                                        result_shm_fd,
                                        0));

  if (result_queue == MAP_FAILED) {
    std::cerr << "[Worker] Error: Failed to map result queue\n";
    munmap(job_queue, sizeof(JobQueueEntry) * MAX_JOBS);
    close(job_shm_fd);
    close(result_shm_fd);
    return 1;
  }

  std::cout << "[Worker] Connected to shared memory queues\n";
  std::cout << "[Worker] Waiting for jobs...\n\n";

  // Main worker loop
  while (keep_running) {
    // Scan job queue for ready jobs
    bool found_job = false;

    for (size_t i = 0; i < MAX_JOBS && keep_running; ++i) {
      if (job_queue[i].ready && !job_queue[i].processed) {
        found_job = true;
        std::string job_id(job_queue[i].job_id);

        std::cout << "[Worker] Processing job: " << job_id
                  << " (type: " << (job_queue[i].problem_type == 0 ? "LP" : "MIP") << ")\n";

        try {
          cuopt::remote::ResultResponse result_response;
          result_response.set_status(cuopt::remote::SUCCESS);

          if (job_queue[i].problem_type == 0) {
            // LP problem
            cuopt::remote::SolveLPRequest lp_request;
            if (!lp_request.ParseFromArray(job_queue[i].data, job_queue[i].data_size)) {
              throw std::runtime_error("Failed to parse LP request");
            }

            auto problem = protobuf_to_problem<int, double>(lp_request.problem());

            std::cout << "[Worker] Solving LP: " << problem.get_n_variables() << " vars, "
                      << problem.get_n_constraints() << " constraints\n";

            cuopt::linear_programming::pdlp_solver_settings_t<int, double> settings;
            auto solution = cuopt::linear_programming::solve_lp(problem, settings);

            std::cout << "[Worker] LP solve completed, status: "
                      << static_cast<int>(solution.get_termination_status()) << "\n";

            lp_solution_to_protobuf(solution, result_response.mutable_lp_solution());

          } else if (job_queue[i].problem_type == 1) {
            // MIP problem
            cuopt::remote::SolveMIPRequest mip_request;
            if (!mip_request.ParseFromArray(job_queue[i].data, job_queue[i].data_size)) {
              throw std::runtime_error("Failed to parse MIP request");
            }

            auto problem = protobuf_to_problem<int, double>(mip_request.problem());

            // Set variable types from is_integer and is_binary fields
            const auto& pb_problem = mip_request.problem();
            std::cout << "[Worker] MIP problem has " << pb_problem.is_integer_size()
                      << " is_integer entries, " << pb_problem.is_binary_size()
                      << " is_binary entries\n";
            if (pb_problem.is_integer_size() > 0 || pb_problem.is_binary_size() > 0) {
              int n_vars = problem.get_n_variables();
              std::vector<cuopt::linear_programming::var_t> var_types(
                n_vars, cuopt::linear_programming::var_t::CONTINUOUS);

              for (int j = 0; j < pb_problem.is_integer_size(); ++j) {
                if (pb_problem.is_integer(j)) {
                  var_types[j] = cuopt::linear_programming::var_t::INTEGER;
                }
              }
              for (int j = 0; j < pb_problem.is_binary_size(); ++j) {
                if (pb_problem.is_binary(j)) {
                  var_types[j] = cuopt::linear_programming::var_t::INTEGER;
                }
              }

              problem.set_variable_types(var_types.data(), var_types.size());
            }

            std::cout << "[Worker] Solving MIP: " << problem.get_n_variables() << " vars, "
                      << problem.get_n_constraints() << " constraints\n";

            cuopt::linear_programming::mip_solver_settings_t<int, double> settings;
            auto solution = cuopt::linear_programming::solve_mip(problem, settings);

            std::cout << "[Worker] MIP solve completed, status: "
                      << static_cast<int>(solution.get_termination_status())
                      << ", objective: " << solution.get_objective_value() << "\n";

            mip_solution_to_protobuf(solution, result_response.mutable_mip_solution());

          } else {
            throw std::runtime_error("Unknown problem type: " +
                                     std::to_string(job_queue[i].problem_type));
          }

          std::string result_data = result_response.SerializeAsString();

          // Find free result slot
          bool stored = false;
          for (size_t j = 0; j < MAX_RESULTS; ++j) {
            if (!result_queue[j].ready) {
              strncpy(result_queue[j].job_id, job_id.c_str(), sizeof(result_queue[j].job_id) - 1);
              result_queue[j].status    = 0;
              result_queue[j].data_size = result_data.size();
              std::memcpy(result_queue[j].data, result_data.data(), result_data.size());
              result_queue[j].retrieved = false;
              result_queue[j].ready     = true;
              stored                    = true;
              break;
            }
          }

          if (!stored) { std::cerr << "[Worker] Warning: Result queue full, result may be lost\n"; }

        } catch (const std::exception& e) {
          std::cerr << "[Worker] Error solving job " << job_id << ": " << e.what() << "\n";

          // Store error result
          for (size_t j = 0; j < MAX_RESULTS; ++j) {
            if (!result_queue[j].ready) {
              strncpy(result_queue[j].job_id, job_id.c_str(), sizeof(result_queue[j].job_id) - 1);
              result_queue[j].status    = 1;  // Error
              result_queue[j].data_size = 0;
              result_queue[j].retrieved = false;
              result_queue[j].ready     = true;
              break;
            }
          }
        }

        // Mark job as processed
        job_queue[i].processed = true;
      }
    }

    if (!found_job) {
      usleep(100000);  // Sleep 100ms if no jobs
    }
  }

  std::cout << "[Worker] Shutting down...\n";

  munmap(job_queue, sizeof(JobQueueEntry) * MAX_JOBS);
  munmap(result_queue, sizeof(ResultQueueEntry) * MAX_RESULTS);
  close(job_shm_fd);
  close(result_shm_fd);

  google::protobuf::ShutdownProtobufLibrary();

  std::cout << "[Worker] Stopped\n";
  return 0;
}
