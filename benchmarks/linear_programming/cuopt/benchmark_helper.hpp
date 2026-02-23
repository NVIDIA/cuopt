/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/linear_programming/optimization_problem.hpp>
#include <cuopt/linear_programming/pdlp/pdlp_hyper_params.cuh>
#include <cuopt/linear_programming/pdlp/solver_solution.hpp>
#include <cuopt/linear_programming/solve.hpp>
#include <mps_parser/parser.hpp>

#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/core/cusparse_macros.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/sparse/linalg/transpose.cuh>

#include <rmm/device_scalar.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/owning_wrapper.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>

inline auto make_async() { return std::make_shared<rmm::mr::cuda_async_memory_resource>(); }
inline auto make_pool()
{
  size_t free_mem, total_mem;
  RAFT_CUDA_TRY(cudaMemGetInfo(&free_mem, &total_mem));
  size_t rmm_alloc_gran = 256;
  double alloc_ratio    = 0.4;
  // allocate 40%
  size_t initial_pool_size = (size_t(free_mem * alloc_ratio) / rmm_alloc_gran) * rmm_alloc_gran;
  return rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(make_async(),
                                                                     initial_pool_size);
}

template <typename T>
void parse_value(std::istringstream& iss, T& value)
{
  iss >> value;
}

template <>
void parse_value(std::istringstream& iss, bool& value)
{
  iss >> std::boolalpha >> value;
}

void fill_pdlp_hyper_params(
  const std::string& pdlp_hyper_params_path,
  cuopt::linear_programming::pdlp_hyper_params::pdlp_hyper_params_t& params)
{
  if (!std::filesystem::exists(pdlp_hyper_params_path)) {
    std::cerr << "PDLP config file path is not a valid: " << pdlp_hyper_params_path << std::endl;
    exit(-1);
  }
  std::ifstream file(pdlp_hyper_params_path);
  std::string line;

  std::map<std::string, double*> double_settings = {
    {"initial_step_size_scaling", &params.initial_step_size_scaling},
    {"default_alpha_pock_chambolle_rescaling", &params.default_alpha_pock_chambolle_rescaling},
    {"default_reduction_exponent", &params.reduction_exponent},
    {"default_growth_exponent", &params.growth_exponent},
    {"default_primal_weight_update_smoothing", &params.primal_weight_update_smoothing},
    {"default_sufficient_reduction_for_restart", &params.sufficient_reduction_for_restart},
    {"default_necessary_reduction_for_restart", &params.necessary_reduction_for_restart},
    {"default_artificial_restart_threshold", &params.default_artificial_restart_threshold},
    {"initial_primal_weight_c_scaling", &params.initial_primal_weight_c_scaling},
    {"initial_primal_weight_b_scaling", &params.initial_primal_weight_b_scaling},
    {"primal_importance", &params.primal_importance},
    {"primal_distance_smoothing", &params.primal_distance_smoothing},
    {"dual_distance_smoothing", &params.dual_distance_smoothing},
    {"reflection_coefficient", &params.reflection_coefficient},
    {"restart_k_p", &params.restart_k_p},
    {"restart_k_i", &params.restart_k_i},
    {"restart_k_d", &params.restart_k_d},
    {"restart_i_smooth", &params.restart_i_smooth}};

  std::map<std::string, int*> int_settings = {
    {"default_l_inf_ruiz_iterations", &params.default_l_inf_ruiz_iterations},
    {"major_iteration", &params.major_iteration},
    {"min_iteration_restart", &params.min_iteration_restart},
    {"restart_strategy", &params.restart_strategy},
  };

  std::map<std::string, bool*> bool_settings = {
    {"do_pock_chambolle_scaling", &params.do_pock_chambolle_scaling},
    {"do_ruiz_scaling", &params.do_ruiz_scaling},
    {"compute_initial_step_size_before_scaling", &params.compute_initial_step_size_before_scaling},
    {"compute_initial_primal_weight_before_scaling",
     &params.compute_initial_primal_weight_before_scaling},
    {"never_restart_to_average", &params.never_restart_to_average},
    {"compute_last_restart_before_new_primal_weight",
     &params.compute_last_restart_before_new_primal_weight},
    {"artificial_restart_in_main_loop", &params.artificial_restart_in_main_loop},
    {"rescale_for_restart", &params.rescale_for_restart},
    {"update_primal_weight_on_initial_solution", &params.update_primal_weight_on_initial_solution},
    {"update_step_size_on_initial_solution", &params.update_step_size_on_initial_solution},
    {"handle_some_primal_gradients_on_finite_bounds_as_residuals",
     &params.handle_some_primal_gradients_on_finite_bounds_as_residuals},
    {"project_initial_primal", &params.project_initial_primal},
    {"use_adaptive_step_size_strategy", &params.use_adaptive_step_size_strategy},
    {"initial_step_size_max_singular_value", &params.initial_step_size_max_singular_value},
    {"initial_primal_weight_combined_bounds", &params.initial_primal_weight_combined_bounds},
    {"bound_objective_rescaling", &params.bound_objective_rescaling},
    {"use_reflected_primal_dual", &params.use_reflected_primal_dual},
    {"use_fixed_point_error", &params.use_fixed_point_error},
    {"use_conditional_major", &params.use_conditional_major}};

  while (std::getline(file, line)) {
    std::istringstream iss(line);
    std::string var;

    if (!(iss >> var >> std::ws && iss.get() == '=')) {
      std::cerr << "Bad line parsed: " << line << std::endl;
      exit(-1);
    }

    auto double_it = double_settings.find(var);
    if (double_it != double_settings.end()) {
      parse_value(iss, *double_it->second);
      continue;
    }

    auto int_it = int_settings.find(var);
    if (int_it != int_settings.end()) {
      parse_value(iss, *int_it->second);
      continue;
    }

    auto bool_it = bool_settings.find(var);
    if (bool_it != bool_settings.end()) {
      parse_value(iss, *bool_it->second);
      continue;
    }

    std::cerr << "Bad parameter: " << var << " is not a valid parameter" << std::endl;
    exit(-1);
  }
}

bool has_file(const std::filesystem::path& file_path) { return std::filesystem::exists(file_path); }

bool has_problem_files(const std::filesystem::path& filename)
{
  if (!std::filesystem::exists(filename) ||
      !std::filesystem::is_directory(filename.parent_path())) {
    std::cerr << "MPS Path '" << filename << "' is not valid" << std::endl;
    exit(-1);
    return false;
  }

  std::vector<std::string> path_names = {
    filename.parent_path().string() + "/A_" + filename.filename().string() + ".bin",
    filename.parent_path().string() + "/A_indices_" + filename.filename().string() + ".bin",
    filename.parent_path().string() + "/A_offsets_" + filename.filename().string() + ".bin",
    filename.parent_path().string() + "/b_" + filename.filename().string() + ".bin",
    filename.parent_path().string() + "/c_" + filename.filename().string() + ".bin",
    filename.parent_path().string() + "/variable_lower_bounds_" + filename.filename().string() +
      ".bin",
    filename.parent_path().string() + "/variable_upper_bounds_" + filename.filename().string() +
      ".bin",
    filename.parent_path().string() + "/constraint_lower_bounds_" + filename.filename().string() +
      ".bin",
    filename.parent_path().string() + "/constraint_upper_bounds_" + filename.filename().string() +
      ".bin",
    filename.parent_path().string() + "/problem_info_" + filename.filename().string() + ".txt"};

  for (const auto& path_name : path_names)
    if (!has_file(path_name)) return false;

  return true;
}

template <typename T>
std::vector<T> read_vector_from_file(const std::string& filename)
{
  std::ifstream in_file(filename, std::ios::binary | std::ios::in);
  if (!in_file.is_open()) {
    std::cerr << "Failed to open file for reading: " << filename << std::endl;
    exit(-1);
    return std::vector<T>();
  }

  std::size_t size;
  in_file.read(reinterpret_cast<char*>(&size), sizeof(size));

  std::vector<T> vec(size);
  in_file.read(reinterpret_cast<char*>(vec.data()), size * sizeof(T));
  in_file.close();

  return vec;
}

template <typename i_t, typename f_t>
void write_problem_info(const cuopt::mps_parser::mps_data_model_t<i_t, f_t>& op_problem,
                        const std::string& filename)
{
  std::ofstream file(filename);
  if (!file) {
    std::cerr << "Failed to open file for writing: " << filename << std::endl;
    exit(-1);
  }
  if (file.is_open()) {
    file << "maximize " << op_problem.get_sense() << "\n";
    file << "objective_scaling_factor " << op_problem.get_objective_scaling_factor() << "\n";
    file << "objective_offset " << op_problem.get_objective_offset() << "\n";
  }
  file.close();
}

template <typename i_t, typename f_t>
void read_problem_info(cuopt::linear_programming::optimization_problem_t<i_t, f_t>& op_problem,
                       const std::string& filename)
{
  std::ifstream file(filename);
  if (!file) {
    std::cerr << "Failed to open file for writing: " << filename << std::endl;
    exit(-1);
  }
  std::string line;
  if (file.is_open()) {
    while (std::getline(file, line)) {
      size_t space_pos  = line.find(' ');
      std::string key   = line.substr(0, space_pos);
      std::string value = line.substr(space_pos + 1);

      if (key == "maximize") {
        op_problem.set_maximize(std::stoi(value));
      } else if (key == "objective_scaling_factor") {
        op_problem.set_objective_scaling_factor(std::stod(value));
      } else if (key == "objective_offset") {
        op_problem.set_objective_offset(std::stod(value));
      }
    }
  }
  file.close();
}

template <typename T>
bool write_vector_to_binary(const std::vector<T>& hvec, const std::string& filename)
{
  std::ofstream out_file(filename, std::ios::out | std::ios::binary);
  if (!out_file) {
    std::cerr << "Failed to open file for writing: " << filename << std::endl;
    exit(-1);
    return false;
  }

  // Write the data from host vector to file
  std::size_t size = hvec.size();
  out_file.write(reinterpret_cast<const char*>(&size), sizeof(size));

  if (size > 0) { out_file.write(reinterpret_cast<const char*>(hvec.data()), size * sizeof(T)); }

  return out_file.good();
}

void mps_file_to_binary(const std::filesystem::path& filename)
{
  const raft::handle_t handle_{};

  std::string p = std::string(filename);

  cuopt::mps_parser::mps_data_model_t<int, double> op_problem =
    cuopt::mps_parser::parse_mps<int, double>(p);

  auto filename_string = filename.filename().string();

  std::vector<std::pair<const std::vector<double>&, std::string>> data_vectors_double = {
    {op_problem.get_constraint_matrix_values(), "/A_" + filename_string + ".bin"},
    {op_problem.get_constraint_bounds(), "/b_" + filename_string + ".bin"},
    {op_problem.get_objective_coefficients(), "/c_" + filename_string + ".bin"},
    {op_problem.get_variable_lower_bounds(), "/variable_lower_bounds_" + filename_string + ".bin"},
    {op_problem.get_variable_upper_bounds(), "/variable_upper_bounds_" + filename_string + ".bin"},
    {op_problem.get_constraint_lower_bounds(),
     "/constraint_lower_bounds_" + filename_string + ".bin"},
    {op_problem.get_constraint_upper_bounds(),
     "/constraint_upper_bounds_" + filename_string + ".bin"}};

  std::vector<std::pair<const std::vector<int>&, std::string>> data_vectors_int = {
    {op_problem.get_constraint_matrix_indices(), "/A_indices_" + filename_string + ".bin"},
    {op_problem.get_constraint_matrix_offsets(), "/A_offsets_" + filename_string + ".bin"},
  };

  for (const auto& [data, path] : data_vectors_double)
    write_vector_to_binary(data, filename.parent_path().string() + path);
  for (const auto& [data, path] : data_vectors_int)
    write_vector_to_binary(data, filename.parent_path().string() + path);

  write_problem_info(
    op_problem,
    filename.parent_path().string() + "/problem_info_" + filename.filename().string() + ".txt");
}
