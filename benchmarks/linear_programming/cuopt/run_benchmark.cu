/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <cuopt/linear_programming/optimization_problem.hpp>
#include <cuopt/linear_programming/pdlp/pdlp_hyper_params.cuh>
#include <cuopt/linear_programming/pdlp/solver_solution.hpp>
#include <cuopt/linear_programming/solve.hpp>
#include <mps_parser/parser.hpp>
#include "benchmark_helper.hpp"

#include <raft/sparse/detail/cusparse_macros.h>
#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/common/nvtx.hpp>
#include <raft/core/handle.hpp>
#include <raft/sparse/linalg/transpose.cuh>

#include <rmm/device_scalar.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>

template <typename i_t, typename f_t>
cuopt::linear_programming::optimization_problem_t<i_t, f_t> get_optimization_problem(
  raft::handle_t& handle_, const std::filesystem::path& filename, bool cache_mps)
{
  if (cache_mps) {
    std::cout << "Using mps cache" << std::endl;
    // Check if all problem is here, all binary files + problem info text
    if (!has_problem_files(filename)) {
      std::cout << "Creating binary cache files" << std::endl;
      mps_file_to_binary(filename);
      std::cout << "Done creating cache files" << std::endl;
    } else
      std::cout << "Binary file alreadu generation" << std::endl;
    auto A         = read_vector_from_file<double>(filename.parent_path().string() + "/A_" +
                                           filename.filename().string() + ".bin");
    auto A_indices = read_vector_from_file<int>(filename.parent_path().string() + "/A_indices_" +
                                                filename.filename().string() + ".bin");
    auto A_offsets = read_vector_from_file<int>(filename.parent_path().string() + "/A_offsets_" +
                                                filename.filename().string() + ".bin");
    auto b         = read_vector_from_file<double>(filename.parent_path().string() + "/b_" +
                                           filename.filename().string() + ".bin");
    auto c         = read_vector_from_file<double>(filename.parent_path().string() + "/c_" +
                                           filename.filename().string() + ".bin");
    auto variable_lower_bounds =
      read_vector_from_file<double>(filename.parent_path().string() + "/variable_lower_bounds_" +
                                    filename.filename().string() + ".bin");
    auto variable_upper_bounds =
      read_vector_from_file<double>(filename.parent_path().string() + "/variable_upper_bounds_" +
                                    filename.filename().string() + ".bin");
    auto constraint_lower_bounds =
      read_vector_from_file<double>(filename.parent_path().string() + "/constraint_lower_bounds_" +
                                    filename.filename().string() + ".bin");
    auto constraint_upper_bounds =
      read_vector_from_file<double>(filename.parent_path().string() + "/constraint_upper_bounds_" +
                                    filename.filename().string() + ".bin");

    cuopt::linear_programming::optimization_problem_t<int, double> op_problem{&handle_};

    op_problem.set_csr_constraint_matrix(
      A.data(), A.size(), A_indices.data(), A_indices.size(), A_offsets.data(), A_offsets.size());
    op_problem.set_constraint_bounds(b.data(), b.size());
    op_problem.set_objective_coefficients(c.data(), c.size());
    op_problem.set_variable_lower_bounds(variable_lower_bounds.data(),
                                         variable_lower_bounds.size());
    op_problem.set_variable_upper_bounds(variable_upper_bounds.data(),
                                         variable_upper_bounds.size());
    op_problem.set_constraint_lower_bounds(constraint_lower_bounds.data(),
                                           constraint_lower_bounds.size());
    op_problem.set_constraint_upper_bounds(constraint_upper_bounds.data(),
                                           constraint_upper_bounds.size());
    read_problem_info(
      op_problem,
      filename.parent_path().string() + "/problem_info_" + filename.filename().string() + ".txt");

    return op_problem;
  } else {
    std::cout << "Using no cache (regular mps parse)" << std::endl;
    cuopt::mps_parser::mps_data_model_t<int, double> op_problem =
      cuopt::mps_parser::parse_mps<int, double>(filename);

    return cuopt::linear_programming::mps_data_model_to_optimization_problem<int, double>(
      &handle_, op_problem);
  }
}

void run_instance(const std::filesystem::path& filename,
                  cuopt::linear_programming::pdlp_solver_settings_t<int, double>& settings,
                  bool cache_mps = true)
{
  bool generate_variable_values = false;

  // Having the same handle accross different instances create memory issues
  raft::handle_t handle_{};

  std::cout << "Instance= " << filename.stem() << std::endl;

  bool file_generation = true;

  cuopt::linear_programming::optimization_problem_t<int, double> op_problem =
    get_optimization_problem<int, double>(handle_, filename, cache_mps);

  auto start_run_solver = std::chrono::high_resolution_clock::now();

  // No problem checking and warm up is already done

  // Those calls should not be needed since we already do them in the solve()
  // But for some unknown reason, without this, a single instance (or cu) is failing under one
  // specific config
  RAFT_CUBLAS_TRY(raft::linalg::detail::cublassetpointermode(
    handle_.get_cublas_handle(), CUBLAS_POINTER_MODE_DEVICE, handle_.get_stream()));
  RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsesetpointermode(
    handle_.get_cusparse_handle(), CUSPARSE_POINTER_MODE_DEVICE, handle_.get_stream()));

  cuopt::linear_programming::optimization_problem_solution_t<int, double> solution =
    cuopt::linear_programming::solve_lp(op_problem, settings, false, false);

  std::cout << "run_solver " << solution.get_solve_time() << std::endl;
  std::cout << "Termination status " << solution.get_termination_status_string() << std::endl;
  std::cout << "iteration="
            << solution.get_additional_termination_information().number_of_steps_taken << std::endl;
  std::cout << std::fixed << std::setprecision(16) << "Primal objective "
            << solution.get_objective_value() << std::endl;

  std::string problem_name = filename.stem();

  if (solution.get_termination_status() ==
      cuopt::linear_programming::pdlp_termination_status_t::TimeLimit) {
    std::string output_file =
      std::filesystem::current_path().string() + "/" + problem_name + "_solution.txt";
    std::ofstream myfile(output_file.data());
    myfile << "timeout" << std::endl;
    return;
  }

  //if (file_generation) {
  //  std::cout << "Generating file" << std::endl;

  //  std::string output_file =
  //    std::filesystem::current_path().string() + "/" + problem_name + "_solution.txt";
  //  solution.write_to_file(output_file, handle_.get_stream(), generate_variable_values);

  //  std::ofstream myfile(output_file.data(), std::ios::app);
  //  myfile.precision(std::numeric_limits<double>::digits10 + 1);

  //  myfile << "run_solver " << solution.get_solve_time() << " milliseconds" << std::endl;

  //  int major_version, minor_version;
  //  cusparseGetProperty(libraryPropertyType_t::MAJOR_VERSION, &major_version);
  //  cusparseGetProperty(libraryPropertyType_t::MINOR_VERSION, &minor_version);
  //  myfile << "cuSparse version = " << major_version << "." << minor_version << std::endl;

  //  cudaDeviceProp prop;
  //  int device;
  //  cudaGetDevice(&device);
  //  cudaGetDeviceProperties(&prop, device);
  //  myfile << "Memory Clock Rate (KHz): " << prop.memoryClockRate << std::endl;
  //  myfile << "Memory Bus Width (bits): " << prop.memoryBusWidth << std::endl;
  //  myfile << prop.name << std::endl;

  //  myfile << "absolute_dual_tolerance = " << settings.get_absolute_dual_tolerance() << std::endl;
  //  myfile << "relative_dual_tolerance = " << settings.get_relative_dual_tolerance() << std::endl;
  //  myfile << "absolute_primal_tolerance = " << settings.get_absolute_primal_tolerance()
  //         << std::endl;
  //  myfile << "relative_primal_tolerance = " << settings.get_relative_primal_tolerance()
  //         << std::endl;
  //  myfile << "absolute_gap_tolerance = " << settings.get_absolute_gap_tolerance() << std::endl;
  //  myfile << "relative_gap_tolerance = " << settings.get_relative_gap_tolerance() << std::endl;
  //}
}

int main(int argc, char* argv[])
{
  if (argc < 4) {
    std::cerr << "Incorrect number of parameters" << std::endl;
    exit(-1);
  }

  std::filesystem::path instances_path = "/home/scratch.nblin_gpu_1/instances_mps";

  std::filesystem::path filename = std::string(argv[1]);
  double tolerance               = 100000000000;

  const raft::handle_t handle_{};
  bool cache_mps = true;

  fill_pdlp_hyper_params(std::string(argv[2]));

  auto memory_resource = make_pool();
  rmm::mr::set_current_device_resource(memory_resource.get());

  try {
    tolerance = std::stod(argv[3]);
    std::cout << "tolerance: " << tolerance << std::endl;
  } catch (const std::invalid_argument& e) {
    std::cerr << "Invalid argument: " << argv[3] << std::endl;
    return -1;
  } catch (const std::out_of_range& e) {
    std::cerr << "Value out of range: " << argv[3] << std::endl;
    return -1;
  }
  if (argc >= 5) {
    std::string should_cache_mps = std::string(argv[4]);
    if (should_cache_mps == std::string("false")) cache_mps = false;
  }

  cuopt::linear_programming::pdlp_solver_settings_t<int, double> settings =
    cuopt::linear_programming::pdlp_solver_settings_t<int, double>{};
  //settings.set_absolute_dual_tolerance(tolerance);
  //settings.set_relative_dual_tolerance(tolerance);
  //settings.set_absolute_primal_tolerance(tolerance);
  //settings.set_relative_primal_tolerance(tolerance);
  //settings.set_absolute_gap_tolerance(tolerance);
  settings.set_optimality_tolerance(tolerance);
  //settings.set_infeasibility_detection(false);
  //settings.iteration_limit = 1000;
  settings.time_limit = 3600;
  //settings.method = cuopt::linear_programming::method_t::PDLP;

  // Single instance solve
  if (filename.extension() == ".mps")  // Direct mps path
  {
    run_instance(filename, settings, cache_mps);
  } else if (filename.extension() == ".txt")  // Multi instance
  {
    std::ifstream file(filename);
    std::string line;

    while (std::getline(file, line)) {
      // Warm up
      std::filesystem::path instance_path =
        instances_path / std::filesystem::path(line) / std::filesystem::path(line + ".mps");
      run_instance(instance_path, settings, cache_mps);
    }
  } else if (std::filesystem::is_directory(filename))  // Directory containing mps file
  {
    for (const auto& entry : std::filesystem::recursive_directory_iterator(filename)) {
      if (entry.is_regular_file() && entry.path().extension() == ".mps") {
        run_instance(entry.path(), settings, cache_mps);
      }
    }
  } else  // Either instance name directly or error
  {
    std::filesystem::path instance_path =
      instances_path / filename / std::filesystem::path(std::string(filename) + ".mps");
    // If mps file exists, solve it, else error
    if (has_file(instance_path)) {
      run_instance(instance_path, settings, cache_mps);
      return 0;
    }

    std::cout << "Error: Bad input file: " << filename << std::endl;
    return -1;
  }

  return 0;
}
