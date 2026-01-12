/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * MsgPack-based serializer for cuOpt remote solve.
 * This demonstrates the pluggable serialization interface.
 *
 * NOTE: This is a CPU-only serializer. For solutions on GPU memory, it will
 * return empty solution vectors. For production use with GPU memory support,
 * convert this to a .cu file and use CUDA memory copy operations.
 */

#include <cuopt/linear_programming/utilities/remote_serialization.hpp>
#include <msgpack.hpp>

#include <cstdio>
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>

namespace cuopt::linear_programming {

// Message type identifiers for sync mode
constexpr uint8_t MSG_LP_REQUEST   = 1;
constexpr uint8_t MSG_MIP_REQUEST  = 2;
constexpr uint8_t MSG_LP_SOLUTION  = 3;
constexpr uint8_t MSG_MIP_SOLUTION = 4;

// Message type identifiers for async mode
constexpr uint8_t MSG_ASYNC_LP_REQUEST  = 10;
constexpr uint8_t MSG_ASYNC_MIP_REQUEST = 11;
constexpr uint8_t MSG_STATUS_REQUEST    = 12;
constexpr uint8_t MSG_GET_RESULT        = 13;
constexpr uint8_t MSG_DELETE_REQUEST    = 14;
constexpr uint8_t MSG_GET_LOGS          = 15;
constexpr uint8_t MSG_CANCEL_REQUEST    = 16;
constexpr uint8_t MSG_WAIT_REQUEST      = 17;

constexpr uint8_t MSG_SUBMIT_RESPONSE = 20;
constexpr uint8_t MSG_STATUS_RESPONSE = 21;
constexpr uint8_t MSG_LOGS_RESPONSE   = 22;
constexpr uint8_t MSG_CANCEL_RESPONSE = 23;

template <typename i_t, typename f_t>
class msgpack_serializer_t : public remote_serializer_t<i_t, f_t> {
 public:
  msgpack_serializer_t() { std::cout << "[msgpack_serializer] Initialized\n"; }

  ~msgpack_serializer_t() override = default;

  std::string format_name() const override { return "msgpack"; }

  uint32_t protocol_version() const override { return 1; }

  //============================================================================
  // LP Request Serialization
  //============================================================================

  std::vector<uint8_t> serialize_lp_request(
    const mps_parser::data_model_view_t<i_t, f_t>& view,
    const pdlp_solver_settings_t<i_t, f_t>& settings) override
  {
    msgpack::sbuffer buffer;
    msgpack::packer<msgpack::sbuffer> pk(&buffer);

    pk.pack_uint8(MSG_LP_REQUEST);
    pk.pack_uint32(protocol_version());
    pack_problem(pk, view);

    // Pack all LP settings (field names match cuOpt API)
    pk.pack_map(28);
    // Termination tolerances
    pk.pack("absolute_gap_tolerance");
    pk.pack(settings.tolerances.absolute_gap_tolerance);
    pk.pack("relative_gap_tolerance");
    pk.pack(settings.tolerances.relative_gap_tolerance);
    pk.pack("primal_infeasible_tolerance");
    pk.pack(settings.tolerances.primal_infeasible_tolerance);
    pk.pack("dual_infeasible_tolerance");
    pk.pack(settings.tolerances.dual_infeasible_tolerance);
    pk.pack("absolute_dual_tolerance");
    pk.pack(settings.tolerances.absolute_dual_tolerance);
    pk.pack("relative_dual_tolerance");
    pk.pack(settings.tolerances.relative_dual_tolerance);
    pk.pack("absolute_primal_tolerance");
    pk.pack(settings.tolerances.absolute_primal_tolerance);
    pk.pack("relative_primal_tolerance");
    pk.pack(settings.tolerances.relative_primal_tolerance);
    // Limits
    pk.pack("time_limit");
    pk.pack(settings.time_limit);
    pk.pack("iteration_limit");
    pk.pack(static_cast<int64_t>(settings.iteration_limit));
    // Solver configuration
    pk.pack("log_to_console");
    pk.pack(settings.log_to_console);
    pk.pack("detect_infeasibility");
    pk.pack(settings.detect_infeasibility);
    pk.pack("strict_infeasibility");
    pk.pack(settings.strict_infeasibility);
    pk.pack("pdlp_solver_mode");
    pk.pack(static_cast<int>(settings.pdlp_solver_mode));
    pk.pack("method");
    pk.pack(static_cast<int>(settings.method));
    pk.pack("presolve");
    pk.pack(settings.presolve);
    pk.pack("dual_postsolve");
    pk.pack(settings.dual_postsolve);
    pk.pack("crossover");
    pk.pack(settings.crossover);
    pk.pack("num_gpus");
    pk.pack(settings.num_gpus);
    // Advanced options
    pk.pack("per_constraint_residual");
    pk.pack(settings.per_constraint_residual);
    pk.pack("cudss_deterministic");
    pk.pack(settings.cudss_deterministic);
    pk.pack("folding");
    pk.pack(settings.folding);
    pk.pack("augmented");
    pk.pack(settings.augmented);
    pk.pack("dualize");
    pk.pack(settings.dualize);
    pk.pack("ordering");
    pk.pack(settings.ordering);
    pk.pack("barrier_dual_initial_point");
    pk.pack(settings.barrier_dual_initial_point);
    pk.pack("eliminate_dense_columns");
    pk.pack(settings.eliminate_dense_columns);
    pk.pack("save_best_primal_so_far");
    pk.pack(settings.save_best_primal_so_far);
    pk.pack("first_primal_feasible");
    pk.pack(settings.first_primal_feasible);

    return std::vector<uint8_t>(buffer.data(), buffer.data() + buffer.size());
  }

  std::vector<uint8_t> serialize_mip_request(
    const mps_parser::data_model_view_t<i_t, f_t>& view,
    const mip_solver_settings_t<i_t, f_t>& settings) override
  {
    msgpack::sbuffer buffer;
    msgpack::packer<msgpack::sbuffer> pk(&buffer);

    pk.pack_uint8(MSG_MIP_REQUEST);
    pk.pack_uint32(protocol_version());
    pack_problem(pk, view);

    // Pack all MIP settings (field names match cuOpt API)
    pk.pack_map(13);
    // Limits
    pk.pack("time_limit");
    pk.pack(settings.time_limit);
    // Tolerances
    pk.pack("relative_mip_gap");
    pk.pack(settings.tolerances.relative_mip_gap);
    pk.pack("absolute_mip_gap");
    pk.pack(settings.tolerances.absolute_mip_gap);
    pk.pack("integrality_tolerance");
    pk.pack(settings.tolerances.integrality_tolerance);
    pk.pack("absolute_tolerance");
    pk.pack(settings.tolerances.absolute_tolerance);
    pk.pack("relative_tolerance");
    pk.pack(settings.tolerances.relative_tolerance);
    pk.pack("presolve_absolute_tolerance");
    pk.pack(settings.tolerances.presolve_absolute_tolerance);
    // Solver configuration
    pk.pack("log_to_console");
    pk.pack(settings.log_to_console);
    pk.pack("heuristics_only");
    pk.pack(settings.heuristics_only);
    pk.pack("num_cpu_threads");
    pk.pack(settings.num_cpu_threads);
    pk.pack("num_gpus");
    pk.pack(settings.num_gpus);
    pk.pack("presolve");
    pk.pack(settings.presolve);
    pk.pack("mip_scaling");
    pk.pack(settings.mip_scaling);

    return std::vector<uint8_t>(buffer.data(), buffer.data() + buffer.size());
  }

  //============================================================================
  // Solution Deserialization (client-side)
  //============================================================================

  optimization_problem_solution_t<i_t, f_t> deserialize_lp_solution(
    const std::vector<uint8_t>& data) override
  {
    try {
      msgpack::object_handle oh =
        msgpack::unpack(reinterpret_cast<const char*>(data.data()), data.size());
      msgpack::object obj = oh.get();

      if (obj.type != msgpack::type::ARRAY || obj.via.array.size < 6) {
        return optimization_problem_solution_t<i_t, f_t>(
          cuopt::logic_error("Invalid LP solution format", cuopt::error_type_t::RuntimeError));
      }

      auto& arr     = obj.via.array;
      auto status   = static_cast<pdlp_termination_status_t>(arr.ptr[1].as<int>());
      f_t obj_value = arr.ptr[2].as<double>();

      std::vector<f_t> primal_sol;
      arr.ptr[3].convert(primal_sol);

      std::vector<f_t> dual_sol;
      arr.ptr[4].convert(dual_sol);

      f_t solve_time = arr.ptr[5].as<double>();

      optimization_problem_solution_t<i_t, f_t> solution(status);
      solution.set_primal_objective(obj_value);
      solution.set_primal_solution_host(std::move(primal_sol));
      solution.set_dual_solution_host(std::move(dual_sol));
      solution.set_solve_time(solve_time);

      return solution;
    } catch (const std::exception& e) {
      return optimization_problem_solution_t<i_t, f_t>(cuopt::logic_error(
        std::string("MsgPack LP parse error: ") + e.what(), cuopt::error_type_t::RuntimeError));
    }
  }

  mip_solution_t<i_t, f_t> deserialize_mip_solution(const std::vector<uint8_t>& data) override
  {
    try {
      msgpack::object_handle oh =
        msgpack::unpack(reinterpret_cast<const char*>(data.data()), data.size());
      msgpack::object obj = oh.get();

      if (obj.type != msgpack::type::ARRAY || obj.via.array.size < 6) {
        return mip_solution_t<i_t, f_t>(
          cuopt::logic_error("Invalid MIP solution format", cuopt::error_type_t::RuntimeError));
      }

      auto& arr     = obj.via.array;
      auto status   = static_cast<mip_termination_status_t>(arr.ptr[1].as<int>());
      f_t obj_value = arr.ptr[2].as<double>();

      std::vector<f_t> sol;
      arr.ptr[3].convert(sol);

      f_t solve_time = arr.ptr[4].as<double>();
      f_t mip_gap    = arr.ptr[5].as<double>();

      solver_stats_t<i_t, f_t> stats;
      stats.total_solve_time = solve_time;

      mip_solution_t<i_t, f_t> solution(std::move(sol),
                                        std::vector<std::string>{},
                                        obj_value,
                                        mip_gap,
                                        status,
                                        0.0,
                                        0.0,
                                        0.0,
                                        stats);

      return solution;
    } catch (const std::exception& e) {
      return mip_solution_t<i_t, f_t>(cuopt::logic_error(
        std::string("MsgPack MIP parse error: ") + e.what(), cuopt::error_type_t::RuntimeError));
    }
  }

  //============================================================================
  // Server-side Operations
  //============================================================================

  bool is_mip_request(const std::vector<uint8_t>& data) override
  {
    if (data.empty()) return false;
    try {
      size_t offset = 0;
      msgpack::object_handle oh =
        msgpack::unpack(reinterpret_cast<const char*>(data.data()), data.size(), offset);
      if (oh.get().type == msgpack::type::POSITIVE_INTEGER) {
        uint8_t msg_type = oh.get().as<uint8_t>();
        // Check both sync and async MIP request types
        return msg_type == MSG_MIP_REQUEST || msg_type == MSG_ASYNC_MIP_REQUEST;
      }
    } catch (...) {
    }
    return false;
  }

  bool deserialize_lp_request(const std::vector<uint8_t>& data,
                              mps_parser::mps_data_model_t<i_t, f_t>& mps_data,
                              pdlp_solver_settings_t<i_t, f_t>& settings) override
  {
    try {
      size_t offset = 0;

      msgpack::object_handle oh1 =
        msgpack::unpack(reinterpret_cast<const char*>(data.data()), data.size(), offset);
      uint8_t msg_type = oh1.get().as<uint8_t>();
      if (msg_type != MSG_LP_REQUEST) return false;

      msgpack::object_handle oh2 =
        msgpack::unpack(reinterpret_cast<const char*>(data.data()), data.size(), offset);
      // version = oh2.get().as<uint32_t>();

      unpack_problem(data, offset, mps_data);

      msgpack::object_handle oh_settings =
        msgpack::unpack(reinterpret_cast<const char*>(data.data()), data.size(), offset);
      auto settings_map = oh_settings.get().as<std::map<std::string, msgpack::object>>();

      // Deserialize all LP settings (field names match cuOpt API)
      // Termination tolerances
      if (settings_map.count("absolute_gap_tolerance")) {
        settings.tolerances.absolute_gap_tolerance =
          settings_map["absolute_gap_tolerance"].as<double>();
      }
      if (settings_map.count("relative_gap_tolerance")) {
        settings.tolerances.relative_gap_tolerance =
          settings_map["relative_gap_tolerance"].as<double>();
      }
      if (settings_map.count("primal_infeasible_tolerance")) {
        settings.tolerances.primal_infeasible_tolerance =
          settings_map["primal_infeasible_tolerance"].as<double>();
      }
      if (settings_map.count("dual_infeasible_tolerance")) {
        settings.tolerances.dual_infeasible_tolerance =
          settings_map["dual_infeasible_tolerance"].as<double>();
      }
      if (settings_map.count("absolute_dual_tolerance")) {
        settings.tolerances.absolute_dual_tolerance =
          settings_map["absolute_dual_tolerance"].as<double>();
      }
      if (settings_map.count("relative_dual_tolerance")) {
        settings.tolerances.relative_dual_tolerance =
          settings_map["relative_dual_tolerance"].as<double>();
      }
      if (settings_map.count("absolute_primal_tolerance")) {
        settings.tolerances.absolute_primal_tolerance =
          settings_map["absolute_primal_tolerance"].as<double>();
      }
      if (settings_map.count("relative_primal_tolerance")) {
        settings.tolerances.relative_primal_tolerance =
          settings_map["relative_primal_tolerance"].as<double>();
      }
      // Limits
      if (settings_map.count("time_limit")) {
        settings.time_limit = settings_map["time_limit"].as<double>();
      }
      if (settings_map.count("iteration_limit")) {
        settings.iteration_limit = settings_map["iteration_limit"].as<i_t>();
      }
      // Solver configuration
      if (settings_map.count("log_to_console")) {
        settings.log_to_console = settings_map["log_to_console"].as<bool>();
      }
      if (settings_map.count("detect_infeasibility")) {
        settings.detect_infeasibility = settings_map["detect_infeasibility"].as<bool>();
      }
      if (settings_map.count("strict_infeasibility")) {
        settings.strict_infeasibility = settings_map["strict_infeasibility"].as<bool>();
      }
      if (settings_map.count("pdlp_solver_mode")) {
        settings.pdlp_solver_mode =
          static_cast<pdlp_solver_mode_t>(settings_map["pdlp_solver_mode"].as<int>());
      }
      if (settings_map.count("method")) {
        settings.method = static_cast<method_t>(settings_map["method"].as<int>());
      }
      if (settings_map.count("presolve")) {
        settings.presolve = settings_map["presolve"].as<bool>();
      }
      if (settings_map.count("dual_postsolve")) {
        settings.dual_postsolve = settings_map["dual_postsolve"].as<bool>();
      }
      if (settings_map.count("crossover")) {
        settings.crossover = settings_map["crossover"].as<bool>();
      }
      if (settings_map.count("num_gpus")) {
        settings.num_gpus = settings_map["num_gpus"].as<int>();
      }
      // Advanced options
      if (settings_map.count("per_constraint_residual")) {
        settings.per_constraint_residual = settings_map["per_constraint_residual"].as<bool>();
      }
      if (settings_map.count("cudss_deterministic")) {
        settings.cudss_deterministic = settings_map["cudss_deterministic"].as<bool>();
      }
      if (settings_map.count("folding")) { settings.folding = settings_map["folding"].as<i_t>(); }
      if (settings_map.count("augmented")) {
        settings.augmented = settings_map["augmented"].as<i_t>();
      }
      if (settings_map.count("dualize")) { settings.dualize = settings_map["dualize"].as<i_t>(); }
      if (settings_map.count("ordering")) {
        settings.ordering = settings_map["ordering"].as<i_t>();
      }
      if (settings_map.count("barrier_dual_initial_point")) {
        settings.barrier_dual_initial_point = settings_map["barrier_dual_initial_point"].as<i_t>();
      }
      if (settings_map.count("eliminate_dense_columns")) {
        settings.eliminate_dense_columns = settings_map["eliminate_dense_columns"].as<bool>();
      }
      if (settings_map.count("save_best_primal_so_far")) {
        settings.save_best_primal_so_far = settings_map["save_best_primal_so_far"].as<bool>();
      }
      if (settings_map.count("first_primal_feasible")) {
        settings.first_primal_feasible = settings_map["first_primal_feasible"].as<bool>();
      }

      return true;
    } catch (const std::exception& e) {
      std::cerr << "[msgpack_serializer] LP request parse error: " << e.what() << "\n";
      return false;
    }
  }

  bool deserialize_mip_request(const std::vector<uint8_t>& data,
                               mps_parser::mps_data_model_t<i_t, f_t>& mps_data,
                               mip_solver_settings_t<i_t, f_t>& settings) override
  {
    try {
      size_t offset = 0;

      msgpack::object_handle oh1 =
        msgpack::unpack(reinterpret_cast<const char*>(data.data()), data.size(), offset);
      uint8_t msg_type = oh1.get().as<uint8_t>();
      if (msg_type != MSG_MIP_REQUEST) return false;

      msgpack::object_handle oh2 =
        msgpack::unpack(reinterpret_cast<const char*>(data.data()), data.size(), offset);

      unpack_problem(data, offset, mps_data);

      msgpack::object_handle oh_settings =
        msgpack::unpack(reinterpret_cast<const char*>(data.data()), data.size(), offset);
      auto settings_map = oh_settings.get().as<std::map<std::string, msgpack::object>>();

      // Deserialize all MIP settings (field names match cuOpt API)
      // Limits
      if (settings_map.count("time_limit")) {
        settings.time_limit = settings_map["time_limit"].as<double>();
      }
      // Tolerances
      if (settings_map.count("relative_mip_gap")) {
        settings.tolerances.relative_mip_gap = settings_map["relative_mip_gap"].as<double>();
      }
      if (settings_map.count("absolute_mip_gap")) {
        settings.tolerances.absolute_mip_gap = settings_map["absolute_mip_gap"].as<double>();
      }
      if (settings_map.count("integrality_tolerance")) {
        settings.tolerances.integrality_tolerance =
          settings_map["integrality_tolerance"].as<double>();
      }
      if (settings_map.count("absolute_tolerance")) {
        settings.tolerances.absolute_tolerance = settings_map["absolute_tolerance"].as<double>();
      }
      if (settings_map.count("relative_tolerance")) {
        settings.tolerances.relative_tolerance = settings_map["relative_tolerance"].as<double>();
      }
      if (settings_map.count("presolve_absolute_tolerance")) {
        settings.tolerances.presolve_absolute_tolerance =
          settings_map["presolve_absolute_tolerance"].as<double>();
      }
      // Solver configuration
      if (settings_map.count("log_to_console")) {
        settings.log_to_console = settings_map["log_to_console"].as<bool>();
      }
      if (settings_map.count("heuristics_only")) {
        settings.heuristics_only = settings_map["heuristics_only"].as<bool>();
      }
      if (settings_map.count("num_cpu_threads")) {
        settings.num_cpu_threads = settings_map["num_cpu_threads"].as<i_t>();
      }
      if (settings_map.count("num_gpus")) {
        settings.num_gpus = settings_map["num_gpus"].as<i_t>();
      }
      if (settings_map.count("presolve")) {
        settings.presolve = settings_map["presolve"].as<bool>();
      }
      if (settings_map.count("mip_scaling")) {
        settings.mip_scaling = settings_map["mip_scaling"].as<bool>();
      }

      return true;
    } catch (const std::exception& e) {
      std::cerr << "[msgpack_serializer] MIP request parse error: " << e.what() << "\n";
      return false;
    }
  }

  std::vector<uint8_t> serialize_lp_solution(
    const optimization_problem_solution_t<i_t, f_t>& solution) override
  {
    msgpack::sbuffer buffer;
    msgpack::packer<msgpack::sbuffer> pk(&buffer);

    // Pack as array: [msg_type, status, obj_value, primal_sol, dual_sol, solve_time]
    pk.pack_array(6);
    pk.pack_uint8(MSG_LP_SOLUTION);
    pk.pack(static_cast<int>(solution.get_termination_status()));
    pk.pack(static_cast<double>(solution.get_objective_value()));

    // Note: If solution is on GPU, we can't access it from pure C++ code
    // For production, this should be a .cu file with CUDA support
    if (!solution.is_device_memory()) {
      auto primal = solution.get_primal_solution_host();
      pk.pack_array(primal.size());
      for (size_t i = 0; i < primal.size(); ++i) {
        pk.pack(static_cast<double>(primal[i]));
      }

      auto dual = solution.get_dual_solution_host();
      pk.pack_array(dual.size());
      for (size_t i = 0; i < dual.size(); ++i) {
        pk.pack(static_cast<double>(dual[i]));
      }
    } else {
      // GPU memory - return empty arrays (limitation of pure C++ serializer)
      pk.pack_array(0);
      pk.pack_array(0);
    }

    pk.pack(static_cast<double>(solution.get_solve_time()));

    return std::vector<uint8_t>(buffer.data(), buffer.data() + buffer.size());
  }

  std::vector<uint8_t> serialize_mip_solution(const mip_solution_t<i_t, f_t>& solution) override
  {
    msgpack::sbuffer buffer;
    msgpack::packer<msgpack::sbuffer> pk(&buffer);

    pk.pack_array(6);
    pk.pack_uint8(MSG_MIP_SOLUTION);
    pk.pack(static_cast<int>(solution.get_termination_status()));
    pk.pack(static_cast<double>(solution.get_objective_value()));

    if (!solution.is_device_memory()) {
      auto sol = solution.get_solution_host();
      pk.pack_array(sol.size());
      for (size_t i = 0; i < sol.size(); ++i) {
        pk.pack(static_cast<double>(sol[i]));
      }
    } else {
      pk.pack_array(0);
    }

    pk.pack(static_cast<double>(solution.get_stats().total_solve_time));
    pk.pack(static_cast<double>(solution.get_mip_gap()));

    return std::vector<uint8_t>(buffer.data(), buffer.data() + buffer.size());
  }

  //============================================================================
  // Async Operations
  //============================================================================

  std::vector<uint8_t> serialize_async_lp_request(
    const mps_parser::data_model_view_t<i_t, f_t>& view,
    const pdlp_solver_settings_t<i_t, f_t>& settings,
    bool blocking) override
  {
    msgpack::sbuffer buffer;
    msgpack::packer<msgpack::sbuffer> pk(&buffer);

    // Header: msg_type, blocking flag, job_id (empty for new submission)
    pk.pack_uint8(MSG_ASYNC_LP_REQUEST);
    pk.pack(blocking);
    pk.pack(std::string(""));  // job_id empty for new submission
    pk.pack(int64_t(0));       // frombyte (unused for LP requests)

    // Pack the problem and settings
    pk.pack_uint32(protocol_version());
    pack_problem(pk, view);

    // Pack all LP settings (field names match cuOpt API)
    pk.pack_map(28);
    // Termination tolerances
    pk.pack("absolute_gap_tolerance");
    pk.pack(settings.tolerances.absolute_gap_tolerance);
    pk.pack("relative_gap_tolerance");
    pk.pack(settings.tolerances.relative_gap_tolerance);
    pk.pack("primal_infeasible_tolerance");
    pk.pack(settings.tolerances.primal_infeasible_tolerance);
    pk.pack("dual_infeasible_tolerance");
    pk.pack(settings.tolerances.dual_infeasible_tolerance);
    pk.pack("absolute_dual_tolerance");
    pk.pack(settings.tolerances.absolute_dual_tolerance);
    pk.pack("relative_dual_tolerance");
    pk.pack(settings.tolerances.relative_dual_tolerance);
    pk.pack("absolute_primal_tolerance");
    pk.pack(settings.tolerances.absolute_primal_tolerance);
    pk.pack("relative_primal_tolerance");
    pk.pack(settings.tolerances.relative_primal_tolerance);
    // Limits
    pk.pack("time_limit");
    pk.pack(settings.time_limit);
    pk.pack("iteration_limit");
    pk.pack(static_cast<int64_t>(settings.iteration_limit));
    // Solver configuration
    pk.pack("log_to_console");
    pk.pack(settings.log_to_console);
    pk.pack("detect_infeasibility");
    pk.pack(settings.detect_infeasibility);
    pk.pack("strict_infeasibility");
    pk.pack(settings.strict_infeasibility);
    pk.pack("pdlp_solver_mode");
    pk.pack(static_cast<int>(settings.pdlp_solver_mode));
    pk.pack("method");
    pk.pack(static_cast<int>(settings.method));
    pk.pack("presolve");
    pk.pack(settings.presolve);
    pk.pack("dual_postsolve");
    pk.pack(settings.dual_postsolve);
    pk.pack("crossover");
    pk.pack(settings.crossover);
    pk.pack("num_gpus");
    pk.pack(settings.num_gpus);
    // Advanced options
    pk.pack("per_constraint_residual");
    pk.pack(settings.per_constraint_residual);
    pk.pack("cudss_deterministic");
    pk.pack(settings.cudss_deterministic);
    pk.pack("folding");
    pk.pack(settings.folding);
    pk.pack("augmented");
    pk.pack(settings.augmented);
    pk.pack("dualize");
    pk.pack(settings.dualize);
    pk.pack("ordering");
    pk.pack(settings.ordering);
    pk.pack("barrier_dual_initial_point");
    pk.pack(settings.barrier_dual_initial_point);
    pk.pack("eliminate_dense_columns");
    pk.pack(settings.eliminate_dense_columns);
    pk.pack("save_best_primal_so_far");
    pk.pack(settings.save_best_primal_so_far);
    pk.pack("first_primal_feasible");
    pk.pack(settings.first_primal_feasible);

    return std::vector<uint8_t>(buffer.data(), buffer.data() + buffer.size());
  }

  std::vector<uint8_t> serialize_async_mip_request(
    const mps_parser::data_model_view_t<i_t, f_t>& view,
    const mip_solver_settings_t<i_t, f_t>& settings,
    bool blocking) override
  {
    msgpack::sbuffer buffer;
    msgpack::packer<msgpack::sbuffer> pk(&buffer);

    // Header: msg_type, blocking flag, job_id (empty for new submission)
    pk.pack_uint8(MSG_ASYNC_MIP_REQUEST);
    pk.pack(blocking);
    pk.pack(std::string(""));  // job_id empty for new submission
    pk.pack(int64_t(0));       // frombyte (unused for MIP requests)

    // Pack the problem and settings
    pk.pack_uint32(protocol_version());
    pack_problem(pk, view);

    // Pack all MIP settings (field names match cuOpt API)
    pk.pack_map(13);
    // Limits
    pk.pack("time_limit");
    pk.pack(settings.time_limit);
    // Tolerances
    pk.pack("relative_mip_gap");
    pk.pack(settings.tolerances.relative_mip_gap);
    pk.pack("absolute_mip_gap");
    pk.pack(settings.tolerances.absolute_mip_gap);
    pk.pack("integrality_tolerance");
    pk.pack(settings.tolerances.integrality_tolerance);
    pk.pack("absolute_tolerance");
    pk.pack(settings.tolerances.absolute_tolerance);
    pk.pack("relative_tolerance");
    pk.pack(settings.tolerances.relative_tolerance);
    pk.pack("presolve_absolute_tolerance");
    pk.pack(settings.tolerances.presolve_absolute_tolerance);
    // Solver configuration
    pk.pack("log_to_console");
    pk.pack(settings.log_to_console);
    pk.pack("heuristics_only");
    pk.pack(settings.heuristics_only);
    pk.pack("num_cpu_threads");
    pk.pack(settings.num_cpu_threads);
    pk.pack("num_gpus");
    pk.pack(settings.num_gpus);
    pk.pack("presolve");
    pk.pack(settings.presolve);
    pk.pack("mip_scaling");
    pk.pack(settings.mip_scaling);

    return std::vector<uint8_t>(buffer.data(), buffer.data() + buffer.size());
  }

  std::vector<uint8_t> serialize_status_request(const std::string& job_id) override
  {
    msgpack::sbuffer buffer;
    msgpack::packer<msgpack::sbuffer> pk(&buffer);
    pk.pack_uint8(MSG_STATUS_REQUEST);
    pk.pack(false);  // blocking (unused)
    pk.pack(job_id);
    pk.pack(int64_t(0));  // frombyte (unused)
    return std::vector<uint8_t>(buffer.data(), buffer.data() + buffer.size());
  }

  std::vector<uint8_t> serialize_get_result_request(const std::string& job_id) override
  {
    msgpack::sbuffer buffer;
    msgpack::packer<msgpack::sbuffer> pk(&buffer);
    pk.pack_uint8(MSG_GET_RESULT);
    pk.pack(false);  // blocking (unused)
    pk.pack(job_id);
    pk.pack(int64_t(0));  // frombyte (unused)
    return std::vector<uint8_t>(buffer.data(), buffer.data() + buffer.size());
  }

  std::vector<uint8_t> serialize_delete_request(const std::string& job_id) override
  {
    msgpack::sbuffer buffer;
    msgpack::packer<msgpack::sbuffer> pk(&buffer);
    pk.pack_uint8(MSG_DELETE_REQUEST);
    pk.pack(false);  // blocking (unused)
    pk.pack(job_id);
    pk.pack(int64_t(0));  // frombyte (unused)
    return std::vector<uint8_t>(buffer.data(), buffer.data() + buffer.size());
  }

  std::vector<uint8_t> serialize_get_logs_request(const std::string& job_id,
                                                  int64_t frombyte = 0) override
  {
    msgpack::sbuffer buffer;
    msgpack::packer<msgpack::sbuffer> pk(&buffer);
    pk.pack_uint8(MSG_GET_LOGS);
    pk.pack(false);  // blocking (unused)
    pk.pack(job_id);
    pk.pack(frombyte);
    return std::vector<uint8_t>(buffer.data(), buffer.data() + buffer.size());
  }

  std::vector<uint8_t> serialize_cancel_request(const std::string& job_id) override
  {
    msgpack::sbuffer buffer;
    msgpack::packer<msgpack::sbuffer> pk(&buffer);
    pk.pack_uint8(MSG_CANCEL_REQUEST);
    pk.pack(false);  // blocking (unused)
    pk.pack(job_id);
    pk.pack(int64_t(0));  // frombyte (unused)
    return std::vector<uint8_t>(buffer.data(), buffer.data() + buffer.size());
  }

  bool deserialize_submit_response(const std::vector<uint8_t>& data,
                                   std::string& job_id,
                                   std::string& error_message) override
  {
    try {
      size_t offset = 0;
      msgpack::object_handle oh_type =
        msgpack::unpack(reinterpret_cast<const char*>(data.data()), data.size(), offset);
      uint8_t msg_type = oh_type.get().as<uint8_t>();

      if (msg_type != MSG_SUBMIT_RESPONSE) {
        error_message = "Invalid response type";
        return false;
      }

      msgpack::object_handle oh_success =
        msgpack::unpack(reinterpret_cast<const char*>(data.data()), data.size(), offset);
      bool success = oh_success.get().as<bool>();

      msgpack::object_handle oh_job_id =
        msgpack::unpack(reinterpret_cast<const char*>(data.data()), data.size(), offset);
      job_id = oh_job_id.get().as<std::string>();

      if (!success) {
        msgpack::object_handle oh_err =
          msgpack::unpack(reinterpret_cast<const char*>(data.data()), data.size(), offset);
        error_message = oh_err.get().as<std::string>();
      }

      return success;
    } catch (const std::exception& e) {
      error_message = std::string("MsgPack parse error: ") + e.what();
      return false;
    }
  }

  typename remote_serializer_t<i_t, f_t>::job_status_t deserialize_status_response(
    const std::vector<uint8_t>& data) override
  {
    using job_status_t = typename remote_serializer_t<i_t, f_t>::job_status_t;
    try {
      size_t offset = 0;
      msgpack::object_handle oh_type =
        msgpack::unpack(reinterpret_cast<const char*>(data.data()), data.size(), offset);
      uint8_t msg_type = oh_type.get().as<uint8_t>();

      if (msg_type != MSG_STATUS_RESPONSE) { return job_status_t::NOT_FOUND; }

      msgpack::object_handle oh_status =
        msgpack::unpack(reinterpret_cast<const char*>(data.data()), data.size(), offset);
      int status = oh_status.get().as<int>();

      // Status codes: 0=QUEUED, 1=PROCESSING, 2=COMPLETED, 3=FAILED, 4=NOT_FOUND, 5=CANCELLED
      switch (status) {
        case 0: return job_status_t::QUEUED;
        case 1: return job_status_t::PROCESSING;
        case 2: return job_status_t::COMPLETED;
        case 3: return job_status_t::FAILED;
        case 4: return job_status_t::NOT_FOUND;
        case 5: return job_status_t::CANCELLED;
        default: return job_status_t::NOT_FOUND;
      }
    } catch (...) {
      return job_status_t::NOT_FOUND;
    }
  }

  typename remote_serializer_t<i_t, f_t>::logs_result_t deserialize_logs_response(
    const std::vector<uint8_t>& data) override
  {
    typename remote_serializer_t<i_t, f_t>::logs_result_t result;
    result.nbytes     = 0;
    result.job_exists = false;

    try {
      size_t offset = 0;
      msgpack::object_handle oh_type =
        msgpack::unpack(reinterpret_cast<const char*>(data.data()), data.size(), offset);
      uint8_t msg_type = oh_type.get().as<uint8_t>();

      if (msg_type != MSG_LOGS_RESPONSE) { return result; }

      msgpack::object_handle oh_exists =
        msgpack::unpack(reinterpret_cast<const char*>(data.data()), data.size(), offset);
      result.job_exists = oh_exists.get().as<bool>();

      msgpack::object_handle oh_nbytes =
        msgpack::unpack(reinterpret_cast<const char*>(data.data()), data.size(), offset);
      result.nbytes = oh_nbytes.get().as<int64_t>();

      msgpack::object_handle oh_lines =
        msgpack::unpack(reinterpret_cast<const char*>(data.data()), data.size(), offset);
      auto lines_array = oh_lines.get().as<std::vector<std::string>>();
      result.log_lines = std::move(lines_array);
    } catch (...) {
    }
    return result;
  }

  typename remote_serializer_t<i_t, f_t>::cancel_result_t deserialize_cancel_response(
    const std::vector<uint8_t>& data) override
  {
    using job_status_t = typename remote_serializer_t<i_t, f_t>::job_status_t;
    typename remote_serializer_t<i_t, f_t>::cancel_result_t result;
    result.success    = false;
    result.message    = "Failed to parse response";
    result.job_status = job_status_t::NOT_FOUND;

    try {
      size_t offset = 0;
      msgpack::object_handle oh_type =
        msgpack::unpack(reinterpret_cast<const char*>(data.data()), data.size(), offset);
      uint8_t msg_type = oh_type.get().as<uint8_t>();

      if (msg_type != MSG_CANCEL_RESPONSE) { return result; }

      msgpack::object_handle oh_success =
        msgpack::unpack(reinterpret_cast<const char*>(data.data()), data.size(), offset);
      result.success = oh_success.get().as<bool>();

      msgpack::object_handle oh_message =
        msgpack::unpack(reinterpret_cast<const char*>(data.data()), data.size(), offset);
      result.message = oh_message.get().as<std::string>();

      msgpack::object_handle oh_status =
        msgpack::unpack(reinterpret_cast<const char*>(data.data()), data.size(), offset);
      int status_code = oh_status.get().as<int>();

      switch (status_code) {
        case 0: result.job_status = job_status_t::QUEUED; break;
        case 1: result.job_status = job_status_t::PROCESSING; break;
        case 2: result.job_status = job_status_t::COMPLETED; break;
        case 3: result.job_status = job_status_t::FAILED; break;
        case 4: result.job_status = job_status_t::NOT_FOUND; break;
        case 5: result.job_status = job_status_t::CANCELLED; break;
        default: result.job_status = job_status_t::NOT_FOUND; break;
      }
    } catch (...) {
    }
    return result;
  }

  optimization_problem_solution_t<i_t, f_t> deserialize_lp_result_response(
    const std::vector<uint8_t>& data) override
  {
    return deserialize_lp_solution(data);
  }

  mip_solution_t<i_t, f_t> deserialize_mip_result_response(
    const std::vector<uint8_t>& data) override
  {
    return deserialize_mip_solution(data);
  }

  //============================================================================
  // Server-side async request detection
  //============================================================================

  bool is_async_request(const std::vector<uint8_t>& data) override
  {
    if (data.empty()) return false;
    try {
      size_t offset = 0;
      msgpack::object_handle oh =
        msgpack::unpack(reinterpret_cast<const char*>(data.data()), data.size(), offset);
      if (oh.get().type == msgpack::type::POSITIVE_INTEGER) {
        uint8_t msg_type = oh.get().as<uint8_t>();
        return msg_type >= MSG_ASYNC_LP_REQUEST && msg_type <= MSG_CANCEL_REQUEST;
      }
    } catch (...) {
    }
    return false;
  }

  bool is_blocking_request(const std::vector<uint8_t>& data) override
  {
    if (data.empty()) return false;
    try {
      size_t offset = 0;
      // Skip msg_type
      msgpack::object_handle oh_type =
        msgpack::unpack(reinterpret_cast<const char*>(data.data()), data.size(), offset);

      // Read blocking flag
      msgpack::object_handle oh_blocking =
        msgpack::unpack(reinterpret_cast<const char*>(data.data()), data.size(), offset);
      return oh_blocking.get().as<bool>();
    } catch (...) {
    }
    return false;
  }

  std::vector<uint8_t> extract_problem_data(const std::vector<uint8_t>& data) override
  {
    // For msgpack, we extract the problem portion by repacking
    // The full request contains: msg_type, blocking, job_id, frombyte, version, problem, settings
    // We need to return a sync-style request: msg_type, version, problem, settings
    if (data.empty()) return {};

    try {
      size_t offset = 0;

      // Read header
      msgpack::object_handle oh_type =
        msgpack::unpack(reinterpret_cast<const char*>(data.data()), data.size(), offset);
      uint8_t msg_type = oh_type.get().as<uint8_t>();

      // Map async type to sync type
      uint8_t sync_type = (msg_type == MSG_ASYNC_MIP_REQUEST) ? MSG_MIP_REQUEST : MSG_LP_REQUEST;

      // Skip blocking, job_id, frombyte
      msgpack::unpack(reinterpret_cast<const char*>(data.data()), data.size(), offset);  // blocking
      msgpack::unpack(reinterpret_cast<const char*>(data.data()), data.size(), offset);  // job_id
      msgpack::unpack(reinterpret_cast<const char*>(data.data()), data.size(), offset);  // frombyte

      // Create sync-style request: msg_type, then rest of data (version, problem, settings)
      msgpack::sbuffer buffer;
      msgpack::packer<msgpack::sbuffer> pk(&buffer);
      pk.pack_uint8(sync_type);

      // Append the rest of the data (version, problem, settings)
      std::vector<uint8_t> result(buffer.data(), buffer.data() + buffer.size());
      result.insert(result.end(), data.begin() + offset, data.end());
      return result;

    } catch (...) {
    }
    return {};
  }

  int64_t get_frombyte(const std::vector<uint8_t>& data) override
  {
    if (data.empty()) return 0;
    try {
      size_t offset = 0;
      // Skip msg_type
      msgpack::unpack(reinterpret_cast<const char*>(data.data()), data.size(), offset);
      // Skip blocking
      msgpack::unpack(reinterpret_cast<const char*>(data.data()), data.size(), offset);
      // Skip job_id
      msgpack::unpack(reinterpret_cast<const char*>(data.data()), data.size(), offset);
      // Read frombyte
      msgpack::object_handle oh_frombyte =
        msgpack::unpack(reinterpret_cast<const char*>(data.data()), data.size(), offset);
      return oh_frombyte.get().as<int64_t>();
    } catch (...) {
    }
    return 0;
  }

  int get_async_request_type(const std::vector<uint8_t>& data) override
  {
    if (data.empty()) return -1;
    try {
      size_t offset = 0;
      msgpack::object_handle oh =
        msgpack::unpack(reinterpret_cast<const char*>(data.data()), data.size(), offset);
      uint8_t msg_type = oh.get().as<uint8_t>();

      // Map msgpack types to the RequestType enum expected by server
      switch (msg_type) {
        case MSG_ASYNC_LP_REQUEST:
        case MSG_ASYNC_MIP_REQUEST: return 0;  // SUBMIT_JOB
        case MSG_STATUS_REQUEST: return 1;     // CHECK_STATUS
        case MSG_GET_RESULT: return 2;         // GET_RESULT
        case MSG_DELETE_REQUEST: return 3;     // DELETE_RESULT
        case MSG_GET_LOGS: return 4;           // GET_LOGS
        case MSG_CANCEL_REQUEST: return 5;     // CANCEL_JOB
        case MSG_WAIT_REQUEST: return 6;       // WAIT_FOR_RESULT
        default: return -1;
      }
    } catch (...) {
    }
    return -1;
  }

  std::string get_job_id(const std::vector<uint8_t>& data) override
  {
    if (data.empty()) return "";
    try {
      size_t offset = 0;
      // Skip msg_type
      msgpack::unpack(reinterpret_cast<const char*>(data.data()), data.size(), offset);
      // Skip blocking
      msgpack::unpack(reinterpret_cast<const char*>(data.data()), data.size(), offset);
      // Read job_id
      msgpack::object_handle oh_job_id =
        msgpack::unpack(reinterpret_cast<const char*>(data.data()), data.size(), offset);
      return oh_job_id.get().as<std::string>();
    } catch (...) {
    }
    return "";
  }

  //============================================================================
  // Server-side response serialization
  //============================================================================

  std::vector<uint8_t> serialize_submit_response(bool success, const std::string& result) override
  {
    msgpack::sbuffer buffer;
    msgpack::packer<msgpack::sbuffer> pk(&buffer);
    pk.pack_uint8(MSG_SUBMIT_RESPONSE);
    pk.pack(success);
    pk.pack(result);                    // job_id on success, error message on failure
    if (!success) { pk.pack(result); }  // error message duplicated for compatibility
    return std::vector<uint8_t>(buffer.data(), buffer.data() + buffer.size());
  }

  std::vector<uint8_t> serialize_status_response(int status_code,
                                                 const std::string& message) override
  {
    msgpack::sbuffer buffer;
    msgpack::packer<msgpack::sbuffer> pk(&buffer);
    pk.pack_uint8(MSG_STATUS_RESPONSE);
    pk.pack(status_code);
    pk.pack(message);
    return std::vector<uint8_t>(buffer.data(), buffer.data() + buffer.size());
  }

  std::vector<uint8_t> serialize_result_response(bool success,
                                                 const std::vector<uint8_t>& result_data,
                                                 const std::string& error_message) override
  {
    // For result response, we prepend success flag then the actual solution data
    msgpack::sbuffer buffer;
    msgpack::packer<msgpack::sbuffer> pk(&buffer);
    pk.pack(success);
    pk.pack(error_message);
    // Append raw solution data
    std::vector<uint8_t> response(buffer.data(), buffer.data() + buffer.size());
    response.insert(response.end(), result_data.begin(), result_data.end());
    return response;
  }

  std::vector<uint8_t> serialize_delete_response(bool success) override
  {
    msgpack::sbuffer buffer;
    msgpack::packer<msgpack::sbuffer> pk(&buffer);
    pk.pack(success);
    return std::vector<uint8_t>(buffer.data(), buffer.data() + buffer.size());
  }

  std::vector<uint8_t> serialize_logs_response(const std::string& job_id,
                                               const std::vector<std::string>& log_lines,
                                               int64_t nbytes,
                                               bool job_exists) override
  {
    msgpack::sbuffer buffer;
    msgpack::packer<msgpack::sbuffer> pk(&buffer);
    pk.pack_uint8(MSG_LOGS_RESPONSE);
    pk.pack(job_exists);
    pk.pack(nbytes);
    pk.pack(log_lines);
    return std::vector<uint8_t>(buffer.data(), buffer.data() + buffer.size());
  }

  std::vector<uint8_t> serialize_cancel_response(bool success,
                                                 const std::string& message,
                                                 int status_code) override
  {
    msgpack::sbuffer buffer;
    msgpack::packer<msgpack::sbuffer> pk(&buffer);
    pk.pack_uint8(MSG_CANCEL_RESPONSE);
    pk.pack(success);
    pk.pack(message);
    pk.pack(status_code);
    return std::vector<uint8_t>(buffer.data(), buffer.data() + buffer.size());
  }

 private:
  void pack_problem(msgpack::packer<msgpack::sbuffer>& pk,
                    const mps_parser::data_model_view_t<i_t, f_t>& view)
  {
    // Field names match data_model_view_t (without trailing underscore)
    auto offsets_span = view.get_constraint_matrix_offsets();
    auto values_span  = view.get_constraint_matrix_values();
    auto obj_span     = view.get_objective_coefficients();

    i_t n_rows = static_cast<i_t>(offsets_span.size()) - 1;
    i_t n_cols = static_cast<i_t>(obj_span.size());
    i_t nnz    = static_cast<i_t>(values_span.size());

    // Count fields: base 20, plus optional fields
    int num_fields        = 20;
    auto init_primal_span = view.get_initial_primal_solution();
    auto init_dual_span   = view.get_initial_dual_solution();
    if (init_primal_span.size() > 0) num_fields++;
    if (init_dual_span.size() > 0) num_fields++;
    if (view.has_quadratic_objective()) num_fields += 3;

    pk.pack_map(num_fields);

    // Problem metadata
    pk.pack("problem_name");
    pk.pack(view.get_problem_name());
    pk.pack("objective_name");
    pk.pack(view.get_objective_name());
    pk.pack("maximize");
    pk.pack(view.get_sense());
    pk.pack("objective_scaling_factor");
    pk.pack(static_cast<double>(view.get_objective_scaling_factor()));
    pk.pack("objective_offset");
    pk.pack(static_cast<double>(view.get_objective_offset()));

    // Dimensions
    pk.pack("n_rows");
    pk.pack(static_cast<int64_t>(n_rows));
    pk.pack("n_cols");
    pk.pack(static_cast<int64_t>(n_cols));
    pk.pack("nnz");
    pk.pack(static_cast<int64_t>(nnz));

    // Constraint matrix A in CSR format (names match data_model_view_t: A_, A_indices_, A_offsets_)
    pk.pack("A_values");
    pk.pack_array(values_span.size());
    for (size_t i = 0; i < values_span.size(); ++i) {
      pk.pack(static_cast<double>(values_span.data()[i]));
    }

    pk.pack("A_indices");
    auto A_idx = view.get_constraint_matrix_indices();
    pk.pack_array(A_idx.size());
    for (size_t i = 0; i < A_idx.size(); ++i) {
      pk.pack(static_cast<int64_t>(A_idx.data()[i]));
    }

    pk.pack("A_offsets");
    pk.pack_array(offsets_span.size());
    for (size_t i = 0; i < offsets_span.size(); ++i) {
      pk.pack(static_cast<int64_t>(offsets_span.data()[i]));
    }

    // Objective coefficients c (name matches data_model_view_t: c_)
    pk.pack("c");
    pk.pack_array(obj_span.size());
    for (size_t i = 0; i < obj_span.size(); ++i) {
      pk.pack(static_cast<double>(obj_span.data()[i]));
    }

    // Variable bounds
    pk.pack("variable_lower_bounds");
    auto vlb = view.get_variable_lower_bounds();
    pk.pack_array(vlb.size());
    for (size_t i = 0; i < vlb.size(); ++i) {
      pk.pack(static_cast<double>(vlb.data()[i]));
    }

    pk.pack("variable_upper_bounds");
    auto vub = view.get_variable_upper_bounds();
    pk.pack_array(vub.size());
    for (size_t i = 0; i < vub.size(); ++i) {
      pk.pack(static_cast<double>(vub.data()[i]));
    }

    // Constraint bounds b (RHS)
    pk.pack("b");
    auto b_span = view.get_constraint_bounds();
    pk.pack_array(b_span.size());
    for (size_t i = 0; i < b_span.size(); ++i) {
      pk.pack(static_cast<double>(b_span.data()[i]));
    }

    // Row types
    pk.pack("row_types");
    auto rt_span = view.get_row_types();
    pk.pack(std::string(rt_span.data(), rt_span.size()));

    // Constraint lower/upper bounds
    pk.pack("constraint_lower_bounds");
    auto clb = view.get_constraint_lower_bounds();
    if (clb.size() > 0) {
      pk.pack_array(clb.size());
      for (size_t i = 0; i < clb.size(); ++i) {
        pk.pack(static_cast<double>(clb.data()[i]));
      }
    } else {
      pk.pack_array(0);
    }

    pk.pack("constraint_upper_bounds");
    auto cub = view.get_constraint_upper_bounds();
    if (cub.size() > 0) {
      pk.pack_array(cub.size());
      for (size_t i = 0; i < cub.size(); ++i) {
        pk.pack(static_cast<double>(cub.data()[i]));
      }
    } else {
      pk.pack_array(0);
    }

    // Variable types (name matches data_model_view_t: variable_types_)
    pk.pack("variable_types");
    auto vt = view.get_variable_types();
    pk.pack(std::string(vt.data(), vt.size()));

    // Initial solutions (if available)
    if (init_primal_span.size() > 0) {
      pk.pack("initial_primal_solution");
      pk.pack_array(init_primal_span.size());
      for (size_t i = 0; i < init_primal_span.size(); ++i) {
        pk.pack(static_cast<double>(init_primal_span.data()[i]));
      }
    }

    if (init_dual_span.size() > 0) {
      pk.pack("initial_dual_solution");
      pk.pack_array(init_dual_span.size());
      for (size_t i = 0; i < init_dual_span.size(); ++i) {
        pk.pack(static_cast<double>(init_dual_span.data()[i]));
      }
    }

    // Quadratic objective matrix Q (for QPS problems)
    if (view.has_quadratic_objective()) {
      pk.pack("Q_values");
      auto q_vals = view.get_quadratic_objective_values();
      pk.pack_array(q_vals.size());
      for (size_t i = 0; i < q_vals.size(); ++i) {
        pk.pack(static_cast<double>(q_vals.data()[i]));
      }

      pk.pack("Q_indices");
      auto q_idx = view.get_quadratic_objective_indices();
      pk.pack_array(q_idx.size());
      for (size_t i = 0; i < q_idx.size(); ++i) {
        pk.pack(static_cast<int64_t>(q_idx.data()[i]));
      }

      pk.pack("Q_offsets");
      auto q_off = view.get_quadratic_objective_offsets();
      pk.pack_array(q_off.size());
      for (size_t i = 0; i < q_off.size(); ++i) {
        pk.pack(static_cast<int64_t>(q_off.data()[i]));
      }
    }
  }

  void unpack_problem(const std::vector<uint8_t>& data,
                      size_t& offset,
                      mps_parser::mps_data_model_t<i_t, f_t>& mps_data)
  {
    // Field names match data_model_view_t (without trailing underscore)
    msgpack::object_handle oh =
      msgpack::unpack(reinterpret_cast<const char*>(data.data()), data.size(), offset);
    auto problem_map = oh.get().as<std::map<std::string, msgpack::object>>();

    // Problem metadata
    if (problem_map.count("problem_name")) {
      mps_data.set_problem_name(problem_map["problem_name"].as<std::string>());
    }
    if (problem_map.count("objective_name")) {
      mps_data.set_objective_name(problem_map["objective_name"].as<std::string>());
    }
    if (problem_map.count("maximize")) {
      mps_data.set_maximize(problem_map["maximize"].as<bool>());
    }
    if (problem_map.count("objective_scaling_factor")) {
      mps_data.set_objective_scaling_factor(problem_map["objective_scaling_factor"].as<double>());
    }
    if (problem_map.count("objective_offset")) {
      mps_data.set_objective_offset(problem_map["objective_offset"].as<double>());
    }

    // Constraint matrix A in CSR format
    std::vector<f_t> A_values;
    problem_map["A_values"].convert(A_values);
    std::vector<i_t> A_indices;
    problem_map["A_indices"].convert(A_indices);
    std::vector<i_t> A_offsets;
    problem_map["A_offsets"].convert(A_offsets);

    mps_data.set_csr_constraint_matrix(A_values.data(),
                                       static_cast<i_t>(A_values.size()),
                                       A_indices.data(),
                                       static_cast<i_t>(A_indices.size()),
                                       A_offsets.data(),
                                       static_cast<i_t>(A_offsets.size()));

    // Objective coefficients c
    std::vector<f_t> c;
    problem_map["c"].convert(c);
    mps_data.set_objective_coefficients(c.data(), static_cast<i_t>(c.size()));

    // Variable bounds
    std::vector<f_t> var_lb, var_ub;
    problem_map["variable_lower_bounds"].convert(var_lb);
    problem_map["variable_upper_bounds"].convert(var_ub);
    mps_data.set_variable_lower_bounds(var_lb.data(), static_cast<i_t>(var_lb.size()));
    mps_data.set_variable_upper_bounds(var_ub.data(), static_cast<i_t>(var_ub.size()));

    // Constraint bounds (prefer lower/upper bounds if available)
    if (problem_map.count("constraint_lower_bounds")) {
      std::vector<f_t> con_lb;
      problem_map["constraint_lower_bounds"].convert(con_lb);
      if (con_lb.size() > 0) {
        std::vector<f_t> con_ub;
        problem_map["constraint_upper_bounds"].convert(con_ub);
        mps_data.set_constraint_lower_bounds(con_lb.data(), static_cast<i_t>(con_lb.size()));
        mps_data.set_constraint_upper_bounds(con_ub.data(), static_cast<i_t>(con_ub.size()));
      }
    }

    // Constraint bounds b (RHS) + row_types format
    if (problem_map.count("b")) {
      std::vector<f_t> b;
      problem_map["b"].convert(b);
      if (b.size() > 0) { mps_data.set_constraint_bounds(b.data(), static_cast<i_t>(b.size())); }
    }

    if (problem_map.count("row_types")) {
      std::string row_types_str = problem_map["row_types"].as<std::string>();
      if (!row_types_str.empty()) {
        mps_data.set_row_types(row_types_str.data(), static_cast<i_t>(row_types_str.size()));
      }
    }

    // Variable types (stored as string, matching data_model_view_t)
    if (problem_map.count("variable_types")) {
      std::string var_types_str = problem_map["variable_types"].as<std::string>();
      if (!var_types_str.empty()) {
        std::vector<char> var_types(var_types_str.begin(), var_types_str.end());
        mps_data.set_variable_types(var_types);
      }
    }

    // Initial solutions (if provided)
    if (problem_map.count("initial_primal_solution")) {
      std::vector<f_t> init_primal;
      problem_map["initial_primal_solution"].convert(init_primal);
      if (init_primal.size() > 0) {
        mps_data.set_initial_primal_solution(init_primal.data(),
                                             static_cast<i_t>(init_primal.size()));
      }
    }

    if (problem_map.count("initial_dual_solution")) {
      std::vector<f_t> init_dual;
      problem_map["initial_dual_solution"].convert(init_dual);
      if (init_dual.size() > 0) {
        mps_data.set_initial_dual_solution(init_dual.data(), static_cast<i_t>(init_dual.size()));
      }
    }

    // Quadratic objective matrix Q (for QPS problems)
    if (problem_map.count("Q_values")) {
      std::vector<f_t> Q_values;
      std::vector<i_t> Q_indices;
      std::vector<i_t> Q_offsets;
      problem_map["Q_values"].convert(Q_values);
      problem_map["Q_indices"].convert(Q_indices);
      problem_map["Q_offsets"].convert(Q_offsets);

      if (Q_values.size() > 0) {
        mps_data.set_quadratic_objective_matrix(Q_values.data(),
                                                static_cast<i_t>(Q_values.size()),
                                                Q_indices.data(),
                                                static_cast<i_t>(Q_indices.size()),
                                                Q_offsets.data(),
                                                static_cast<i_t>(Q_offsets.size()));
      }
    }
  }
};

}  // namespace cuopt::linear_programming

//============================================================================
// Factory Functions (exported for dynamic loading)
//============================================================================

extern "C" {

std::unique_ptr<cuopt::linear_programming::remote_serializer_t<int32_t, double>>
create_cuopt_serializer_i32_f64()
{
  return std::make_unique<cuopt::linear_programming::msgpack_serializer_t<int32_t, double>>();
}

}  // extern "C"
