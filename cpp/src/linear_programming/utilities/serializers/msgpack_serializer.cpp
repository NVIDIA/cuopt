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

// Message type identifiers
constexpr uint8_t MSG_LP_REQUEST   = 1;
constexpr uint8_t MSG_MIP_REQUEST  = 2;
constexpr uint8_t MSG_LP_SOLUTION  = 3;
constexpr uint8_t MSG_MIP_SOLUTION = 4;

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

    // Pack LP settings
    pk.pack_map(4);
    pk.pack("time_limit");
    pk.pack(settings.time_limit);
    pk.pack("iteration_limit");
    pk.pack(static_cast<int64_t>(settings.iteration_limit));
    pk.pack("abs_gap_tol");
    pk.pack(settings.tolerances.absolute_gap_tolerance);
    pk.pack("rel_gap_tol");
    pk.pack(settings.tolerances.relative_gap_tolerance);

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

    pk.pack_map(2);
    pk.pack("time_limit");
    pk.pack(settings.time_limit);
    pk.pack("mip_gap");
    pk.pack(settings.tolerances.relative_mip_gap);

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
        return oh.get().as<uint8_t>() == MSG_MIP_REQUEST;
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

      if (settings_map.count("time_limit")) {
        settings.time_limit = settings_map["time_limit"].as<double>();
      }
      if (settings_map.count("iteration_limit")) {
        settings.iteration_limit = settings_map["iteration_limit"].as<i_t>();
      }
      if (settings_map.count("abs_gap_tol")) {
        settings.tolerances.absolute_gap_tolerance = settings_map["abs_gap_tol"].as<double>();
      }
      if (settings_map.count("rel_gap_tol")) {
        settings.tolerances.relative_gap_tolerance = settings_map["rel_gap_tol"].as<double>();
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

      if (settings_map.count("time_limit")) {
        settings.time_limit = settings_map["time_limit"].as<double>();
      }
      if (settings_map.count("mip_gap")) {
        settings.tolerances.relative_mip_gap = settings_map["mip_gap"].as<double>();
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
  // Async Operations (simplified for testing)
  //============================================================================

  std::vector<uint8_t> serialize_async_lp_request(
    const mps_parser::data_model_view_t<i_t, f_t>& view,
    const pdlp_solver_settings_t<i_t, f_t>& settings,
    bool blocking) override
  {
    return serialize_lp_request(view, settings);
  }

  std::vector<uint8_t> serialize_async_mip_request(
    const mps_parser::data_model_view_t<i_t, f_t>& view,
    const mip_solver_settings_t<i_t, f_t>& settings,
    bool blocking) override
  {
    return serialize_mip_request(view, settings);
  }

  std::vector<uint8_t> serialize_status_request(const std::string& job_id) override
  {
    msgpack::sbuffer buffer;
    msgpack::packer<msgpack::sbuffer> pk(&buffer);
    pk.pack_map(1);
    pk.pack("job_id");
    pk.pack(job_id);
    return std::vector<uint8_t>(buffer.data(), buffer.data() + buffer.size());
  }

  std::vector<uint8_t> serialize_get_result_request(const std::string& job_id) override
  {
    return serialize_status_request(job_id);
  }

  std::vector<uint8_t> serialize_delete_request(const std::string& job_id) override
  {
    return serialize_status_request(job_id);
  }

  bool deserialize_submit_response(const std::vector<uint8_t>& data,
                                   std::string& job_id,
                                   std::string& error_message) override
  {
    job_id = "msgpack_job";
    return true;
  }

  typename remote_serializer_t<i_t, f_t>::job_status_t deserialize_status_response(
    const std::vector<uint8_t>& data) override
  {
    return remote_serializer_t<i_t, f_t>::job_status_t::COMPLETED;
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

 private:
  void pack_problem(msgpack::packer<msgpack::sbuffer>& pk,
                    const mps_parser::data_model_view_t<i_t, f_t>& view)
  {
    auto offsets_span = view.get_constraint_matrix_offsets();
    auto values_span  = view.get_constraint_matrix_values();
    auto obj_span     = view.get_objective_coefficients();

    i_t n_rows = static_cast<i_t>(offsets_span.size()) - 1;
    i_t n_cols = static_cast<i_t>(obj_span.size());
    i_t nnz    = static_cast<i_t>(values_span.size());

    pk.pack_map(13);

    pk.pack("n_rows");
    pk.pack(static_cast<int64_t>(n_rows));

    pk.pack("n_cols");
    pk.pack(static_cast<int64_t>(n_cols));

    pk.pack("nnz");
    pk.pack(static_cast<int64_t>(nnz));

    pk.pack("maximize");
    pk.pack(view.get_sense());

    // CSR matrix
    pk.pack("A_values");
    auto A_vals = view.get_constraint_matrix_values();
    pk.pack_array(A_vals.size());
    for (size_t i = 0; i < A_vals.size(); ++i) {
      pk.pack(static_cast<double>(A_vals.data()[i]));
    }

    pk.pack("A_indices");
    auto A_idx = view.get_constraint_matrix_indices();
    pk.pack_array(A_idx.size());
    for (size_t i = 0; i < A_idx.size(); ++i) {
      pk.pack(static_cast<int64_t>(A_idx.data()[i]));
    }

    pk.pack("A_offsets");
    auto A_off = view.get_constraint_matrix_offsets();
    pk.pack_array(A_off.size());
    for (size_t i = 0; i < A_off.size(); ++i) {
      pk.pack(static_cast<int64_t>(A_off.data()[i]));
    }

    pk.pack("obj_coeffs");
    auto obj = view.get_objective_coefficients();
    pk.pack_array(obj.size());
    for (size_t i = 0; i < obj.size(); ++i) {
      pk.pack(static_cast<double>(obj.data()[i]));
    }

    pk.pack("var_lb");
    auto vlb = view.get_variable_lower_bounds();
    pk.pack_array(vlb.size());
    for (size_t i = 0; i < vlb.size(); ++i) {
      pk.pack(static_cast<double>(vlb.data()[i]));
    }

    pk.pack("var_ub");
    auto vub = view.get_variable_upper_bounds();
    pk.pack_array(vub.size());
    for (size_t i = 0; i < vub.size(); ++i) {
      pk.pack(static_cast<double>(vub.data()[i]));
    }

    // Constraint bounds - derive from row_types if needed
    pk.pack("con_lb");
    auto clb = view.get_constraint_lower_bounds();
    if (clb.size() > 0) {
      pk.pack_array(clb.size());
      for (size_t i = 0; i < clb.size(); ++i) {
        pk.pack(static_cast<double>(clb.data()[i]));
      }
    } else {
      auto b_span  = view.get_constraint_bounds();
      auto rt_span = view.get_row_types();
      pk.pack_array(n_rows);
      for (i_t i = 0; i < n_rows; ++i) {
        char rt = (rt_span.size() > 0) ? rt_span.data()[i] : 'E';
        f_t b   = (b_span.size() > 0) ? b_span.data()[i] : 0;
        f_t lb  = (rt == 'G' || rt == 'E') ? b : -1e20;
        pk.pack(static_cast<double>(lb));
      }
    }

    pk.pack("con_ub");
    auto cub = view.get_constraint_upper_bounds();
    if (cub.size() > 0) {
      pk.pack_array(cub.size());
      for (size_t i = 0; i < cub.size(); ++i) {
        pk.pack(static_cast<double>(cub.data()[i]));
      }
    } else {
      auto b_span  = view.get_constraint_bounds();
      auto rt_span = view.get_row_types();
      pk.pack_array(n_rows);
      for (i_t i = 0; i < n_rows; ++i) {
        char rt = (rt_span.size() > 0) ? rt_span.data()[i] : 'E';
        f_t b   = (b_span.size() > 0) ? b_span.data()[i] : 0;
        f_t ub  = (rt == 'L' || rt == 'E') ? b : 1e20;
        pk.pack(static_cast<double>(ub));
      }
    }

    // Variable types
    pk.pack("var_types");
    auto vt = view.get_variable_types();
    pk.pack_array(n_cols);
    for (i_t i = 0; i < n_cols; ++i) {
      char t = (vt.size() > 0) ? vt.data()[i] : 'C';
      pk.pack(std::string(1, t));
    }
  }

  void unpack_problem(const std::vector<uint8_t>& data,
                      size_t& offset,
                      mps_parser::mps_data_model_t<i_t, f_t>& mps_data)
  {
    msgpack::object_handle oh =
      msgpack::unpack(reinterpret_cast<const char*>(data.data()), data.size(), offset);
    auto problem_map = oh.get().as<std::map<std::string, msgpack::object>>();

    i_t n_cols = problem_map["n_cols"].as<int64_t>();

    mps_data.set_maximize(problem_map["maximize"].as<bool>());

    // CSR matrix
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

    std::vector<f_t> obj_coeffs;
    problem_map["obj_coeffs"].convert(obj_coeffs);
    mps_data.set_objective_coefficients(obj_coeffs.data(), static_cast<i_t>(obj_coeffs.size()));

    std::vector<f_t> var_lb, var_ub;
    problem_map["var_lb"].convert(var_lb);
    problem_map["var_ub"].convert(var_ub);
    mps_data.set_variable_lower_bounds(var_lb.data(), static_cast<i_t>(var_lb.size()));
    mps_data.set_variable_upper_bounds(var_ub.data(), static_cast<i_t>(var_ub.size()));

    std::vector<f_t> con_lb, con_ub;
    problem_map["con_lb"].convert(con_lb);
    problem_map["con_ub"].convert(con_ub);
    mps_data.set_constraint_lower_bounds(con_lb.data(), static_cast<i_t>(con_lb.size()));
    mps_data.set_constraint_upper_bounds(con_ub.data(), static_cast<i_t>(con_ub.size()));

    // Variable types
    std::vector<char> var_types;
    if (problem_map["var_types"].type == msgpack::type::ARRAY) {
      auto arr = problem_map["var_types"].via.array;
      for (size_t i = 0; i < arr.size; ++i) {
        if (arr.ptr[i].type == msgpack::type::STR) {
          std::string s;
          arr.ptr[i].convert(s);
          var_types.push_back(s.empty() ? 'C' : s[0]);
        } else {
          var_types.push_back('C');
        }
      }
    } else {
      var_types.resize(n_cols, 'C');
    }
    mps_data.set_variable_types(var_types);
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
