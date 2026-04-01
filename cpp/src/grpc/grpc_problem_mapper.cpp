/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#include "grpc_problem_mapper.hpp"

#include <cuopt/linear_programming/constants.h>
#include <cuopt_remote.pb.h>
#include <cuopt_remote_service.pb.h>
#include <cuopt/linear_programming/cpu_optimization_problem.hpp>
#include <cuopt/linear_programming/mip/solver_settings.hpp>
#include <cuopt/linear_programming/optimization_problem_interface.hpp>
#include <cuopt/linear_programming/pdlp/solver_settings.hpp>
#include "grpc_settings_mapper.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>

namespace cuopt::linear_programming {

template <typename i_t, typename f_t>
void map_problem_to_proto(const cpu_optimization_problem_t<i_t, f_t>& cpu_problem,
                          cuopt::remote::OptimizationProblem* pb_problem)
{
  // Basic problem metadata
  pb_problem->set_problem_name(cpu_problem.get_problem_name());
  pb_problem->set_objective_name(cpu_problem.get_objective_name());
  pb_problem->set_maximize(cpu_problem.get_sense());
  pb_problem->set_objective_scaling_factor(cpu_problem.get_objective_scaling_factor());
  pb_problem->set_objective_offset(cpu_problem.get_objective_offset());

  // Get constraint matrix data from host memory
  auto values  = cpu_problem.get_constraint_matrix_values_host();
  auto indices = cpu_problem.get_constraint_matrix_indices_host();
  auto offsets = cpu_problem.get_constraint_matrix_offsets_host();

  // Constraint matrix A in CSR format
  for (const auto& val : values) {
    pb_problem->add_a(static_cast<double>(val));
  }
  for (const auto& idx : indices) {
    pb_problem->add_a_indices(static_cast<int32_t>(idx));
  }
  for (const auto& off : offsets) {
    pb_problem->add_a_offsets(static_cast<int32_t>(off));
  }

  // Objective coefficients
  auto obj_coeffs = cpu_problem.get_objective_coefficients_host();
  for (const auto& c : obj_coeffs) {
    pb_problem->add_c(static_cast<double>(c));
  }

  // Variable bounds
  auto var_lb = cpu_problem.get_variable_lower_bounds_host();
  auto var_ub = cpu_problem.get_variable_upper_bounds_host();
  for (const auto& lb : var_lb) {
    pb_problem->add_variable_lower_bounds(static_cast<double>(lb));
  }
  for (const auto& ub : var_ub) {
    pb_problem->add_variable_upper_bounds(static_cast<double>(ub));
  }

  // Constraint bounds
  auto con_lb = cpu_problem.get_constraint_lower_bounds_host();
  auto con_ub = cpu_problem.get_constraint_upper_bounds_host();

  if (!con_lb.empty() && !con_ub.empty()) {
    for (const auto& lb : con_lb) {
      pb_problem->add_constraint_lower_bounds(static_cast<double>(lb));
    }
    for (const auto& ub : con_ub) {
      pb_problem->add_constraint_upper_bounds(static_cast<double>(ub));
    }
  }

  // Row types (if available)
  auto row_types = cpu_problem.get_row_types_host();
  if (!row_types.empty()) {
    pb_problem->set_row_types(std::string(row_types.begin(), row_types.end()));
  }

  // Constraint bounds (RHS) - if available
  auto b = cpu_problem.get_constraint_bounds_host();
  if (!b.empty()) {
    for (const auto& rhs : b) {
      pb_problem->add_b(static_cast<double>(rhs));
    }
  }

  // Variable names
  const auto& var_names = cpu_problem.get_variable_names();
  for (const auto& name : var_names) {
    pb_problem->add_variable_names(name);
  }

  // Row names
  const auto& row_names = cpu_problem.get_row_names();
  for (const auto& name : row_names) {
    pb_problem->add_row_names(name);
  }

  // Variable types (for MIP problems)
  auto var_types = cpu_problem.get_variable_types_host();
  if (!var_types.empty()) {
    // Convert var_t enum to char representation
    std::string var_types_str;
    var_types_str.reserve(var_types.size());
    for (const auto& vt : var_types) {
      switch (vt) {
        case var_t::CONTINUOUS: var_types_str.push_back('C'); break;
        case var_t::INTEGER: var_types_str.push_back('I'); break;
        default:
          throw std::runtime_error("map_problem_to_proto: unknown var_t value " +
                                   std::to_string(static_cast<int>(vt)));
      }
    }
    pb_problem->set_variable_types(var_types_str);
  }

  // Quadratic objective matrix Q (for QPS problems)
  if (cpu_problem.has_quadratic_objective()) {
    const auto& q_values  = cpu_problem.get_quadratic_objective_values();
    const auto& q_indices = cpu_problem.get_quadratic_objective_indices();
    const auto& q_offsets = cpu_problem.get_quadratic_objective_offsets();

    for (const auto& val : q_values) {
      pb_problem->add_q_values(static_cast<double>(val));
    }
    for (const auto& idx : q_indices) {
      pb_problem->add_q_indices(static_cast<int32_t>(idx));
    }
    for (const auto& off : q_offsets) {
      pb_problem->add_q_offsets(static_cast<int32_t>(off));
    }
  }
}

template <typename i_t, typename f_t>
void map_proto_to_problem(const cuopt::remote::OptimizationProblem& pb_problem,
                          cpu_optimization_problem_t<i_t, f_t>& cpu_problem)
{
  // Basic problem metadata
  cpu_problem.set_problem_name(pb_problem.problem_name());
  cpu_problem.set_objective_name(pb_problem.objective_name());
  cpu_problem.set_maximize(pb_problem.maximize());
  cpu_problem.set_objective_scaling_factor(pb_problem.objective_scaling_factor());
  cpu_problem.set_objective_offset(pb_problem.objective_offset());

  // Constraint matrix A in CSR format
  std::vector<f_t> values(pb_problem.a().begin(), pb_problem.a().end());
  std::vector<i_t> indices(pb_problem.a_indices().begin(), pb_problem.a_indices().end());
  std::vector<i_t> offsets(pb_problem.a_offsets().begin(), pb_problem.a_offsets().end());

  cpu_problem.set_csr_constraint_matrix(values.data(),
                                        static_cast<i_t>(values.size()),
                                        indices.data(),
                                        static_cast<i_t>(indices.size()),
                                        offsets.data(),
                                        static_cast<i_t>(offsets.size()));

  // Objective coefficients
  std::vector<f_t> obj(pb_problem.c().begin(), pb_problem.c().end());
  cpu_problem.set_objective_coefficients(obj.data(), static_cast<i_t>(obj.size()));

  // Variable bounds
  std::vector<f_t> var_lb(pb_problem.variable_lower_bounds().begin(),
                          pb_problem.variable_lower_bounds().end());
  std::vector<f_t> var_ub(pb_problem.variable_upper_bounds().begin(),
                          pb_problem.variable_upper_bounds().end());
  cpu_problem.set_variable_lower_bounds(var_lb.data(), static_cast<i_t>(var_lb.size()));
  cpu_problem.set_variable_upper_bounds(var_ub.data(), static_cast<i_t>(var_ub.size()));

  // Constraint bounds (prefer lower/upper bounds if available)
  if (pb_problem.constraint_lower_bounds_size() > 0 &&
      pb_problem.constraint_upper_bounds_size() > 0 &&
      pb_problem.constraint_lower_bounds_size() == pb_problem.constraint_upper_bounds_size()) {
    std::vector<f_t> con_lb(pb_problem.constraint_lower_bounds().begin(),
                            pb_problem.constraint_lower_bounds().end());
    std::vector<f_t> con_ub(pb_problem.constraint_upper_bounds().begin(),
                            pb_problem.constraint_upper_bounds().end());
    cpu_problem.set_constraint_lower_bounds(con_lb.data(), static_cast<i_t>(con_lb.size()));
    cpu_problem.set_constraint_upper_bounds(con_ub.data(), static_cast<i_t>(con_ub.size()));
  } else if (pb_problem.b_size() > 0) {
    // Use b (RHS) + row_types format
    std::vector<f_t> b(pb_problem.b().begin(), pb_problem.b().end());
    cpu_problem.set_constraint_bounds(b.data(), static_cast<i_t>(b.size()));

    if (!pb_problem.row_types().empty()) {
      const std::string& row_types_str = pb_problem.row_types();
      cpu_problem.set_row_types(row_types_str.data(), static_cast<i_t>(row_types_str.size()));
    }
  }

  // Variable names
  if (pb_problem.variable_names_size() > 0) {
    std::vector<std::string> var_names(pb_problem.variable_names().begin(),
                                       pb_problem.variable_names().end());
    cpu_problem.set_variable_names(var_names);
  }

  // Row names
  if (pb_problem.row_names_size() > 0) {
    std::vector<std::string> row_names(pb_problem.row_names().begin(),
                                       pb_problem.row_names().end());
    cpu_problem.set_row_names(row_names);
  }

  // Variable types
  if (!pb_problem.variable_types().empty()) {
    const std::string& var_types_str = pb_problem.variable_types();
    // Convert char representation to var_t enum
    std::vector<var_t> var_types;
    var_types.reserve(var_types_str.size());
    for (char c : var_types_str) {
      switch (c) {
        case 'C': var_types.push_back(var_t::CONTINUOUS); break;
        case 'I':
        case 'B': var_types.push_back(var_t::INTEGER); break;
        default:
          throw std::runtime_error(std::string("Unknown variable type character '") + c +
                                   "' in variable_types string (expected 'C', 'I', or 'B')");
      }
    }
    cpu_problem.set_variable_types(var_types.data(), static_cast<i_t>(var_types.size()));
  }

  // Quadratic objective matrix Q (for QPS problems)
  if (pb_problem.q_values_size() > 0) {
    std::vector<f_t> q_values(pb_problem.q_values().begin(), pb_problem.q_values().end());
    std::vector<i_t> q_indices(pb_problem.q_indices().begin(), pb_problem.q_indices().end());
    std::vector<i_t> q_offsets(pb_problem.q_offsets().begin(), pb_problem.q_offsets().end());

    cpu_problem.set_quadratic_objective_matrix(q_values.data(),
                                               static_cast<i_t>(q_values.size()),
                                               q_indices.data(),
                                               static_cast<i_t>(q_indices.size()),
                                               q_offsets.data(),
                                               static_cast<i_t>(q_offsets.size()));
  }

  // Infer problem category from variable types
  if (!pb_problem.variable_types().empty()) {
    const std::string& var_types_str = pb_problem.variable_types();
    bool has_integers                = false;
    for (char c : var_types_str) {
      if (c == 'I' || c == 'B') {
        has_integers = true;
        break;
      }
    }
    cpu_problem.set_problem_category(has_integers ? problem_category_t::MIP
                                                  : problem_category_t::LP);
  } else {
    cpu_problem.set_problem_category(problem_category_t::LP);
  }
}

// ============================================================================
// Size estimation
// ============================================================================

template <typename i_t, typename f_t>
size_t estimate_problem_proto_size(const cpu_optimization_problem_t<i_t, f_t>& cpu_problem)
{
  size_t est = 0;

  // Constraint matrix CSR arrays
  auto values  = cpu_problem.get_constraint_matrix_values_host();
  auto indices = cpu_problem.get_constraint_matrix_indices_host();
  auto offsets = cpu_problem.get_constraint_matrix_offsets_host();
  est += values.size() * sizeof(double);  // packed repeated double
  est += indices.size() * 5;              // varint int32 (worst case 5 bytes each)
  est += offsets.size() * 5;

  // Objective coefficients
  est += cpu_problem.get_objective_coefficients_host().size() * sizeof(double);

  // Variable bounds
  est += cpu_problem.get_variable_lower_bounds_host().size() * sizeof(double);
  est += cpu_problem.get_variable_upper_bounds_host().size() * sizeof(double);

  // Constraint bounds
  est += cpu_problem.get_constraint_lower_bounds_host().size() * sizeof(double);
  est += cpu_problem.get_constraint_upper_bounds_host().size() * sizeof(double);
  est += cpu_problem.get_constraint_bounds_host().size() * sizeof(double);

  // Row types and variable types
  est += cpu_problem.get_row_types_host().size();
  est += cpu_problem.get_variable_types_host().size();

  // Quadratic objective
  if (cpu_problem.has_quadratic_objective()) {
    est += cpu_problem.get_quadratic_objective_values().size() * sizeof(double);
    est += cpu_problem.get_quadratic_objective_indices().size() * 5;
    est += cpu_problem.get_quadratic_objective_offsets().size() * 5;
  }

  // String arrays (rough estimate)
  for (const auto& name : cpu_problem.get_variable_names()) {
    est += name.size() + 2;  // string + tag + length varint
  }
  for (const auto& name : cpu_problem.get_row_names()) {
    est += name.size() + 2;
  }

  // Protobuf overhead for tags, submessage lengths, etc.
  est += 512;

  return est;
}

// ============================================================================
// Chunked header population (client-side, for CHUNKED_ARRAYS upload)
// ============================================================================

template <typename i_t, typename f_t>
void populate_chunked_header_lp(const cpu_optimization_problem_t<i_t, f_t>& cpu_problem,
                                const pdlp_solver_settings_t<i_t, f_t>& settings,
                                cuopt::remote::ChunkedProblemHeader* header)
{
  // Request header
  auto* rh = header->mutable_header();
  rh->set_version(1);
  rh->set_problem_category(cuopt::remote::LP);

  header->set_maximize(cpu_problem.get_sense());
  header->set_objective_scaling_factor(cpu_problem.get_objective_scaling_factor());
  header->set_objective_offset(cpu_problem.get_objective_offset());
  header->set_problem_name(cpu_problem.get_problem_name());
  header->set_objective_name(cpu_problem.get_objective_name());

  // Variable/row names are sent as chunked arrays, not in the header,
  // to avoid the header exceeding gRPC max message size for large problems.

  // LP settings
  map_pdlp_settings_to_proto(settings, header->mutable_lp_settings());
}

template <typename i_t, typename f_t>
void populate_chunked_header_mip(const cpu_optimization_problem_t<i_t, f_t>& cpu_problem,
                                 const mip_solver_settings_t<i_t, f_t>& settings,
                                 bool enable_incumbents,
                                 cuopt::remote::ChunkedProblemHeader* header)
{
  // Request header
  auto* rh = header->mutable_header();
  rh->set_version(1);
  rh->set_problem_category(cuopt::remote::MIP);

  header->set_maximize(cpu_problem.get_sense());
  header->set_objective_scaling_factor(cpu_problem.get_objective_scaling_factor());
  header->set_objective_offset(cpu_problem.get_objective_offset());
  header->set_problem_name(cpu_problem.get_problem_name());
  header->set_objective_name(cpu_problem.get_objective_name());

  // Variable/row names are sent as chunked arrays, not in the header.

  // MIP settings
  map_mip_settings_to_proto(settings, header->mutable_mip_settings());
  header->set_enable_incumbents(enable_incumbents);
}

// ============================================================================
// Chunked header reconstruction (server-side)
// ============================================================================

template <typename i_t, typename f_t>
void map_chunked_header_to_problem(const cuopt::remote::ChunkedProblemHeader& header,
                                   cpu_optimization_problem_t<i_t, f_t>& cpu_problem)
{
  cpu_problem.set_problem_name(header.problem_name());
  cpu_problem.set_objective_name(header.objective_name());
  cpu_problem.set_maximize(header.maximize());
  cpu_problem.set_objective_scaling_factor(header.objective_scaling_factor());
  cpu_problem.set_objective_offset(header.objective_offset());

  // String arrays
  if (header.variable_names_size() > 0) {
    std::vector<std::string> var_names(header.variable_names().begin(),
                                       header.variable_names().end());
    cpu_problem.set_variable_names(var_names);
  }
  if (header.row_names_size() > 0) {
    std::vector<std::string> row_names(header.row_names().begin(), header.row_names().end());
    cpu_problem.set_row_names(row_names);
  }

  // Problem category inferred later when variable_types array is set
}

// ============================================================================
// Chunked array reconstruction (server-side, consolidates all array mapping)
// ============================================================================

template <typename i_t, typename f_t>
void map_chunked_arrays_to_problem(const cuopt::remote::ChunkedProblemHeader& header,
                                   const std::map<int32_t, std::vector<uint8_t>>& arrays,
                                   cpu_optimization_problem_t<i_t, f_t>& cpu_problem)
{
  map_chunked_header_to_problem(header, cpu_problem);

  auto get_doubles = [&](int32_t field_id) -> std::vector<f_t> {
    auto it = arrays.find(field_id);
    if (it == arrays.end() || it->second.empty()) return {};
    if (it->second.size() % sizeof(double) != 0) return {};
    size_t n = it->second.size() / sizeof(double);
    if constexpr (std::is_same_v<f_t, double>) {
      std::vector<double> v(n);
      std::memcpy(v.data(), it->second.data(), n * sizeof(double));
      return v;
    } else {
      std::vector<double> tmp(n);
      std::memcpy(tmp.data(), it->second.data(), n * sizeof(double));
      return std::vector<f_t>(tmp.begin(), tmp.end());
    }
  };

  auto get_ints = [&](int32_t field_id) -> std::vector<i_t> {
    auto it = arrays.find(field_id);
    if (it == arrays.end() || it->second.empty()) return {};
    if (it->second.size() % sizeof(int32_t) != 0) return {};
    size_t n = it->second.size() / sizeof(int32_t);
    if constexpr (std::is_same_v<i_t, int32_t>) {
      std::vector<int32_t> v(n);
      std::memcpy(v.data(), it->second.data(), n * sizeof(int32_t));
      return v;
    } else {
      std::vector<int32_t> tmp(n);
      std::memcpy(tmp.data(), it->second.data(), n * sizeof(int32_t));
      return std::vector<i_t>(tmp.begin(), tmp.end());
    }
  };

  auto get_bytes = [&](int32_t field_id) -> std::string {
    auto it = arrays.find(field_id);
    if (it == arrays.end() || it->second.empty()) return {};
    return std::string(reinterpret_cast<const char*>(it->second.data()), it->second.size());
  };

  auto get_string_list = [&](int32_t field_id) -> std::vector<std::string> {
    auto it = arrays.find(field_id);
    if (it == arrays.end() || it->second.empty()) return {};
    std::vector<std::string> names;
    const char* s     = reinterpret_cast<const char*>(it->second.data());
    const char* s_end = s + it->second.size();
    while (s < s_end) {
      const char* nul = static_cast<const char*>(std::memchr(s, '\0', s_end - s));
      if (!nul) nul = s_end;
      names.emplace_back(s, nul);
      if (nul == s_end) break;
      s = nul + 1;
    }
    return names;
  };

  // CSR constraint matrix
  auto a_values  = get_doubles(cuopt::remote::FIELD_A_VALUES);
  auto a_indices = get_ints(cuopt::remote::FIELD_A_INDICES);
  auto a_offsets = get_ints(cuopt::remote::FIELD_A_OFFSETS);
  if (!a_values.empty() && !a_indices.empty() && !a_offsets.empty()) {
    cpu_problem.set_csr_constraint_matrix(a_values.data(),
                                          static_cast<i_t>(a_values.size()),
                                          a_indices.data(),
                                          static_cast<i_t>(a_indices.size()),
                                          a_offsets.data(),
                                          static_cast<i_t>(a_offsets.size()));
  }

  // Objective coefficients
  auto c_vec = get_doubles(cuopt::remote::FIELD_C);
  if (!c_vec.empty()) {
    cpu_problem.set_objective_coefficients(c_vec.data(), static_cast<i_t>(c_vec.size()));
  }

  // Variable bounds
  auto var_lb = get_doubles(cuopt::remote::FIELD_VARIABLE_LOWER_BOUNDS);
  auto var_ub = get_doubles(cuopt::remote::FIELD_VARIABLE_UPPER_BOUNDS);
  if (!var_lb.empty()) {
    cpu_problem.set_variable_lower_bounds(var_lb.data(), static_cast<i_t>(var_lb.size()));
  }
  if (!var_ub.empty()) {
    cpu_problem.set_variable_upper_bounds(var_ub.data(), static_cast<i_t>(var_ub.size()));
  }

  // Constraint bounds
  auto con_lb = get_doubles(cuopt::remote::FIELD_CONSTRAINT_LOWER_BOUNDS);
  auto con_ub = get_doubles(cuopt::remote::FIELD_CONSTRAINT_UPPER_BOUNDS);
  if (!con_lb.empty()) {
    cpu_problem.set_constraint_lower_bounds(con_lb.data(), static_cast<i_t>(con_lb.size()));
  }
  if (!con_ub.empty()) {
    cpu_problem.set_constraint_upper_bounds(con_ub.data(), static_cast<i_t>(con_ub.size()));
  }

  auto b_vec = get_doubles(cuopt::remote::FIELD_B);
  if (!b_vec.empty()) {
    cpu_problem.set_constraint_bounds(b_vec.data(), static_cast<i_t>(b_vec.size()));
  }

  // Row types
  auto row_types_str = get_bytes(cuopt::remote::FIELD_ROW_TYPES);
  if (!row_types_str.empty()) {
    cpu_problem.set_row_types(row_types_str.data(), static_cast<i_t>(row_types_str.size()));
  }

  // Variable types + problem category
  auto var_types_str = get_bytes(cuopt::remote::FIELD_VARIABLE_TYPES);
  if (!var_types_str.empty()) {
    std::vector<var_t> vtypes;
    vtypes.reserve(var_types_str.size());
    bool has_ints = false;
    for (char c : var_types_str) {
      switch (c) {
        case 'C': vtypes.push_back(var_t::CONTINUOUS); break;
        case 'I':
        case 'B':
          vtypes.push_back(var_t::INTEGER);
          has_ints = true;
          break;
        default:
          throw std::runtime_error(std::string("Unknown variable type character '") + c +
                                   "' in chunked variable_types (expected 'C', 'I', or 'B')");
      }
    }
    cpu_problem.set_variable_types(vtypes.data(), static_cast<i_t>(vtypes.size()));
    cpu_problem.set_problem_category(has_ints ? problem_category_t::MIP : problem_category_t::LP);
  } else {
    cpu_problem.set_problem_category(problem_category_t::LP);
  }

  // Quadratic objective
  auto q_values  = get_doubles(cuopt::remote::FIELD_Q_VALUES);
  auto q_indices = get_ints(cuopt::remote::FIELD_Q_INDICES);
  auto q_offsets = get_ints(cuopt::remote::FIELD_Q_OFFSETS);
  if (!q_values.empty() && !q_indices.empty() && !q_offsets.empty()) {
    cpu_problem.set_quadratic_objective_matrix(q_values.data(),
                                               static_cast<i_t>(q_values.size()),
                                               q_indices.data(),
                                               static_cast<i_t>(q_indices.size()),
                                               q_offsets.data(),
                                               static_cast<i_t>(q_offsets.size()));
  }

  // String arrays (may also be in header; these override if present as chunked arrays)
  auto var_names = get_string_list(cuopt::remote::FIELD_VARIABLE_NAMES);
  if (!var_names.empty()) { cpu_problem.set_variable_names(var_names); }
  auto row_names = get_string_list(cuopt::remote::FIELD_ROW_NAMES);
  if (!row_names.empty()) { cpu_problem.set_row_names(row_names); }
}

// =============================================================================
// Chunked array request building (client-side)
// =============================================================================

namespace {

template <typename T>
void chunk_typed_array(std::vector<cuopt::remote::SendArrayChunkRequest>& out,
                       cuopt::remote::ArrayFieldId field_id,
                       const std::vector<T>& data,
                       const std::string& upload_id,
                       int64_t chunk_data_budget)
{
  if (data.empty()) return;

  const int64_t elem_size      = static_cast<int64_t>(sizeof(T));
  const int64_t total_elements = static_cast<int64_t>(data.size());

  int64_t elems_per_chunk = chunk_data_budget / elem_size;
  if (elems_per_chunk <= 0) elems_per_chunk = 1;

  const auto* raw = reinterpret_cast<const uint8_t*>(data.data());

  for (int64_t offset = 0; offset < total_elements; offset += elems_per_chunk) {
    int64_t count       = std::min(elems_per_chunk, total_elements - offset);
    int64_t byte_offset = offset * elem_size;
    int64_t byte_count  = count * elem_size;

    cuopt::remote::SendArrayChunkRequest req;
    req.set_upload_id(upload_id);
    auto* ac = req.mutable_chunk();
    ac->set_field_id(field_id);
    ac->set_element_offset(offset);
    ac->set_total_elements(total_elements);
    ac->set_data(raw + byte_offset, byte_count);
    out.push_back(std::move(req));
  }
}

void chunk_byte_blob(std::vector<cuopt::remote::SendArrayChunkRequest>& out,
                     cuopt::remote::ArrayFieldId field_id,
                     const std::vector<uint8_t>& data,
                     const std::string& upload_id,
                     int64_t chunk_data_budget)
{
  chunk_typed_array(out, field_id, data, upload_id, chunk_data_budget);
}

}  // namespace

template <typename i_t, typename f_t>
std::vector<cuopt::remote::SendArrayChunkRequest> build_array_chunk_requests(
  const cpu_optimization_problem_t<i_t, f_t>& problem,
  const std::string& upload_id,
  int64_t chunk_size_bytes)
{
  std::vector<cuopt::remote::SendArrayChunkRequest> requests;

  auto values  = problem.get_constraint_matrix_values_host();
  auto indices = problem.get_constraint_matrix_indices_host();
  auto offsets = problem.get_constraint_matrix_offsets_host();
  auto obj     = problem.get_objective_coefficients_host();
  auto var_lb  = problem.get_variable_lower_bounds_host();
  auto var_ub  = problem.get_variable_upper_bounds_host();
  auto con_lb  = problem.get_constraint_lower_bounds_host();
  auto con_ub  = problem.get_constraint_upper_bounds_host();
  auto b       = problem.get_constraint_bounds_host();

  chunk_typed_array(requests, cuopt::remote::FIELD_A_VALUES, values, upload_id, chunk_size_bytes);
  chunk_typed_array(requests, cuopt::remote::FIELD_A_INDICES, indices, upload_id, chunk_size_bytes);
  chunk_typed_array(requests, cuopt::remote::FIELD_A_OFFSETS, offsets, upload_id, chunk_size_bytes);
  chunk_typed_array(requests, cuopt::remote::FIELD_C, obj, upload_id, chunk_size_bytes);
  chunk_typed_array(
    requests, cuopt::remote::FIELD_VARIABLE_LOWER_BOUNDS, var_lb, upload_id, chunk_size_bytes);
  chunk_typed_array(
    requests, cuopt::remote::FIELD_VARIABLE_UPPER_BOUNDS, var_ub, upload_id, chunk_size_bytes);
  chunk_typed_array(
    requests, cuopt::remote::FIELD_CONSTRAINT_LOWER_BOUNDS, con_lb, upload_id, chunk_size_bytes);
  chunk_typed_array(
    requests, cuopt::remote::FIELD_CONSTRAINT_UPPER_BOUNDS, con_ub, upload_id, chunk_size_bytes);
  chunk_typed_array(requests, cuopt::remote::FIELD_B, b, upload_id, chunk_size_bytes);

  auto row_types = problem.get_row_types_host();
  if (!row_types.empty()) {
    std::vector<uint8_t> rt_bytes(row_types.begin(), row_types.end());
    chunk_byte_blob(
      requests, cuopt::remote::FIELD_ROW_TYPES, rt_bytes, upload_id, chunk_size_bytes);
  }

  auto var_types = problem.get_variable_types_host();
  if (!var_types.empty()) {
    std::vector<uint8_t> vt_bytes;
    vt_bytes.reserve(var_types.size());
    for (const auto& vt : var_types) {
      switch (vt) {
        case var_t::CONTINUOUS: vt_bytes.push_back('C'); break;
        case var_t::INTEGER: vt_bytes.push_back('I'); break;
        default:
          throw std::runtime_error("chunk_problem_to_proto: unknown var_t value " +
                                   std::to_string(static_cast<int>(vt)));
      }
    }
    chunk_byte_blob(
      requests, cuopt::remote::FIELD_VARIABLE_TYPES, vt_bytes, upload_id, chunk_size_bytes);
  }

  if (problem.has_quadratic_objective()) {
    const auto& q_values  = problem.get_quadratic_objective_values();
    const auto& q_indices = problem.get_quadratic_objective_indices();
    const auto& q_offsets = problem.get_quadratic_objective_offsets();
    chunk_typed_array(
      requests, cuopt::remote::FIELD_Q_VALUES, q_values, upload_id, chunk_size_bytes);
    chunk_typed_array(
      requests, cuopt::remote::FIELD_Q_INDICES, q_indices, upload_id, chunk_size_bytes);
    chunk_typed_array(
      requests, cuopt::remote::FIELD_Q_OFFSETS, q_offsets, upload_id, chunk_size_bytes);
  }

  auto names_to_blob = [](const std::vector<std::string>& names) -> std::vector<uint8_t> {
    if (names.empty()) return {};
    size_t total = 0;
    for (const auto& n : names)
      total += n.size() + 1;
    std::vector<uint8_t> blob(total);
    size_t pos = 0;
    for (const auto& n : names) {
      std::memcpy(blob.data() + pos, n.data(), n.size());
      pos += n.size();
      blob[pos++] = '\0';
    }
    return blob;
  };

  auto var_names_blob = names_to_blob(problem.get_variable_names());
  auto row_names_blob = names_to_blob(problem.get_row_names());
  chunk_byte_blob(
    requests, cuopt::remote::FIELD_VARIABLE_NAMES, var_names_blob, upload_id, chunk_size_bytes);
  chunk_byte_blob(
    requests, cuopt::remote::FIELD_ROW_NAMES, row_names_blob, upload_id, chunk_size_bytes);

  return requests;
}

// Explicit template instantiations
#if CUOPT_INSTANTIATE_FLOAT
template void map_problem_to_proto(const cpu_optimization_problem_t<int32_t, float>& cpu_problem,
                                   cuopt::remote::OptimizationProblem* pb_problem);
template void map_proto_to_problem(const cuopt::remote::OptimizationProblem& pb_problem,
                                   cpu_optimization_problem_t<int32_t, float>& cpu_problem);
template size_t estimate_problem_proto_size(
  const cpu_optimization_problem_t<int32_t, float>& cpu_problem);
template void populate_chunked_header_lp(
  const cpu_optimization_problem_t<int32_t, float>& cpu_problem,
  const pdlp_solver_settings_t<int32_t, float>& settings,
  cuopt::remote::ChunkedProblemHeader* header);
template void populate_chunked_header_mip(
  const cpu_optimization_problem_t<int32_t, float>& cpu_problem,
  const mip_solver_settings_t<int32_t, float>& settings,
  bool enable_incumbents,
  cuopt::remote::ChunkedProblemHeader* header);
template void map_chunked_header_to_problem(
  const cuopt::remote::ChunkedProblemHeader& header,
  cpu_optimization_problem_t<int32_t, float>& cpu_problem);
template void map_chunked_arrays_to_problem(
  const cuopt::remote::ChunkedProblemHeader& header,
  const std::map<int32_t, std::vector<uint8_t>>& arrays,
  cpu_optimization_problem_t<int32_t, float>& cpu_problem);
template std::vector<cuopt::remote::SendArrayChunkRequest> build_array_chunk_requests(
  const cpu_optimization_problem_t<int32_t, float>& problem,
  const std::string& upload_id,
  int64_t chunk_size_bytes);
#endif

#if CUOPT_INSTANTIATE_DOUBLE
template void map_problem_to_proto(const cpu_optimization_problem_t<int32_t, double>& cpu_problem,
                                   cuopt::remote::OptimizationProblem* pb_problem);
template void map_proto_to_problem(const cuopt::remote::OptimizationProblem& pb_problem,
                                   cpu_optimization_problem_t<int32_t, double>& cpu_problem);
template size_t estimate_problem_proto_size(
  const cpu_optimization_problem_t<int32_t, double>& cpu_problem);
template void populate_chunked_header_lp(
  const cpu_optimization_problem_t<int32_t, double>& cpu_problem,
  const pdlp_solver_settings_t<int32_t, double>& settings,
  cuopt::remote::ChunkedProblemHeader* header);
template void populate_chunked_header_mip(
  const cpu_optimization_problem_t<int32_t, double>& cpu_problem,
  const mip_solver_settings_t<int32_t, double>& settings,
  bool enable_incumbents,
  cuopt::remote::ChunkedProblemHeader* header);
template void map_chunked_header_to_problem(
  const cuopt::remote::ChunkedProblemHeader& header,
  cpu_optimization_problem_t<int32_t, double>& cpu_problem);
template void map_chunked_arrays_to_problem(
  const cuopt::remote::ChunkedProblemHeader& header,
  const std::map<int32_t, std::vector<uint8_t>>& arrays,
  cpu_optimization_problem_t<int32_t, double>& cpu_problem);
template std::vector<cuopt::remote::SendArrayChunkRequest> build_array_chunk_requests(
  const cpu_optimization_problem_t<int32_t, double>& problem,
  const std::string& upload_id,
  int64_t chunk_size_bytes);
#endif

}  // namespace cuopt::linear_programming
