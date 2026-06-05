/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <pdlp/distributed_pdlp/partition_loader.hpp>

#include <cuopt/error.hpp>

#include <fstream>
#include <unordered_set>
#include <utility>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
std::vector<i_t> partition_loader_t<i_t, f_t>::parse_distributed_pdlp_partition_file(
  std::string const& file)
{
  std::ifstream part_file(file);
  cuopt_expects(part_file.is_open(),
                error_type_t::ValidationError,
                "Failed to open partition file: %s",
                file.c_str());

  // One integer per line; operator>> skips whitespace so blank lines and
  // trailing newlines are tolerated.
  std::vector<i_t> parts;
  i_t part = 0;
  while (part_file >> part) {
    parts.push_back(part);
  }

  // We must have hit EOF cleanly; any other state means a malformed token.
  cuopt_expects(part_file.eof(),
                error_type_t::ValidationError,
                "Malformed partition file (expected one integer per line): %s",
                file.c_str());

  return parts;
}

template <typename i_t, typename f_t>
void partition_loader_t<i_t, f_t>::export_distributed_pdlp_partition_file(
  std::string const& file, std::vector<i_t> const& parts)
{
  std::ofstream part_file(file);
  cuopt_expects(part_file.is_open(),
                error_type_t::ValidationError,
                "Failed to open partition file for export: %s",
                file.c_str());
  for (auto const& part : parts) {
    part_file << part << "\n";
  }
  cuopt_expects(part_file.good(),
                error_type_t::RuntimeError,
                "Failed while writing partition file: %s",
                file.c_str());
}

template <typename i_t, typename f_t>
std::vector<rank_data_t<i_t, f_t>> partition_loader_t<i_t, f_t>::create_rank_data_from_parts(
  const std::vector<i_t>& parts,
  const std::vector<i_t>& A_row_offsets,
  const std::vector<i_t>& A_col_indices,
  const std::vector<f_t>& A_values,
  const std::vector<i_t>& A_t_row_offsets,
  const std::vector<i_t>& A_t_col_indices,
  const std::vector<f_t>& A_t_values,
  i_t nb_parts,
  i_t nb_cstr,
  i_t nb_vars,
  i_t nnz)
{
  std::vector<rank_data_t<i_t, f_t>> rank_data(nb_parts, rank_data_t<i_t, f_t>(nb_parts));
  cuopt_expects(static_cast<i_t>(parts.size()) == nb_cstr + nb_vars,
                error_type_t::ValidationError,
                "parts size mismatch: expected nb_cstr + nb_vars");

  // Same as two vectors but faster
  auto cstr_owner = [&](i_t c) { return parts[c]; };
  auto var_owner  = [&](i_t v) { return parts[nb_cstr + v]; };

  std::vector<i_t> owned_cstr_counts(nb_parts, 0);
  std::vector<i_t> owned_var_counts(nb_parts, 0);
  std::vector<i_t> owned_A_nnz(nb_parts, 0);
  std::vector<i_t> owned_A_t_nnz(nb_parts, 0);

  // Pre-count ownership and nnz to reserve exact capacities and avoid
  // repeated growth/reallocation across huge vectors.
  for (i_t i = 0; i < nb_cstr; ++i) {
    const i_t owner = cstr_owner(i);
    ++owned_cstr_counts[owner];
    owned_A_nnz[owner] += (A_row_offsets[i + 1] - A_row_offsets[i]);
  }
  for (i_t j = 0; j < nb_vars; ++j) {
    const i_t owner = var_owner(j);
    ++owned_var_counts[owner];
    owned_A_t_nnz[owner] += (A_t_row_offsets[j + 1] - A_t_row_offsets[j]);
  }

  for (i_t rank = 0; rank < nb_parts; ++rank) {
    rank_data[rank].owned_cstr_indices.reserve(static_cast<std::size_t>(owned_cstr_counts[rank]));
    rank_data[rank].owned_var_indices.reserve(static_cast<std::size_t>(owned_var_counts[rank]));
  }

  // 1. Compute ownership
  for (i_t i = 0; i < nb_cstr; i++) {
    rank_data[cstr_owner(i)].owned_cstr_indices.push_back(i);
  }
  for (i_t i = 0; i < nb_vars; i++) {
    rank_data[var_owner(i)].owned_var_indices.push_back(i);
  }

  // 2. Compute local matrices and rank_data
#pragma omp parallel for
  for (i_t rank = 0; rank < nb_parts; rank++) {
    auto& rd           = rank_data[rank];
    rd.owned_var_size  = rd.owned_var_indices.size();
    rd.owned_cstr_size = rd.owned_cstr_indices.size();
    // ---- A side ----
    std::vector<i_t> local_A_row_offsets;
    std::vector<i_t> local_A_col_indices;
    std::vector<f_t> local_A_values;
    local_A_row_offsets.reserve(static_cast<std::size_t>(rd.owned_cstr_size) + 1);
    local_A_col_indices.reserve(static_cast<std::size_t>(owned_A_nnz[rank]));
    local_A_values.reserve(static_cast<std::size_t>(owned_A_nnz[rank]));

    i_t local_A_nnz = 0;
    local_A_row_offsets.push_back(local_A_nnz);

    // For each owned constraint, build local matrix A. We walk both the
    // unscaled and scaled global value arrays in lockstep so the produced
    // local arrays share identical (offsets, col_indices) and differ only
    // in values.
    for (auto owned_cstr : rd.owned_cstr_indices) {
      i_t cstr_len  = A_row_offsets[owned_cstr + 1] - A_row_offsets[owned_cstr];
      i_t row_start = A_row_offsets[owned_cstr];
      local_A_col_indices.insert(
        local_A_col_indices.end(), A_col_indices.begin() + row_start, A_col_indices.begin() + row_start + cstr_len);
      local_A_values.insert(
        local_A_values.end(), A_values.begin() + row_start, A_values.begin() + row_start + cstr_len);
      local_A_nnz += cstr_len;
      local_A_row_offsets.push_back(local_A_nnz);
    }

    std::vector<std::vector<i_t>> needed_var_from_peer(nb_parts);
    std::unordered_set<i_t> seen_needed_vars;
    // size / 2 + 1 is a heuristic to avoid overestimating and resizing
    seen_needed_vars.reserve(local_A_col_indices.size() / 2 + 1);
    for (auto indice : local_A_col_indices) {
      const i_t owner = var_owner(indice);
      if (owner != rank && seen_needed_vars.insert(indice).second) {
        needed_var_from_peer[owner].push_back(indice);
      }
    }

    for (i_t peer = 0; peer < nb_parts; peer++) {
      i_t nb_recv_from_peer    = needed_var_from_peer[peer].size();
      rd.var_recv_counts[peer] = nb_recv_from_peer;
      rd.var_recv_offsets[peer] =
        peer == 0 ? 0 : rd.var_recv_offsets[peer - 1] + rd.var_recv_counts[peer - 1];
      rank_data[peer].var_send_per_peer[rank] = std::move(needed_var_from_peer[peer]);
    }

    rd.h_A_row_offsets   = std::move(local_A_row_offsets);
    rd.h_A_col_indices   = std::move(local_A_col_indices);
    rd.h_A_values        = std::move(local_A_values);

    // ---- A_t side ----
    std::vector<i_t> local_A_t_row_offsets;
    std::vector<i_t> local_A_t_col_indices;
    std::vector<f_t> local_A_t_values;
    local_A_t_row_offsets.reserve(static_cast<std::size_t>(rd.owned_var_size) + 1);
    local_A_t_col_indices.reserve(static_cast<std::size_t>(owned_A_t_nnz[rank]));
    local_A_t_values.reserve(static_cast<std::size_t>(owned_A_t_nnz[rank]));
    i_t local_A_t_nnz = 0;
    local_A_t_row_offsets.push_back(local_A_t_nnz);

    for (auto owned_var : rd.owned_var_indices) {
      i_t var_len   = A_t_row_offsets[owned_var + 1] - A_t_row_offsets[owned_var];
      i_t row_start = A_t_row_offsets[owned_var];
      local_A_t_col_indices.insert(local_A_t_col_indices.end(),
                                   A_t_col_indices.begin() + row_start,
                                   A_t_col_indices.begin() + row_start + var_len);
      local_A_t_values.insert(
        local_A_t_values.end(), A_t_values.begin() + row_start, A_t_values.begin() + row_start + var_len);
      local_A_t_nnz += var_len;
      local_A_t_row_offsets.push_back(local_A_t_nnz);
    }

    std::vector<std::vector<i_t>> needed_cstr_from_peer(nb_parts);
    std::unordered_set<i_t> seen_needed_cstrs;
    seen_needed_cstrs.reserve(local_A_t_col_indices.size() / 2 + 1);
    for (auto indice : local_A_t_col_indices) {
      const i_t owner = cstr_owner(indice);
      if (owner != rank && seen_needed_cstrs.insert(indice).second) {
        needed_cstr_from_peer[owner].push_back(indice);
      }
    }

    for (i_t peer = 0; peer < nb_parts; peer++) {
      i_t nb_recv_from_peer     = needed_cstr_from_peer[peer].size();
      rd.cstr_recv_counts[peer] = nb_recv_from_peer;
      rd.cstr_recv_offsets[peer] =
        peer == 0 ? 0 : rd.cstr_recv_offsets[peer - 1] + rd.cstr_recv_counts[peer - 1];
      rank_data[peer].cstr_send_per_peer[rank] = std::move(needed_cstr_from_peer[peer]);
    }

    rd.h_A_t_row_offsets   = std::move(local_A_t_row_offsets);
    rd.h_A_t_col_indices   = std::move(local_A_t_col_indices);
    rd.h_A_t_values        = std::move(local_A_t_values);

    rd.total_var_size  = rd.owned_var_size + static_cast<i_t>(seen_needed_vars.size());
    rd.total_cstr_size = rd.owned_cstr_size + static_cast<i_t>(seen_needed_cstrs.size());

    // Pad row-offset arrays so cuSPARSE sees the local matrices as
    // (total_cstr x total_var) for A and (total_var x total_cstr) for A_T
    const i_t a_last_nnz = rd.h_A_row_offsets.empty() ? i_t{0} : rd.h_A_row_offsets.back();
    rd.h_A_row_offsets.resize(rd.total_cstr_size + 1, a_last_nnz);

    const i_t at_last_nnz = rd.h_A_t_row_offsets.empty() ? i_t{0} : rd.h_A_t_row_offsets.back();
    rd.h_A_t_row_offsets.resize(rd.total_var_size + 1, at_last_nnz);
  }

  // 3. Generate local indices for contiguous [[self], [peer1], ..., [peer_k]]
  //    Build scatter_gather_maps
  // 3. Build local<->global maps in parallel across ranks.
#pragma omp parallel for
  for (i_t rank = 0; rank < nb_parts; rank++) {
    auto& rd = rank_data[rank];
    rd.global_to_local_cstr.reserve(static_cast<std::size_t>(rd.total_cstr_size));
    rd.global_to_local_var.reserve(static_cast<std::size_t>(rd.total_var_size));
    rd.local_to_global_cstr.reserve(static_cast<std::size_t>(rd.total_cstr_size));
    rd.local_to_global_var.reserve(static_cast<std::size_t>(rd.total_var_size));

    i_t curr_id = 0;
    for (auto owned_cstr : rd.owned_cstr_indices) {
      rd.global_to_local_cstr[owned_cstr] = curr_id;
      rd.local_to_global_cstr.push_back(owned_cstr);
      curr_id++;
    }
    for (i_t peer = 0; peer < nb_parts; peer++) {
      if (peer == rank) continue;
      for (auto recv_cstr : rank_data[peer].cstr_send_per_peer[rank]) {
        rd.global_to_local_cstr[recv_cstr] = curr_id;
        rd.local_to_global_cstr.push_back(recv_cstr);
        curr_id++;
      }
    }

    curr_id = 0;
    for (auto owned_var : rd.owned_var_indices) {
      rd.global_to_local_var[owned_var] = curr_id;
      rd.local_to_global_var.push_back(owned_var);
      curr_id++;
    }
    for (i_t peer = 0; peer < nb_parts; peer++) {
      if (peer == rank) continue;
      for (auto recv_var : rank_data[peer].var_send_per_peer[rank]) {
        rd.global_to_local_var[recv_var] = curr_id;
        rd.local_to_global_var.push_back(recv_var);
        curr_id++;
      }
    }
  }

  // 4. Remap global -> local everywhere
#pragma omp parallel for
  for (i_t rank = 0; rank < nb_parts; rank++) {
    auto& rd = rank_data[rank];

    for (auto& send_vec : rd.var_send_per_peer) {
      for (auto& v : send_vec)
        v = rd.global_to_local_var.at(v);
    }
    for (auto& send_vec : rd.cstr_send_per_peer) {
      for (auto& v : send_vec)
        v = rd.global_to_local_cstr.at(v);
    }

    for (auto& v : rd.h_A_col_indices)
      v = rd.global_to_local_var.at(v);
    for (auto& v : rd.h_A_t_col_indices)
      v = rd.global_to_local_cstr.at(v);
  }

  return rank_data;
}

template struct partition_loader_t<int, double>;

}  // namespace cuopt::linear_programming::detail
