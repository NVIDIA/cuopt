/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <pdlp/distributed_pdlp/partition_loader.hpp>

#include <cuopt/error.hpp>

#include <fstream>
#include <set>
#include <utility>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
std::vector<i_t> partition_loader_t<i_t, f_t>::parse_distributed_pdlp_partition_file(
  std::string const& file)
{
  std::ifstream part_file(file);
  cuopt_expects(
    part_file.is_open(), error_type_t::ValidationError, "Failed to open partition file: " + file);

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
                "Malformed partition file (expected one integer per line): " + file);

  return parts;
}

template <typename i_t, typename f_t>
std::vector<rank_data_t<i_t, f_t>> partition_loader_t<i_t, f_t>::create_rank_data_from_parts(
  const std::vector<i_t>& parts,
  const std::vector<i_t>& A_row_offsets,
  const std::vector<i_t>& A_col_indices,
  const std::vector<f_t>& A_values,
  const std::vector<f_t>& A_values_scaled,
  const std::vector<i_t>& A_t_row_offsets,
  const std::vector<i_t>& A_t_col_indices,
  const std::vector<f_t>& A_t_values,
  const std::vector<f_t>& A_t_values_scaled,
  i_t nb_parts,
  i_t nb_cstr,
  i_t nb_vars,
  i_t nnz)
{
  cuopt_expects(A_values.size() == A_values_scaled.size(),
                error_type_t::ValidationError,
                "A_values and A_values_scaled must have the same length");
  cuopt_expects(A_t_values.size() == A_t_values_scaled.size(),
                error_type_t::ValidationError,
                "A_t_values and A_t_values_scaled must have the same length");

  std::vector<rank_data_t<i_t, f_t>> rank_data(nb_parts, rank_data_t<i_t, f_t>(nb_parts));
  std::vector<i_t> cstr_parts(parts.begin(), parts.begin() + nb_cstr);
  std::vector<i_t> var_parts(parts.begin() + nb_cstr, parts.begin() + nb_cstr + nb_vars);

  // 1. Compute ownership
  for (i_t i = 0; i < nb_cstr; i++) {
    rank_data[cstr_parts[i]].owned_cstr_indices.push_back(i);
  }
  for (i_t i = 0; i < nb_vars; i++) {
    rank_data[var_parts[i]].owned_var_indices.push_back(i);
  }

  // 2. Compute local matrices and rank_data
  for (i_t rank = 0; rank < nb_parts; rank++) {
    auto& rd           = rank_data[rank];
    rd.owned_var_size  = rd.owned_var_indices.size();
    rd.owned_cstr_size = rd.owned_cstr_indices.size();
    // ---- A side ----
    std::vector<i_t> local_A_row_offsets;
    std::vector<i_t> local_A_col_indices;
    std::vector<f_t> local_A_values;
    std::vector<f_t> local_A_values_scaled;

    i_t local_A_nnz = 0;
    local_A_row_offsets.push_back(local_A_nnz);

    // For each owned constraint, build local matrix A. We walk both the
    // unscaled and scaled global value arrays in lockstep so the produced
    // local arrays share identical (offsets, col_indices) and differ only
    // in values.
    for (auto owned_cstr : rd.owned_cstr_indices) {
      i_t cstr_len  = A_row_offsets[owned_cstr + 1] - A_row_offsets[owned_cstr];
      i_t row_start = A_row_offsets[owned_cstr];
      for (i_t v = 0; v < cstr_len; v++) {
        local_A_col_indices.push_back(A_col_indices[row_start + v]);
        local_A_values.push_back(A_values[row_start + v]);
        local_A_values_scaled.push_back(A_values_scaled[row_start + v]);
      }
      local_A_nnz += cstr_len;
      local_A_row_offsets.push_back(local_A_nnz);
    }

    std::set<i_t> needed_vars;
    for (auto indice : local_A_col_indices) {
      if (var_parts[indice] != rank) needed_vars.insert(indice);
    }

    for (i_t peer = 0; peer < nb_parts; peer++) {
      std::vector<i_t> needed_var_from_peer;
      for (auto needed_var : needed_vars) {
        if (var_parts[needed_var] == peer) needed_var_from_peer.push_back(needed_var);
      }
      i_t nb_recv_from_peer    = needed_var_from_peer.size();
      rd.var_recv_counts[peer] = nb_recv_from_peer;
      rd.var_recv_offsets[peer] =
        peer == 0 ? 0 : rd.var_recv_offsets[peer - 1] + rd.var_recv_counts[peer - 1];
      rank_data[peer].var_send_per_peer[rank] = std::move(needed_var_from_peer);
    }

    rd.h_A_row_offsets   = std::move(local_A_row_offsets);
    rd.h_A_col_indices   = std::move(local_A_col_indices);
    rd.h_A_values        = std::move(local_A_values);
    rd.h_A_values_scaled = std::move(local_A_values_scaled);

    // ---- A_t side ----
    std::vector<i_t> local_A_t_row_offsets;
    std::vector<i_t> local_A_t_col_indices;
    std::vector<f_t> local_A_t_values;
    std::vector<f_t> local_A_t_values_scaled;
    i_t local_A_t_nnz = 0;
    local_A_t_row_offsets.push_back(local_A_t_nnz);

    for (auto owned_var : rd.owned_var_indices) {
      i_t var_len   = A_t_row_offsets[owned_var + 1] - A_t_row_offsets[owned_var];
      i_t row_start = A_t_row_offsets[owned_var];
      for (i_t v = 0; v < var_len; v++) {
        local_A_t_col_indices.push_back(A_t_col_indices[row_start + v]);
        local_A_t_values.push_back(A_t_values[row_start + v]);
        local_A_t_values_scaled.push_back(A_t_values_scaled[row_start + v]);
      }
      local_A_t_nnz += var_len;
      local_A_t_row_offsets.push_back(local_A_t_nnz);
    }

    std::set<i_t> needed_cstrs;
    for (auto indice : local_A_t_col_indices) {
      if (cstr_parts[indice] != rank) needed_cstrs.insert(indice);
    }

    for (i_t peer = 0; peer < nb_parts; peer++) {
      std::vector<i_t> needed_cstr_from_peer;
      for (auto needed_cstr : needed_cstrs) {
        if (cstr_parts[needed_cstr] == peer) needed_cstr_from_peer.push_back(needed_cstr);
      }
      i_t nb_recv_from_peer     = needed_cstr_from_peer.size();
      rd.cstr_recv_counts[peer] = nb_recv_from_peer;
      rd.cstr_recv_offsets[peer] =
        peer == 0 ? 0 : rd.cstr_recv_offsets[peer - 1] + rd.cstr_recv_counts[peer - 1];
      rank_data[peer].cstr_send_per_peer[rank] = std::move(needed_cstr_from_peer);
    }

    rd.h_A_t_row_offsets   = std::move(local_A_t_row_offsets);
    rd.h_A_t_col_indices   = std::move(local_A_t_col_indices);
    rd.h_A_t_values        = std::move(local_A_t_values);
    rd.h_A_t_values_scaled = std::move(local_A_t_values_scaled);

    rd.total_var_size  = rd.owned_var_size + needed_vars.size();
    rd.total_cstr_size = rd.owned_cstr_size + needed_cstrs.size();
  }

  // 3. Generate local indices for contiguous [[self], [peer1], ..., [peer_k]]
  //    Build scatter_gather_maps
  for (i_t rank = 0; rank < nb_parts; rank++) {
    auto& rd = rank_data[rank];

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
        // rd.local_to_global_cstr.push_back(recv_cstr); // Not needed, we only do local_to_global
        // on owned side
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
        // rd.local_to_global_var.push_back(recv_var); // same as over
        curr_id++;
      }
    }
  }

  // 4. Remap global -> local everywhere
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
