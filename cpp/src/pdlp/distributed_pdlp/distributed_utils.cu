/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <pdlp/distributed_pdlp/distributed_utils.hpp>

#include <cuopt/error.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <unordered_set>
#include <utility>

namespace cuopt::mathematical_optimization::pdlp {

template <typename i_t, typename f_t>
std::vector<rank_data_t<i_t, f_t>> create_rank_data_from_parts(
  const std::vector<int>& parts,
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
  // Reserve exact capacities
  for (i_t rank = 0; rank < nb_parts; ++rank) {
    rank_data[rank].owned_cstr_indices.reserve(owned_cstr_counts[rank]);
    rank_data[rank].owned_var_indices.reserve(owned_var_counts[rank]);
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
    local_A_row_offsets.reserve(rd.owned_cstr_size + 1);
    local_A_col_indices.reserve(owned_A_nnz[rank]);
    local_A_values.reserve(owned_A_nnz[rank]);

    i_t local_A_nnz = 0;
    local_A_row_offsets.push_back(local_A_nnz);

    // For each owned constraint, add associated matrix row to local A
    // Keep the same indices here but re-order later
    for (auto owned_cstr : rd.owned_cstr_indices) {
      i_t cstr_len  = A_row_offsets[owned_cstr + 1] - A_row_offsets[owned_cstr];
      i_t row_start = A_row_offsets[owned_cstr];
      local_A_col_indices.insert(local_A_col_indices.end(),
                                 A_col_indices.begin() + row_start,
                                 A_col_indices.begin() + row_start + cstr_len);
      local_A_values.insert(local_A_values.end(),
                            A_values.begin() + row_start,
                            A_values.begin() + row_start + cstr_len);
      local_A_nnz += cstr_len;
      local_A_row_offsets.push_back(local_A_nnz);
    }

    // Compute halo
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

    // Compute counts and offsets of halo data to stack them all at the end of the vector
    for (i_t peer = 0; peer < nb_parts; peer++) {
      i_t nb_recv_from_peer    = needed_var_from_peer[peer].size();
      rd.var_recv_counts[peer] = nb_recv_from_peer;
      rd.var_recv_offsets[peer] =
        peer == 0 ? 0 : rd.var_recv_offsets[peer - 1] + rd.var_recv_counts[peer - 1];
      rank_data[peer].var_send_per_peer[rank] = std::move(needed_var_from_peer[peer]);
    }

    rd.h_A_row_offsets = std::move(local_A_row_offsets);
    rd.h_A_col_indices = std::move(local_A_col_indices);
    rd.h_A_values      = std::move(local_A_values);

    // ---- A_t side ----
    // conceptually same as A side
    std::vector<i_t> local_A_t_row_offsets;
    std::vector<i_t> local_A_t_col_indices;
    std::vector<f_t> local_A_t_values;
    local_A_t_row_offsets.reserve(rd.owned_var_size + 1);
    local_A_t_col_indices.reserve(owned_A_t_nnz[rank]);
    local_A_t_values.reserve(owned_A_t_nnz[rank]);
    i_t local_A_t_nnz = 0;
    local_A_t_row_offsets.push_back(local_A_t_nnz);

    for (auto owned_var : rd.owned_var_indices) {
      i_t var_len   = A_t_row_offsets[owned_var + 1] - A_t_row_offsets[owned_var];
      i_t row_start = A_t_row_offsets[owned_var];
      local_A_t_col_indices.insert(local_A_t_col_indices.end(),
                                   A_t_col_indices.begin() + row_start,
                                   A_t_col_indices.begin() + row_start + var_len);
      local_A_t_values.insert(local_A_t_values.end(),
                              A_t_values.begin() + row_start,
                              A_t_values.begin() + row_start + var_len);
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

    rd.h_A_t_row_offsets = std::move(local_A_t_row_offsets);
    rd.h_A_t_col_indices = std::move(local_A_t_col_indices);
    rd.h_A_t_values      = std::move(local_A_t_values);

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
#pragma omp parallel for
  for (i_t rank = 0; rank < nb_parts; rank++) {
    auto& rd = rank_data[rank];
    rd.global_to_local_cstr.reserve(rd.total_cstr_size);
    rd.global_to_local_var.reserve(rd.total_var_size);
    rd.local_to_global_cstr.reserve(rd.total_cstr_size);
    rd.local_to_global_var.reserve(rd.total_var_size);

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
  // Including in A_local and At_local
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

namespace {

template <typename i_t, typename index_t>
std::vector<i_t> narrow_vector(const std::vector<index_t>& src)
{
  std::vector<i_t> dst(src.size());
  std::transform(
    src.begin(), src.end(), dst.begin(), [](index_t v) { return static_cast<i_t>(v); });
  return dst;
}

}  // namespace

template <typename i_t, typename index_t, typename f_t>
std::vector<rank_data_t<i_t, f_t>> narrow_rank_data(std::vector<rank_data_t<index_t, f_t>>&& wide)
{
  if constexpr (std::is_same_v<i_t, index_t>) {
    return std::move(wide);
  }
  constexpr index_t max_i_t = static_cast<index_t>(std::numeric_limits<i_t>::max());

  std::vector<rank_data_t<i_t, f_t>> narrow;
  narrow.reserve(wide.size());

  for (std::size_t rank = 0; rank < wide.size(); ++rank) {
    auto& src                  = wide[rank];
    const std::size_t nb_parts = src.var_send_per_peer.size();

    // Column indices have already been remapped to shard-local ids, so they are bounded by
    // total_var_size / total_cstr_size and cannot overflow. Only the row offsets can: their
    // last entry is this shard's nonzero count, which is what the partition has to keep
    // inside i_t for the per-GPU solvers to be 32-bit.
    const index_t a_nnz = src.h_A_row_offsets.empty() ? index_t{0} : src.h_A_row_offsets.back();
    const index_t a_t_nnz =
      src.h_A_t_row_offsets.empty() ? index_t{0} : src.h_A_t_row_offsets.back();
    cuopt_expects(a_nnz <= max_i_t && a_t_nnz <= max_i_t,
                  error_type_t::ValidationError,
                  "Shard %d holds too many nonzeros for a 32-bit local index; increase num_gpus",
                  static_cast<int>(rank));

    rank_data_t<i_t, f_t> dst(nb_parts);
    dst.owned_var_size  = static_cast<i_t>(src.owned_var_size);
    dst.total_var_size  = static_cast<i_t>(src.total_var_size);
    dst.owned_cstr_size = static_cast<i_t>(src.owned_cstr_size);
    dst.total_cstr_size = static_cast<i_t>(src.total_cstr_size);

    dst.owned_var_indices  = narrow_vector<i_t>(src.owned_var_indices);
    dst.owned_cstr_indices = narrow_vector<i_t>(src.owned_cstr_indices);

    for (std::size_t peer = 0; peer < nb_parts; ++peer) {
      dst.var_send_per_peer[peer]  = narrow_vector<i_t>(src.var_send_per_peer[peer]);
      dst.cstr_send_per_peer[peer] = narrow_vector<i_t>(src.cstr_send_per_peer[peer]);
    }
    dst.var_recv_counts   = narrow_vector<i_t>(src.var_recv_counts);
    dst.var_recv_offsets  = narrow_vector<i_t>(src.var_recv_offsets);
    dst.cstr_recv_counts  = narrow_vector<i_t>(src.cstr_recv_counts);
    dst.cstr_recv_offsets = narrow_vector<i_t>(src.cstr_recv_offsets);

    dst.global_to_local_var.reserve(src.global_to_local_var.size());
    for (const auto& [global_id, local_id] : src.global_to_local_var) {
      dst.global_to_local_var.emplace(static_cast<i_t>(global_id), static_cast<i_t>(local_id));
    }
    dst.global_to_local_cstr.reserve(src.global_to_local_cstr.size());
    for (const auto& [global_id, local_id] : src.global_to_local_cstr) {
      dst.global_to_local_cstr.emplace(static_cast<i_t>(global_id), static_cast<i_t>(local_id));
    }
    dst.local_to_global_var  = narrow_vector<i_t>(src.local_to_global_var);
    dst.local_to_global_cstr = narrow_vector<i_t>(src.local_to_global_cstr);

    dst.h_A_row_offsets = narrow_vector<i_t>(src.h_A_row_offsets);
    dst.h_A_col_indices = narrow_vector<i_t>(src.h_A_col_indices);
    // Same f_t on both sides: move rather than copy, this is the bulk of the bytes.
    dst.h_A_values        = std::move(src.h_A_values);
    dst.h_A_t_row_offsets = narrow_vector<i_t>(src.h_A_t_row_offsets);
    dst.h_A_t_col_indices = narrow_vector<i_t>(src.h_A_t_col_indices);
    dst.h_A_t_values      = std::move(src.h_A_t_values);

    narrow.push_back(std::move(dst));
    // Release the wide shard as we go so both representations are never fully live at once.
    src = rank_data_t<index_t, f_t>(0);
  }

  return narrow;
}

template std::vector<rank_data_t<int, double>> create_rank_data_from_parts<int, double>(
  const std::vector<int>& parts,
  const std::vector<int>& A_row_offsets,
  const std::vector<int>& A_col_indices,
  const std::vector<double>& A_values,
  const std::vector<int>& A_t_row_offsets,
  const std::vector<int>& A_t_col_indices,
  const std::vector<double>& A_t_values,
  int nb_parts,
  int nb_cstr,
  int nb_vars,
  int nnz);

template std::vector<rank_data_t<int64_t, double>> create_rank_data_from_parts<int64_t, double>(
  const std::vector<int>& parts,
  const std::vector<int64_t>& A_row_offsets,
  const std::vector<int64_t>& A_col_indices,
  const std::vector<double>& A_values,
  const std::vector<int64_t>& A_t_row_offsets,
  const std::vector<int64_t>& A_t_col_indices,
  const std::vector<double>& A_t_values,
  int64_t nb_parts,
  int64_t nb_cstr,
  int64_t nb_vars,
  int64_t nnz);

template std::vector<rank_data_t<int, double>> narrow_rank_data<int, int, double>(
  std::vector<rank_data_t<int, double>>&&);
template std::vector<rank_data_t<int, double>> narrow_rank_data<int, int64_t, double>(
  std::vector<rank_data_t<int64_t, double>>&&);

}  // namespace cuopt::mathematical_optimization::pdlp
