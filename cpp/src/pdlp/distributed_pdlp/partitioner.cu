/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <pdlp/distributed_pdlp/kaminpar_partitioner.hpp>
#include <pdlp/distributed_pdlp/partitioner.hpp>

#include <cuopt/error.hpp>

#include <algorithm>
#include <cstddef>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
std::vector<i_t> dummy_partitioner_t<i_t, f_t>::partition(
  partitioner_input_t<i_t, f_t> const& input) const
{
  cuopt_expects(input.nb_parts > 0,
                error_type_t::ValidationError,
                "dummy_partitioner: nb_parts must be positive");
  cuopt_expects(input.nb_cstr >= 0 && input.nb_vars >= 0,
                error_type_t::ValidationError,
                "dummy_partitioner: invalid problem dimensions");

  const std::size_t nvtx =
    static_cast<std::size_t>(input.nb_cstr) + static_cast<std::size_t>(input.nb_vars);
  std::vector<i_t> parts(nvtx);
  for (std::size_t i = 0; i < nvtx; ++i) {
    parts[i] = static_cast<i_t>(i % static_cast<std::size_t>(input.nb_parts));
  }
  validate_partition(parts,
                     static_cast<int>(input.nb_cstr),
                     static_cast<int>(input.nb_vars),
                     static_cast<int>(input.nb_parts),
                     "dummy_partitioner");
  return parts;
}

void validate_partition(
  std::vector<int> const& parts, int nb_cstr, int nb_vars, int nb_parts, char const* context)
{
  const std::size_t expected =
    static_cast<std::size_t>(nb_cstr) + static_cast<std::size_t>(nb_vars);
  cuopt_expects(parts.size() == expected,
                error_type_t::ValidationError,
                "%s: expected %zu part entries (cstrs + vars), got %zu",
                context,
                expected,
                parts.size());
  cuopt_expects(
    nb_parts > 0, error_type_t::ValidationError, "%s: nb_parts must be positive", context);
  if (parts.empty()) { return; }
  const auto [min_it, max_it] = std::minmax_element(parts.begin(), parts.end());
  cuopt_expects(*min_it >= 0,
                error_type_t::ValidationError,
                "%s: partition ids must be non-negative (min=%d)",
                context,
                static_cast<int>(*min_it));
  cuopt_expects(*max_it < nb_parts,
                error_type_t::ValidationError,
                "%s: partition ids must be in [0, %d) (max=%d)",
                context,
                static_cast<int>(nb_parts),
                static_cast<int>(*max_it));
}

template <typename i_t, typename f_t>
std::unique_ptr<partitioner_i<i_t, f_t>> make_partitioner(partitioner_kind_t kind)
{
  switch (kind) {
    case partitioner_kind_t::Dummy: return std::make_unique<dummy_partitioner_t<i_t, f_t>>();
    case partitioner_kind_t::KaMinPar: return std::make_unique<kaminpar_partitioner_t<i_t, f_t>>();
  }
  cuopt_expects(
    false, error_type_t::RuntimeError, "make_partitioner: unsupported partitioner kind");
  return nullptr;
}

template class dummy_partitioner_t<int, double>;
template std::unique_ptr<partitioner_i<int, double>> make_partitioner<int, double>(
  partitioner_kind_t);

}  // namespace cuopt::linear_programming::detail
