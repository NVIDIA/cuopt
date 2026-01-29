/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <utilities/cuda_helpers.cuh>
#include "routing/structures.hpp"

namespace cuopt {
namespace routing {
namespace detail {

template <request_t REQUEST>
struct request_id_t;

template <>
struct request_id_t<request_t::PDP> {
  HDI request_id_t() {}
  HDI request_id_t(int pickup_, int delivery_) : pickup(pickup_), delivery(delivery_) {}
  HDI int id() const { return pickup; }
  HDI int& id() { return pickup; }
  HDI void check([[maybe_unused]] int n_orders) const
  {
    cuopt_assert(pickup >= 0, "pickup id must be strictly positive");
    cuopt_assert(pickup < n_orders, "delivery id must be lower than number of orders");
    cuopt_assert(delivery >= 0, "delivery id must be strictly positive");
    cuopt_assert(delivery < n_orders, "delivery id must be lower than number of orders");
  }
  int pickup{};
  int delivery{};
};
template <>
struct request_id_t<request_t::VRP> {
  HDI request_id_t() {}
  HDI request_id_t(int pickup_or_delivery_) : pickup_or_delivery(pickup_or_delivery_) {}
  HDI int id() const { return pickup_or_delivery; }
  HDI int& id() { return pickup_or_delivery; }
  HDI void check([[maybe_unused]] int n_orders) const
  {
    cuopt_assert(pickup_or_delivery >= 0, "id must be strictly positive");
    cuopt_assert(pickup_or_delivery < n_orders, "id must be lower than number of orders");
  }
  int pickup_or_delivery{};
};

template <typename i_t>
struct info_pos_pair_t {
  HDI info_pos_pair_t() {}
  HDI info_pos_pair_t(uint32_t request_id, uint32_t pos)
  {
    request_id_ = request_id;
    pos_        = pos;
  }

  HDI uint32_t get_request_id() const { return request_id_; }

  HDI uint32_t get_pos() const { return pos_; }

 private:
  uint32_t request_id_;
  uint32_t pos_ = std::numeric_limits<uint32_t>::max();
};

template <typename i_t, request_t REQUEST, typename Enable = void>
class request_info_t;

template <typename i_t, request_t REQUEST>
class request_info_t<i_t, REQUEST, std::enable_if_t<REQUEST == request_t::PDP>> {
 public:
  request_info_t() = default;

  DI request_info_t(detail::NodeInfo<i_t> info_, detail::NodeInfo<i_t> brother_info_)
    : info(info_), brother_info(brother_info_)
  {
    if (info_.is_service_node()) {
      cuopt_assert(info_.node() != brother_info.node(), "Node cannot be its own brother");
    }
  }

  DI bool operator==(request_info_t const& rhs) const noexcept
  {
    return info == rhs.info && brother_info == rhs.brother_info;
  }

  static constexpr auto primary_node_type() { return node_type_t::PICKUP; }
  static constexpr i_t size() { return 2; }
  DI bool is_pickup() const { return info.is_pickup(); }
  DI bool is_delivery() const { return info.is_delivery(); }
  DI i_t brother_id() const { return brother_info.node(); }
  HDI bool is_valid(bool depot_included) const
  {
    if (depot_included) {
      return info.node() > 0 && brother_info.node() > 0;
    } else {
      return info.node() >= 0 && brother_info.node() >= 0;
    }
  }

  constexpr void print()
  {
    printf("node: %i, brother_node: %i\n", info.node(), brother_info.node());
  }

  DI bool is_valid(i_t n_orders) const
  {
    return info.node() < n_orders && brother_info.node() < n_orders;
  }

  detail::NodeInfo<i_t> info;
  detail::NodeInfo<i_t> brother_info;
};

template <typename i_t, request_t REQUEST>
class request_info_t<i_t, REQUEST, std::enable_if_t<REQUEST == request_t::VRP>> {
 public:
  request_info_t() = default;

  DI request_info_t(detail::NodeInfo<i_t> info_) : info(info_) {}
  DI request_info_t(detail::NodeInfo<i_t> info_, detail::NodeInfo<i_t> brother_info_) : info(info_)
  {
  }

  DI bool operator==(request_info_t const& rhs) const noexcept { return info == rhs.info; }

  static constexpr auto primary_node_type() { return node_type_t::DELIVERY; }
  static constexpr i_t size() { return 1; }
  constexpr void print() { printf("node: %i\n", info.node()); }

  DI bool is_valid(bool depot_included) const
  {
    if (depot_included) {
      return info.node() > 0;
    } else {
      return info.node() >= 0;
    }
  }
  DI bool is_valid(i_t n_orders) const { return info.node() < n_orders; }

  detail::NodeInfo<i_t> info;
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
