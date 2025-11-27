/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include "../routing_helpers.cuh"

#include <array>
#include <csignal>
#include <cstdio>
#define DEPO 0

// Array indexes for evaluation of distinct features
// distance
constexpr int DIST = 0;
// time
constexpr int TIME = 1;
// pdp capacity ( positive/negative supply )
constexpr int CAP = 2;

constexpr int PRIZE = 3;

constexpr int TASKS = 4;

constexpr int SERVICE_TIME = 5;

constexpr int MISMATCH = 6;

constexpr int BREAK = 7;

constexpr int VEHICLE_FIXED_COST = 8;

constexpr int NDIM = 9;

#define MACHINE_EPSILON 0.000001
#define MOVE_EPSILON    0.0001

//! A type storing all dimensions data of a solution
using costs = std::array<double, NDIM>;

inline double apply_costs(const costs& in, const costs& weights) noexcept
{
  double cost = 0.;
  cuopt::routing::detail::constexpr_for<0, NDIM, 1>([&](auto I) { cost += in[I] * weights[I]; });
  return cost;
}

// #define RUNTIME_RUNTIME_TEST

#ifdef RUNTIME_RUNTIME_TEST
#define INVARIANT(__) (__);
#define RUNTIME_TEST(__)                                               \
  if (!(__)) {                                                         \
    printf("test invalid: %s, %s:%d\n", __func__, __FILE__, __LINE__); \
    std::raise(SIGTRAP);                                               \
  }
#else
#define RUNTIME_TEST(_)
#define INVARIANT(_)
#endif
