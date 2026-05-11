/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.linear_programming;

/**
 * Solver statistics for LP solves. Returned by {@code Problem.lpStats()}.
 *
 * <p>Fields not measured by the underlying solver are reported as
 * {@code Double.NaN} (doubles) or {@code -1L} (longs).
 *
 * @param primalResidual  primal infeasibility residual at termination
 * @param dualResidual    dual infeasibility residual at termination
 * @param primalObjective primal objective value
 * @param dualObjective   dual objective value
 * @param iterations      total iteration count
 * @param gap             relative primal-dual gap at termination
 */
public record LpStats(
    double primalResidual,
    double dualResidual,
    double primalObjective,
    double dualObjective,
    long iterations,
    double gap
) {}
