/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.linear_programming;

/**
 * Solver statistics for MILP solves. Returned by
 * {@code Problem.milpStats()} (empty {@link java.util.Optional} for LP-only problems).
 *
 * @param mipGap             relative MIP gap at termination ({@code (UB - LB) / |UB|})
 * @param bestBound          best dual bound found
 * @param nodesExplored      number of branch-and-bound nodes explored
 * @param incumbentsFound    number of distinct incumbent solutions found
 * @param presolveTime       wall-clock time spent in presolve, seconds
 * @param rootRelaxationTime wall-clock time spent on the root LP relaxation, seconds
 */
public record MilpStats(
    double mipGap,
    double bestBound,
    long nodesExplored,
    long incumbentsFound,
    double presolveTime,
    double rootRelaxationTime
) {}
