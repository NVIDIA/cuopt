/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.spi;

import com.nvidia.cuopt.linear_programming.ErrorStatus;
import com.nvidia.cuopt.linear_programming.LpStats;
import com.nvidia.cuopt.linear_programming.MIPStats;
import com.nvidia.cuopt.linear_programming.SolverMethod;
import com.nvidia.cuopt.linear_programming.TerminationStatus;

/**
 * Internal data carrier from the FFM implementation back to the
 * {@link com.nvidia.cuopt.linear_programming.Problem} or
 * {@link com.nvidia.cuopt.linear_programming.Problem} that requested
 * the solve. Pre-extracted from the native {@code cuOptSolution}
 * handle before that handle is freed.
 *
 * <p>This is on the SPI boundary because {@code Problem} (Java 21)
 * cannot reference Java 22 FFM types. Pure Java types only.
 */
public record SolveResult(
    double[] primalSolution,
    double[] dualSolution,
    double[] reducedCost,
    double[] slack,
    TerminationStatus terminationStatus,
    String terminationReason,
    ErrorStatus errorStatus,
    String errorMessage,
    double objectiveValue,
    double dualObjectiveValue,
    double solveTime,
    SolverMethod solvedBy,
    LpStats lpStats,
    MIPStats mipStats
) {}
