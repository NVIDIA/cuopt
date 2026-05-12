/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.linear_programming;

/**
 * User-provided callback invoked by the MIP solver each time a new
 * incumbent solution is found. Register via
 * {@link SolverSettings#setMIPGetSolutionCallback(MIPGetSolutionCallback)}.
 *
 * <p>The arrays and scalars passed in are valid <b>only for the duration
 * of the callback</b>. Copy any data you need to keep beyond the call.
 *
 * <p>Exceptions thrown by the callback are caught and dropped; they will
 * not propagate to the native solver.
 *
 * <p>Do not call back into the solver (e.g. {@code Problem.solve()}) from
 * within this callback — the solver is single-threaded and re-entry will
 * deadlock or corrupt state.
 *
 * @see MIPSetSolutionCallback
 */
@FunctionalInterface
public interface MIPGetSolutionCallback {

    /**
     * @param solution        The incumbent's primal values, one per variable
     *                        (length = number of variables in the problem).
     * @param objectiveValue  The incumbent's objective value.
     * @param solutionBound   The current dual / user bound on the optimal value.
     */
    void onSolution(double[] solution, double objectiveValue, double solutionBound);
}
