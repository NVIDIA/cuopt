/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.linear_programming;

/**
 * User-provided callback invoked by the MIP solver when it would like
 * the user to inject a candidate primal solution. Register via
 * {@link SolverSettings#setMIPSetSolutionCallback(MIPSetSolutionCallback)}.
 *
 * <p><b>Registering a set-solution callback disables presolve.</b>
 *
 * <p>The callback must write the candidate's primal values into
 * {@code outSolution} (one per variable) and its objective value into
 * {@code outObjective[0]}. If the user has no candidate to inject, the
 * callback should still leave the buffer in a defined state; the
 * solver will use whatever is written. The arrays are valid only for
 * the duration of the callback.
 *
 * <p>Exceptions thrown by the callback are caught and dropped; they will
 * not propagate to the native solver. In that case, the buffer is not
 * written back and the solver proceeds without an injected solution.
 *
 * <p>Do not call back into the solver (e.g. {@code Problem.solve()}) from
 * within this callback — the solver is single-threaded and re-entry will
 * deadlock or corrupt state.
 *
 * @see MIPGetSolutionCallback
 */
@FunctionalInterface
public interface MIPSetSolutionCallback {

    /**
     * @param outSolution    Output buffer for primal values (length = number
     *                       of variables). Write your candidate values here.
     * @param outObjective   Single-element output buffer. Write the candidate's
     *                       objective value into {@code outObjective[0]}.
     * @param solutionBound  The current dual / user bound on the optimal value.
     */
    void provideSolution(double[] outSolution, double[] outObjective, double solutionBound);
}
