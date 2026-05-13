/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.linear_programming;

import static com.nvidia.cuopt.cuOpt.*;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class QpSolverIT {

    /**
     * Mirrors cuOpt Python {@code test_qp.py::test_solver}:
     *   minimize x1^2 + 4*x2^2 - 8*x1 - 16*x2
     *   s.t. x1 + x2 >= 5
     *        x1 >= 3
     *        x2 >= 0
     */
    @Test
    void simple_qp_minimize() {
        try (var problem = new Problem("qp_test")) {
            Variable x1 = problem.addVariable(3.0, 10.0, CONTINUOUS, "x1");
            Variable x2 = problem.addVariable(0.0, 10.0, CONTINUOUS, "x2");

            LinearExpr c = new LinearExpr()
                .addTerm(1.0, x1).addTerm(1.0, x2);
            problem.addConstraint(c, GREATER_EQUAL, 5.0, "sum");

            QuadraticExpr obj = new QuadraticExpr()
                .addQuadraticTerm(1.0, x1, x1)
                .addQuadraticTerm(4.0, x2, x2)
                .addTerm(-8.0, x1)
                .addTerm(-16.0, x2);
            problem.setObjective(obj, MINIMIZE);

            assertEquals(ProblemCategory.QP, problem.problemCategory());

            problem.solve();

            // QP should terminate with OPTIMAL or a feasible status.
            TerminationStatus s = problem.status();
            assertTrue(s == TerminationStatus.OPTIMAL
                    || s == TerminationStatus.PRIMAL_FEASIBLE,
                "QP should be feasible and optimal, got " + s);

            // Solution within bounds.
            assertTrue(x1.value() >= 3.0 - 1e-6);
            assertTrue(x2.value() >= 0.0 - 1e-6);
            assertTrue(x1.value() + x2.value() >= 5.0 - 1e-6,
                "sum constraint should be satisfied");
        }
    }
}
