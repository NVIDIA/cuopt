/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.linear_programming;

import static com.nvidia.cuopt.cuOpt.*;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

class LinearExprTest {

    @Test
    void addTerm_and_addTerms_are_interchangeable() {
        try (var p = new Problem()) {
            Variable x = p.addVariable(0, INF, CONTINUOUS, "x");
            Variable y = p.addVariable(0, INF, CONTINUOUS, "y");
            Variable z = p.addVariable(0, INF, CONTINUOUS, "z");

            LinearExpr a = new LinearExpr()
                .addTerm(2.0, x).addTerm(3.0, y).addTerm(1.0, z);

            LinearExpr b = new LinearExpr()
                .addTerms(new double[]{2.0, 3.0, 1.0}, new Variable[]{x, y, z});

            assertEquals(a.numTerms(), b.numTerms());
            assertEquals(a.terms(), b.terms());
        }
    }

    @Test
    void same_variable_coefficients_accumulate() {
        try (var p = new Problem()) {
            Variable x = p.addVariable(0, INF, CONTINUOUS, "x");
            LinearExpr e = new LinearExpr()
                .addTerm(2.0, x).addTerm(3.0, x);
            assertEquals(1, e.numTerms());
            assertEquals(5.0, e.terms().get(x), 1e-12);
        }
    }

    @Test
    void length_mismatch_throws() {
        try (var p = new Problem()) {
            Variable x = p.addVariable(0, INF, CONTINUOUS, "x");
            LinearExpr e = new LinearExpr();
            assertThrows(IllegalArgumentException.class,
                () -> e.addTerms(new double[]{1.0, 2.0}, new Variable[]{x}));
        }
    }

    @Test
    void mixed_problem_variables_throws() {
        try (var p1 = new Problem(); var p2 = new Problem()) {
            Variable x1 = p1.addVariable(0, INF, CONTINUOUS, "x1");
            Variable x2 = p2.addVariable(0, INF, CONTINUOUS, "x2");
            LinearExpr e = new LinearExpr().addTerm(1.0, x1);
            assertThrows(IllegalArgumentException.class,
                () -> e.addTerm(1.0, x2));
        }
    }
}
