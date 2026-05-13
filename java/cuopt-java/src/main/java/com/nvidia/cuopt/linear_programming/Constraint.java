/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.linear_programming;

import com.nvidia.cuopt.CuOptException;
import java.util.Objects;

/**
 * A constraint owned by a {@link Problem}.
 *
 * <p>Like {@link Variable}, this is an immutable handle. Post-solve
 * accessors ({@link #dualValue()}, {@link #slack()}) delegate back to
 * the owning Problem.
 *
 * <p>A constraint has a sense ({@link CType#LE}, {@link CType#GE},
 * {@link CType#EQ}) and a right-hand-side value.
 */
public final class Constraint {

    private final Problem owner;
    private final int index;
    private final CType sense;
    private final double rhs;
    private final String name;

    // Package-private constructor.
    Constraint(Problem owner, int index, CType sense, double rhs, String name) {
        this.owner = Objects.requireNonNull(owner, "owner");
        this.index = index;
        this.sense = Objects.requireNonNull(sense, "sense");
        this.rhs = rhs;
        this.name = name == null ? "" : name;
    }

    public int index() {
        return index;
    }

    public String name() {
        return name;
    }

    /** Constraint sense ({@link CType#LE}, {@link CType#GE}, {@link CType#EQ}). */
    public CType sense() {
        return sense;
    }

    /** Right-hand side of the constraint. */
    public double rhs() {
        return rhs;
    }

    // ── post-solve accessors ─────────────────────────────────────

    /**
     * Dual value (shadow price) of this constraint in the current
     * LP solution. LP only — for MIP solves, returns {@code Double.NaN}.
     *
     * @throws CuOptException if the owning Problem has not been solved
     *                        or has been closed.
     */
    public double dualValue() {
        return owner.dualValueOf(index);
    }

    /**
     * Slack of this constraint in the current solution.
     *
     * @throws CuOptException if the owning Problem has not been solved
     *                        or has been closed.
     */
    public double slack() {
        return owner.slackOf(index);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Constraint other)) return false;
        return owner == other.owner && index == other.index;
    }

    @Override
    public int hashCode() {
        return System.identityHashCode(owner) * 31 + index;
    }

    @Override
    public String toString() {
        return name.isEmpty() ? "Constraint[" + index + "]" : "Constraint[" + name + "]";
    }

    Problem owner() {
        return owner;
    }
}
