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
 * <p>Constraints can be one-sided (a single {@code rhs} with a
 * {@code sense} in {@link CType}) or ranged (lower &lt;= lhs &lt;= upper);
 * {@link #isRanged()} discriminates.
 */
public final class Constraint {

    private final Problem owner;
    private final int index;
    private final CType sense;        // null for ranged constraints
    private final double rhs;         // single rhs for one-sided
    private final double lowerBound;  // for ranged
    private final double upperBound;  // for ranged
    private final boolean ranged;
    private final String name;

    // Package-private constructor for one-sided constraints.
    Constraint(Problem owner, int index, CType sense, double rhs, String name) {
        this.owner = Objects.requireNonNull(owner, "owner");
        this.index = index;
        this.sense = Objects.requireNonNull(sense, "sense");
        this.rhs = rhs;
        this.lowerBound = Double.NaN;
        this.upperBound = Double.NaN;
        this.ranged = false;
        this.name = name == null ? "" : name;
    }

    // Package-private constructor for ranged constraints.
    Constraint(Problem owner, int index, double lowerBound, double upperBound, String name) {
        this.owner = Objects.requireNonNull(owner, "owner");
        this.index = index;
        this.sense = null;
        this.rhs = Double.NaN;
        this.lowerBound = lowerBound;
        this.upperBound = upperBound;
        this.ranged = true;
        this.name = name == null ? "" : name;
    }

    public int index() {
        return index;
    }

    public String name() {
        return name;
    }

    /**
     * Constraint sense ({@link CType#LE}, {@link CType#GE},
     * {@link CType#EQ}) for one-sided constraints.
     *
     * @throws IllegalStateException if this is a ranged constraint.
     */
    public CType sense() {
        if (ranged) {
            throw new IllegalStateException(
                "sense() is not defined for ranged constraints; use lowerBound()/upperBound()");
        }
        return sense;
    }

    /**
     * Right-hand side for one-sided constraints.
     *
     * @throws IllegalStateException if this is a ranged constraint.
     */
    public double rhs() {
        if (ranged) {
            throw new IllegalStateException(
                "rhs() is not defined for ranged constraints; use lowerBound()/upperBound()");
        }
        return rhs;
    }

    /** Whether this is a ranged constraint (lower &lt;= lhs &lt;= upper). */
    public boolean isRanged() {
        return ranged;
    }

    /**
     * Lower bound for ranged constraints. For one-sided constraints,
     * returns the appropriate one-sided bound (e.g. rhs for GE; -infinity for LE).
     */
    public double lowerBound() {
        if (ranged) return lowerBound;
        return switch (sense) {
            case LE -> Double.NEGATIVE_INFINITY;
            case GE, EQ -> rhs;
        };
    }

    /**
     * Upper bound for ranged constraints. For one-sided constraints,
     * returns the appropriate one-sided bound (e.g. rhs for LE; +infinity for GE).
     */
    public double upperBound() {
        if (ranged) return upperBound;
        return switch (sense) {
            case LE, EQ -> rhs;
            case GE -> Double.POSITIVE_INFINITY;
        };
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
