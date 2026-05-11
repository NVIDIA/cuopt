/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.linear_programming;

import com.nvidia.cuopt.CuOptException;
import java.util.Objects;

/**
 * A decision variable owned by a {@link Problem}.
 *
 * <p>{@code Variable} is an immutable handle — all fields are
 * {@code final}. Post-solve accessors ({@link #value()},
 * {@link #reducedCost()}) delegate back to the owning Problem rather
 * than holding state themselves.
 *
 * <p>Variables are equal iff they refer to the same Problem and have
 * the same index.
 */
public final class Variable {

    private final Problem owner;
    private final int index;
    private final double lowerBound;
    private final double upperBound;
    private final VType variableType;
    private final String name;

    // Package-private — only Problem can create Variables.
    Variable(Problem owner, int index, double lowerBound, double upperBound,
             VType variableType, String name) {
        this.owner = Objects.requireNonNull(owner, "owner");
        this.index = index;
        this.lowerBound = lowerBound;
        this.upperBound = upperBound;
        this.variableType = Objects.requireNonNull(variableType, "variableType");
        this.name = name == null ? "" : name;
    }

    // ── identity / spec accessors ────────────────────────────────

    /** Zero-based index of this variable in its owning Problem. */
    public int index() {
        return index;
    }

    /** Variable name; empty string if none provided. */
    public String name() {
        return name;
    }

    public double lowerBound() {
        return lowerBound;
    }

    public double upperBound() {
        return upperBound;
    }

    public VType variableType() {
        return variableType;
    }

    // ── post-solve accessors (delegate to owning Problem) ────────

    /**
     * Primal value of this variable in the current solution.
     *
     * @throws CuOptException if the owning Problem has not been solved
     *                        or has been closed.
     */
    public double value() {
        return owner.primalValueOf(index);
    }

    /**
     * Reduced cost of this variable in the current LP solution. LP only;
     * returns {@code Double.NaN} for MIP solves.
     *
     * @throws CuOptException if the owning Problem has not been solved
     *                        or has been closed.
     */
    public double reducedCost() {
        return owner.reducedCostOf(index);
    }

    // ── equality / hashing ───────────────────────────────────────

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Variable other)) return false;
        return owner == other.owner && index == other.index;
    }

    @Override
    public int hashCode() {
        return System.identityHashCode(owner) * 31 + index;
    }

    @Override
    public String toString() {
        return name.isEmpty() ? "Variable[" + index + "]" : "Variable[" + name + "]";
    }

    // Package-private accessor for owner identity checks in expressions.
    Problem owner() {
        return owner;
    }
}
