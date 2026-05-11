/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.linear_programming;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * A mutable linear expression of the form {@code Σ cᵢ·xᵢ + constant}.
 *
 * <p>Supports two construction styles, both chainable, freely
 * interchangeable. Same-variable coefficients accumulate (Gurobi
 * convention):
 *
 * <pre>{@code
 *   // Hand-written, per-term:
 *   LinearExpr e = new LinearExpr()
 *       .addTerm(2.0, x)
 *       .addTerm(3.0, y);
 *
 *   // Bulk arrays (good for data-driven generation):
 *   LinearExpr e = new LinearExpr().addTerms(coeffs, vars);
 *
 *   // Mix and match:
 *   LinearExpr e = new LinearExpr()
 *       .addTerms(baseCoeffs, baseVars)
 *       .addTerm(0.5, special)
 *       .addConstant(10);
 * }</pre>
 *
 * <p>All terms in one expression must reference Variables owned by the
 * same {@link Problem}. Mixing across problems throws
 * {@link IllegalArgumentException}.
 */
public final class LinearExpr {

    // LinkedHashMap preserves insertion order (useful for deterministic
    // CSR builds), keys are Variables, values are summed coefficients.
    private final Map<Variable, Double> terms = new LinkedHashMap<>();
    private double constant = 0.0;
    private Problem owner; // null until first term is added

    public LinearExpr() {}

    /**
     * Adds a single term {@code coeff · var} to this expression.
     * Returns this for chaining. If {@code var} is already present, the
     * coefficients sum.
     */
    public LinearExpr addTerm(double coeff, Variable var) {
        Objects.requireNonNull(var, "var");
        bindOwner(var);
        terms.merge(var, coeff, Double::sum);
        return this;
    }

    /**
     * Adds n terms in bulk. {@code coeffs} and {@code vars} must have
     * the same length. Same-variable coefficients accumulate.
     */
    public LinearExpr addTerms(double[] coeffs, Variable[] vars) {
        Objects.requireNonNull(coeffs, "coeffs");
        Objects.requireNonNull(vars, "vars");
        if (coeffs.length != vars.length) {
            throw new IllegalArgumentException(
                "coefficient array length " + coeffs.length
                + " != variable array length " + vars.length);
        }
        for (int i = 0; i < coeffs.length; i++) {
            addTerm(coeffs[i], vars[i]);
        }
        return this;
    }

    /** Adds a scalar offset to the expression. Returns this for chaining. */
    public LinearExpr addConstant(double c) {
        this.constant += c;
        return this;
    }

    /** Adds another linear expression to this one. Returns this. */
    public LinearExpr add(LinearExpr other) {
        Objects.requireNonNull(other, "other");
        for (Map.Entry<Variable, Double> e : other.terms.entrySet()) {
            addTerm(e.getValue(), e.getKey());
        }
        this.constant += other.constant;
        return this;
    }

    /** Number of distinct variables in this expression. */
    public int numTerms() {
        return terms.size();
    }

    /** Constant offset. */
    public double constant() {
        return constant;
    }

    /**
     * Read-only view of the terms map. Order is insertion order
     * (deterministic). Same-variable coefficients have already been
     * summed at construction time.
     */
    public Map<Variable, Double> terms() {
        return java.util.Collections.unmodifiableMap(terms);
    }

    /**
     * Owning Problem (the Problem all variables in this expression
     * belong to). {@code null} if no terms have been added.
     */
    public Problem owner() {
        return owner;
    }

    private void bindOwner(Variable v) {
        if (owner == null) {
            owner = v.owner();
        } else if (owner != v.owner()) {
            throw new IllegalArgumentException(
                "Cannot mix variables from different Problems in one LinearExpr");
        }
    }
}
