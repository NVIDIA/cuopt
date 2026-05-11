/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.linear_programming;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * A mutable quadratic expression of the form
 * {@code Σ qᵢⱼ·xᵢ·xⱼ + Σ cᵢ·xᵢ + constant}.
 *
 * <p>The linear part is stored as a {@link LinearExpr}; quadratic terms
 * are stored as parallel arrays of (coefficient, var1, var2) triples
 * (one entry per Q-matrix non-zero). All chainable methods return
 * {@code this}.
 *
 * <pre>{@code
 *   QuadraticExpr obj = new QuadraticExpr()
 *       .addTerms(linearCoeffs, x)              // linear part bulk
 *       .addQuadraticTerm(1.0, x[0], x[0])      // x[0]² term
 *       .addQuadraticTerm(0.5, x[0], x[1]);     // 0.5 · x[0]·x[1] term
 * }</pre>
 */
public final class QuadraticExpr {

    private final LinearExpr linearPart = new LinearExpr();
    private final List<Variable> qVar1 = new ArrayList<>();
    private final List<Variable> qVar2 = new ArrayList<>();
    private final List<Double> qCoeff = new ArrayList<>();
    private Problem owner;

    public QuadraticExpr() {}

    /** Adds a linear term {@code coeff · var}. Same convention as {@link LinearExpr#addTerm}. */
    public QuadraticExpr addTerm(double coeff, Variable var) {
        bindOwner(Objects.requireNonNull(var, "var"));
        linearPart.addTerm(coeff, var);
        return this;
    }

    /** Adds a bulk linear part. */
    public QuadraticExpr addTerms(double[] coeffs, Variable[] vars) {
        Objects.requireNonNull(coeffs, "coeffs");
        Objects.requireNonNull(vars, "vars");
        if (coeffs.length != vars.length) {
            throw new IllegalArgumentException(
                "coefficient array length " + coeffs.length
                + " != variable array length " + vars.length);
        }
        for (int i = 0; i < vars.length; i++) {
            addTerm(coeffs[i], vars[i]);
        }
        return this;
    }

    /**
     * Adds a quadratic term {@code coeff · v1 · v2} to the objective.
     * For squared terms, pass the same variable twice.
     */
    public QuadraticExpr addQuadraticTerm(double coeff, Variable v1, Variable v2) {
        Objects.requireNonNull(v1, "v1");
        Objects.requireNonNull(v2, "v2");
        bindOwner(v1);
        bindOwner(v2);
        qVar1.add(v1);
        qVar2.add(v2);
        qCoeff.add(coeff);
        return this;
    }

    /** Adds a scalar offset. */
    public QuadraticExpr addConstant(double c) {
        linearPart.addConstant(c);
        return this;
    }

    /** Adds another quadratic expression. */
    public QuadraticExpr add(QuadraticExpr other) {
        Objects.requireNonNull(other, "other");
        linearPart.add(other.linearPart);
        for (int i = 0; i < other.qVar1.size(); i++) {
            addQuadraticTerm(other.qCoeff.get(i), other.qVar1.get(i), other.qVar2.get(i));
        }
        return this;
    }

    /** Adds a linear expression to the linear part. */
    public QuadraticExpr add(LinearExpr linear) {
        Objects.requireNonNull(linear, "linear");
        if (linear.owner() != null) {
            if (owner == null) owner = linear.owner();
            else if (owner != linear.owner()) {
                throw new IllegalArgumentException(
                    "Cannot mix variables from different Problems in one QuadraticExpr");
            }
        }
        linearPart.add(linear);
        return this;
    }

    /** Number of quadratic (non-linear) terms. */
    public int numQuadraticTerms() {
        return qVar1.size();
    }

    public LinearExpr linearPart() {
        return linearPart;
    }

    public Problem owner() {
        return owner != null ? owner : linearPart.owner();
    }

    // ── internal accessors used by the FFM implementation
    //    (public for cross-package access from internal/; not part of
    //    the stable user-facing API — do not rely on these in user code) ─

    public Variable quadVar1(int i) {
        return qVar1.get(i);
    }

    public Variable quadVar2(int i) {
        return qVar2.get(i);
    }

    public double quadCoeff(int i) {
        return qCoeff.get(i);
    }

    private void bindOwner(Variable v) {
        if (owner == null) {
            owner = v.owner();
        } else if (owner != v.owner()) {
            throw new IllegalArgumentException(
                "Cannot mix variables from different Problems in one QuadraticExpr");
        }
    }
}
