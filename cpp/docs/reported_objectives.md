# Reported LP/QP objectives

For the minimization model

\[
  \min_{l\le x\le u,\;l_A\le Ax\le u_A}
  s\left(c^Tx+\tfrac12x^THx+o\right),
\]

the input quadratic setter stores the symmetric Hessian `H = Q + Q^T`.
The returned multipliers use the coefficient units of `c` and `H`; the separate
objective factor `s` is applied when reporting objective values.

With the support function \(\sigma_S(v)=\sup_{w\in S}v^Tw\), define

\[
 P=s(c^Tx+\tfrac12x^THx+o),\qquad
 D=s\left(-\sigma_{[l_A,u_A]}(-y)-\sigma_{[l,u]}(-z)
             -\tfrac12x^THx+o\right).
\]

The dual equality is \(Hx+c=A^Ty+z\). For LP, set \(H=0\).
A positive multiplier selects the lower bound; a negative multiplier selects the
upper bound. A zero multiplier contributes zero, including on an unbounded interval.
A nonzero multiplier selecting an infinite endpoint does not yield a finite dual
objective. Such a value is reported unavailable (`NaN`), rather than silently dropped.

For maximization, the returned multipliers have the objective sign restored. Select
bounds using `-y` and `-z`, while retaining the returned multipliers in the products.
This computes the corresponding upper-bound objective when the dual conditions hold.

\[
 \mathrm{gap}=|P-D|,\qquad
 \mathrm{relative\_gap}=\frac{|P-D|}{1+|P|+|D|}.
\]

The offset is included in both objective values and therefore in the relative-gap
denominator, matching the reported API quantities. The factor `s` is applied once.
These formulas do not assert feasibility: with inexact dual equality, `D` is an
objective estimate, not a certified bound. Residuals and multiplier signs must be
checked separately.

Native Barrier and DualSimplex results are evaluated after outer postsolve on the
original model and returned vectors. Concurrent results use the winning method.
No iterate, missing dual recovery, invalid multipliers, or unsupported quadratic
constraint multipliers leave dual objective and gap unavailable. Solver termination
status and residual definitions are unchanged. PDLP uses its existing reporting path.
