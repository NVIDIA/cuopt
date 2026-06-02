\ Convex SOC-friendly QCQP with closed-form optimum.  Companion problem
\ to QC_Test_1.mps: same wire-format aspects (multiple QCs, off-diagonal
\ QCMATRIX cross-terms, linear-in-QC-row terms, normal LP constraint
\ alongside QCs) but with a non-trivial objective that the SOCP/barrier
\ path must actually optimize, so end-to-end gRPC tests can verify both
\ transport and solver correctness.
\
\ SOC conversion strictly requires rhs = 0 for all QC rows and forbids
\ mixing off-diagonal Q entries with linear-in-row terms in the same
\ QC, so the wire-format features are split:
\   QC0 = rotated SOC  (off-diagonal cross-term, no linear-in-row)
\   QC1 = affine SOC   (linear-in-row, no off-diagonal)
\
\ Problem:
\   min  -x - y - z
\   s.t. x + y + z <= 10            (LIN0: linear, slack at optimum)
\        z^2 - 2*x*y <= 0           (QC0: rotated SOC, cross-term)
\        x^2 + y^2 - z <= 0         (QC1: affine SOC, linear-in-row)
\        x, y, z >= 0
\
\ Both QCs convert cleanly to second-order cones:
\   QC0: cross-term -2*x*y plus +z^2  =>  z^2 <= 2*x*y
\   QC1: x^2 + y^2 with linear -z     =>  z >= x^2 + y^2
\
\ QC0 implies x*y >= 0; QC1 implies z >= 0 and feasible region is
\ bounded.  By symmetry of objective and constraints under x <-> y, the
\ KKT conditions force x = y = a at optimum.  With x = y = a, QC0 binds
\ at z = a*sqrt(2), QC1 binds at z = 2*a^2.  Both binding => a = 1/sqrt(2),
\ z = 1.  Multipliers are strictly positive at this point so both QCs
\ are simultaneously active.
\
\   Optimal solution:  x = y = 1/sqrt(2) ~= 0.70710678, z = 1
\   Optimal objective: -(1 + sqrt(2)) ~= -2.41421356

Minimize
 OBJ: - x - y - z
Subject To
 LIN0: x + y + z <= 10
 QC0: [ - 2 x * y + z ^ 2 ] <= 0
 QC1: - z + [ x ^ 2 + y ^ 2 ] <= 0
Bounds
 x >= 0
 y >= 0
 z >= 0
End
