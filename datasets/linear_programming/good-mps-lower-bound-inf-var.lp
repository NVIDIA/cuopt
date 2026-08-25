\ Equivalent of good-mps-lower-bound-inf-var.mps
\  VAR1 has MI (lower -inf, upper default +inf); VAR2 default [0, +inf)

Minimize
  0.2 VAR1 + 0.1 VAR2
Subject To
  ROW1: 3 VAR1 + 4 VAR2 <= 5.4
  ROW2: 2.7 VAR1 + 10.1 VAR2 <= 4.9
Bounds
  -inf <= VAR1
End
