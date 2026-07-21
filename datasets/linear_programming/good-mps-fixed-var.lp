\ Equivalent of good-mps-fixed-var.mps
\  VAR1 fixed at 2; VAR2 default [0, inf)

Minimize
  0.2 VAR1 + 0.1 VAR2
Subject To
  ROW1: 3 VAR1 + 4 VAR2 <= 5.4
  ROW2: 2.7 VAR1 + 10.1 VAR2 <= 4.9
Bounds
  VAR1 = 2
End
