\ Equivalent of good-mps-some-var-bounds.mps
\  -1 <= VAR1 <= inf; 0 <= VAR2 <= 2

Minimize
  0.2 VAR1 + 0.1 VAR2
Subject To
  ROW1: 3 VAR1 + 4 VAR2 <= 5.4
  ROW2: 2.7 VAR1 + 10.1 VAR2 <= 4.9
Bounds
  -1 <= VAR1
  VAR2 <= 2
End
