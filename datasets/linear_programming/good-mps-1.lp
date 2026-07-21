\ Equivalent of good-mps-1.mps
\  min 0.2*VAR1 + 0.1*VAR2
\  s.t. 3*VAR1 + 4*VAR2 <= 5.4
\       2.7*VAR1 + 10.1*VAR2 <= 4.9
\  VAR1, VAR2 >= 0 (default)

Minimize
  0.2 VAR1 + 0.1 VAR2
Subject To
  ROW1: 3 VAR1 + 4 VAR2 <= 5.4
  ROW2: 2.7 VAR1 + 10.1 VAR2 <= 4.9
End
