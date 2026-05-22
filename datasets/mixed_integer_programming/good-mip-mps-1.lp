\ Equivalent of good-mip-mps-1.mps
\ maximize 100*VAR1 + 150*VAR2
\  s.t. 8000*VAR1 + 4000*VAR2 <= 40000
\       15*VAR1 + 30*VAR2 <= 200
\       0 <= VAR1, VAR2 <= 10 and integer

Maximize
  100 VAR1 + 150 VAR2
Subject To
  ROW1: 8000 VAR1 + 4000 VAR2 <= 40000
  ROW2: 15 VAR1 + 30 VAR2 <= 200
Bounds
  VAR1 <= 10
  VAR2 <= 10
Generals
  VAR1 VAR2
End
