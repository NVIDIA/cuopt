\ Equivalent of good-mip-mps-partial-bounds.mps
\ maximize 100*VAR1 + 150*VAR2
\ VAR1 integer with no explicit bounds (MPS default [0,1] ⇒ Binaries in LP)
\ VAR2 continuous with [0, 10] from BOUNDS section

Maximize
  100 VAR1 + 150 VAR2
Subject To
  ROW1: 8000 VAR1 + 4000 VAR2 <= 40000
  ROW2: 15 VAR1 + 30 VAR2 <= 200
Bounds
  VAR2 <= 10
Binaries
  VAR1
End
