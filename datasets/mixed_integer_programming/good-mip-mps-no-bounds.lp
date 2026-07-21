\ Equivalent of good-mip-mps-no-bounds.mps
\ maximize 100*VAR1 + 150*VAR2; VAR1 integer, VAR2 continuous.
\ MPS defaults VAR1 (unbounded integer) to [0, 1]; LP "Binaries" matches.

Maximize
  100 VAR1 + 150 VAR2
Subject To
  ROW1: 8000 VAR1 + 4000 VAR2 <= 40000
  ROW2: 15 VAR1 + 30 VAR2 <= 200
Binaries
  VAR1
End
