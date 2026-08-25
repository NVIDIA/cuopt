\ Equivalent of lp_model_with_var_bounds.mps
\  min 2x - y
\  s.t. x + y <= 3
\       0 <= x <= 1
\       1 <= y <= 2

Minimize
  2 x - y
Subject To
  con: x + y <= 3
Bounds
  0 <= x <= 1
  1 <= y <= 2
End
