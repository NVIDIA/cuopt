*SENSE:Minimize
NAME    presolve_infeasible_small
ROWS
 N obj
 G c1
 L c2
 G c3
 L c4
COLUMNS
  MARK  'MARKER'        'INTORG'
  x    c1    1.000000000000e+00
  x    c2    1.000000000000e+00
  x    c3    1.000000000000e+00
  x    c4    1.000000000000e+00
  x    obj   1.000000000000e+00
  y    c1    1.000000000000e+00
  y    c2    1.000000000000e+00
  y    c3   -1.000000000000e+00
  y    obj   1.000000000000e+00
  z    c1    1.000000000000e+00
  z    c2    1.000000000000e+00
  z    c4    1.000000000000e+00
  z    obj   1.000000000000e+00
  MARK  'MARKER'        'INTEND'
RHS
  RHS   c1    2.000000000000e+00
  RHS   c2    1.000000000000e+00
  RHS   c3    0.000000000000e+00
  RHS   c4    2.000000000000e+00
BOUNDS
 BV BND   x
 BV BND   y
 BV BND   z
ENDATA
