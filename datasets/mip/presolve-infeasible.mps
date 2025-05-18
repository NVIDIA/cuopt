NAME          infeasible_integer_example
ROWS
 N  COST
 E  C1
 E  C2
COLUMNS
    MARK0000  'MARKER'                 'INTORG'
    x1        COST    10
    x1        C1      2
    x1        C2      1
    x2        COST    15
    x2        C1      1
    x2        C2      3
    MARK0001  'MARKER'                 'INTEND'
RHS
    RHS1      C1      3
    RHS1      C2      1
BOUNDS
 LO BND1      x1      0
 LO BND1      x2      0
ENDATA
