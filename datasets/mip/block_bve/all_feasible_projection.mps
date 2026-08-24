NAME          BVE_ALL_FEASIBLE
ROWS
 N  Obj
 L  r0
 L  r1
 G  r2
COLUMNS
    MARK0000  'MARKER'                 'INTORG'
    a         r0        1
    a         r1        1
    a         r2        1
    b         Obj       1
    b         r0       -1
    b         r2       -1
    c         Obj       1
    c         r1       -1
    c         r2       -1
    MARK0001  'MARKER'                 'INTEND'
RHS
    RHS_V     r2       -1
BOUNDS
 BV BOUND     a
 BV BOUND     b
 BV BOUND     c
ENDATA
