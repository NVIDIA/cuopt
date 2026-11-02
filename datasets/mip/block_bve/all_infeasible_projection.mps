NAME          BVE_ALL_INFEASIBLE
ROWS
 N  Obj
 G  p000
 G  p001
 G  p010
 G  p011
 G  p100
 G  p101
 G  p110
 G  p111
 L  link
COLUMNS
    MARK0000  'MARKER'                 'INTORG'
    a         p000      1
    a         p001      1
    a         p010      1
    a         p011      1
    a         p100     -1
    a         p101     -1
    a         p110     -1
    a         p111     -1
    a         link      1
    b         p000      1
    b         p001      1
    b         p010     -1
    b         p011     -1
    b         p100      1
    b         p101      1
    b         p110     -1
    b         p111     -1
    c         p000      1
    c         p001     -1
    c         p010      1
    c         p011     -1
    c         p100      1
    c         p101     -1
    c         p110      1
    c         p111     -1
    x         Obj       1
    x         link     -1
    MARK0001  'MARKER'                 'INTEND'
RHS
    RHS_V     p000      1
    RHS_V     p011     -1
    RHS_V     p101     -1
    RHS_V     p110     -1
    RHS_V     p111     -2
BOUNDS
 BV BOUND     a
 BV BOUND     b
 BV BOUND     c
 BV BOUND     x
ENDATA
