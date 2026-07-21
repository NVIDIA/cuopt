NAME          SOCPSAMPLE
ROWS
 N  OBJ
 G  C1
 L  QCSTD
 L  QCROT
COLUMNS
    A         C1               1.0
    B         C1               1.0
    S         OBJ              1.0
    P         OBJ              1.0
    Q         OBJ              1.0
RHS
    RHS       C1               2.0
BOUNDS
 FR BND       A
 FR BND       B
QCMATRIX   QCSTD
    A         A                1.0
    B         B                1.0
    S         S               -1.0
QCMATRIX   QCROT
    A         A                1.0
    B         B                1.0
    P         Q               -0.5
    Q         P               -0.5
ENDATA
