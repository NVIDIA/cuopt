* Min  2x - y
* s.t. x + y <= 3
*      0 <= x <= 1
*      1 <= y <= 2
NAME lp_model_with_var_bounds
ROWS
 N  OBJ
 L  con
COLUMNS
     x        con      1
     x        OBJ      2
     y        con      1
     y        OBJ      -1
RHS
    rhs       con      3
BOUNDS
 LO bounds1    y        1
 UP bounds2    y        2
 LO bounds3    x        0
 UP bounds4    x        1
ENDATA
