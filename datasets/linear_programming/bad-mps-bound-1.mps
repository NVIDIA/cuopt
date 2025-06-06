* optimize
*  cost = 0.2 * VAR1 + 0.1 * VAR2
* subject to
*  3 * VAR1 + 4 * VAR2 <= 5.4
*  2.7 * VAR1 + 10.1 * VAR2 <= 4.9
*  0 = VAR1 or 1 <= VAR1 <= 2 semi-cont integer variable
*  0  <= VAR2 <= inf
NAME   bad-mps-bound-SC
ROWS
 N  COST
 L  ROW1
 L  ROW2
COLUMNS
    VAR1      COST      0.2
    VAR1      ROW1      3              ROW2      2.7
    VAR2      COST      0.1
    VAR2      ROW1      4              ROW2      10.1
RHS
    RHS1      ROW1      5.4            ROW2      4.9
BOUNDS
 SC bnd       VAR1      2
ENDATA