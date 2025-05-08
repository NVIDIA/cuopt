* optimize
*  cost = 0.2 * VAR1 + 0.1 * VAR2
* subject to
*  1.2 <= 3 * VAR1 + 4 * VAR2 <= 5.4
*  1.5 <= 2.7 * VAR1 + 10.1 * VAR2 <= 4.9
*  7.4 * VAR1 + 0.2 * VAR2 == 7.9
*  9.4 * VAR1 + 0.4 * VAR2 == 6.9
NAME   good-mps-fixed-ranges
ROWS
 N  COST
 L  ROW1
 G  ROW2
 E  ROW3
 E  ROW4
COLUMNS
    VAR1      COST      0.2
    VAR1      ROW1      3              ROW2      2.7
    VAR1      ROW3      7.4            ROW4      9.4
    VAR2      COST      0.1
    VAR2      ROW1      4              ROW2      10.1
    VAR2      ROW3      9.4            ROW4      0.4
RHS
    RHS1      ROW1      5.4            ROW2      1.5
    RHS1      ROW3      9.5            ROW4      3.5
RANGES
    RANGE     ROW1      4.2            ROW2      3.4
    RANGE     ROW0      -1.6
    RANGE     ROW4      3.4
ENDATA
