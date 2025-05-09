* optimize
*  cost = 0.2 * VAR1 + 0.1 * VAR2
* subject to
*  3 * VAR1 + 4 * VAR2 <= 5.4
*  2.7 * VAR1 + 10.1 * VAR2 <= 4.9
*
* contains spaces in between the names (which is still a valid MPS)
NAME   good-1
ROWS
 N  COST
 L  RO W1
 L  ROW2
COLUMNS
    VA R1     COST      0.2
    VA R1     RO W1     3              ROW2      2.7
    VAR2      COST      0.1
    VAR2      RO W1     4              ROW2      10.1
RHS
    RHS1      RO W1     5.4            ROW2      4.9
ENDATA
