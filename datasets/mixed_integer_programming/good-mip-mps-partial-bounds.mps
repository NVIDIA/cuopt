* maximize
*  cost = 100.0 * VAR1 + 150.0 * VAR2
* subject to
*  8000.0 * VAR1 + 4000.0 * VAR2 <= 40000.0
*  15.0 * VAR1 + 30.0 * VAR2 <= 200.0
*  VAR2 >= 0
NAME   mps_ref
OBJSENSE
    MAX
ROWS
 N  COST
 L  ROW1
 L  ROW2
COLUMNS
    MARK000   'MARKER'                 'INTORG'
    VAR1      COST      100.0
    VAR1      ROW1      8000.0         ROW2      15.0
    MARK001   'MARKER'                 'INTEND'
    VAR2      COST      150.0
    VAR2      ROW1      4000.0           ROW2      30.0
RHS
    RHS1      ROW1      40000.0        ROW2      200.0
BOUNDS
 UP BOUND      VAR2        10
 LO BOUND      VAR2        0
ENDATA