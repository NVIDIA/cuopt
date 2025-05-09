* maximize
*  cost = 100.0 * VAR1 + 150.0 * VAR2
* subject to
*  8000.0 * VAR1 + 4000.0 * VAR2 <= 40000.0
*  15.0 * VAR1 + 30.0 * VAR2 <= 200.0
*  10 >= VAR1, VAR2 >= 0 and integer
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
    VAR2      COST      150.0
    VAR2      ROW1      4000.0           ROW2      30.0
    MARK001   'MARKER'                 'INTEND'
RHS
    RHS1      ROW1      40000.0        ROW2      200.0
BOUNDS
 UI BOUND      VAR1        10
 LO BOUND      VAR1        0
 UI BOUND      VAR2        10
 LO BOUND      VAR2        0
ENDATA
