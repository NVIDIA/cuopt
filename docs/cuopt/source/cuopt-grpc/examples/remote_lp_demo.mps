NAME          good-1
OBJSENSE
 MAXIMIZE
ROWS
 N  COST
 L  ROW1
 L  ROW2
COLUMNS
    VAR1      COST                 0.2
    VAR1      ROW1                 3.0   ROW2                 2.7
    VAR2      COST                 0.1
    VAR2      ROW1                 4.0   ROW2                10.1
RHS
    RHS1      ROW1                 5.4   ROW2                 4.9
BOUNDS
 LO BND1      VAR1                 0.0
 UP BND1      VAR1                 2.0
 LO BND1      VAR2                 0.0
ENDATA
