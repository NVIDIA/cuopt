=====================
Release Notes
=====================

====================
Release Notes 25.05
====================

New Features
------------

- Added crossover from PDLP to Dual Simplex
- Added concurrent mode that runs PDLP and Dual Simplex together
- Added a LP and MILP C API for cuOpt 
- PDLP enhancements 
- Limit minimal absolute tolerance to 1e-12 and NoTermination is now NumericalError 

Breaking Changes
----------------

- FIx concurrent LP crashes and limit minimal absolute tolerance are not breaking changes. 
- Limit minimal absolute tolerance to 1e-12 and NoTermination is now NumericalError 
- Split cuOpt as libcuopt and cuopt wheel 

Improvements
------------

- FIx concurrent LP crashes and limit minimal absolute tolerance are not breaking changes.
- Hook up MILP Gap parameters and add info about number of nodes explored and simplex iterations, other fixes 
- FJ bug fixes, tests and improvements 
- Cleanup solver settings 
- Allow no time limit on MILP 
- Refactor routing  
- Probing cache optimization 
- Split cuOpt as libcuopt and cuopt wheel 
- Diversity improvements for routing
- Enable more compile warnings and faster compile by bypassing rapids fetch 
- Constraint prop based on load balanced bounds update 
- Logger file handling and bug fixes on MILP 
- Add shellcheck to pre-commit and fix warnings 

Bug Fixes
---------

- In the solution, ``termination_status`` should be cast to correct enum.
- Fixed a bug using vehicle IDs in construct feasible solution algorithm.
- FP recombiner probing bug fix.
- Fix print relative dual residual. 
- Handle empty problems gracefully.
- Improve breaks to allow dimensions at arbitrary places in the route.
- Free var elimination with a substitute variable for each free variable.
- Fixed race condition when resetting vehicle IDs in heterogenous mode.
- cuOpt self-hosted client, some MILPs will not have all fields in ``lp_stats``.
- Fixed RAPIDS logger usage.
- Handle LP state more cleanly, per solution.
- Fixed routing solver intermittent failures.
- Gracefully exit when the problem is infeasible after presolve.
- Fixed bug on dual resizing.


Documentation
-------------
- Restructure documementation to accomdate new APIs