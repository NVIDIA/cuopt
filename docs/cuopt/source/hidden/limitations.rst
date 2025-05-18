======================================
NVIDIA cuOpt Limitations
======================================

Routing
=======

#. Number of tasks:
    - For ``H100`` : ``Tasks <= 15000`` per solve request. 
    - For ``A100`` : ``Tasks <= 10000`` per solve request.

.. note::
   There may be cases where cuOpt might fail due to out of memory error in case the input problem has numerous constraints.

#. Number of vehicles:
    - ``Vehicles <= 3000`` per solve request is suggested.

#. Capacity and demand constraints:
    - Only ``3`` dimensions are supported.

#. Time windows:
    - All time windows are expected to be => 0, and all time windows are int32 types. So the time window range supported would be ``[0, 2^31-1]``.
    - This translates to ``[January 1, 1970, 00:00:00, January 19, 2038, 03:14:07]`` in UTC.

#. Type of vehicles, which also dictates number of cost and time matrix for each vehicle:
    - ``Types <= 10`` is suggested.

Linear Programming
==================

-  For ``H100``,
   - 10M rows/constraints 10M columns/variables and 2B non-zeros in the constraint matrix
   - 74.5M rows/constraints 74.5M columns/variables and 1.49B non-zeros in the constraint matrix 

Mixed Integer Linear Programming
================================

-  Number of non-zeros/coefficient matrix size supported
    -  For ``H100`` - 27 million