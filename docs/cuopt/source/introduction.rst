==========================
Introduction
==========================

**NVIDIA® cuOpt™** is a GPU-accelerated optimization library that solves `Mixed Integer Linear Programming (MILP) <https://en.wikipedia.org/wiki/Linear_programming#Integer_unknowns>`_, `Linear Programming (LP) <https://en.wikipedia.org/wiki/Linear_programming>`_, and `Vehicle Routing Problems (VRP) <https://en.wikipedia.org/wiki/Vehicle_routing_problem>`_. It enables solutions for large-scale problems with millions of variables and constraints, offering seamless deployment across hybrid and multi-cloud environments.

Using accelerated computing, NVIDIA® cuOpt optimizes operations research and logistics by enabling better, faster decisions.

As part of `NVIDIA AI Enterprise <https://www.nvidia.com/en-us/data-center/products/ai-enterprise/>`_, NVIDIA cuOpt offers a secure, efficient way to rapidly generate world-class route optimization solutions. Using a single optimized container, you can deploy the AI microservice in under 5 minutes on accelerated NVIDIA GPU systems in the cloud, data center, workstations, or PCs. A license for NVIDIA AI Enterprise or membership in the NVIDIA Developer Program is required. For more information about NVAIE licensing, accessing NGC registry, and pulling container images, please refer to the `FAQ section <faq.html>`_.

.. note::
   NVAIE support is extended to only cuOpt Routing service API. LP and MILP are not supported as part of it, they are just add-ons.

.. note::
   Check out this `FAQ <https://forums.developer.nvidia.com/t/nvidia-nim-faq/300317>`__ for more information about the NVIDIA Developer Program. 


Routing (TSP, VRP, and PDP)
=============================

The **Vehicle Routing Problem (VRP)** and **Pickup and Delivery Problems (PDP)** are derived from the **Traveling Salesperson Problem (TSP)**, which is one of the most studied problems in operations research and, more generally, in computer science. 

TSP asks the following question: 

  -  Given a list of destinations and a matrix of distances between each pair of destinations, what is the shortest possible route that visits each destination exactly one time and returns to the original location? 

For example, the TSP has several applications in planning and logistics, where a good solution can save significant travel time and fuel costs in the transportation and delivery of goods. VRP and PDP are essentially extensions of TSP with additional complexity.

The VRP generalizes the TSP to solve for the optimal set of routes for a fleet of vehicles in order to deliver to a given set of customers. The PDP adds the possibility of two different types of services, namely pickup or delivery, whereas in VRP all customers require the same service be performed at a customer location.


How cuOpt Solves the Routing Problem
-------------------------------------

cuOpt first generates an initial population of solutions, then iteratively improves the population until the time limit is reached, and picks the best solution from the population.


The Necessity for Heuristics
------------------------------

Given the time and computational resources required for brute-force enumeration, obtaining the exact optimal solution is not realistic. However, there are well-studied heuristics that yield near-optimal solutions for very large problems within a reasonable time, and NVIDIA cuOpt focuses on using these heuristics.



Linear Programming (LP)
=======================

**Linear Programming** is a technique for optimizing a linear objective function over a feasible region defined by a set of linear inequality and equality constraints. For example, consider the following system constraints

                          2x + 4y  >= 230

                          3x + 2y  <= 190

                          x >= 0

                          y >= 0,

and suppose we want to maximize the objective function

                          f(x,y) = 5x + 3y.

This is a linear program.


How cuOpt Solves the Linear Programming Problem
------------------------------------------------
cuOpt includes an LP solver based on `PDLP <https://arxiv.org/abs/2106.04756>`__, a new First-Order Method (FOM) used to solve large-scale LPs. This solver implements gradient descent, enhanced by heuristics, and performing massively parallel operations efficiently by leveraging the latest NVIDIA GPUs. 

In addition to PDLP, cuOpt includes a dual simplex solver that runs on the CPU. Both algorithms can be run concurrently on the GPU and CPU.

Mixed Integer Linear Programming (MILP)
=========================================

A **Mixed Integer Linear Program** is a variant of a Linear Program where some of the variables are restricted to take on only integer values, while other variables can vary continuously. NVIDIA cuOpt uses a hybrid GPU/CPU method: running primal heuristics on the GPU and improving the dual bound on the CPU.

For example, consider the following system of constraints:

                          2x + 4y  >= 230

                          3x + 2y  <= 190

                          x >= 0 and x is integer

                          y >= 0 and y is continuous,

and suppose we wish to maximize the objective function 

                          f(x,y) = 5x + 3y.

This is a mixed integer linear program.

Although MILPs seems similar to a LPs, they require much more computation to solve.

How cuOpt Solves the Mixed-Integer Linear Programming Problem
-------------------------------------------------------------

The MILP solver is a hybrid GPU/CPU algorithm. Primal heuristics including local search, feasibility pump, and feasibility jump are performed on the GPU to improve the primal bound. Branch and bound is performed on the CPU to improve the dual bound. Integer feasible solutions are shared between both algorithms. 


=============================
Supported APIs
=============================

cuOpt supports the following APIs:

- C API support
   - Linear Programming (LP)
   - Mixed Integer Linear Programming (MILP)
- C++ API support
   - cuOpt is written in C++ and includes a native C++ API. However, we do not provide documentation for the C++ API at this time. We anticipate that the C++ API will change significantly in the future. Use it at your own risk.
- Python support
   - Routing (TSP, VRP, and PDP)
   - Linear Programming (LP) and Mixed Integer Linear Programming (MILP)
       - cuOpt includes a Python API that is used as the backend of the cuOpt server. However, we do not provide documentation for the Python API at this time. We suggest using cuOpt server to access cuOpt via Python. We anticipate that the Python API will change significantly in the future. Use it at your own risk.
- Server support
   - Linear Programming (LP)
   - Mixed Integer Linear Programming (MILP)
   - Routing (TSP, VRP, and PDP)
