Command Line Interface
======================

The cuopt_cli is a command-line interface for LP/MILP solvers that accepts MPS, QPS, or LP format files as input models. The format is dispatched automatically from the file extension (case-insensitive): ``.lp`` (with optional ``.gz`` / ``.bz2``) goes to the LP parser, ``.mps`` / ``.qps`` (with optional ``.gz`` / ``.bz2``) goes to the MPS parser, and unknown extensions are rejected. It provides command-line arguments to control all solver settings and parameters when solving linear and mixed-integer programming problems.

.. toctree::
   :maxdepth: 3
   :caption: Command Line Interface Overview
   :name: Command Line Interface Overview
   :titlesonly:

   quick-start.rst

.. toctree::
   :maxdepth: 3
   :caption: Usage
   :name: Usage
   :titlesonly:

   cli-examples.rst
