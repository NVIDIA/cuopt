===============================
Third-Party Modeling Languages
===============================


--------------------------
AMPL Support
--------------------------

AMPL can be used with near zero code changes: simply switch to cuOpt as a solver to solve linear and mixed-integer programming problems. Please refer to the `AMPL documentation <https://www.ampl.com/>`_ for more information. Also, see the example notebook in the `colab <https://colab.research.google.com/drive/1eEQik_pae4g_tJQ61QJFlO1fFBXazpBr?usp=sharing>`_.

--------------------------
GAMS and GAMSPy Support
--------------------------

GAMS and GAMSPy models can be used with near zero code changes after setting up the solver link: simply switch to cuOpt as a solver to solve linear and mixed-integer programming problems (e.g. ``gams trnsport lp=cuopt``). Please refer to the `GAMS cuOpt link repository <https://github.com/GAMS-dev/cuoptlink-builder>`_ for more information on how to setup GAMS and GAMSPy for cuOpt. Also, see the example notebook in the `cuopt-examples <https://github.com/NVIDIA/cuopt-examples>`_ repository.

--------------------------
PuLP Support
--------------------------

PuLP can be used with near zero code changes: simply switch to cuOpt as a solver to solve linear and mixed-integer programming problems.
Please refer to the `PuLP documentation <https://pypi.org/project/PuLP/>`_ for more information. Also, see the example notebook in the `cuopt-examples <https://github.com/NVIDIA/cuopt-examples>`_ repository.

--------------------------
JuMP Support
--------------------------

JuMP can be used with near zero code changes: simply switch to cuOpt as a solver to solve linear and mixed-integer programming problems.
Please refer to the `JuMP documentation <https://github.com/jump-dev/cuOpt.jl>`_ for more information.
