Java Quick Start
================

The experimental Java bindings live in ``java/cuopt`` and are built explicitly
from source. Repository CI and release workflows also build and test the module
against the matching ``libcuopt`` artifact. It is not part of the top-level
cuOpt build, and a supported Maven distribution has not yet been defined.

Requirements
------------

The Java module requires:

* Java 11 or newer, with ``JAVA_HOME`` pointing to a JDK;
* a C++20 compiler;
* an existing cuOpt installation containing ``libcuopt.so``; and
* a CUDA-enabled runtime for solving problems.

The module uses Maven for Java compilation and a Java-local CMake project for
the JNI library. The standalone native build links to
``$CUOPT_PREFIX/lib/libcuopt.so`` and places ``libcuopt_jni.so`` under
``java/cuopt/build/native``.

.. code-block:: bash

   cd /path/to/cuopt/java/cuopt
   export JAVA_HOME=/path/to/jdk-11
   export CUOPT_PREFIX=/path/to/cuopt/conda/environment
   bash scripts/build_native.sh

This builds ``java/cuopt/build/native/libcuopt_jni.so``. Java is intentionally
not part of the default cuOpt build.

To build the native library in a different directory, set
``CUOPT_JAVA_NATIVE_BUILD_DIR``. If CUDA headers are installed outside the
usual locations, pass ``-DCUOPT_CUDA_INCLUDE_DIR=/path/to/cuda/include`` to
the CMake configure step.

Native Loading
--------------

At runtime the bindings load ``libcuopt_jni``. For local development, point Java
at the directory containing the built native library:

.. code-block:: bash

   cd java/cuopt
   export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
   export CUOPT_PREFIX=/path/to/cuopt/conda/environment
   export LD_LIBRARY_PATH=$CUOPT_PREFIX/targets/x86_64-linux/lib:$CUOPT_PREFIX/lib:build/native
   mvn test -Dcuopt.native.dir=build/native

The helper script combines the native build and Maven test steps:

.. code-block:: bash

   cd /path/to/cuopt/java/cuopt
   export JAVA_HOME=/path/to/jdk-11
   export CUOPT_PREFIX=/path/to/cuopt/conda/environment
   bash scripts/test.sh

To run one test class, pass its Maven property to the helper:

.. code-block:: bash

   bash scripts/test.sh -Dtest=ProblemIntegrationTest

Application code can use the same property:

.. code-block:: bash

   java -Dcuopt.native.dir=/path/to/java/cuopt/build/native ...

The Java classes load ``libcuopt_jni`` when the first binding object is
created. ``cuopt.native.dir`` must contain that library, and the cuOpt and
CUDA runtime libraries must be discoverable through ``LD_LIBRARY_PATH`` or the
native library's runtime path. The standalone native build embeds the CUDA
runtime path for the configured ``CUOPT_PREFIX``; the helper script also
exports it for Maven.

LP Example
----------

A ``Problem`` owns the variables and constraints. Expressions are assembled
with methods that return a new expression, and a constraint is formed by
comparing one against a bound with ``le``, ``ge`` or ``eq``.

.. code-block:: java

   import com.nvidia.cuopt.mathematicaloptimization.*;

   Problem problem = new Problem("simple");
   Variable x = problem.addVariable(0, Double.POSITIVE_INFINITY, 0,
       VariableType.CONTINUOUS, "x");
   Variable y = problem.addVariable(0, Double.POSITIVE_INFINITY, 0,
       VariableType.CONTINUOUS, "y");

   problem.addConstraint(LinearExpression.of(x).plus(y).ge(1.0), "c0");
   problem.setObjective(LinearExpression.of(x).plus(y), ObjectiveSense.MINIMIZE);

   try (SolverSettings settings = new SolverSettings().setMethod(SolverMethod.PDLP);
        Solution solution = problem.solve(settings)) {
     System.out.println(solution.getTerminationStatus());
     System.out.println(solution.getPrimalObjective());
   }

MIP Example
-----------

.. code-block:: java

   Problem problem = new Problem("integer");
   Variable x = problem.addVariable(0, 10, 1.0, VariableType.INTEGER, "x");
   problem.addConstraint(LinearExpression.of(x).ge(1.0));

   try (SolverSettings settings = new SolverSettings()
            .setSetting(CuOptConstants.CUOPT_TIME_LIMIT, 10.0);
        Solution solution = problem.solve(settings)) {
     System.out.println(solution.getMIPGap());
     System.out.println(solution.getSolutionBound());
   }

QP Example
----------

.. code-block:: java

   try (Problem problem = new Problem("quadratic")) {
     Variable x = problem.addVariable(0.0, 10.0, 0.0, VariableType.CONTINUOUS, "x");
     Variable y = problem.addVariable(0.0, 10.0, 0.0, VariableType.CONTINUOUS, "y");
     problem.addConstraint(LinearExpression.of(x).plus(y).ge(5.0));
     problem.setObjective(
         QuadraticExpression.of(x, x, 1.0).plus(y, y, 4.0),
         ObjectiveSense.MINIMIZE);
     try (Solution solution = problem.solve()) {
       System.out.println(solution.getPrimalObjective());
     }
   }

MPS I/O
-------

.. code-block:: java

   try (Problem problem = Problem.read("problem.mps")) {
     problem.write("roundtrip.mps");
   }

Lifecycle
---------

``SolverSettings`` and ``Solution`` own native handles and implement
``AutoCloseable``, so close them with try-with-resources. They also register a
``Cleaner`` fallback, but closing them deterministically keeps native memory
pressure predictable.

Expressions are built and compared through methods — ``plus``, ``minus``,
``le``, ``ge`` and ``eq`` — each returning a new object rather than mutating
the receiver. The following pages document the implemented LP/MIP/QP/QCQP/SOCP
surface.
