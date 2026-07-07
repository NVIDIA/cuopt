Java Quick Start
================

The Java bindings live in ``java/cuopt`` and are built explicitly as a
standalone beta module. They are not part of the repository-level cuOpt build
and do not modify the main C/C++ or Python bindings.

Requirements
------------

The Java module requires:

* Java 11 or newer, with ``JAVA_HOME`` pointing to a JDK;
* a C++20 compiler;
* an existing cuOpt installation containing ``libcuopt.so``; and
* a CUDA-enabled runtime for solving models.

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

   bash scripts/test.sh -Dtest=PythonParityTest

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

The modeling API mirrors the Python concepts while using Java builder methods
instead of operator overloading.

.. code-block:: java

   import com.nvidia.cuopt.linearprogramming.*;

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
     System.out.println(solution.getLpStats().getSolvedBy());
   }

MILP Example
------------

.. code-block:: java

   Problem problem = new Problem("integer");
   Variable x = problem.addVariable(0, 10, 1.0, VariableType.INTEGER, "x");
   problem.addConstraint(LinearExpression.of(x).ge(1.0));

   try (SolverSettings settings = new SolverSettings()
            .setParameter(CuOptConstants.CUOPT_TIME_LIMIT, 10.0);
        Solution solution = problem.solve(settings)) {
     System.out.println(solution.getMipGap());
     System.out.println(solution.getMipStats().getNumNodes());
   }

QP Example
----------

``DataModel`` exposes the lower-level CSR API and quadratic hooks from the C API.

.. code-block:: java

   CsrMatrix matrix = new CsrMatrix(
       new int[] {0, 2},
       new int[] {0, 1},
       new double[] {1.0, 1.0});

   try (DataModel model = DataModel.createProblem(
          1, 2, ObjectiveSense.MINIMIZE, 0.0,
          new double[] {-8.0, -16.0},
          matrix,
          new byte[] {(byte) 'G'},
          new double[] {5.0},
          new double[] {0.0, 0.0},
          new double[] {10.0, 10.0},
          new byte[] {(byte) 'C', (byte) 'C'})) {
     Problem shell = new Problem();
     Variable x0 = shell.addVariable();
     Variable x1 = shell.addVariable();
     model.setQuadraticObjective(
         QuadraticExpression.of(x0, x0, 1.0).plus(x1, x1, 4.0));
   }

MPS I/O
-------

.. code-block:: java

   try (DataModel model = DataModel.read("model.mps")) {
     model.writeMPS("roundtrip.mps");
   }

Lifecycle
---------

``DataModel``, ``SolverSettings``, and ``Solution`` own native handles and
implement ``AutoCloseable``. Prefer try-with-resources. They also register a
``Cleaner`` fallback, but deterministic close keeps native memory pressure
predictable.

The Java module is not a drop-in translation of Python syntax. Java uses
fluent expression methods such as ``plus``, ``minus``, ``le``, ``ge``, and
``eq`` instead of Python operator overloads. The following pages document the
implemented LP/MILP/QP surface and its Java names.
