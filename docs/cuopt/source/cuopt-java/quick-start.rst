Java Quick Start
================

The experimental Java bindings live in ``java/cuopt``. There are three ways to
get them, depending on your setup:

* the official cuOpt Docker images already contain a prebuilt ``cuopt.jar``
  and ``libcuopt_jni.so`` — see `Using the Docker Image`_;
* the ``com.nvidia.cuopt:cuopt`` Maven artifact is a self-contained classifier
  jar that embeds the native library — see `Using the Maven Artifact`_; or
* building from source, which this section covers first and which repository
  CI and release workflows use to produce both of the above.

It is not part of the top-level cuOpt build.

Requirements
------------

The Java module requires:

* Java 17 or newer, with ``JAVA_HOME`` pointing to a JDK;
* a C++20 compiler;
* an existing cuOpt installation containing ``libcuopt.so``; and
* a CUDA-enabled runtime for solving problems.

The module uses Maven for Java compilation and a Java-local CMake project for
the JNI library. The standalone native build links to
``$CUOPT_PREFIX/lib/libcuopt.so`` and places ``libcuopt_jni.so`` under
``java/cuopt/build/native``.

.. code-block:: bash

   cd /path/to/cuopt/java/cuopt
   export JAVA_HOME=/path/to/jdk-17
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
   export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
   export CUOPT_PREFIX=/path/to/cuopt/conda/environment
   export LD_LIBRARY_PATH=$CUOPT_PREFIX/targets/x86_64-linux/lib:$CUOPT_PREFIX/lib:build/native
   mvn test -Dcuopt.native.dir=build/native

The helper script combines the native build and Maven test steps:

.. code-block:: bash

   cd /path/to/cuopt/java/cuopt
   export JAVA_HOME=/path/to/jdk-17
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

Using the Docker Image
----------------------

The official cuOpt Docker images ship ``cuopt.jar`` and ``libcuopt_jni.so``
under ``/opt/cuopt/java``, built against the image's own ``libcuopt.so``. No
build step is needed; point ``cuopt.native.dir`` at that directory:

.. code-block:: bash

   docker run --rm --gpus all -v $(pwd):/work -w /work <cuopt-image> bash -c '
     javac -cp /opt/cuopt/java/cuopt.jar -d . MyProgram.java
     java -Dcuopt.native.dir=/opt/cuopt/java -cp /opt/cuopt/java/cuopt.jar:. MyProgram
   '

Using the Maven Artifact
------------------------

``com.nvidia.cuopt:cuopt`` publishes classifier jars (``cuda12``,
``cuda12-arm64``, ``cuda13``, ``cuda13-arm64``) to the Sonatype snapshot and
release repositories. Each classifier jar embeds ``libcuopt_jni.so`` and
cuOpt's own native dependencies (``libcuopt``, rmm, cuDSS, NCCL, TBB), which
``NativeLibraryLoader`` extracts to a temp directory and loads automatically —
no ``cuopt.native.dir`` is required:

.. code-block:: xml

   <dependency>
     <groupId>com.nvidia.cuopt</groupId>
     <artifactId>cuopt</artifactId>
     <version>26.10.0-SNAPSHOT</version>
     <classifier>cuda12</classifier>
   </dependency>

The embedded libraries do not include the CUDA toolkit's own math libraries
(``libcublas``, ``libcusolver``, etc.) — those must already be present on the
target system. Loading the jar on a system without them fails with an
``UnsatisfiedLinkError`` naming the missing CUDA library. An
``nvidia/cuda:*-runtime-*`` base image satisfies this without installing
cuOpt itself:

.. code-block:: bash

   docker run --rm --gpus all -v $(pwd):/work -w /work \
     nvidia/cuda:12.9.0-runtime-ubuntu24.04 bash -c '
       apt-get update -qq && apt-get install -y -qq openjdk-17-jdk-headless maven
       mvn -q dependency:copy-dependencies -DoutputDirectory=lib
       javac -cp "lib/cuopt-*-cuda12.jar" -d . MyProgram.java
       java -cp "lib/cuopt-*-cuda12.jar:." MyProgram
     '

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
