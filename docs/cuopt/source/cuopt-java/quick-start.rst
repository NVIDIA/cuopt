Java Quickstart Guide
=====================

NVIDIA cuOpt provides experimental Java bindings for LP, MIP, QP, QCQP, and
SOCP, built from ``java/cuopt``. It is not part of the top-level cuOpt build.

Installation
============

Choose your install method below; the selector is pre-set for Java. Copy the
Docker command and run it in your environment — ``cuopt.jar`` and
``libcuopt_jni.so`` are already at ``/opt/cuopt/java`` inside the container,
so no build step is needed. Use ``-cp /opt/cuopt/java/cuopt.jar`` for both
compilation and execution, and pass ``-Dcuopt.native.dir=/opt/cuopt/java``
only to the ``java`` command. See :doc:`../install` for all interfaces and
options.

.. install-selector::
   :default-iface: java

Using the Maven Artifact
-------------------------

``com.nvidia.cuopt:cuopt`` publishes classifier jars (``cuda12``,
``cuda12-arm64``, ``cuda13``, ``cuda13-arm64``) to the Sonatype snapshot and
release repositories. Each classifier jar embeds ``libcuopt_jni.so`` and
cuOpt's own native dependencies (``libcuopt``, rmm, cuDSS, NCCL, TBB), which
``NativeLibraryLoader`` extracts to a temp directory and loads automatically —
no ``cuopt.native.dir`` is required:

.. code-block:: xml

   <repositories>
     <repository>
       <id>sonatype-snapshots</id>
       <url>https://central.sonatype.com/repository/maven-snapshots</url>
       <releases><enabled>false</enabled></releases>
       <snapshots><enabled>true</enabled></snapshots>
     </repository>
   </repositories>

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
cuOpt itself; outside Docker, install the matching
``cuda-libraries-<major>-<minor>`` package (e.g. ``cuda-libraries-12-9``) from
`NVIDIA's CUDA repository <https://developer.nvidia.com/cuda-downloads>`_ via
``apt-get`` or ``dnf`` instead of the full CUDA toolkit.

Building from source is covered in ``java/cuopt/README.md``.

Smoke Test
----------

After installation, verify cuOpt Java is working by compiling and running a
minimal LP inside the container:

.. code-block:: bash

   cat > SmokeTest.java <<'EOF'
   import com.nvidia.cuopt.mathematicaloptimization.*;

   public class SmokeTest {
     public static void main(String[] args) throws Exception {
       try (Problem problem = new Problem("smoke-test")) {
         Variable x = problem.addVariable(0, Double.POSITIVE_INFINITY, 0,
             VariableType.CONTINUOUS, "x");
         Variable y = problem.addVariable(0, Double.POSITIVE_INFINITY, 0,
             VariableType.CONTINUOUS, "y");
         problem.addConstraint(LinearExpression.of(x).plus(y).ge(1.0), "c0");
         problem.setObjective(LinearExpression.of(x).plus(y), ObjectiveSense.MINIMIZE);
         try (Solution solution = problem.solve()) {
           System.out.println(solution.getTerminationStatus());
           System.out.println(solution.getPrimalObjective());
         }
       }
     }
   }
   EOF
   javac -cp /opt/cuopt/java/cuopt.jar -d . SmokeTest.java
   java -Dcuopt.native.dir=/opt/cuopt/java -cp /opt/cuopt/java/cuopt.jar:. SmokeTest

Example Response:

.. code-block:: text

   OPTIMAL
   1.0

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
