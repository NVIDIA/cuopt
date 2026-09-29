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
minimal LP inside the container.

:download:`SmokeTest.java <examples/SmokeTest.java>`

.. literalinclude:: examples/SmokeTest.java
   :language: java
   :linenos:

.. code-block:: bash

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

:download:`LpExample.java <examples/LpExample.java>`

.. literalinclude:: examples/LpExample.java
   :language: java
   :linenos:

MIP Example
-----------

:download:`MipExample.java <examples/MipExample.java>`

.. literalinclude:: examples/MipExample.java
   :language: java
   :linenos:

QP Example
----------

:download:`QpQuickstart.java <examples/QpQuickstart.java>`

.. literalinclude:: examples/QpQuickstart.java
   :language: java
   :linenos:

MPS I/O
-------

:download:`MpsRoundtrip.java <convex/examples/MpsRoundtrip.java>` and
:download:`sample.mps <convex/examples/sample.mps>`

.. literalinclude:: convex/examples/MpsRoundtrip.java
   :language: java
   :linenos:

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
