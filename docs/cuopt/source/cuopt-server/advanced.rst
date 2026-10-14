..
   SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

=======================
Advanced Configuration
=======================

Some **HTTP proxy server** settings can be configured with environment
variables. A command-line option overrides the corresponding environment
variable. See the :doc:`quick start <quick-start>` for start commands.

HTTP Proxy Server Variables
---------------------------

These variables are read by
``python -m cuopt_server.cuopt_proxy``:

.. list-table::
   :header-rows: 1
   :widths: 24 18 58

   * - Variable
     - Default
     - Description
   * - ``CUOPT_SERVER_IP``
     - ``0.0.0.0``
     - Bind address. Override with ``--ip``.
   * - ``CUOPT_SERVER_PORT``
     - ``5000``
     - HTTP listen port. Override with ``--port``.
   * - ``CUOPT_GRPC_HOST``
     - ``127.0.0.1``
     - Hostname or IP of ``cuopt_grpc_server``. Override with ``--grpc-host``.
   * - ``CUOPT_GRPC_PORT``
     - ``5001``
     - Port of ``cuopt_grpc_server``. Override with ``--grpc-port``.

``python -m cuopt_server.cuopt_proxy --help`` lists the remaining HTTP proxy
server options and their environment variables, including logging, TLS,
request and result limits, file modes, and data directories.

Additional Container Entrypoint Variables
-----------------------------------------

The container's default command starts the HTTP proxy server and
``cuopt_grpc_server``. The HTTP proxy server listens on port 5000 and forwards
each request to gRPC on ``127.0.0.1`` port 5001.

The container entrypoint preserves all the HTTP proxy server configuration
variables above. It also reads these variables to select the server mode and
configure the ``cuopt_grpc_server`` process that it starts:

In Kubernetes, set ``CUOPT_SERVER_TYPE=proxy`` on the pod. Leave the variable
unset for that same default. When the variable is unset, the container command
``proxy`` selects this mode. The variable overrides the container command.

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Variable
     - Default
     - Description
   * - ``CUOPT_SERVER_TYPE``
     - *(unset)*
     - ``proxy``, or unset: the HTTP proxy server and ``cuopt_grpc_server``. The variable overrides the container command.
   * - ``CUOPT_GPU_COUNT``
     - *(unset)*
     - Worker processes for the gRPC server the HTTP proxy server forwards to. One worker per GPU. When unset, ``--workers`` is omitted (server default, 1).
   * - ``CUOPT_GRPC_ARGS``
     - *(empty)*
     - Extra flags for that gRPC server, split on whitespace (TLS, ``--max-message-mb``, ``--log-to-console``). Paths with spaces: run ``cuopt_grpc_server`` directly.

(``CUOPT_GRPC_PORT`` is shared by the HTTP proxy server and the container
entrypoint. In ``proxy`` mode, the entrypoint starts ``cuopt_grpc_server`` on
that port and the HTTP proxy server dials it. Default is still 5001)

Publish port 5000. ``cuopt_grpc_server`` listens on port 5001 inside the container so the HTTP proxy server can reach it. Docker does not publish that port unless you add ``-p 5001:5001``. In Kubernetes, expose port 5000 through the Service. A NetworkPolicy that permits ingress only to port 5000 keeps other pods off the pod's gRPC port.

.. note::

   Examples use ``--gpus all``. That requires NVIDIA GPUs on the host and Docker with the `NVIDIA Container Toolkit <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>`_ so devices are visible inside the container.

.. code-block:: bash

   docker run --gpus all -p 5000:5000 \
     nvcr.io/nvidia/cuopt/cuopt:latest-cu12

``CUOPT_SERVER_TYPE=legacy`` starts the legacy HTTP server. See the :doc:`Legacy HTTP Server <legacy-http-server>` page.
