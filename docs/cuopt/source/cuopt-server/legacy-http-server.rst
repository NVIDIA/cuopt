..
   SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

.. _legacy-http-server:

==================
Legacy HTTP Server
==================

.. note::

   The legacy HTTP server is **deprecated**. It is still shipped in 26.10 and
   will be removed in 26.12. It is being replaced by the
   :doc:`HTTP proxy <index>`, which forwards requests to
   ``cuopt_grpc_server``. With no command, the container starts the HTTP
   proxy and ``cuopt_grpc_server``.

For the legacy server's command line, HTTP API, and general usage, see the
`26.08 server guide
<https://archive.docs.nvidia.com/cuopt/user-guide/26.08.00/cuopt-server/index.html>`_.

.. _legacy-http-container:

Starting the Container
----------------------

The command ``legacy`` starts the legacy HTTP server. It listens on port 5000:

.. code-block:: bash

   docker run --gpus all -it --rm -p 5000:5000 <CONTAINER_IMAGE_PATH> legacy

In Kubernetes, set ``CUOPT_SERVER_TYPE=legacy`` on the pod. The variable overrides the container command. When it is unset, the container command ``legacy`` selects this server.

Legacy behavior the HTTP proxy does not carry forward is listed in :ref:`legacy-http-differences`.
