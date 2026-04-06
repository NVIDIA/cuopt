..
   SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

=======================
Advanced configuration
=======================

This page lists **configuration parameters** first, then **usage** walkthroughs (TLS, Docker, private CA). Complete :doc:`quick-start` first (install, plain TCP server, and minimal example).

For RPC summaries and server behavior, see :doc:`api` and :doc:`grpc-server-architecture`. Example entry points with ``CUOPT_REMOTE_*``: :doc:`examples`. Contributor-only internals: ``cpp/docs/grpc-server-architecture.md`` in the repository.

Configuration parameters
========================

``cuopt_grpc_server`` (host or explicit container command)
------------------------------------------------------------

Run ``cuopt_grpc_server --help`` for the full list. Typical flags (also passable inside ``CUOPT_GRPC_ARGS`` when using the container entrypoint):

.. code-block:: text

   cuopt_grpc_server [options]

     -p, --port PORT              gRPC listen port (default: 5001)
     -w, --workers NUM            Number of worker processes (default: 1)
         --max-message-mb N       Max gRPC message size in MiB (default: 256; clamped to [4 KiB, ~2 GiB])
         --max-message-bytes N    Max gRPC message size in bytes (exact; min 4096)
         --chunk-timeout N        Per-chunk timeout in seconds for streaming (0=disabled, default: 60)
         --log-to-console         Echo solver logs to server console
     -v, --verbose                Increase verbosity (default: on)
     -q, --quiet                  Reduce verbosity (verbose is the default)
         --server-log PATH        Path to server operational log file (in addition to console)

   TLS Options:
         --tls                    Enable TLS encryption
         --tls-cert PATH          Server certificate (PEM)
         --tls-key PATH           Server private key (PEM)
         --tls-root PATH          Root CA certificate (for client verification)
         --require-client-cert    Require client certificate (mTLS)

NVIDIA cuOpt container (gRPC via entrypoint)
--------------------------------------------

These variables apply when the container **entrypoint** builds a ``cuopt_grpc_server`` command (see *Docker: gRPC server in container* under Usage). If you pass an explicit command after the image name, this table does not apply.

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Variable
     - Default
     - Description
   * - ``CUOPT_SERVER_TYPE``
     - *(unset)*
     - Set to ``grpc`` for entrypoint-built gRPC. Unset with no explicit command: **Python REST** server.
   * - ``CUOPT_SERVER_PORT``
     - ``5001``
     - Passed as ``--port`` to ``cuopt_grpc_server``.
   * - ``CUOPT_GPU_COUNT``
     - *(unset)*
     - When set, passed as ``--workers``. When unset, ``--workers`` is omitted (server default, typically 1).
   * - ``CUOPT_GRPC_ARGS``
     - *(empty)*
     - Extra flags split on **whitespace** and appended (TLS, ``--max-message-mb``, ``--log-to-console``, etc.). Paths with spaces: prefer mounts without spaces or run ``cuopt_grpc_server`` manually with proper quoting.

The REST server path in the same image still uses ``CUOPT_SERVER_PORT`` for HTTP in other docs; that is separate from the gRPC defaults above.

Bundled remote client (Python, C API, ``cuopt_cli``)
----------------------------------------------------

Remote mode is active when **both** ``CUOPT_REMOTE_HOST`` and ``CUOPT_REMOTE_PORT`` are set. A **custom** gRPC client does not read these automatically; it must configure the channel and protos itself (see :doc:`api`).

.. list-table::
   :header-rows: 1
   :widths: 26 14 18 42

   * - Variable
     - Required
     - Default
     - Description
   * - ``CUOPT_REMOTE_HOST``
     - For remote
     - —
     - Server hostname or IP
   * - ``CUOPT_REMOTE_PORT``
     - For remote
     - —
     - Server port (e.g. ``5001``)
   * - ``CUOPT_TLS_ENABLED``
     - No
     - ``0``
     - Non-zero enables TLS on the client
   * - ``CUOPT_TLS_ROOT_CERT``
     - If TLS
     - —
     - PEM path to verify the **server** certificate
   * - ``CUOPT_TLS_CLIENT_CERT``
     - mTLS
     - —
     - Client certificate PEM
   * - ``CUOPT_TLS_CLIENT_KEY``
     - mTLS
     - —
     - Client private key PEM
   * - ``CUOPT_CHUNK_SIZE``
     - No
     - 16 MiB (lib)
     - Chunk size in **bytes** for large transfers (clamped in library code)
   * - ``CUOPT_MAX_MESSAGE_BYTES``
     - No
     - 256 MiB (lib)
     - Client gRPC max message size in **bytes** (clamped in library code)
   * - ``CUOPT_GRPC_DEBUG``
     - No
     - ``0``
     - Non-zero: extra gRPC client logging

Usage
=====

Start the server with TLS
--------------------------

Basic (no TLS), plain TCP, is in :doc:`quick-start`. Encrypted server:

.. code-block:: bash

   cuopt_grpc_server --port 5001 \
     --tls \
     --tls-cert server.crt \
     --tls-key server.key

mTLS (mutual TLS):

.. code-block:: bash

   cuopt_grpc_server --port 5001 \
     --tls \
     --tls-cert server.crt \
     --tls-key server.key \
     --tls-root ca.crt \
     --require-client-cert

How mTLS works
--------------

With mTLS the server verifies every client, and the client verifies the server. Trust is based on **Certificate Authorities** (CAs), not individual certificate lists:

* ``--tls-root ca.crt`` tells the server which CA to trust; any client cert signed by that CA is accepted. The server does not store per-client certificates.
* ``--require-client-cert`` makes client verification **mandatory**. Without it, the server may still allow connections without a client cert.
* On the client, ``CUOPT_TLS_ROOT_CERT`` is the CA that signed the **server** certificate so the client can verify the server.

Restricting access with a private CA
------------------------------------

To limit which clients can connect, run your own CA and issue client certs only to authorized actors.

**1. Create a private CA (one-time):**

.. code-block:: bash

   openssl genrsa -out ca.key 4096
   openssl req -new -x509 -key ca.key -sha256 -days 3650 \
     -subj "/CN=cuopt-internal-ca" -out ca.crt

**2. Issue a client certificate:**

.. code-block:: bash

   openssl genrsa -out client.key 2048
   openssl req -new -key client.key \
     -subj "/CN=team-member-alice" -out client.csr
   openssl x509 -req -in client.csr -CA ca.crt -CAkey ca.key \
     -CAcreateserial -days 365 -sha256 -out client.crt

Repeat for each authorized client. Keep ``ca.key`` private; distribute ``ca.crt`` to the server and per-client ``client.crt`` + ``client.key`` pairs.

**3. Issue a server certificate (same CA):**

.. code-block:: bash

   openssl genrsa -out server.key 2048
   openssl req -new -key server.key \
     -subj "/CN=server.example.com" -out server.csr

   cat > server.ext <<EOF
   subjectAltName=DNS:server.example.com,DNS:localhost,IP:127.0.0.1
   EOF

   openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key \
     -CAcreateserial -days 365 -sha256 -extfile server.ext -out server.crt

``server.crt`` must be signed by the CA you give to clients, and **subjectAltName** must match the hostname or IP clients use. gRPC hostname verification expects SAN; **CN alone is not sufficient**.

**4. Start the server:**

.. code-block:: bash

   cuopt_grpc_server --port 5001 \
     --tls \
     --tls-cert server.crt \
     --tls-key server.key \
     --tls-root ca.crt \
     --require-client-cert

**5. Configure an authorized client:**

.. code-block:: bash

   export CUOPT_REMOTE_HOST=server.example.com
   export CUOPT_REMOTE_PORT=5001
   export CUOPT_TLS_ENABLED=1
   export CUOPT_TLS_ROOT_CERT=ca.crt
   export CUOPT_TLS_CLIENT_CERT=client.crt
   export CUOPT_TLS_CLIENT_KEY=client.key

**Revocation:** built-in gRPC TLS does **not** implement CRL or OCSP. To revoke a client, rotate the CA, stop issuing from a compromised CA, or terminate TLS at a reverse proxy (e.g., Envoy) that supports revocation.

Docker: gRPC server in container
---------------------------------

The official NVIDIA cuOpt image includes the REST server and ``cuopt_grpc_server``. The entrypoint behaves as follows:

1. **Explicit command** after the image name (e.g. ``cuopt_grpc_server …``) runs as-is; env-based gRPC wiring is skipped.
2. **`CUOPT_SERVER_TYPE=grpc`** builds a ``cuopt_grpc_server`` command from the **NVIDIA cuOpt container** table in *Configuration parameters*.
3. **Default** — if ``CUOPT_SERVER_TYPE`` is unset and there is no explicit command, the Python **REST** server starts.

.. note::

   Examples use ``--gpus all``. That requires NVIDIA GPUs on the host and Docker with the `NVIDIA Container Toolkit <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>`_ so devices are visible inside the container.

Typical run:

.. code-block:: bash

   docker run --gpus all -p 5001:5001 \
     -e CUOPT_SERVER_TYPE=grpc \
     nvcr.io/nvidia/cuopt/cuopt:latest

TLS example with a cert volume:

.. code-block:: bash

   docker run --gpus all -p 5001:5001 \
     -e CUOPT_SERVER_TYPE=grpc \
     -e CUOPT_GRPC_ARGS="--tls --tls-cert /certs/server.crt --tls-key /certs/server.key --log-to-console" \
     -v ./certs:/certs:ro \
     nvcr.io/nvidia/cuopt/cuopt:latest

Bypass the entrypoint:

.. code-block:: bash

   docker run --gpus all -p 5001:5001 \
     nvcr.io/nvidia/cuopt/cuopt:latest \
     cuopt_grpc_server --port 5001 --workers 2

Client environment (examples)
------------------------------

**Required** for remote (see *Bundled remote client* table for all variables):

.. code-block:: bash

   export CUOPT_REMOTE_HOST=<server-hostname>
   export CUOPT_REMOTE_PORT=5001

**TLS** (optional):

.. code-block:: bash

   export CUOPT_TLS_ENABLED=1
   export CUOPT_TLS_ROOT_CERT=ca.crt

For mTLS, also:

.. code-block:: bash

   export CUOPT_TLS_CLIENT_CERT=client.crt
   export CUOPT_TLS_CLIENT_KEY=client.key

Limitations and scope
=====================

* **Problem types** — **LP**, **MILP**, and **QP** are supported on the gRPC remote path. **Routing** (VRP, TSP, PDP) is **not** supported yet; use the :doc:`REST self-hosted server <../cuopt-server/index>` for remote routing until a future release adds routing over ``CuOptRemoteService``.
* **Message size** — Large problems use chunking; very large models can still hit gRPC max message / timeout limits. Tune ``CUOPT_CHUNK_SIZE``, ``CUOPT_MAX_MESSAGE_BYTES``, server ``--max-message-mb``, and solver ``time_limit`` as needed.
* **``CUOPT_GRPC_ARGS``** — Parsed on whitespace only; arguments containing spaces are awkward unless you invoke ``cuopt_grpc_server`` directly.
* **CRL / OCSP** — Not handled by the bundled gRPC TLS stack; use a private CA rotation strategy or a TLS-terminating proxy if you need revocation workflows.

Troubleshooting
===============

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Symptom
     - Check
   * - Connection refused
     - Server running; host/port match; firewalls and Docker port mapping.
   * - TLS handshake failure
     - ``CUOPT_TLS_ENABLED=1``; correct CA and cert paths; SAN matches server name.
   * - Cannot open TLS file
     - Path exists and is readable inside the client/server environment (including container mounts).
   * - Timeout on large problems
     - Increase solver ``time_limit`` and client/server message limits.

Further reading
===============

* :doc:`quick-start` — Plain TCP quick path.
* :doc:`examples` — Links to Python, C, and CLI example sections (use with ``CUOPT_REMOTE_*`` on the client).
* :doc:`grpc-server-architecture` — Process model and job behavior (operator overview).
