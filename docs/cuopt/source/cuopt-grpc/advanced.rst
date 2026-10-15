..
   SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

=======================
Advanced Configuration
=======================

This page lists **configuration parameters** for ``cuopt_grpc_server``, then **usage** walkthroughs (TLS and private CA). Complete :doc:`quick-start` first (install, plain TCP server, and minimal example).

For RPC summaries, see the :doc:`gRPC API reference <api>`. Example entry
points using ``CUOPT_REMOTE_*`` can be found under :doc:`Examples <examples>`.
Developers can read about the server architecture in :doc:`gRPC Server
Behavior <grpc-server-architecture>`. Implementation details are in
``cpp/docs/grpc-server-architecture.md`` in the repository.

Configuration Parameters
========================

Command-Line Flags
------------------

Run ``cuopt_grpc_server --help`` for the full list. Typical flags (also passable inside ``CUOPT_GRPC_ARGS`` when the container starts ``cuopt_grpc_server``):

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

gRPC Container Variables
------------------------

In Kubernetes, set ``CUOPT_SERVER_TYPE=grpc`` on the pod. When the variable is unset, the container command ``grpc`` starts ``cuopt_grpc_server`` only. Command-line start is on the :doc:`gRPC quick start <quick-start>`.

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Variable
     - Default
     - Description
   * - ``CUOPT_SERVER_TYPE``
     - *(unset)*
     - ``grpc``: ``cuopt_grpc_server`` only. The variable overrides the container command.
   * - ``CUOPT_GRPC_PORT``
     - ``5001``
     - Passed as ``--port``. When unset, ``CUOPT_SERVER_PORT`` is used, then ``5001``.
   * - ``CUOPT_GPU_COUNT``
     - *(unset)*
     - Passed as ``--workers``. When unset, ``--workers`` is omitted (server default, 1). Set this to the number of visible GPUs, one worker per GPU.
   * - ``CUOPT_GRPC_ARGS``
     - *(empty)*
     - Extra flags split on whitespace (TLS, ``--max-message-mb``, ``--log-to-console``). Paths with spaces: run ``cuopt_grpc_server`` directly.

These examples publish the gRPC port and pass extra server flags. Set ``CUOPT_SERVER_TYPE=grpc``, or pass the command ``grpc`` when that variable is unset. ``--gpus all`` requires NVIDIA GPUs on the host and Docker with the `NVIDIA Container Toolkit <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>`_.

TLS example with a cert volume:

.. code-block:: bash

   docker run --gpus all -p 5001:5001 \
     -e CUOPT_GRPC_ARGS="--tls --tls-cert /certs/server.crt --tls-key /certs/server.key --log-to-console" \
     -v ./certs:/certs:ro \
     nvcr.io/nvidia/cuopt/cuopt:latest-cu12 \
     grpc

Bypass the entrypoint:

.. code-block:: bash

   docker run --gpus all -p 5001:5001 \
     nvcr.io/nvidia/cuopt/cuopt:latest-cu12 \
     cuopt_grpc_server --port 5001 --workers 2

Integrated Remote Client (Python, C API, ``cuopt_cli``)
-------------------------------------------------------

These variables apply to **remote execution**: the client integrated into the
Python solver APIs, the C API (``cuOptSolve``), and ``cuopt_cli``. Remote mode
is active when **both** ``CUOPT_REMOTE_HOST`` and ``CUOPT_REMOTE_PORT`` are set.

The :doc:`gRPC linear programming client <python-async-client>` and the
:doc:`gRPC routing client <routing>` do not read ``CUOPT_REMOTE_HOST`` /
``CUOPT_REMOTE_PORT``; you pass host and port to ``Client(...)`` or
``RoutingClient(...)``. See *gRPC Python clients* below for variables that
apply to those clients. A **custom** gRPC client must configure the channel
itself (see :doc:`api`).

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
     - GPU server hostname or IP
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

gRPC Python Clients
-------------------

``Client(host, port, tls=...)`` (LP/MIP/QP, :doc:`gRPC linear programming
client <python-async-client>`) and ``RoutingClient(host, port, tls=...)``
(VRP, :doc:`gRPC routing client <routing>`) take the server address in code.
Neither reads ``CUOPT_REMOTE_HOST`` or ``CUOPT_REMOTE_PORT``.

When ``tls`` is omitted (``None``), both clients honor the same
``CUOPT_TLS_*`` variables as remote execution. Pass ``tls=False`` for plain
TCP, or ``tls=TlsConfig(...)`` for explicit PEM paths. See the documentation
for each client.

.. list-table::
   :header-rows: 1
   :widths: 26 14 18 42

   * - Variable
     - Required
     - Default
     - Description
   * - ``CUOPT_TLS_ENABLED``
     - No
     - ``0``
     - Used when ``tls=None``; non-zero enables TLS
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
   * - ``CUOPT_GRPC_DEBUG``
     - No
     - ``0``
     - Non-zero: extra gRPC client logging

``CUOPT_MAX_MESSAGE_BYTES`` also applies to both clients when set (same
default as the integrated remote client) -- it raises the gRPC channel's
max message size regardless of chunking support. ``CUOPT_CHUNK_SIZE`` only
has an effect for ``Client``; ``RoutingClient`` has no chunking to size (see
:doc:`routing`'s Limitations).

Usage
=====

Start the Server with TLS
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

How mTLS Works
--------------

With mTLS the server verifies every client, and the client verifies the server. Trust is based on **Certificate Authorities** (CAs), not individual certificate lists:

* ``--tls-root ca.crt`` tells the server which CA to trust; any client cert signed by that CA is accepted. The server does not store per-client certificates.
* ``--require-client-cert`` makes client verification **mandatory**. Without it, the server may still allow connections without a client cert.
* On the client, ``CUOPT_TLS_ROOT_CERT`` is the CA that signed the **server** certificate so the client can verify the server.

Restricting Access with a Private CA
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

Client Environment (Examples)
------------------------------

**Remote execution** — required host/port (see *Integrated remote client* table):

.. code-block:: bash

   export CUOPT_REMOTE_HOST=<server-hostname>
   export CUOPT_REMOTE_PORT=5001

**TLS** (optional; also used by the gRPC linear programming client and the gRPC routing client when ``tls=None``):

.. code-block:: bash

   export CUOPT_TLS_ENABLED=1
   export CUOPT_TLS_ROOT_CERT=ca.crt

For mTLS, also:

.. code-block:: bash

   export CUOPT_TLS_CLIENT_CERT=client.crt
   export CUOPT_TLS_CLIENT_KEY=client.key

**gRPC linear programming client** — pass host and port to ``Client(...)`` (not
``CUOPT_REMOTE_*``). With ``tls=None`` (default), the same ``CUOPT_TLS_*``
variables above apply. For explicit PEM paths, use ``TlsConfig``:

.. code-block:: python

   from cuopt.grpc.linear_programming import Client, TlsConfig

   # TLS: verify the server with a CA (or omit root_certs for the system trust store)
   client = Client(
       "server.example.com",
       5001,
       tls=TlsConfig(root_certs="ca.crt"),
   )

   # mTLS: also present a client certificate
   client = Client(
       "server.example.com",
       5001,
       tls=TlsConfig(
           root_certs="ca.crt",
           client_cert="client.crt",
           client_key="client.key",
       ),
   )

   # Plain TCP (ignore CUOPT_TLS_* even if set)
   client = Client("localhost", 5001, tls=False)

See :doc:`python-async-client` for the full job API.

Limitations and Scope
=====================

* **Problem types** — **LP**, **MIP**, and **QP** support remote execution and the :doc:`gRPC linear programming client <python-async-client>`. **Routing** (VRP, TSP, PDP) uses the :doc:`gRPC routing client <routing>` or a custom client. Routing does not read ``CUOPT_REMOTE_HOST``/``CUOPT_REMOTE_PORT``.
* **Message size** — Large problems use chunking; very large models can still hit gRPC max message / timeout limits. Tune ``CUOPT_CHUNK_SIZE``, ``CUOPT_MAX_MESSAGE_BYTES``, server ``--max-message-mb``, and solver ``time_limit`` as needed.
* **``CUOPT_GRPC_ARGS``** — Parsed on whitespace only; arguments containing spaces are awkward unless you invoke ``cuopt_grpc_server`` directly.
* **CRL / OCSP** — Not handled by the integrated gRPC TLS stack; use a private CA rotation strategy or a TLS-terminating proxy if you need revocation workflows.

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

Further Reading
===============

* :doc:`quick-start` — Plain TCP quick path.
* :doc:`examples` — Links to Python, C, and CLI example sections (use with ``CUOPT_REMOTE_*`` on the client for remote execution).
* :doc:`python-async-client` — Explicit Python gRPC client.
* :doc:`grpc-server-architecture` — Process model and job behavior (operator overview).
