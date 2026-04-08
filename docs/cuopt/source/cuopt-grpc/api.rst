..
   SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

======================
gRPC API (reference)
======================

The **CuOptRemoteService** gRPC API is defined in Protocol Buffers under the ``cuopt.remote`` package. Source files in the repository:

* ``cpp/src/grpc/cuopt_remote_service.proto`` — service and job/chunk/log RPCs
* ``cpp/src/grpc/cuopt_remote.proto`` — LP/MIP problem, settings, and result messages

Most users do **not** call these RPCs directly: the NVIDIA cuOpt **Python** API, **C API**, and **cuopt_cli** submit jobs using solver APIs plus :doc:`environment variables <advanced>`. **Custom** clients call ``CuOptRemoteService`` over gRPC using these definitions. This page summarizes the service for custom integrators and debugging.

Service: ``CuOptRemoteService``
================================

Asynchronous jobs
-----------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - RPC
     - Purpose
   * - ``SubmitJob``
     - Submit an LP or MILP job in one message (within gRPC message size limits).
   * - ``CheckStatus``
     - Poll job status by ``job_id``.
   * - ``GetResult``
     - Fetch a completed result (unary, when the payload fits one message).
   * - ``DeleteResult``
     - Remove a stored result from server memory.
   * - ``CancelJob``
     - Cancel a queued or running job.
   * - ``WaitForCompletion``
     - Block until the job finishes (status only; use ``GetResult`` for the solution).

Chunked upload (large problems)
--------------------------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - RPC
     - Purpose
   * - ``StartChunkedUpload``
     - Begin a session; send problem metadata and settings (arrays follow as chunks).
   * - ``SendArrayChunk``
     - Upload one slice of a numeric array field.
   * - ``FinishChunkedUpload``
     - Finalize the upload and return ``job_id`` (same as ``SubmitJob``).

Chunked download (large results)
--------------------------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - RPC
     - Purpose
   * - ``StartChunkedDownload``
     - Begin a download session; returns scalar result fields and array descriptors.
   * - ``GetResultChunk``
     - Fetch one chunk of a result array.
   * - ``FinishChunkedDownload``
     - End the download session and release server state.

Streaming and callbacks
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - RPC
     - Purpose
   * - ``StreamLogs``
     - Server-streaming solver log lines for a job.
   * - ``GetIncumbents``
     - MILP incumbent solutions since a given index.

Messages and constraints
========================

* **Problem types** — LP and MILP in the enum; the problem payload can include quadratic objective data for **QP**-style solves where the client API supports it. **Routing** over this gRPC service is **not** available yet; it is planned for an **upcoming** release (use REST for remote routing today).
* **Solver settings** — Carried as ``PDLPSolverSettings`` or ``MIPSolverSettings`` inside the request or chunked header, aligned with the NVIDIA cuOpt solver options documentation.
* **Errors** — gRPC status codes carry failures (see comments at the end of ``cuopt_remote_service.proto``).

Further reading
===============

* :doc:`grpc-server-architecture` — Server process model and job lifecycle (overview); :doc:`advanced` for ``cuopt_grpc_server`` flags. Contributor details: ``cpp/docs/grpc-server-architecture.md``.
* :doc:`advanced` — TLS, Docker, client environment variables, and limitations.
