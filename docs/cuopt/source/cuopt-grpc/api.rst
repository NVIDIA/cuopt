..
   SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

======================
gRPC API (Reference)
======================

The **CuOptRemoteService** gRPC API is defined in Protocol Buffers under the ``cuopt.remote`` package. Source files in the repository:

* ``cpp/src/grpc/cuopt_remote_service.proto`` — service and job/chunk/log RPCs
* ``cpp/src/grpc/cuopt_remote.proto`` — LP/MIP problem, settings, and result messages

Most users do **not** call these RPCs directly:

* **Remote execution** — Python, C (``cuOptSolve``), and ``cuopt_cli`` forward
  solves when ``CUOPT_REMOTE_HOST`` and ``CUOPT_REMOTE_PORT`` are set
  (:doc:`quick-start`, :doc:`advanced`).
* **Python async gRPC client** — ``cuopt.grpc.linear_programming.Client``
  (:doc:`python-async-client`).

**Custom** clients call ``CuOptRemoteService`` over gRPC using these definitions.
This page summarizes the service for custom integrators and debugging.

Service: ``CuOptRemoteService``
================================

Asynchronous Jobs
-----------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - RPC
     - Purpose
   * - ``SubmitJob``
     - Submit an LP or MIP job in one message (within gRPC message size limits).
   * - ``CheckStatus``
     - Poll job status by ``job_id``.
   * - ``GetResult``
     - Fetch a completed result (unary, when the payload fits one message).
   * - ``DeleteResult``
     - Cancel the job if it is still queued or running, then remove all server-side state for that ``job_id``.
   * - ``CancelJob``
     - Cancel a queued or running job.
   * - ``WaitForCompletion``
     - Block until the job finishes (status only; use ``GetResult`` for the solution).

Chunked Upload (Large Problems)
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

Chunked Download (Large Results)
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

Streaming and Callbacks
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - RPC
     - Purpose
   * - ``StreamLogs``
     - Server-streaming solver log lines for a job.
   * - ``GetIncumbents``
     - MIP incumbent solutions since a given index (only if the job was
       submitted with ``enable_incumbents``; otherwise the list is empty).

Messages and Constraints
========================

* **Problem types** — Wire categories are LP/QP or MIP. QP is submitted as
  ``lp_request`` (``SolveLPRequest``) with quadratic fields on
  ``OptimizationProblem``. **Routing** over this gRPC service is **not**
  available yet (planned; use REST for remote routing today).
* **Solver settings** — Carried as ``PDLPSolverSettings`` or ``MIPSolverSettings`` inside the request or chunked header, aligned with the NVIDIA cuOpt solver options documentation.
* **Errors** — Transport failures use gRPC status codes. Some outcomes use
  ``Status::OK`` with response fields: ``CheckStatus`` reports unknown jobs as
  ``job_status=NOT_FOUND``; ``GetResult`` uses transport ``NOT_FOUND`` /
  ``UNAVAILABLE`` (not ready) and ``status=ERROR_SOLVE_FAILED`` for failed
  solves; ``DeleteResult`` / ``CancelJob`` report outcomes in the response.
  See ``cuopt_remote_service.proto``.

Further Reading
===============

* :doc:`python-async-client` / :doc:`python-async-client-api` — Python job client (``cuopt.grpc``) built on these RPCs.
* :doc:`grpc-server-architecture` — Server process model and job lifecycle (overview); :doc:`advanced` for ``cuopt_grpc_server`` flags. Contributor details: ``cpp/docs/grpc-server-architecture.md``.
* :doc:`advanced` — TLS, Docker, client environment variables, and limitations.
