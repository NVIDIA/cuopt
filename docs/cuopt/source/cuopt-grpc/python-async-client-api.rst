..
   SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

==========================================
Python Async gRPC Client API Reference
==========================================

Import path: ``cuopt.grpc.linear_programming``.

Client
======

.. autoclass:: cuopt.grpc.linear_programming.Client
   :members:
   :undoc-members:
   :exclude-members: _spawn_client, _as_data_model, _backfill_log_stream, _run_log_stream, _stream_logs, _run_incumbent_stream, _poll_incumbents

Supporting Types
================

.. autoclass:: cuopt.grpc.linear_programming.TlsConfig
   :members:
   :undoc-members:

.. autoclass:: cuopt.grpc.linear_programming.JobStatus
   :members:
   :undoc-members:
   :member-order: bysource
   :exclude-members: __new__, __init__, _generate_next_value_, as_integer_ratio, bit_count, bit_length, conjugate, denominator, from_bytes, imag, is_integer, numerator, real, to_bytes

Exceptions
==========

.. autoexception:: cuopt.grpc.linear_programming.GrpcError
   :members:
   :show-inheritance:

.. autoexception:: cuopt.grpc.linear_programming.JobNotReadyError
   :members:
   :show-inheritance:

See Also
=========

* :doc:`python-async-client` — overview and when to use this client
* :doc:`python-async-client-examples` — log and incumbent streaming examples
* :doc:`api` — ``CuOptRemoteService`` proto / RPC reference
