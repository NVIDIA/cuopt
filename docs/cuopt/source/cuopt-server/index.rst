======
Server
======

.. note::

   The :doc:`legacy HTTP server <legacy-http-server>` is **deprecated**. It is
   still shipped in 26.10 and will be removed in 26.12. It is being replaced
   with a HTTP proxy server which forwards requests to ``cuopt_grpc_server``.
   These pages describe the HTTP proxy server.

The **NVIDIA cuOpt HTTP proxy server** provides a **REST** (HTTP/JSON)
interface which forwards requests to ``cuopt_grpc_server``. Use
:doc:`quick-start` for deployment, :doc:`server-api/index` for the API, and
:doc:`client-api/index` for clients (including cuopt-sh-client).

.. _legacy-http-differences:

Deprecated Interfaces Not Provided by the Proxy Server
======================================================

This section lists HTTP behavior the proxy does not carry forward from the legacy server.

* ``POST /cuopt/solution`` (an uploaded solution) returns 501.
* On ``POST /cuopt/request``, the query parameters ``cache`` and ``reqId`` return 501.
* A JSON list of LP problems (batch LP) returns 501.
* ``DELETE /cuopt/request/*`` returns 501.
* On ``DELETE /cuopt/request``, the query flags ``running``, ``queued``, and ``cached`` return 501.
* ``initialId`` is a completed routing solution from an earlier request. An uploaded solution cannot be passed as ``initialId``. ``initialId`` on an LP or MILP request returns 501.
* ``DELETE /cuopt/log/{id}`` is a no-op. The call returns 200 when the log can be fetched and leaves the file in place. ``cuopt_grpc_server`` removes that file when the job is deleted.

.. toctree::
   :maxdepth: 2
   :caption: In this section
   :name: cuopt-server-contents

   quick-start.rst
   advanced.rst
   server-api/index.rst
   client-api/index.rst
   examples/index.rst
   csp-guides/index.rst
   nim-operator/index.rst
