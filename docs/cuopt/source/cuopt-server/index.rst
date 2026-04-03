Server
======

The **NVIDIA cuOpt self-hosted server** is a **REST** (HTTP/JSON) service for integrations that speak HTTP. Use :doc:`quick-start` for deployment, :doc:`server-api/index` for the API, and :doc:`client-api/index` for clients (including cuopt-sh-client).

For **gRPC remote execution** (Python, C API, ``cuopt_cli``, or custom clients to ``cuopt_grpc_server``), see :doc:`../cuopt-grpc/index` — it uses a different protocol and is not part of the HTTP REST surface.

.. image:: images/cuOpt-self-hosted.png
  :width: 500
  :align: center

Please refer to the following sections for REST deployment, API reference, and examples.

.. toctree::
   :caption: Quickstart
   :name: Quickstart
   :titlesonly:

   quick-start.rst

.. toctree::
   :caption: Server API
   :name: Server API
   :titlesonly:

   Server-API<server-api/index.rst>

.. toctree::
   :caption: Client API
   :name: Client API
   :titlesonly:

   Client-API<client-api/index.rst>

.. toctree::
   :caption: Examples
   :name: Examples
   :titlesonly:

   Examples<examples/index.rst>

.. toctree::
   :caption: CSP Guides
   :name: CSP Guides
   :titlesonly:

   CSP-Guides<csp-guides/index.rst>

.. toctree::
   :caption: NIM Operator
   :name: NIM Operator
   :titlesonly:

   NIM-Operator<nim-operator/index.rst>
