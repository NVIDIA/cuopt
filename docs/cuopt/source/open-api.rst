========================================
cuOpt Open-API Reference - Swagger
========================================

The OpenAPI document is generated from the cuOpt self-hosted server and reflects request validation. For **LP and MILP** payloads, ``solver_config`` is intentionally minimal in this schema: the service passes through every engine parameter; names and semantics are documented in :doc:`lp-qp-milp-settings`, and the macro values in ``cpp/include/cuopt/linear_programming/constants.h`` define the canonical snake_case strings. Do not treat an absent property in Swagger as unsupported.

.. swagger-plugin:: cuopt_spec.yaml
   :id: cuopt-api
