.. _cuopt-nim-operator:

NIM Operator Deployment
=======================

This guide walks you through deploying NVIDIA cuOpt as a Kubernetes service using the
`NIM Operator <https://docs.nvidia.com/nim-operator/latest/index.html>`_.

Overview
--------

The NIM Operator simplifies the deployment and management of NVIDIA NIM microservices
on Kubernetes. This deployment method provides:

* **Automated lifecycle management** - The operator handles scaling, updates, and health checks
* **Native Kubernetes integration** - Uses standard K8s patterns (CRDs, services, ingress)
* **GPU resource management** - Leverages the GPU Operator for optimal GPU utilization
* **Built-in monitoring** - Prometheus metrics and service monitors

Supported GPUs
--------------

The following NVIDIA GPUs are supported:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - GPU
     - Architecture
     - Notes
   * - A100
     - Ampere
     - Recommended for production
   * - H100
     - Hopper
     - High performance
   * - H200
     - Hopper
     - High performance with extended memory
   * - B200
     - Blackwell
     - Latest generation
   * - RTX 6000 Pro
     - Ada Lovelace
     - Workstation GPU

Quick Start
-----------

1. Complete all :ref:`prerequisites <cuopt-nim-prerequisites>`

2. Set your NGC API Key:

   .. code-block:: bash

      export NGC_API_KEY=<your-ngc-api-key>

3. Run the deployment script:

   .. code-block:: bash

      ./deploy.sh

4. Verify the deployment:

   .. code-block:: bash

      kubectl get nimservice -n nim-service
      kubectl get pods -n nim-service

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2

   prerequisites
   deployment
   configuration

Additional Resources
--------------------

* `NVIDIA cuOpt Documentation <https://docs.nvidia.com/cuopt/>`_
* `NIM Operator Documentation <https://docs.nvidia.com/nim-operator/latest/index.html>`_
* `GPU Operator Documentation <https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/index.html>`_
