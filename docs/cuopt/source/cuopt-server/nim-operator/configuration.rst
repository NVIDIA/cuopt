.. _cuopt-nim-configuration:

Configuration Guide
===================

This guide covers configuration options for the CuOpt NIM Operator deployment.

Image Configuration
-------------------

CuOpt Image Versions
^^^^^^^^^^^^^^^^^^^^

Update the image tag in ``cuopt-nimservice.yaml``:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - CUDA Version
     - Image Tag
   * - CUDA 12.9
     - ``25.12.0-cuda12.9-py3.13``

.. code-block:: yaml

   spec:
     image:
       repository: nvcr.io/nvidia/cuopt/cuopt
       tag: "25.12.0-cuda12.9-py3.13"
       pullPolicy: IfNotPresent

Resource Configuration
----------------------

GPU Resources
^^^^^^^^^^^^^

Configure GPU allocation:

.. code-block:: yaml

   spec:
     resources:
       limits:
         nvidia.com/gpu: 1    # Number of GPUs

Memory Resources
^^^^^^^^^^^^^^^^

For workloads requiring specific memory allocation:

.. code-block:: yaml

   spec:
     resources:
       limits:
         nvidia.com/gpu: 1
         memory: "32Gi"
       requests:
         memory: "16Gi"

Environment Variables
---------------------

CuOpt supports several environment variables for configuration:

.. code-block:: yaml

   spec:
     env:
       - name: CUOPT_DATA_DIR
         value: /model-store
       - name: CUOPT_SERVER_LOG_LEVEL
         value: info          # Options: debug, info, warning, error
       - name: CUOPT_SERVER_PORT
         value: "8000"

Storage Configuration
---------------------

The deployment optionally uses persistent storage so that datasets can be passed through the filesystem
rather than over http. If data is sent over http (the default), this storage is not needed.

.. code-block:: yaml

   spec:
     storage:
       pvc:
         create: true
         size: 10Gi
         storageClass: ""           # Uses default storage class
         volumeAccessMode: "ReadWriteOnce"

For custom storage class:

.. code-block:: yaml

   spec:
     storage:
       pvc:
         create: true
         size: 20Gi
         storageClass: "fast-ssd"
         volumeAccessMode: "ReadWriteOnce"

Networking Configuration
------------------------

Service Configuration
^^^^^^^^^^^^^^^^^^^^^

Default ClusterIP service:

.. code-block:: yaml

   spec:
     expose:
       service:
         type: ClusterIP
         port: 8000

For NodePort access:

.. code-block:: yaml

   spec:
     expose:
       service:
         type: NodePort
         port: 8000
         nodePort: 30800

For LoadBalancer (cloud environments):
.. note:: Currently the cuopt service does not support scaling; there can only be 1 instance of the pod per service. Therefore a LoadBalancer service is unnecessary.

.. code-block:: yaml

   spec:
     expose:
       service:
         type: LoadBalancer
         port: 8000

Ingress Configuration
^^^^^^^^^^^^^^^^^^^^^

To expose CuOpt externally via ingress:

.. code-block:: yaml

   spec:
     expose:
       service:
         type: ClusterIP
         port: 8000
       ingress:
         enabled: true
         spec:
           ingressClassName: nginx
           rules:
             - host: cuopt.example.com
               http:
                 paths:
                 - backend:
                     service:
                       name: cuopt-service
                       port:
                         number: 8000
                   path: /
                   pathType: Prefix

With TLS:

.. code-block:: yaml

   spec:
     expose:
       ingress:
         enabled: true
         spec:
           ingressClassName: nginx
           tls:
             - hosts:
                 - cuopt.example.com
               secretName: cuopt-tls-secret
           rules:
             - host: cuopt.example.com
               http:
                 paths:
                 - backend:
                     service:
                       name: cuopt-service
                       port:
                         number: 8000
                   path: /
                   pathType: Prefix

Scaling Configuration
---------------------

Currently the cuOpt service does not support scaling. Only a single instance of the pod per service is supported.

Health Probes
-------------

Liveness Probe
^^^^^^^^^^^^^^

Determines if the container is running:

.. code-block:: yaml

   spec:
     livenessProbe:
       enabled: true
       probe:
         failureThreshold: 3
         httpGet:
           path: /v2/health/live
           port: api
         initialDelaySeconds: 15
         periodSeconds: 10
         successThreshold: 1
         timeoutSeconds: 1

Readiness Probe
^^^^^^^^^^^^^^^

Determines if the container is ready to accept traffic:

.. code-block:: yaml

   spec:
     readinessProbe:
       enabled: true
       probe:
         failureThreshold: 30
         httpGet:
           path: /v2/health/ready
           port: api
         initialDelaySeconds: 30
         periodSeconds: 10
         successThreshold: 1
         timeoutSeconds: 1

Startup Probe
^^^^^^^^^^^^^

For slower starting containers:

.. code-block:: yaml

   spec:
     startupProbe:
       enabled: true
       probe:
         failureThreshold: 30
         httpGet:
           path: /v2/health/ready
           port: api
         periodSeconds: 10

Monitoring Configuration
------------------------

Enable Prometheus metrics and ServiceMonitor:

.. code-block:: yaml

   spec:
     metrics:
       enabled: true
       serviceMonitor:
         additionalLabels:
           release: kube-prometheus-stack

Full Configuration Example
--------------------------

Here's a complete production-ready configuration:

:download:`cuopt-nimservice-full.yaml <guide/cuopt-nimservice-full.yaml>`

.. literalinclude:: guide/cuopt-nimservice-full.yaml
   :language: yaml
   :linenos:
