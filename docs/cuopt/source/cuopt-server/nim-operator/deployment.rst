.. _cuopt-nim-deployment:

Deployment Guide
================

This guide covers deploying CuOpt using the NIM Operator.

Automated Deployment
--------------------

The easiest way to deploy CuOpt is using the provided deployment script.

Using the Deploy Script
^^^^^^^^^^^^^^^^^^^^^^^

:download:`deploy.sh <guide/deploy.sh>`

.. literalinclude:: guide/deploy.sh
   :language: bash
   :linenos:
   :lines: 1-50
   :caption: deploy.sh (showing first 50 lines)

1. Set your NGC API Key:

   .. code-block:: bash

      export NGC_API_KEY=<your-ngc-api-key>

2. Run the deployment script:

   .. code-block:: bash

      ./deploy.sh

3. With custom options:

   .. code-block:: bash

      ./deploy.sh --namespace my-cuopt --wait 600

Script Options
^^^^^^^^^^^^^^

.. code-block:: text

   Usage: deploy.sh [OPTIONS]

   Options:
       -n, --namespace NAME      Kubernetes namespace (default: nim-service)
       -t, --tag TAG             CuOpt image tag
       -u, --uninstall           Uninstall CuOpt deployment
       -s, --skip-prerequisites  Skip prerequisite checks
       -w, --wait SECONDS        Timeout for waiting on resources (default: 300)
       -h, --help                Show help message

Manual Deployment
-----------------

If you prefer to deploy manually, follow these steps.

Step 1: Create Namespace
^^^^^^^^^^^^^^^^^^^^^^^^

:download:`namespace.yaml <guide/namespace.yaml>`

.. literalinclude:: guide/namespace.yaml
   :language: yaml
   :linenos:

Apply the namespace:

.. code-block:: bash

   kubectl apply -f namespace.yaml

Step 2: Create Secrets
^^^^^^^^^^^^^^^^^^^^^^

Create the image pull secret:

.. code-block:: bash

   kubectl create secret -n nim-service docker-registry ngc-secret \
       --docker-server=nvcr.io \
       --docker-username='$oauthtoken' \
       --docker-password=${NGC_API_KEY}

Create the NGC API key secret:

.. code-block:: bash

   kubectl create secret -n nim-service generic ngc-api-secret \
       --from-literal=NGC_API_KEY=${NGC_API_KEY}

Step 3: Deploy CuOpt NIMService
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:download:`cuopt-nimservice.yaml <guide/cuopt-nimservice.yaml>`

.. literalinclude:: guide/cuopt-nimservice.yaml
   :language: yaml
   :linenos:

Apply the NIMService:

.. code-block:: bash

   kubectl apply -f cuopt-nimservice.yaml

Verifying the Deployment
------------------------

Check NIMService Status
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   kubectl get nimservice -n nim-service

Expected output:

.. code-block:: text

   NAME            STATE   AGE
   cuopt-service   Ready   5m

Check Pods
^^^^^^^^^^

.. code-block:: bash

   kubectl get pods -n nim-service

Check Service
^^^^^^^^^^^^^

.. code-block:: bash

   kubectl get service -n nim-service

Expected output:

.. code-block:: text

   NAME            TYPE        CLUSTER-IP       EXTERNAL-IP   PORT(S)    AGE
   cuopt-service   ClusterIP   10.101.219.185   <none>        8000/TCP   5m

Check Logs
^^^^^^^^^^

.. code-block:: bash

   kubectl logs -f deployment/cuopt-service -n nim-service

You should see output indicating the server is running:

.. code-block:: text

   2025-11-19 20:52:50.655 INFO cuopt server version 25.10.01
   2025-11-19 20:52:50.767 INFO Application startup complete.
   2025-11-19 20:52:50.767 INFO Uvicorn running on http://0.0.0.0:5000

Testing the Deployment
----------------------

Port Forward for Local Testing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   kubectl port-forward svc/cuopt-service -n nim-service 8000:8000

Then access the service at ``http://localhost:8000``.

Test with cuopt_sh CLI
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Get the service ClusterIP
   CUOPT_IP=$(kubectl get svc cuopt-service -n nim-service -o jsonpath='{.spec.clusterIP}')

   # Run a test
   cuopt_sh -i $CUOPT_IP -t LP /path/to/your/problem.mps

Example output:

.. code-block:: text

   2025-11-19 21:29:22.391 cuopt_sh_client.cuopt_self_host_client INFO Optimal
   {'response': {'solver_response': {'status': 'Optimal', ...}}}

Health Endpoints
^^^^^^^^^^^^^^^^

* Liveness: ``GET /v2/health/live``
* Readiness: ``GET /v2/health/ready``

.. code-block:: bash

   curl http://localhost:8000/v2/health/ready

Cleanup
-------

Using the Script
^^^^^^^^^^^^^^^^

.. code-block:: bash

   ./deploy.sh --uninstall

Manual Cleanup
^^^^^^^^^^^^^^

.. code-block:: bash

   kubectl delete -f cuopt-nimservice.yaml
   kubectl delete secret ngc-secret ngc-api-secret -n nim-service
   kubectl delete namespace nim-service

Troubleshooting
---------------

Pod Not Starting
^^^^^^^^^^^^^^^^

Check pod events:

.. code-block:: bash

   kubectl describe pod -l app=cuopt-service -n nim-service

Check GPU operator status:

.. code-block:: bash

   kubectl get pods -n gpu-operator

Image Pull Errors
^^^^^^^^^^^^^^^^^

Verify your NGC credentials:

.. code-block:: bash

   kubectl get secret ngc-secret -n nim-service -o yaml

Health Check Failures
^^^^^^^^^^^^^^^^^^^^^

Check the readiness and liveness probe logs:

.. code-block:: bash

   kubectl logs -f deployment/cuopt-service -n nim-service | grep -i health

Next Steps
----------

See :ref:`configuration <cuopt-nim-configuration>` for advanced configuration options.
