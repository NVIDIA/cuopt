.. _cuopt-nim-prerequisites:

Prerequisites
=============

This document covers all prerequisites needed before deploying CuOpt with the NIM Operator.

Kubernetes Cluster Setup
------------------------

You need a Kubernetes cluster with GPU-enabled nodes. Choose one of the following
installation methods:

Option 1: kubeadm (Manual Setup)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Standard Kubernetes installation using kubeadm. Follow the
`official Kubernetes documentation <https://kubernetes.io/docs/setup/production-environment/tools/kubeadm/>`_.

Option 2: Cloud Native Stack (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use NVIDIA Cloud Native Stack Ansible playbooks for automated setup including GPU Operator:

.. code-block:: bash

   git clone https://github.com/NVIDIA/cloud-native-stack.git
   cd cloud-native-stack/playbooks
   # Follow the playbook instructions

This method automatically deploys necessary operators including the GPU Operator.

Option 3: Minikube (Development/Testing)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For local development and testing:

.. code-block:: bash

   minikube start --driver=docker --gpus all

GPU Operator Installation
-------------------------

If not using Cloud Native Stack, install the GPU Operator manually.

Add NVIDIA Helm Repository
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
   helm repo update

Install GPU Operator
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   helm install --wait --generate-name \
      -n gpu-operator --create-namespace \
      nvidia/gpu-operator

This typically takes 3-5 minutes to install the driver and set up the cloud native
stack for GPU usage.

Verify GPU Operator
^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   kubectl get pods -n gpu-operator

All pods should be in ``Running`` state.

Storage Provisioner
-------------------

CuOpt requires persistent storage. Deploy a storage provisioner if your cluster
doesn't have one.

Local Path Provisioner (Development/Single Node)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   kubectl apply -f https://raw.githubusercontent.com/rancher/local-path-provisioner/v0.0.31/deploy/local-path-storage.yaml

Wait for the provisioner to be ready:

.. code-block:: bash

   kubectl rollout status deployment/local-path-provisioner -n local-path-storage --timeout=120s

Set Default Storage Class
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   kubectl patch storageclass local-path -p '{"metadata": {"annotations":{"storageclass.kubernetes.io/is-default-class":"true"}}}'

Verify Storage Class
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   kubectl get storageclass

You should see ``local-path`` marked as ``(default)``.

Production Storage Options
^^^^^^^^^^^^^^^^^^^^^^^^^^

For production deployments, consider:

* **Cloud providers**: Use native storage classes (AWS EBS, GCP PD, Azure Disk)
* **On-premises**: Longhorn, OpenEBS, Rook-Ceph

NIM Operator Installation
-------------------------

Create Namespace
^^^^^^^^^^^^^^^^

.. code-block:: bash

   kubectl create namespace nim-operator

Install NIM Operator
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   helm upgrade --install nim-operator nvidia/k8s-nim-operator \
       -n nim-operator \
       --version=3.0.2

Verify Installation
^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   kubectl get pods -n nim-operator
   kubectl get crd | grep nvidia

You should see the ``nimservices.apps.nvidia.com`` CRD registered.

NGC API Key
-----------

You need an NGC API key to pull NVIDIA container images.

Obtain NGC API Key
^^^^^^^^^^^^^^^^^^

1. Go to `NGC <https://ngc.nvidia.com/>`_
2. Sign in or create an account
3. Navigate to **Setup** → **API Key**
4. Generate a new API key

Set Environment Variable
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   export NGC_API_KEY=<your-api-key>

For persistent configuration, add to your shell profile:

.. code-block:: bash

   echo 'export NGC_API_KEY=<your-api-key>' >> ~/.bashrc
   source ~/.bashrc

Verification Checklist
----------------------

Before proceeding with CuOpt deployment, verify:

* Kubernetes cluster is running (``kubectl cluster-info``)
* GPU nodes are available (``kubectl get nodes -l nvidia.com/gpu.present=true``)
* GPU Operator pods are running (``kubectl get pods -n gpu-operator``)
* Storage class is configured (``kubectl get storageclass``)
* NIM Operator is installed (``kubectl get pods -n nim-operator``)
* NGC API key is set (``echo $NGC_API_KEY``)

Next Steps
----------

Once all prerequisites are met, proceed to :ref:`deployment <cuopt-nim-deployment>`.
