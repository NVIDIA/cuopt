=================
Quickstart Guide
=================

NVIDIA cuOpt provides a Python API for routing optimization and LP/QP/MILP that enables users to solve complex optimization problems efficiently.

Installation
============

Choose your install method below; the selector is pre-set for the Python API. Copy the command and run it in your environment. See :doc:`../install` for all interfaces and options.

.. install-selector::
   :default-iface: python

NVIDIA Launchable
-------------------

NVIDIA cuOpt can be tested with `NVIDIA Launchable <https://brev.nvidia.com/launchable/deploy?launchableID=env-2qIG6yjGKDtdMSjXHcuZX12mDNJ>`_ with `example notebooks <https://github.com/NVIDIA/cuopt-examples>`_. For more details, please refer to the `NVIDIA Launchable documentation <https://docs.nvidia.com/brev/latest/>`_.

Smoke Test
----------

After installation, you can verify that NVIDIA cuOpt is working correctly by running a simple test.

Copy and paste this script directly into your terminal (:download:`smoke_test_example.sh <routing/examples/smoke_test_example.sh>`):

.. literalinclude:: routing/examples/smoke_test_example.sh
   :language: bash
   :linenos:
   :start-after: # Users can copy


Example Response:

.. code-block:: text

        route  arrival_stamp  truck_id  location      type
           0            0.0         0         0     Depot
           2            2.0         0         2  Delivery
           1            4.0         0         1  Delivery
           0            6.0         0         0     Depot


      ****************** Display Routes *************************
      Vehicle-0 starts at: 0.0, completes at: 6.0, travel time: 6.0,  Route :
        0(Dpt)->2(D)->1(D)->0(Dpt)

      This results in a travel time of 6.0 to deliver all routes
