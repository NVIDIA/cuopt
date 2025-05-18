======================
Azure Quickstart Guide
======================

By the end of this guide, you will have registered for NVIDIA AI Enterprise, created a Virtual Machine (VM) on Microsoft Azure that uses the NVIDIA AI Enterprise image, run a sample cuOpt workload, and you will be ready to run an instance of cuOpt on an Azure VM.
To complement this guide, you can watch the video tutorial `here <https://www.youtube.com/watch?v=W7-jMYp58rc>`__. The video tutorial and this user guide cover the same deployment steps.

Step 1: Create an Azure VM with NVAIE Image
--------------------------------------------

#. Log in to Azure and navigate to Azure Marketplace.

#. Look up **NVIDIA AI Enterprise** in the search bar.

   .. image:: images/azure1.png
                :width: 9.5in
                :height: 5in

#. You should see the following results. Click on the first option, the **NVIDIA AI Enterprise** Virtual Machine.

   .. image:: images/azure2.png
                :width: 9.5in
                :height: 5in

#. The following page will open. Click **Create**.

   .. image:: images/azure3.png
                :width: 9.5in
                :height: 5in

#. Configure your VM. If you have deployed a VM through Azure in the past, you should already have an SSH key you can reuse. If this is your first time, create a new SSH key.

   .. note::
        When selecting the hardware configuration, make sure to select a GPU of A100 or higher.

#. After configuring your VM, wait for it to deploy. This might take a few minutes to complete.

#. Upon deployment completion, you will see the following screen. Select **Go to resource**.

   .. image:: images/azure4.png
                :width: 9.5in
                :height: 5in

#. This will take you to the Overview page of your VM. Select **Connect** at the top of the page.

   .. image:: images/azure5.png
                :width: 9.5in
                :height: 5in

#. You can connect to the VM through Azure CLI or through your terminal (Native SSH). The public IP address of your VM is written at the top of the page.

   .. image:: images/azure6.png
                :width: 9.5in
                :height: 5in

Step 2: Activate NVAIE Subscription
------------------------------------

Once connected to the VM, generate an identity token. Activate your NVIDIA AI Enterprise subscription using that identity token on NGC. Follow the instructions `here <https://docs.nvidia.com/ai-enterprise/deployment-guide-cloud/0.1.0/azure-ai-enterprise-vmi.html#accessing-the-nc-on-ngc>`__.

Step 3: Run cuOpt
------------------

To run cuOpt, you will need to log in to the NVIDIA Container Registry, pull the cuOpt container, and then run it. To test that it is successfully running, you can run a sample cuOpt request. This process is the same for deploying cuOpt on your own infrastructure. Refer `Self-Hosted Service Quickstart Guide </cuopt-server/quick-start.html#container-from-nvidia-ngc>`__.


Step 4: Mapping Visualization with Azure
-----------------------------------------

Upon running your optimization problems and getting the optimized routes from cuOpt, you may wish to post-process your data and visualize your data on a map. One tool you can use for this is Azure Maps. Linked below are a demo application and sample code you can use as a reference.

-  `Demo application <https://samples.azuremaps.com/rest-services/mio>`__

-  `Sample code <https://github.com/Azure-Samples/AzureMapsCodeSamples/blob/main/Samples/REST%20Services/MIO/mio.html>`__
