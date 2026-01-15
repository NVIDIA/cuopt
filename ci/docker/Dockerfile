# syntax=docker/dockerfile:1.2
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ARG CUDA_VER=unset
ARG CUOPT_VER=unset
ARG PYTHON_SHORT_VER=unset
ARG LINUX_VER=unset

FROM nvidia/cuda:${CUDA_VER}-runtime-ubuntu${LINUX_VER} AS cuda-libs

# CUDA headers (e.g., cuda_fp16.h) are not present in `runtime`/`base` images.
# Create a lightweight `devel` stage solely to copy `/usr/local/cuda/include`
# into the final image for CuPy/NVRTC runtime compilation.
FROM nvidia/cuda:${CUDA_VER}-devel-ubuntu${LINUX_VER} AS cuda-headers

# Install cuOpt
FROM nvidia/cuda:${CUDA_VER}-base-ubuntu${LINUX_VER} AS python-env

ARG CUDA_VER
ARG CUOPT_VER
ARG PYTHON_SHORT_VER

ENV DEBIAN_FRONTEND=noninteractive

# gcc is required for building psutils
RUN apt-get update && apt-get install -y --no-install-recommends build-essential software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -y --no-install-recommends \
        wget \
        unzip \
        gcc \
        gnupg2 \
        python${PYTHON_SHORT_VER} \
        python${PYTHON_SHORT_VER}-dev \
        python${PYTHON_SHORT_VER}-venv \
    && apt-get install -y --no-install-recommends --only-upgrade gnupg2 \
    && rm -rf /var/lib/apt/lists/* && \
    python${PYTHON_SHORT_VER} -m ensurepip --upgrade && \
    python${PYTHON_SHORT_VER} -m pip install --upgrade pip

ENV DEBIAN_FRONTEND=""

RUN ln -sf /usr/bin/python${PYTHON_SHORT_VER} /usr/bin/python

FROM python-env AS install-env

ARG CUOPT_VER
ARG PYTHON_SHORT_VER

# Install cuOpt as root to make it available to all users
RUN \
    cuda_suffix=cu$(echo ${CUDA_VER} | cut -d'.' -f1) && \
    cuda_major_minor=$(echo ${CUDA_VER} | cut -d'.' -f1-2) && \
    python -m pip install \
      --extra-index-url https://pypi.nvidia.com \
      --extra-index-url https://pypi.anaconda.org/rapidsai-wheels-nightly/simple \
      --no-cache-dir \
      "pyyaml" \
      "cuopt-server-${cuda_suffix}==${CUOPT_VER}" \
      "cuopt-sh-client==${CUOPT_VER}" && \
    python -m pip list

# Remove build-only deps to save space and reduce CVE surface area
RUN apt-get purge -y gcc gnupg2 && rm -rf /var/lib/apt/lists/*

FROM install-env AS cuopt-final

ARG PYTHON_SHORT_VER

# Consolidate all directory creation, permissions, and file operations into a single layer
RUN mkdir -p /opt/cuopt && \
    chmod 777 /opt/cuopt && \
    # Create profile.d script for universal access
    echo '#!/bin/bash' > /etc/profile.d/cuopt.sh && \
    echo 'export PATH="/usr/local/cuda/bin:/usr/bin:/usr/local/bin:/usr/local/nvidia/bin/:/usr/local/lib/python${PYTHON_SHORT_VER}/dist-packages/libcuopt/bin:$PATH"' >> /etc/profile.d/cuopt.sh && \
    echo 'export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/lib/aarch64-linux-gnu:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib/wsl/lib:/usr/lib/wsl/lib/libnvidia-container:/usr/lib/nvidia:/usr/lib/nvidia-current:/usr/local/lib/python${PYTHON_SHORT_VER}/dist-packages/libcuopt/lib/:/usr/local/lib/python${PYTHON_SHORT_VER}/dist-packages/rapids_logger/lib64:${LD_LIBRARY_PATH}"' >> /etc/profile.d/cuopt.sh && \
    chmod +x /etc/profile.d/cuopt.sh && \
    # Set in /etc/environment for system-wide access
    echo 'PATH="/usr/local/cuda/bin:/usr/bin:/usr/local/bin:/usr/local/nvidia/bin/:/usr/local/lib/python${PYTHON_SHORT_VER}/dist-packages/libcuopt/bin:$PATH"' >> /etc/environment && \
    echo 'LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/lib/aarch64-linux-gnu:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib/wsl/lib:/usr/lib/wsl/lib/libnvidia-container:/usr/lib/nvidia:/usr/lib/nvidia-current:/usr/local/lib/python${PYTHON_SHORT_VER}/dist-packages/libcuopt/lib/:/usr/local/lib/python${PYTHON_SHORT_VER}/dist-packages/rapids_logger/lib64:${LD_LIBRARY_PATH}"' >> /etc/environment && \
    # Set proper permissions for cuOpt installation
    chmod -R 755 /usr/local/lib/python${PYTHON_SHORT_VER}/dist-packages/cuopt* && \
    chmod -R 755 /usr/local/lib/python${PYTHON_SHORT_VER}/dist-packages/libcuopt* && \
    chmod -R 755 /usr/local/lib/python${PYTHON_SHORT_VER}/dist-packages/cuopt_* && \
    chmod -R 755 /usr/local/bin/* && \
    # Create entrypoint script in a single operation
    echo '#!/bin/bash' > /opt/cuopt/entrypoint.sh && \
    echo 'set -e' >> /opt/cuopt/entrypoint.sh && \
    echo '' >> /opt/cuopt/entrypoint.sh && \
    echo '# Get current user info from Docker environment variables' >> /opt/cuopt/entrypoint.sh && \
    echo 'CURRENT_UID=${UID:-1000}' >> /opt/cuopt/entrypoint.sh && \
    echo 'CURRENT_GID=${GID:-1000}' >> /opt/cuopt/entrypoint.sh && \
    echo '' >> /opt/cuopt/entrypoint.sh && \
    echo '# Set environment variables for the current user' >> /opt/cuopt/entrypoint.sh && \
    echo 'export HOME="/opt/cuopt"' >> /opt/cuopt/entrypoint.sh && \
    echo '' >> /opt/cuopt/entrypoint.sh && \
    echo '# Execute the command' >> /opt/cuopt/entrypoint.sh && \
    echo 'exec "$@"' >> /opt/cuopt/entrypoint.sh && \
    chmod +x /opt/cuopt/entrypoint.sh

# Set the default working directory to the cuopt folder
WORKDIR /opt/cuopt

# Copy all static files in a single layer
COPY ./LICENSE ./VERSION ./THIRD_PARTY_LICENSES ./COMMIT_SHA ./COMMIT_TIME /opt/cuopt/

# Copy CUDA libraries from the cuda-libs stage
COPY --from=cuda-libs /usr/local/cuda/lib64/libnvrtc* /usr/local/cuda/lib64/
COPY --from=cuda-libs /usr/local/cuda/lib64/libnvJitLink* /usr/local/cuda/lib64/

# Copy CUDA headers needed for runtime compilation (e.g., CuPy NVRTC).
COPY --from=cuda-headers /usr/local/cuda/include/ /usr/local/cuda/include/

# Use the flexible entrypoint
ENTRYPOINT ["/opt/cuopt/entrypoint.sh"]
CMD ["python", "-m", "cuopt_server.cuopt_service"]
