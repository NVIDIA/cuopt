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

# Apply all available OS security patches (fixes CVE-2026-45447 openssl and others),
# then install Python. wget/unzip/gnupg2/software-properties-common are only needed
# to add the deadsnakes PPA — purge them in the same layer so they never persist.
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
        software-properties-common \
        gnupg2 \
        wget \
        unzip \
    && ( for i in 1 2 3 4 5; do \
            add-apt-repository -y ppa:deadsnakes/ppa && exit 0; \
            echo "add-apt-repository failed (attempt ${i}/5), retrying in 5s..."; \
            sleep 5; \
         done; \
         exit 1 ) \
    && apt-get install -y --no-install-recommends \
        python${PYTHON_SHORT_VER} \
        python${PYTHON_SHORT_VER}-dev \
        python${PYTHON_SHORT_VER}-venv \
    && apt-get purge -y --autoremove \
        software-properties-common \
        gnupg2 \
        wget \
        unzip \
    && rm -rf /var/lib/apt/lists/* \
    && python${PYTHON_SHORT_VER} -m ensurepip --upgrade \
    && python${PYTHON_SHORT_VER} -m pip install --upgrade pip

ENV DEBIAN_FRONTEND=""

RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python${PYTHON_SHORT_VER} 100 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_SHORT_VER} 100

FROM python-env AS install-env

ARG CUDA_VER
ARG CUOPT_VER
ARG PYTHON_SHORT_VER

# Install gcc (needed to build psutils), pip-install cuopt, then remove gcc
# — all in one layer so the build-only dep never persists in the final image.
RUN \
    cuda_suffix=cu$(echo ${CUDA_VER} | cut -d'.' -f1) && \
    apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && python -m pip install \
      --extra-index-url https://pypi.nvidia.com \
      --extra-index-url https://pypi.anaconda.org/rapidsai-wheels-nightly/simple \
      --no-cache-dir \
      "pyyaml" \
      "cuopt-server-${cuda_suffix}==${CUOPT_VER}" \
      "cuopt-${cuda_suffix}==${CUOPT_VER}" \
      "libcuopt-${cuda_suffix}==${CUOPT_VER}" \
      "cuopt-sh-client==${CUOPT_VER}" \
    && python -m pip list \
    && apt-get purge -y --autoremove gcc \
    && rm -rf /var/lib/apt/lists/*

FROM install-env AS cuopt-final

ARG PYTHON_SHORT_VER

# Make cuopt_grpc_server, cuopt_cli, and shared libraries available to all processes
# (profile.d scripts are only sourced by login shells; ENV works for all containers)
ENV PATH="/usr/local/cuda/bin:/usr/bin:/usr/local/bin:/usr/local/nvidia/bin/:/usr/local/lib/python${PYTHON_SHORT_VER}/dist-packages/libcuopt/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/lib/aarch64-linux-gnu:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib/wsl/lib:/usr/lib/wsl/lib/libnvidia-container:/usr/lib/nvidia:/usr/lib/nvidia-current:/usr/local/lib/python${PYTHON_SHORT_VER}/dist-packages/libcuopt/lib/:/usr/local/lib/python${PYTHON_SHORT_VER}/dist-packages/rapids_logger/lib64:${LD_LIBRARY_PATH}"

# Directory creation, permissions
RUN mkdir -p /opt/cuopt && \
    chmod 777 /opt/cuopt && \
    chmod -R 755 /usr/local/lib/python${PYTHON_SHORT_VER}/dist-packages/cuopt* && \
    chmod -R 755 /usr/local/lib/python${PYTHON_SHORT_VER}/dist-packages/libcuopt* && \
    chmod -R 755 /usr/local/lib/python${PYTHON_SHORT_VER}/dist-packages/cuopt_* && \
    chmod -R 755 /usr/local/bin/*

# Set the default working directory to the cuopt folder
WORKDIR /opt/cuopt

# Copy all static files in a single layer
COPY ./LICENSE ./VERSION ./THIRD_PARTY_LICENSES ./COMMIT_SHA ./COMMIT_TIME /opt/cuopt/

# Copy CUDA libraries from the cuda-libs stage
COPY --from=cuda-libs /usr/local/cuda/lib64/libnvrtc* /usr/local/cuda/lib64/
COPY --from=cuda-libs /usr/local/cuda/lib64/libnvJitLink* /usr/local/cuda/lib64/

# Copy CUDA headers needed for runtime compilation (e.g., CuPy NVRTC).
COPY --from=cuda-headers /usr/local/cuda/include/ /usr/local/cuda/include/

# Entrypoint supports server selection:
#   Default:                  Python REST server
#   CUOPT_SERVER_TYPE=grpc:   gRPC server (uses CUOPT_SERVER_PORT, CUOPT_GPU_COUNT)
#   Explicit command:         docker run <image> cuopt_grpc_server [args...]
COPY ./entrypoint.sh /opt/cuopt/entrypoint.sh
ENTRYPOINT ["/opt/cuopt/entrypoint.sh"]
CMD ["python", "-m", "cuopt_server.cuopt_service"]
