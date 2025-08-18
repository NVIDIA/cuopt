# syntax=docker/dockerfile:1.2
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG CUDA_VER=unset
ARG CUOPT_VER=unset
ARG PYTHON_SHORT_VER=unset
ARG LINUX_VER=unset

FROM nvidia/cuda:${CUDA_VER}-runtime-ubuntu${LINUX_VER} AS cuda-libs

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
        python${PYTHON_SHORT_VER} \
        python${PYTHON_SHORT_VER}-dev \
        python${PYTHON_SHORT_VER}-venv \
    && rm -rf /var/lib/apt/lists/* && \
    python${PYTHON_SHORT_VER} -m ensurepip --upgrade && \
    python${PYTHON_SHORT_VER} -m pip install --upgrade pip

ENV DEBIAN_FRONTEND=""

RUN ln -sf /usr/bin/python${PYTHON_SHORT_VER} /usr/bin/python && \
    groupadd -r cuopt && \
    useradd -r -g cuopt cuopt && \
    chown -R cuopt:cuopt /usr/local/lib/python${PYTHON_SHORT_VER}/dist-packages

USER cuopt

FROM python-env AS install-env

WORKDIR /home/cuopt

ARG CUOPT_VER
ARG PYTHON_SHORT_VER

RUN cuda_suffix=cu$(echo ${CUDA_VER} | cut -d'.' -f1) && \
    cuda_major_minor=$(echo ${CUDA_VER} | cut -d'.' -f1-2) && \
    python -m pip install \
      --extra-index-url https://pypi.nvidia.com \
      --extra-index-url https://pypi.anaconda.org/rapidsai-wheels-nightly/simple \
      --user \
      --no-cache-dir \
      "cuopt-server-${cuda_suffix}==${CUOPT_VER}" \
      "cuopt-sh-client==${CUOPT_VER}" \
      "nvidia-cuda-runtime-${cuda_suffix}==${cuda_major_minor}.*" && \
    python -m pip list

USER root

# Remove gcc to save space, gcc was required for building psutils
RUN apt-get purge -y gcc && rm -rf /var/lib/apt/lists/*

USER cuopt

COPY ./LICENSE /home/cuopt/LICENSE
COPY ./VERSION /home/cuopt/VERSION
COPY ./THIRD_PARTY_LICENSES /home/cuopt/THIRD_PARTY_LICENSES

FROM install-env AS cuopt-final

ARG PYTHON_SHORT_VER

# Set environment variables in .bashrc for all future shells
RUN echo 'export PATH="/usr/local/cuda/bin:/usr/bin:/usr/local/bin:/usr/local/nvidia/bin/:/home/cuopt/.local/lib/python${PYTHON_SHORT_VER}/dist-packages/libcuopt/bin/:/home/cuopt/.local/bin:$PATH"' >> /home/cuopt/.bashrc && \
    echo 'export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/lib/aarch64-linux-gnu:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib/wsl/lib:/usr/lib/wsl/lib/libnvidia-container:/usr/lib/nvidia:/usr/lib/nvidia-current:/home/cuopt/.local/lib/python${PYTHON_SHORT_VER}/dist-packages/libcuopt/lib/:/home/cuopt/.local/lib/python${PYTHON_SHORT_VER}/dist-packages/rapids_logger/lib64:${LD_LIBRARY_PATH}"' >> /home/cuopt/.bashrc


# Create a .bash_profile that sources .bashrc if it exists
RUN echo 'if [ -f ~/.bashrc ]; then . ~/.bashrc; fi' > /home/cuopt/.bash_profile

COPY --from=cuda-libs /usr/local/cuda/lib64/libnvrtc* /usr/local/cuda/lib64/
COPY --from=cuda-libs /usr/local/cuda/lib64/libnvJitLink* /usr/local/cuda/lib64/

# Use a shell as entrypoint to handle both service and interactive modes
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["python -m cuopt_server.cuopt_service"]
