# syntax=docker/dockerfile:1.2
# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

ARG CUDA_VER=12.8.0
ARG CUOPT_VER=25.8.0
ARG PYTHON_VER=3.12.11
ARG PYTHON_SHORT_VER=3.12

FROM nvidia/cuda:${CUDA_VER}-runtime-ubuntu22.04 AS cuda-libs

# Install cuOpt
FROM nvidia/cuda:${CUDA_VER}-base-ubuntu22.04 AS python-env

ARG CUDA_VER
ARG CUOPT_VER
ARG PYTHON_VER

RUN apt-get update && apt-get install -y \
    wget \
    git \
    gcc \
    build-essential \
    zlib1g-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    libssl-dev \
    libreadline-dev \
    libffi-dev \
    libsqlite3-dev \
    libbz2-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://www.python.org/ftp/python/${PYTHON_VER}/Python-${PYTHON_VER}.tgz \
    && tar xzf Python-${PYTHON_VER}.tgz \
    && cd Python-${PYTHON_VER} \
    && ./configure --enable-optimizations \
    && make -j$(nproc) \
    && make install \
    && cd .. \
    && rm -rf Python-${PYTHON_VER} Python-${PYTHON_VER}.tgz \
    && python3 -m pip install --upgrade pip

RUN apt-get purge -y \
    build-essential \
    zlib1g-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    libssl-dev \
    libreadline-dev \
    libffi-dev \
    libsqlite3-dev \
    libbz2-dev \
    && apt-get autoremove -y \
    && apt-get clean

RUN ln -sf /usr/local/bin/python3 /usr/local/bin/python
RUN groupadd -r cuopt && useradd -r -g cuopt cuopt
RUN chown -R cuopt:cuopt /usr/local/lib/python3.12/site-packages
USER cuopt

FROM python-env AS install-env

WORKDIR /home/cuopt

ARG cuda-suffix=cu$(echo ${CUDA_VER} | cut -d'.' -f1)
ARG CUOPT_VERSION

ENV PIP_EXTRA_INDEX_URL="https://pypi.nvidia.com https://pypi.anaconda.org/rapidsai-wheels-nightly/simple"

SHELL ["/bin/bash", "-c"]
RUN if [[ "${CUDA_VER}" =~ ^12\. ]]; then \
        cuda_suffix="cu12"; \
        python -m pip install nvidia-cuda-runtime-cu12==$(echo ${CUDA_VER} | cut -d'.' -f1-2).*; \
    fi && \
    python -m pip install --user --no-cache-dir cuopt-server-${cuda_suffix}==${CUOPT_VER} cuopt-sh-client==${CUOPT_VER} && \
    python -m pip list

USER root

RUN apt-get purge -y gcc git \
    && apt-get update \
    && apt-get install -y unzip \
    && apt-get autoremove -y \
    && apt-get clean

USER cuopt

COPY ./LICENSE /home/cuopt/LICENSE
COPY ./VERSION /home/cuopt/VERSION

FROM install-env AS cuopt-final 

ARG PYTHON_SHORT_VER

ENV PATH="/usr/local/cuda/bin:/usr/bin:/usr/local/bin:/usr/local/nvidia/bin/:/home/cuopt/.local/lib/python3.12/site-packages/libcuopt/bin/:/home/cuopt/.local/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/lib/aarch64-linux-gnu:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib/wsl/lib:/usr/lib/wsl/lib/libnvidia-container:/usr/lib/nvidia:/usr/lib/nvidia-current:/home/cuopt/.local/lib/python3.12/site-packages/libcuopt/lib/:/home/cuopt/.local/lib/python3.12/site-packages/rapids_logger/lib64:${LD_LIBRARY_PATH}"
# Make pip packages accessible to all users
ENV PYTHONPATH="/home/cuopt/.local/lib/python${PYTHON_SHORT_VER}/site-packages:/usr/local/lib/python${PYTHON_SHORT_VER}/site-packages:${PYTHONPATH:-}"

COPY --from=cuda-libs /usr/local/cuda/lib64/libnvrtc* /usr/local/cuda/lib64/
COPY --from=cuda-libs /usr/local/cuda/lib64/libnvJitLink* /usr/local/cuda/lib64/

ENV CUOPT_SERVER_PORT=8000

ENTRYPOINT ["python", "-m", "cuopt_server.cuopts_service"]