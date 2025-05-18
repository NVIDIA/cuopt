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

ARG arch=amd
ARG cuda_ver=12.5.1

# To copy cuda files
FROM nvcr.io/nvidia/cuda:${cuda_ver}-runtime-ubuntu22.04 AS cuda-env


# To copy nvvm
FROM rapidsai/ci-wheel:cuda${cuda_ver}-rockylinux8-py3.11 AS cuopt_build


# Install cuOpt
FROM python:3.12.8-slim-bookworm AS install-env

COPY --chown=nvs:nvs ./wheels /tmp/wheels/
ARG cuda-suffix=cu12

RUN apt-get update && apt-get install -y git gcc

ENV PIP_EXTRA_INDEX_URL="https://pypi.nvidia.com https://pypi.anaconda.org/rapidsai-wheels-nightly/simple"

RUN python -m pip install nvidia-cuda-runtime-cu12==12.5.*
RUN python -m pip install /tmp/wheels/cuopt_mps_parser* /tmp/wheels/cuopt_${cuda_suffix}* /tmp/wheels/cuopt_server_${cuda_suffix}*
RUN python -m pip uninstall setuptools -y


# Build release container
FROM nvcr.io/nvidian/distroless/python:3.12-v3.4.4-${arch}64

ARG cuda_ver=12.5.1
COPY --from=install-env --chown=nvs:nvs \
    /usr/local/lib/python3.12/site-packages \
    /usr/local/lib/python3.12/dist-packages

COPY --from=cuda-env --chown=nvs:nvs \
    /usr/local/cuda-* \
    /usr/local/cuda

COPY --from=cuopt_build --chown=nvs:nvs /usr/local/cuda/nvvm/ /usr/local/cuda/nvvm/

ARG nspect_id
ARG server_port=5000

ENV CUOPT_SERVER_PORT=${server_port}

ENV CUOPT_SERVER_NSPECT_ID=${nspect_id}

ENV RMM_DEBUG_LOG_FILE=/tmp/rmm_log.txt
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64
ENV CUPY_CACHE_DIR=/tmp/.cupy

WORKDIR /cache

COPY ./LICENSE /cache/LICENSE
COPY ./container-builder/README.md /cache/
COPY ./container-builder/CHANGELOG.md /cache/
COPY ./git_info.txt /cache/

CMD ["python3", "-m", "cuopt_server.cuopt_service"]
