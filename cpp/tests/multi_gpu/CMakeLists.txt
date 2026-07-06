# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

# Multi-GPU smoke test. The *_MG_TEST suffix marks it as requiring >=2 GPUs:
# ci/test_cpp_multi_gpu.sh runs these on the 2-GPU runner, while the single-GPU
# ci/run_ctests.sh skips them.
ConfigureTest(MULTI_GPU_SMOKE_MG_TEST multi_gpu_smoke_test.cu LABELS numopt)
