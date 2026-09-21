/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.mathematicaloptimization;

/**
 * Graph partitioning strategy used to split a problem across GPUs when distributed (multi-GPU)
 * PDLP is used, backed by constants generated from the C++ public header.
 */
public enum DistributedPdlpPartitioner {
  AUTO(CuOptConstants.CUOPT_DISTRIBUTED_PDLP_PARTITIONER_AUTO),
  KAMINPAR(CuOptConstants.CUOPT_DISTRIBUTED_PDLP_PARTITIONER_KAMINPAR),
  ROUND_ROBIN(CuOptConstants.CUOPT_DISTRIBUTED_PDLP_PARTITIONER_ROUND_ROBIN);

  private final int nativeValue;

  DistributedPdlpPartitioner(int nativeValue) {
    this.nativeValue = nativeValue;
  }

  public int nativeValue() {
    return nativeValue;
  }
}
