/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt;

/**
 * Unchecked exception thrown by cuopt-java when a native call fails or
 * the runtime environment is misconfigured (e.g., the native library
 * cannot be loaded, or the JVM is older than Java 22).
 */
public class CuOptException extends RuntimeException {

    private static final long serialVersionUID = 1L;

    public CuOptException(String message) {
        super(message);
    }

    public CuOptException(String message, Throwable cause) {
        super(message, cause);
    }

    public CuOptException(Throwable cause) {
        super(cause);
    }
}
