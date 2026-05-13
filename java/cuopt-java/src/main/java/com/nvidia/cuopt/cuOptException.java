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
public class cuOptException extends RuntimeException {

    private static final long serialVersionUID = 1L;

    public cuOptException(String message) {
        super(message);
    }

    public cuOptException(String message, Throwable cause) {
        super(message, cause);
    }

    public cuOptException(Throwable cause) {
        super(cause);
    }
}
