/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.mathematicalprogramming;

import java.lang.ref.Cleaner;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public final class SolverSettings implements AutoCloseable {
  private static final Cleaner CLEANER = Cleaner.create();
  private final NativeHandle nativeHandle;
  private final Cleaner.Cleanable cleanable;
  private final List<MIPCallback> mipCallbacks = new ArrayList<>();

  public SolverSettings() {
    this.nativeHandle = new NativeHandle(NativeCuOpt.createSolverSettings());
    this.cleanable = CLEANER.register(this, nativeHandle);
  }

  long handle() {
    nativeHandle.requireOpen();
    return nativeHandle.handle;
  }

  public static Set<String> getSolverSettingNames() {
    return new LinkedHashSet<>(Arrays.asList(NativeCuOpt.getSolverSettingNames()));
  }

  public static String getSolverSetting(String name) {
    try (SolverSettings settings = new SolverSettings()) {
      return settings.getSettingAsString(name);
    }
  }

  public static <T> T getSolverSetting(String name, Class<T> type) {
    return parseValue(getSolverSetting(name), type);
  }

  public static String getSolverSettingAsString(String name) {
    try (SolverSettings settings = new SolverSettings()) {
      return settings.getSettingAsString(name);
    }
  }

  public SolverSettings setSetting(String name, String value) {
    NativeCuOpt.setSetting(handle(), name, value);
    return this;
  }

  public SolverSettings setSetting(String name, int value) {
    NativeCuOpt.setIntegerSetting(handle(), name, value);
    return this;
  }

  public SolverSettings setSetting(String name, double value) {
    NativeCuOpt.setFloatSetting(handle(), name, value);
    return this;
  }

  public SolverSettings setSetting(String name, boolean value) {
    NativeCuOpt.setIntegerSetting(handle(), name, value ? 1 : 0);
    return this;
  }

  public String getSetting(String name) {
    return getSettingAsString(name);
  }

  public <T> T getSetting(String name, Class<T> type) {
    return parseValue(getSettingAsString(name), type);
  }

  public String getSettingAsString(String name) {
    return NativeCuOpt.getSetting(handle(), name);
  }

  public SolverSettings setMethod(SolverMethod method) {
    return setSetting(CuOptConstants.CUOPT_METHOD, method.nativeValue());
  }

  public SolverSettings setPDLPSolverMode(PDLPSolverMode mode) {
    return setSetting(CuOptConstants.CUOPT_PDLP_SOLVER_MODE, mode.nativeValue());
  }

  public SolverSettings setOptimalityTolerance(double tolerance) {
    for (String setting : getSolverSettingNames()) {
      if (setting.endsWith("tolerance")
          && !setting.startsWith("mip")
          && !setting.contains("infeasible")) {
        setSetting(setting, tolerance);
      }
    }
    return this;
  }

  public SolverSettings addMIPStart(double[] values) {
    NativeCuOpt.addMIPStart(handle(), Arrays.copyOf(values, values.length));
    return this;
  }

  /**
   * Warm-start PDLP with a primal solution of length {@code numVariables}.
   *
   * <p>The Python API exposes this on {@code DataModel}; in Java it lives here, matching the C API,
   * where it is a solver setting rather than part of the problem.
   */
  public SolverSettings setInitialPrimalSolution(double[] values) {
    NativeCuOpt.setInitialPrimalSolution(handle(), Arrays.copyOf(values, values.length));
    return this;
  }

  /**
   * Warm-start PDLP with a dual solution of length {@code numConstraints}.
   *
   * <p>The Python API exposes this on {@code DataModel}; in Java it lives here, matching the C API,
   * where it is a solver setting rather than part of the problem.
   */
  public SolverSettings setInitialDualSolution(double[] values) {
    NativeCuOpt.setInitialDualSolution(handle(), Arrays.copyOf(values, values.length));
    return this;
  }

  public SolverSettings setMIPCallback(
      MIPSolutionCallback callback, Object userData, int numVariables) {
    NativeCuOpt.registerMIPGetSolutionCallback(handle(), callback, userData, numVariables);
    mipCallbacks.add(callback);
    return this;
  }

  public SolverSettings setMIPCallback(
      MIPSetSolutionCallback callback, Object userData, int numVariables) {
    NativeCuOpt.registerMIPSetSolutionCallback(handle(), callback, userData, numVariables);
    mipCallbacks.add(callback);
    return this;
  }

  public List<MIPCallback> getMIPCallbacks() {
    return Collections.unmodifiableList(mipCallbacks);
  }

  public boolean dumpSettingsToFile(String path, boolean hyperparametersOnly) {
    return NativeCuOpt.dumpSettingsToFile(handle(), path, hyperparametersOnly);
  }

  public boolean dumpSettingsToFile(String path) {
    return dumpSettingsToFile(path, true);
  }

  public SolverSettings loadSettingsFromFile(String path) {
    NativeCuOpt.loadSettingsFromFile(handle(), path);
    return this;
  }

  public Map<String, Object> toDict() {
    Map<String, Object> result = new LinkedHashMap<>();
    Map<String, Object> tolerances = new LinkedHashMap<>();
    for (String setting : getSolverSettingNames()) {
      Object value = inferredValue(getSettingAsString(setting));
      if (setting.endsWith("tolerance")) {
        tolerances.put(setting, value);
      } else {
        result.put(setting, value instanceof Double && ((Double) value).isInfinite() ? null : value);
      }
    }
    result.put("tolerances", tolerances);
    return Collections.unmodifiableMap(result);
  }

  private static <T> T parseValue(String value, Class<T> type) {
    if (type == String.class) {
      return type.cast(value);
    }
    if (type == Boolean.class) {
      if (!"true".equalsIgnoreCase(value) && !"false".equalsIgnoreCase(value)) {
        throw new IllegalArgumentException("Setting value is not a boolean: " + value);
      }
      return type.cast(Boolean.valueOf(value));
    }
    try {
      if (type == Integer.class) {
        return type.cast(Integer.valueOf(value));
      }
      if (type == Double.class) {
        return type.cast(Double.valueOf(value));
      }
    } catch (NumberFormatException exception) {
      throw new IllegalArgumentException(
          "Setting value cannot be converted to " + type.getSimpleName() + ": " + value,
          exception);
    }
    throw new IllegalArgumentException(
        "Unsupported setting type: " + type.getName()
            + "; use String, Boolean, Integer, or Double");
  }

  private static Object inferredValue(String value) {
    if ("true".equalsIgnoreCase(value)) {
      return true;
    }
    if ("false".equalsIgnoreCase(value)) {
      return false;
    }
    try {
      return Integer.valueOf(value);
    } catch (NumberFormatException ignored) {
      try {
        return Double.valueOf(value);
      } catch (NumberFormatException ignoredAgain) {
        return value;
      }
    }
  }

  @Override
  public void close() {
    cleanable.clean();
  }

  private static final class NativeHandle implements Runnable {
    private long handle;

    NativeHandle(long handle) {
      this.handle = handle;
    }

    void requireOpen() {
      if (handle == 0) {
        throw new IllegalStateException("SolverSettings is closed");
      }
    }

    @Override
    public void run() {
      if (handle != 0) {
        NativeCuOpt.destroySolverSettings(handle);
        handle = 0;
      }
    }
  }
}
