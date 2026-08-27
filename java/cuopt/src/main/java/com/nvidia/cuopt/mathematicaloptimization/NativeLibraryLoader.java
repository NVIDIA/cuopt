/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.mathematicaloptimization;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

/**
 * Locates and loads {@code libcuopt_jni}, in three steps.
 *
 * <ol>
 *   <li>{@code -Dcuopt.native.dir}, for a library built from source;
 *   <li>a copy embedded in this JAR, which is how the classifier artifacts ship;
 *   <li>{@code System.loadLibrary}, for a library already on the library path.
 * </ol>
 */
final class NativeLibraryLoader {
  private static final String LIBRARY_NAME = "cuopt_jni";

  private NativeLibraryLoader() {}

  static void load() {
    String nativeDir = System.getProperty("cuopt.native.dir");
    if (nativeDir != null && !nativeDir.isBlank()) {
      System.load(Path.of(nativeDir, System.mapLibraryName(LIBRARY_NAME)).toAbsolutePath().toString());
      return;
    }

    Path embedded = extractEmbeddedLibrary();
    if (embedded != null) {
      System.load(embedded.toString());
      return;
    }

    System.loadLibrary(LIBRARY_NAME);
  }

  /**
   * The path an embedded library occupies, which is also the layout the packaging step writes.
   * {@code os.arch} reports {@code amd64} on x86_64 JVMs and {@code aarch64} on ARM ones.
   */
  static String resourcePath(String osArch, String libraryFileName) {
    String directory;
    switch (osArch) {
      case "amd64":
      case "x86_64":
        directory = "amd64";
        break;
      case "aarch64":
      case "arm64":
        directory = "aarch64";
        break;
      default:
        throw new IllegalStateException(
            "cuOpt has no native library for architecture '" + osArch + "'");
    }
    return "/" + directory + "/Linux/" + libraryFileName;
  }

  /**
   * Copies the embedded library out of the JAR, or returns null when this JAR does not carry one.
   *
   * <p>The library is written to a per-user directory keyed by its name and size rather than to a
   * fresh temporary file, because it is hundreds of megabytes and re-extracting it on every JVM
   * start would dominate startup.
   */
  private static Path extractEmbeddedLibrary() {
    String fileName = System.mapLibraryName(LIBRARY_NAME);
    String resource;
    try {
      resource = resourcePath(System.getProperty("os.arch", ""), fileName);
    } catch (IllegalStateException e) {
      return null;
    }

    URL url = NativeLibraryLoader.class.getResource(resource);
    if (url == null) {
      return null;
    }

    try {
      Path directory =
          Path.of(System.getProperty("java.io.tmpdir"), "cuopt-native-" + System.getProperty("user.name", "shared"));
      Files.createDirectories(directory);
      Path target = directory.resolve(fileName);

      long expectedSize = url.openConnection().getContentLengthLong();
      if (expectedSize >= 0 && Files.isRegularFile(target) && Files.size(target) == expectedSize) {
        return target;
      }

      // A partially written file from an interrupted run would fail to load, so write to a
      // sibling first and move it into place, which is atomic on the same filesystem.
      Path staging = Files.createTempFile(directory, fileName + ".", ".part");
      try (InputStream in = url.openStream()) {
        Files.copy(in, staging, StandardCopyOption.REPLACE_EXISTING);
        Files.move(staging, target, StandardCopyOption.REPLACE_EXISTING);
      } finally {
        Files.deleteIfExists(staging);
      }
      return target;
    } catch (IOException e) {
      throw new UncheckedIOException("failed to extract " + resource + " from the cuOpt JAR", e);
    }
  }
}
