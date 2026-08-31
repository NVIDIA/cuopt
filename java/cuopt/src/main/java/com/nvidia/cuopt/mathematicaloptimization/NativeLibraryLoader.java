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

  /**
   * rmm, rapids_logger and TBB have no static build, so they travel beside the JNI library.
   * libcudss_mtlayer_gomp.so.0 is cuDSS's OpenMP threading backend, which cudssSetThreadingLayer
   * dlopen()s at runtime rather than linking directly; without it that call fails and cuDSS
   * writes the failure straight to the process's native stdout, corrupting Maven Surefire's
   * forked-JVM protocol exactly like the raw writes NativeLogSink was built to intercept -- but
   * from a source outside cuopt's own logger entirely, so no logging fix here can catch it.
   */
  private static final String[] COMPANION_LIBRARIES = {"librmm.so", "librapids_logger.so", "libtbb.so.12", "libnccl.so.2", "libcudss.so.0", "libcudss_mtlayer_gomp.so.0"};

  private NativeLibraryLoader() {}

  static void load() {
    String nativeDir = System.getProperty("cuopt.native.dir");
    if (nativeDir != null && !nativeDir.isBlank()) {
      System.load(
          Path.of(nativeDir, System.mapLibraryName(LIBRARY_NAME)).toAbsolutePath().toString());
      return;
    }

    Path embedded = extractEmbeddedLibraries();
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
   * Copies the packaged libraries out of the JAR and returns the path of the JNI one, or null when
   * this JAR does not carry them.
   *
   * <p>The companions are not loaded here. The JNI library's {@code $ORIGIN} RPATH resolves them
   * once they sit in the same directory, so they only have to be on disk before it is loaded.
   */
  private static Path extractEmbeddedLibraries() {
    String osArch = System.getProperty("os.arch", "");
    String fileName = System.mapLibraryName(LIBRARY_NAME);
    String resource;
    try {
      resource = resourcePath(osArch, fileName);
    } catch (IllegalStateException e) {
      return null;
    }
    if (NativeLibraryLoader.class.getResource(resource) == null) {
      return null;
    }

    try {
      Path directory =
          Path.of(
              System.getProperty("java.io.tmpdir"),
              "cuopt-native-" + System.getProperty("user.name", "shared"));
      Files.createDirectories(directory);

      for (String companion : COMPANION_LIBRARIES) {
        extractResource(resourcePath(osArch, companion), directory, companion);
      }
      return extractResource(resource, directory, fileName);
    } catch (IOException e) {
      throw new UncheckedIOException("failed to extract native libraries from the cuOpt JAR", e);
    }
  }

  /**
   * Copies one packaged file into {@code directory} and returns it, or null when the JAR does not
   * contain it.
   *
   * <p>A file already there with the expected size is reused rather than rewritten, because the JNI
   * library is hundreds of megabytes and re-extracting it on every JVM start would dominate
   * startup. It is written to a sibling and moved into place, so an interrupted run cannot leave a
   * truncated library behind for the next one to load.
   */
  private static Path extractResource(String resource, Path directory, String fileName)
      throws IOException {
    URL url = NativeLibraryLoader.class.getResource(resource);
    if (url == null) {
      return null;
    }
    Path target = directory.resolve(fileName);
    long expectedSize = url.openConnection().getContentLengthLong();
    if (expectedSize >= 0 && Files.isRegularFile(target) && Files.size(target) == expectedSize) {
      return target;
    }
    Path staging = Files.createTempFile(directory, fileName + ".", ".part");
    try (InputStream in = url.openStream()) {
      Files.copy(in, staging, StandardCopyOption.REPLACE_EXISTING);
      Files.move(staging, target, StandardCopyOption.REPLACE_EXISTING);
    } finally {
      Files.deleteIfExists(staging);
    }
    return target;
  }
}
