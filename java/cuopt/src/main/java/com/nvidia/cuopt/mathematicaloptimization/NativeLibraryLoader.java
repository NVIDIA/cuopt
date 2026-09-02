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
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.attribute.PosixFilePermissions;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Arrays;

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
   * libgomp.so.1, libstdc++.so.6 and libgcc_s.so.1 travel too: this library is built against the
   * build host's GCC runtime libraries, which can require symbol versions (e.g. OMP_5.0.1,
   * GLIBCXX_3.4.30, GCC_14.0.0) newer than a consumer's own system copies ship -- observed with
   * Rocky Linux 8's defaults, which only go up to OMP_3.1, GLIBCXX_3.4.29 and GCC_7.0.0
   * respectively.
   */
  private static final String[] COMPANION_LIBRARIES = {"librmm.so", "librapids_logger.so", "libtbb.so.12", "libnccl.so.2", "libcudss.so.0", "libcudss_mtlayer_gomp.so.0", "libgomp.so.1", "libstdc++.so.6", "libgcc_s.so.1"};

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
      Path directory = privateExtractionDirectory();

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
   * <p>A file already there whose digest matches the packaged resource is reused rather than
   * rewritten, because the JNI library is hundreds of megabytes and re-extracting it on every JVM
   * start would dominate startup. Comparing digests rather than just size means a file another
   * process happened to leave at the same size cannot be mistaken for the real library. It is
   * written to a sibling and moved into place, so an interrupted run cannot leave a truncated
   * library behind for the next one to load.
   */
  private static Path extractResource(String resource, Path directory, String fileName)
      throws IOException {
    URL url = NativeLibraryLoader.class.getResource(resource);
    if (url == null) {
      return null;
    }
    Path target = directory.resolve(fileName);
    byte[] expectedDigest;
    try (InputStream in = url.openStream()) {
      expectedDigest = digest(in);
    }
    if (Files.isRegularFile(target) && Arrays.equals(expectedDigest, digest(target))) {
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

  private static byte[] digest(Path path) throws IOException {
    try (InputStream in = Files.newInputStream(path)) {
      return digest(in);
    }
  }

  private static byte[] digest(InputStream in) throws IOException {
    MessageDigest sha256;
    try {
      sha256 = MessageDigest.getInstance("SHA-256");
    } catch (NoSuchAlgorithmException e) {
      // Mandatory per the Java platform spec; every conforming JVM provides it.
      throw new IllegalStateException("SHA-256 unavailable", e);
    }
    byte[] buffer = new byte[1 << 16];
    int n;
    while ((n = in.read(buffer)) != -1) {
      sha256.update(buffer, 0, n);
    }
    return sha256.digest();
  }

  /**
   * A directory private to the current OS user, reused across JVM runs so the (potentially
   * hundreds-of-megabytes) native libraries are extracted once rather than on every start.
   *
   * <p>{@code java.io.tmpdir} is typically world-writable, so a fixed, predictable path under it
   * is only safe to reuse if it is verified private on every use: otherwise another local user
   * could pre-create it -- as a symlink elsewhere, or simply owned by them -- ahead of this
   * process and have {@link #extractResource} write into a location of their choosing before this
   * process ever runs, or read files this process wrote expecting them to be private. Refuse to
   * proceed rather than silently extracting into an untrusted directory.
   */
  private static Path privateExtractionDirectory() throws IOException {
    Path directory =
        Path.of(
            System.getProperty("java.io.tmpdir"),
            "cuopt-native-" + System.getProperty("user.name", "shared"));

    if (!Files.exists(directory, LinkOption.NOFOLLOW_LINKS)) {
      Files.createDirectories(directory);
      try {
        Files.setPosixFilePermissions(directory, PosixFilePermissions.fromString("rwx------"));
      } catch (UnsupportedOperationException e) {
        // Non-POSIX filesystem (e.g. Windows), which has no equivalent world-writable-tmpdir
        // risk to guard against here.
      }
      return directory;
    }

    if (Files.isSymbolicLink(directory)) {
      throw new IOException(directory + " is a symlink; refusing to extract native libraries "
          + "through it");
    }
    try {
      String owner = Files.getOwner(directory).getName();
      String currentUser = System.getProperty("user.name");
      if (currentUser != null && !currentUser.equals(owner)) {
        throw new IOException(
            directory + " is owned by '" + owner + "', not the current user; refusing to "
                + "extract native libraries into it");
      }
    } catch (UnsupportedOperationException e) {
      // Non-POSIX filesystem; ownership isn't a meaningful concept to check here.
    }
    return directory;
  }
}
