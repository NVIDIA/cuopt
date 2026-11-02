/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.mathematicaloptimization;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.UncheckedIOException;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.nio.file.attribute.PosixFilePermissions;
import java.util.ArrayList;
import java.util.List;

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
   * The companion filename, one per line, is read from {@code companions.txt}, packaged
   * alongside the JNI library by build_cuopt_java_jar.sh -- not hardcoded here -- because exact
   * SONAMEs are build-environment-dependent: e.g. conda-forge's TBB is {@code libtbb.so.12}, but
   * Rocky 8's dnf tbb-devel package is the much older {@code libtbb.so.2}. See that script's own
   * comment for the full list of what travels this way and why (rmm, rapids_logger, TBB, NCCL,
   * cuDSS, and the build host's GCC runtime), and for why {@code libcudss_mtlayer_gomp.so.0} is
   * always included even though it is dlopen()'d rather than linked.
   */
  private static final String COMPANION_MANIFEST = "companions.txt";

  /**
   * The CUDA math libraries the JNI library links dynamically rather than statically (see
   * java/cuopt/CMakeLists.txt) -- too large to embed, and expected to already be present on a
   * system with a CUDA-capable driver. A correctly configured CUDA image registers these with
   * the dynamic linker, so this is normally a no-op; see {@link #preloadCudaLibraries}.
   */
  private static final String[] CUDA_RUNTIME_LIBRARIES = {
    "libcublas.so", "libcublasLt.so", "libcusparse.so", "libcusolver.so", "libnvJitLink.so"
  };

  /**
   * cuOpt's own cuDSS threading-layer plugin, built against a modern libgomp instead of cuDSS's
   * prebuilt one (see java/cuopt/ci/build_static_libcuopt.sh, cpp/CMakeLists.txt's
   * CUOPT_BUILD_CUSTOM_CUDSS_MTLAYER). cudssSetThreadingLayer() dlopen()s it by bare filename
   * from code running inside libcudss.so.0 itself -- not from this library -- so
   * libcuopt_jni.so's own {@code $ORIGIN} RPATH trick (see the class javadoc) never applies to
   * it: that dlopen() call uses libcudss.so.0's own search path instead (which is empty; a
   * dnf-installed system library carries no RPATH of its own), plus {@code LD_LIBRARY_PATH} and
   * the ld.so cache, none of which cover this JAR's private extraction directory. Explicitly
   * loading it by absolute path first works around that: once a library with a given SONAME is
   * resident anywhere in the process, glibc's dynamic linker satisfies a later bare-filename
   * dlopen() of the same SONAME from the already-loaded copy instead of searching again -- the
   * same trick {@link #preloadCudaLibraries} uses. Without this, cudssSetThreadingLayer() fails
   * with CUDSS_STATUS_INVALID_VALUE.
   */
  private static final String DLOPEN_ONLY_LIBRARY = "libcudss_mtlayer_cuopt.so";

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
      preloadCudaLibraries();
      preloadDlopenOnlyLibraries(embedded.getParent());
      System.load(embedded.toString());
      return;
    }

    System.loadLibrary(LIBRARY_NAME);
  }

  /**
   * Best-effort: loads {@link #DLOPEN_ONLY_LIBRARY} by absolute path if this JAR packaged one, so
   * cuDSS's own later dlopen() of it by bare filename resolves without needing it on any standard
   * search path. See that field's comment for why. Silently does nothing if this JAR did not
   * package it (an older build, or a build without cuDSS) or if loading fails; the CUDSS call
   * that needs it fails with a clear status code of its own later, which is more useful than a
   * failure in this best-effort step.
   */
  private static void preloadDlopenOnlyLibraries(Path directory) {
    Path library = directory.resolve(DLOPEN_ONLY_LIBRARY);
    if (Files.isRegularFile(library)) {
      try {
        System.load(library.toString());
      } catch (UnsatisfiedLinkError e) {
        // See the failure-handling note above.
      }
    }
  }

  /**
   * Best-effort: if the CUDA math libraries the JNI library needs are not already resolvable
   * through the dynamic linker's default search (RPATH, {@code LD_LIBRARY_PATH}, or the
   * ldconfig cache), find and load them explicitly by path first.
   *
   * <p>This works because of how the dynamic linker resolves a shared object's own dependencies:
   * once something with a given name is loaded anywhere in the process, a later library's
   * reference to that same name is satisfied by the already-loaded copy rather than searched for
   * again. It does not require {@code LD_LIBRARY_PATH} itself to change, which a running JVM
   * cannot portably do for its own process.
   *
   * <p>A correctly configured CUDA image registers these with the dynamic linker already, so
   * this is normally a no-op. It exists because that registration was observed missing on at
   * least one CI runner's arm64 image despite the library being present on disk -- the same
   * class of environment inconsistency a real consumer's image could hit too. Failures here are
   * silent: if a library still cannot be found or loaded, the later {@code System.load} of the
   * JNI library itself fails with a clearer {@code UnsatisfiedLinkError} naming exactly what is
   * missing, which is more useful than a failure in this best-effort step.
   */
  private static void preloadCudaLibraries() {
    for (String library : CUDA_RUNTIME_LIBRARIES) {
      if (isAlreadyResolvable(library)) {
        continue;
      }
      Path found = findCudaLibrary(library);
      if (found != null) {
        try {
          System.load(found.toString());
        } catch (UnsatisfiedLinkError e) {
          // See the failure-handling note above.
        }
      }
    }
  }

  private static boolean isAlreadyResolvable(String libraryFileName) {
    String name = libraryFileName.startsWith("lib") ? libraryFileName.substring(3) : libraryFileName;
    if (name.endsWith(".so")) {
      name = name.substring(0, name.length() - 3);
    }
    try {
      System.loadLibrary(name);
      return true;
    } catch (UnsatisfiedLinkError e) {
      return false;
    }
  }

  /**
   * Searches the CUDA toolkit layout every RAPIDS/NVIDIA CUDA image uses --
   * {@code /usr/local/cuda-<version>/targets/<arch>/lib} -- for {@code libraryFileName}, matching
   * on the file actually being present rather than the first directory found: some images ship
   * more than one targets directory, not all of them populated.
   */
  private static Path findCudaLibrary(String libraryFileName) {
    Path cudaRoot = Paths.get("/usr/local");
    if (!Files.isDirectory(cudaRoot)) {
      return null;
    }
    try (DirectoryStream<Path> cudaDirs = Files.newDirectoryStream(cudaRoot, "cuda*")) {
      for (Path cudaDir : cudaDirs) {
        Path targetsDir = cudaDir.resolve("targets");
        if (!Files.isDirectory(targetsDir)) {
          continue;
        }
        try (DirectoryStream<Path> archDirs = Files.newDirectoryStream(targetsDir)) {
          for (Path archDir : archDirs) {
            Path candidate = archDir.resolve("lib").resolve(libraryFileName);
            if (Files.isRegularFile(candidate)) {
              return candidate;
            }
          }
        } catch (IOException e) {
          // Keep searching other cuda* directories.
        }
      }
    } catch (IOException e) {
      return null;
    }
    return null;
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

      for (String companion : readCompanionManifest(osArch)) {
        extractResource(resourcePath(osArch, companion), directory, companion);
      }
      return extractResource(resource, directory, fileName);
    } catch (IOException e) {
      throw new UncheckedIOException("failed to extract native libraries from the cuOpt JAR", e);
    }
  }

  /**
   * Reads the companion filenames packaged alongside the JNI library for this architecture. See
   * {@link #COMPANION_MANIFEST} for why this list is not hardcoded. An older JAR built before
   * this manifest existed would have none, so a missing manifest is treated as an empty list
   * rather than an error.
   */
  private static List<String> readCompanionManifest(String osArch) throws IOException {
    String resource = resourcePath(osArch, COMPANION_MANIFEST);
    try (InputStream in = NativeLibraryLoader.class.getResourceAsStream(resource)) {
      if (in == null) {
        return List.of();
      }
      List<String> companions = new ArrayList<>();
      try (BufferedReader reader = new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8))) {
        String line;
        while ((line = reader.readLine()) != null) {
          line = line.trim();
          if (!line.isEmpty()) {
            companions.add(line);
          }
        }
      }
      return companions;
    }
  }

  /**
   * Copies one packaged file into {@code directory} and returns it, or null when the JAR does not
   * contain it.
   *
   * <p>A file already there with the expected size is reused rather than rewritten, because the
   * JNI library is hundreds of megabytes and re-extracting it on every JVM start would dominate
   * startup. This is only safe because {@link #privateExtractionDirectory} guarantees {@code
   * directory} is private to the current OS user: relying on size alone in a directory anyone
   * could write to would let another user's same-size file pass as the real library.
   *
   * <p>It is written to a sibling and moved into place, so an interrupted run cannot leave a
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
