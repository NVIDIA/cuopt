/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * cuOpt install selector - generates install commands from user choices.
 * Stable version comes from window.CUOPT_INSTALL_VERSION (injected at build from cuopt.__version__).
 * Next = stable + 2 on minor with year rollover (YY.MM format). Update COMMANDS structure when commands change.
 */
(function () {
  "use strict";

  var ver = typeof window !== "undefined" && window.CUOPT_INSTALL_VERSION;
  if (!ver || !ver.conda || !ver.pip) {
    var root = document.getElementById("cuopt-install-selector");
    if (root) {
      root.innerHTML =
        '<div class="cuopt-install-selector-wrap cuopt-install-error">' +
        '<p><strong>Install selector error:</strong> Version was not injected. ' +
        'Build the documentation (e.g. <code>make html</code>) so that <code>cuopt-install-version.js</code> is generated from the package version.</p>' +
        "</div>";
    }
    return;
  }

  var V_CONDA = ver.conda;
  var V = ver.pip;
  var parts = V_CONDA.split(".");
  var major = parseInt(parts[0], 10);
  var minor = parseInt(parts[1], 10) || 0;
  var nextMinor = minor + 2;
  var nextMajor = major;
  if (nextMinor > 12) {
    nextMajor = major + 1;
    nextMinor = nextMinor - 12;
  }
  var V_CONDA_NEXT = nextMajor + "." + (nextMinor < 10 ? "0" : "") + nextMinor;
  var V_NEXT = nextMajor + "." + nextMinor;

  function pipInstall(pkg, cudaSuffix, version, nightly) {
    var name = pkg + (cudaSuffix ? "-" + cudaSuffix : "");
    var flags = nightly
      ? "--pre --extra-index-url=https://pypi.nvidia.com --extra-index-url=https://pypi.anaconda.org/rapidsai-wheels-nightly/simple/"
      : "--extra-index-url=https://pypi.nvidia.com";
    return "pip install " + flags + " '" + name + "==" + version + ".*'";
  }

  function condaInstall(pkg, version, cudaVersion, nightly) {
    var cmd = "conda install -c " + (nightly ? "rapidsai-nightly" : "rapidsai") +
      " -c conda-forge -c nvidia " + pkg + "=" + version + ".*";
    return cudaVersion ? cmd + " cuda-version=" + cudaVersion : cmd;
  }

  /* Shared Docker image lines: same tags are typically published to Docker Hub and NGC */
  var CONTAINER_CUOPT_LIB = {
    stable: {
      cu12: {
        default: "docker pull nvidia/cuopt:latest-cu12",
        run: "docker run --gpus all -it --rm nvidia/cuopt:latest-cu12 /bin/bash",
      },
      cu13: {
        default: "docker pull nvidia/cuopt:latest-cu13",
        run: "docker run --gpus all -it --rm nvidia/cuopt:latest-cu13 /bin/bash",
      },
      "cu13-ubi10": {
        default: "docker pull nvidia/cuopt:latest-cu13-ubi10",
        run: "docker run --gpus all -it --rm nvidia/cuopt:latest-cu13-ubi10 /bin/bash",
      },
    },
    nightly: {
      cu12: {
        default: "docker pull nvidia/cuopt:" + V_NEXT + ".0a-cu12",
        run: "docker run --gpus all -it --rm nvidia/cuopt:" + V_NEXT + ".0a-cu12 /bin/bash",
      },
      cu13: {
        default: "docker pull nvidia/cuopt:" + V_NEXT + ".0a-cu13",
        run: "docker run --gpus all -it --rm nvidia/cuopt:" + V_NEXT + ".0a-cu13 /bin/bash",
      },
      "cu13-ubi10": {
        default: "docker pull nvidia/cuopt:" + V_NEXT + ".0a-cu13-ubi10",
        run: "docker run --gpus all -it --rm nvidia/cuopt:" + V_NEXT + ".0a-cu13-ubi10 /bin/bash",
      },
    },
  };

  var COMMANDS = {
    python: {
      pip: {
        stable: {
          cu12:
            "pip install --extra-index-url=https://pypi.nvidia.com 'cuopt-cu12==" +
            V +
            ".*'",
          cu13:
            "pip install --extra-index-url=https://pypi.nvidia.com 'cuopt-cu13==" +
            V +
            ".*'",
        },
        nightly: {
          cu12:
            "pip install --pre --extra-index-url=https://pypi.nvidia.com --extra-index-url=https://pypi.anaconda.org/rapidsai-wheels-nightly/simple/ 'cuopt-cu12==" +
            V_NEXT +
            ".*'",
          cu13:
            "pip install --pre --extra-index-url=https://pypi.nvidia.com --extra-index-url=https://pypi.anaconda.org/rapidsai-wheels-nightly/simple/ 'cuopt-cu13==" +
            V_NEXT +
            ".*'",
        },
      },
      conda: {
        stable: {
          cu12:
            "conda install -c rapidsai -c conda-forge -c nvidia cuopt=" +
            V_CONDA +
            ".* cuda-version=12.9",
          cu13:
            "conda install -c rapidsai -c conda-forge -c nvidia cuopt=" +
            V_CONDA +
            ".* cuda-version=13.0",
        },
        nightly: {
          cu12:
            "conda install -c rapidsai-nightly -c conda-forge -c nvidia cuopt=" +
            V_CONDA_NEXT +
            ".* cuda-version=12.9",
          cu13:
            "conda install -c rapidsai-nightly -c conda-forge -c nvidia cuopt=" +
            V_CONDA_NEXT +
            ".* cuda-version=13.0",
        },
      },
      container: CONTAINER_CUOPT_LIB,
    },
    c: {
      pip: {
        stable: {
          cu12:
            "pip install --extra-index-url=https://pypi.nvidia.com 'libcuopt-cu12==" +
            V +
            ".*'",
          cu13:
            "pip install --extra-index-url=https://pypi.nvidia.com 'libcuopt-cu13==" +
            V +
            ".*'",
        },
        nightly: {
          cu12:
            "pip install --pre --extra-index-url=https://pypi.nvidia.com --extra-index-url=https://pypi.anaconda.org/rapidsai-wheels-nightly/simple/ 'libcuopt-cu12==" +
            V_NEXT +
            ".*'",
          cu13:
            "pip install --pre --extra-index-url=https://pypi.nvidia.com --extra-index-url=https://pypi.anaconda.org/rapidsai-wheels-nightly/simple/ 'libcuopt-cu13==" +
            V_NEXT +
            ".*'",
        },
      },
      conda: {
        stable: {
          cu12:
            "conda install -c rapidsai -c conda-forge -c nvidia libcuopt=" +
            V_CONDA +
            ".* cuda-version=12.9",
          cu13:
            "conda install -c rapidsai -c conda-forge -c nvidia libcuopt=" +
            V_CONDA +
            ".* cuda-version=13.0",
        },
        nightly: {
          cu12:
            "conda install -c rapidsai-nightly -c conda-forge -c nvidia libcuopt=" +
            V_CONDA_NEXT +
            ".* cuda-version=12.9",
          cu13:
            "conda install -c rapidsai-nightly -c conda-forge -c nvidia libcuopt=" +
            V_CONDA_NEXT +
            ".* cuda-version=13.0",
        },
      },
      container: CONTAINER_CUOPT_LIB,
    },
    server: {
      pip: {
        stable: {
          cu12:
            "pip install --extra-index-url=https://pypi.nvidia.com 'nvidia-cuda-runtime-cu12==12.9.*' 'cuopt-server-cu12==" +
            V +
            ".*' 'cuopt-sh-client==" +
            V_CONDA +
            ".*'",
          cu13:
            "pip install --extra-index-url=https://pypi.nvidia.com 'nvidia-cuda-runtime==13.0.*' 'cuopt-server-cu13==" +
            V +
            ".*' 'cuopt-sh-client==" +
            V_CONDA +
            ".*'",
        },
        nightly: {
          cu12:
            "pip install --pre --extra-index-url=https://pypi.nvidia.com --extra-index-url=https://pypi.anaconda.org/rapidsai-wheels-nightly/simple/ 'cuopt-server-cu12==" +
            V_NEXT +
            ".*' 'cuopt-sh-client==" +
            V_CONDA_NEXT +
            ".*'",
          cu13:
            "pip install --pre --extra-index-url=https://pypi.nvidia.com --extra-index-url=https://pypi.anaconda.org/rapidsai-wheels-nightly/simple/ 'cuopt-server-cu13==" +
            V_NEXT +
            ".*' 'cuopt-sh-client==" +
            V_CONDA_NEXT +
            ".*'",
        },
      },
      conda: {
        stable: {
          default:
            "conda install -c rapidsai -c conda-forge -c nvidia cuopt-server=" +
            V_CONDA +
            ".* cuopt-sh-client=" +
            V_CONDA +
            ".*",
        },
        nightly: {
          default:
            "conda install -c rapidsai-nightly -c conda-forge -c nvidia cuopt-server=" +
            V_CONDA_NEXT +
            ".* cuopt-sh-client=" +
            V_CONDA_NEXT +
            ".*",
        },
      },
      container: {
        stable: {
          cu12: {
            default: "docker pull nvidia/cuopt:latest-cu12",
            run: "docker run --gpus all -it --rm -p 8000:8000 -e CUOPT_SERVER_PORT=8000 nvidia/cuopt:latest-cu12",
          },
          cu13: {
            default: "docker pull nvidia/cuopt:latest-cu13",
            run: "docker run --gpus all -it --rm -p 8000:8000 -e CUOPT_SERVER_PORT=8000 nvidia/cuopt:latest-cu13",
          },
          "cu13-ubi10": {
            default: "docker pull nvidia/cuopt:latest-cu13-ubi10",
            run: "docker run --gpus all -it --rm -p 8000:8000 -e CUOPT_SERVER_PORT=8000 nvidia/cuopt:latest-cu13-ubi10",
          },
        },
        nightly: {
          cu12: {
            default: "docker pull nvidia/cuopt:" + V_NEXT + ".0a-cu12",
            run: "docker run --gpus all -it --rm -p 8000:8000 -e CUOPT_SERVER_PORT=8000 nvidia/cuopt:" + V_NEXT + ".0a-cu12",
          },
          cu13: {
            default: "docker pull nvidia/cuopt:" + V_NEXT + ".0a-cu13",
            run: "docker run --gpus all -it --rm -p 8000:8000 -e CUOPT_SERVER_PORT=8000 nvidia/cuopt:" + V_NEXT + ".0a-cu13",
          },
          "cu13-ubi10": {
            default: "docker pull nvidia/cuopt:" + V_NEXT + ".0a-cu13-ubi10",
            run: "docker run --gpus all -it --rm -p 8000:8000 -e CUOPT_SERVER_PORT=8000 nvidia/cuopt:" + V_NEXT + ".0a-cu13-ubi10",
          },
        },
      },
    },
    /* Java has no pip/conda package; the Docker images already contain cuopt.jar. */
    java: {
      container: CONTAINER_CUOPT_LIB,
      /* Marker only -- getCommand() builds the actual pom.xml snippet, since it
         also depends on arch (classifier suffix), not just release/cuda. */
      maven: { stable: { cu12: true, cu13: true }, nightly: { cu12: true, cu13: true } },
    },
  };

  /* Single-component C API installs: libcuopt still ships everything standalone (pip) or
     depends on all three (conda), so these are for callers who want just one piece. Client
     links no CUDA/rmm, so it has one universal package, no cu12/cu13 split. */
  var LIBCUOPT_COMPONENTS = {
    client: {
      pip: {
        stable: { default: pipInstall("libcuopt-client", "", V, false) },
        nightly: { default: pipInstall("libcuopt-client", "", V_NEXT, true) },
      },
      conda: {
        stable: { default: condaInstall("libcuopt-client", V_CONDA, "", false) },
        nightly: { default: condaInstall("libcuopt-client", V_CONDA_NEXT, "", true) },
      },
    },
    mathopt: {
      pip: {
        stable: { cu12: pipInstall("libcuopt-mathopt", "cu12", V, false), cu13: pipInstall("libcuopt-mathopt", "cu13", V, false) },
        nightly: { cu12: pipInstall("libcuopt-mathopt", "cu12", V_NEXT, true), cu13: pipInstall("libcuopt-mathopt", "cu13", V_NEXT, true) },
      },
      conda: {
        stable: { cu12: condaInstall("libcuopt-mathopt", V_CONDA, "12.9", false), cu13: condaInstall("libcuopt-mathopt", V_CONDA, "13.0", false) },
        nightly: { cu12: condaInstall("libcuopt-mathopt", V_CONDA_NEXT, "12.9", true), cu13: condaInstall("libcuopt-mathopt", V_CONDA_NEXT, "13.0", true) },
      },
    },
    routing: {
      pip: {
        stable: { cu12: pipInstall("libcuopt-routing", "cu12", V, false), cu13: pipInstall("libcuopt-routing", "cu13", V, false) },
        nightly: { cu12: pipInstall("libcuopt-routing", "cu12", V_NEXT, true), cu13: pipInstall("libcuopt-routing", "cu13", V_NEXT, true) },
      },
      conda: {
        stable: { cu12: condaInstall("libcuopt-routing", V_CONDA, "12.9", false), cu13: condaInstall("libcuopt-routing", V_CONDA, "13.0", false) },
        nightly: { cu12: condaInstall("libcuopt-routing", V_CONDA_NEXT, "12.9", true), cu13: condaInstall("libcuopt-routing", V_CONDA_NEXT, "13.0", true) },
      },
    },
  };

  var SUPPORTED_METHODS = {
    python: ["pip", "conda", "container"],
    c: ["pip", "conda", "container"],
    server: ["pip", "conda", "container"],
    cli: ["pip", "conda", "container"],
    java: ["container", "maven"],
  };

  function getSelectedValue(name) {
    var el = document.querySelector('input[name="' + name + '"]:checked');
    return el ? el.value : "";
  }

  /* component is only meaningful for iface "c" + method pip/conda; "full" means the
     regular COMMANDS table, anything else looks up LIBCUOPT_COMPONENTS instead. */
  function resolveData(iface, method, component) {
    if (component && component !== "full") {
      return LIBCUOPT_COMPONENTS[component] && LIBCUOPT_COMPONENTS[component][method];
    }
    return COMMANDS[iface] && COMMANDS[iface][method];
  }

  function hasCudaVariants(data) {
    if (!data || !data.stable) return false;
    return !!(data.stable.cu12 && data.stable.cu13);
  }

  function getCommand() {
    var iface = getSelectedValue("cuopt-iface");
    var method = getSelectedValue("cuopt-method");
    var release = getSelectedValue("cuopt-release");
    var cuda = getSelectedValue("cuopt-cuda");
    var component = iface === "c" ? (getSelectedValue("cuopt-component") || "full") : "full";

    /* CLI uses libcuopt (c) install; cuopt_cli is shipped with libcuopt. */
    if (iface === "cli") {
      iface = "c";
      release = "stable";
      cuda = "cu12";
      component = "full";
    }
    if (method === "container") component = "full";

    var data = resolveData(iface, method, component);
    if (!data || !data[release]) return "";

    var cmd = "";
    if (method === "container") {
      var variant = getSelectedValue("cuopt-variant") || "ubuntu";
      var baseCuda = cuda || "cu12";
      var cudaKey = baseCuda + (variant === "ubi10" ? "-ubi10" : "");
      var fallbackKey = (variant === "ubi10") ? "cu13-ubi10" : "cu12";
      var c = data[release][cudaKey] || data[release][fallbackKey];
      var hubPull = c.default;
      var tag = "latest-cu12";
      var tm = hubPull.match(/docker pull nvidia\/cuopt:(\S+)/);
      if (tm) tag = tm[1];
      var registry = getSelectedValue("cuopt-registry") || "hub";
      var runLine = c.run;
      if (registry === "ngc" && release === "nightly") {
        cmd =
          "# Nightly cuOpt container images are not published to NVIDIA NGC; use Docker Hub for nightly builds.\n" +
          "# (Select \"Docker Hub\" above for the same commands without this note.)\n\n" +
          "# Docker Hub (docker.io) — no registry login required for public pulls\n" +
          hubPull +
          "\n\n" +
          "# Run the container:\n" +
          runLine;
      } else if (registry === "ngc") {
        runLine = runLine.replace(/nvidia\/cuopt:/g, "nvcr.io/nvidia/cuopt/cuopt:");
        cmd =
          "# NVIDIA NGC (nvcr.io) — authenticate once per session, then pull:\n" +
          "docker login nvcr.io\n" +
          "# Username: $oauthtoken\n" +
          "# Password: <NGC API key>\n\n" +
          "docker pull nvcr.io/nvidia/cuopt/cuopt:" +
          tag +
          "\n\n" +
          "# Run the container:\n" +
          runLine;
      } else {
        cmd =
          "# Docker Hub (docker.io) — no registry login required for public pulls\n" +
          hubPull +
          "\n\n" +
          "# Run the container:\n" +
          runLine;
      }
    } else if (method === "maven") {
      var mvnVersion = release === "nightly" ? V_NEXT + ".0-SNAPSHOT" : V;
      var arch = getSelectedValue("cuopt-arch") || "amd64";
      var classifier = (cuda || "cu12").replace("cu", "cuda") + (arch === "arm64" ? "-arm64" : "");
      var repoBlock =
        release === "nightly"
          ? "<repositories>\n" +
            "  <repository>\n" +
            "    <id>sonatype-snapshots</id>\n" +
            "    <url>https://central.sonatype.com/repository/maven-snapshots</url>\n" +
            "    <releases><enabled>false</enabled></releases>\n" +
            "    <snapshots><enabled>true</enabled></snapshots>\n" +
            "  </repository>\n" +
            "</repositories>\n\n"
          : "";
      cmd =
        repoBlock +
        "<dependency>\n" +
        "  <groupId>com.nvidia</groupId>\n" +
        "  <artifactId>cuopt</artifactId>\n" +
        "  <version>" + mvnVersion + "</version>\n" +
        "  <classifier>" + classifier + "</classifier>\n" +
        "</dependency>";
    } else {
      var key = data[release].cu12 && data[release].cu13 ? cuda : "default";
      cmd = data[release][key] || data[release].cu12 || data[release].cu13 || data[release].default || "";
    }
    return cmd;
  }

  function updateOutput() {
    var out = document.getElementById("cuopt-cmd-out");
    var copyBtn = document.getElementById("cuopt-copy-btn");
    var cmd = getCommand();
    out.value = cmd;
    out.style.display = cmd ? "block" : "none";
    copyBtn.style.display = cmd ? "inline-flex" : "none";
  }

  var lastMethod = "";

  function updateVisibility() {
    var method = getSelectedValue("cuopt-method");
    var iface = getSelectedValue("cuopt-iface");
    var allowed = SUPPORTED_METHODS[iface] || [];

    /* The published Maven jar's own unclassified default is cuda13 (see
       assemble_maven_repo.sh); default the CUDA radio to match on entering
       this method, without overriding an explicit choice made while still in it. */
    if (method === "maven" && lastMethod !== "maven") {
      var cu13 = document.querySelector('input[name="cuopt-cuda"][value="cu13"]');
      if (cu13) cu13.checked = true;
    }
    lastMethod = method;
    var methodInputs = document.querySelectorAll('input[name="cuopt-method"]');
    methodInputs.forEach(function (input) {
      var enabled = allowed.indexOf(input.value) !== -1;
      input.disabled = !enabled;
      var label = input.closest("label");
      if (label) label.style.display = enabled ? "" : "none";
    });
    if (allowed.indexOf(method) === -1 && allowed.length) {
      var fallback = document.querySelector('input[name="cuopt-method"][value="' + allowed[0] + '"]');
      if (fallback) {
        fallback.checked = true;
        method = allowed[0];
      }
    }
    var cudaRow = document.getElementById("cuopt-cuda-row");
    var releaseRow = document.getElementById("cuopt-release-row");
    var releaseVisible = iface !== "cli";
    var ifaceForVariants = iface === "cli" ? "c" : iface;
    var component = iface === "c" && method !== "container" ? (getSelectedValue("cuopt-component") || "full") : "full";
    var showCuda =
      releaseVisible &&
      (method === "pip" || method === "conda" || method === "container" || method === "maven") &&
      hasCudaVariants(resolveData(ifaceForVariants, method, component));
    cudaRow.style.display = showCuda ? "table-row" : "none";
    releaseRow.style.display = releaseVisible ? "table-row" : "none";
    var variantRow = document.getElementById("cuopt-variant-row");
    if (variantRow) {
      variantRow.style.display = method === "container" ? "table-row" : "none";
    }
    var registryRow = document.getElementById("cuopt-registry-row");
    if (registryRow) {
      registryRow.style.display = method === "container" ? "table-row" : "none";
    }
    var archRow = document.getElementById("cuopt-arch-row");
    if (archRow) {
      archRow.style.display = iface === "java" && method === "maven" ? "table-row" : "none";
    }
    var componentRow = document.getElementById("cuopt-component-row");
    if (componentRow) {
      componentRow.style.display = iface === "c" && (method === "pip" || method === "conda") ? "table-row" : "none";
    }
    updateOutput();
  }

  function copyToClipboard() {
    var out = document.getElementById("cuopt-cmd-out");
    if (!out.value) return;
    out.select();
    out.setSelectionRange(0, 99999);
    try {
      document.execCommand("copy");
      var btn = document.getElementById("cuopt-copy-btn");
      var orig = btn.textContent;
      btn.textContent = "Copied!";
      setTimeout(function () {
        btn.textContent = orig;
      }, 1500);
    } catch (e) {}
  }

  function render() {
    var root = document.getElementById("cuopt-install-selector");
    if (!root) return;

    root.innerHTML =
      '<div class="cuopt-install-selector-wrap">' +
      '<table class="docutils cuopt-install-selector-table">' +
      '<tr><td class="cuopt-opt-label">Interface</td><td class="cuopt-opt-group" role="group" aria-label="Interface">' +
      '<label class="cuopt-opt"><input type="radio" name="cuopt-iface" value="python" checked> Python (cuopt)</label>' +
      '<label class="cuopt-opt"><input type="radio" name="cuopt-iface" value="c"> C (libcuopt)</label>' +
      '<label class="cuopt-opt"><input type="radio" name="cuopt-iface" value="server"> Server</label>' +
      '<label class="cuopt-opt"><input type="radio" name="cuopt-iface" value="cli"> CLI (cuopt_cli)</label>' +
      '<label class="cuopt-opt"><input type="radio" name="cuopt-iface" value="java"> Java (experimental)</label>' +
      '</td></tr>' +
      '<tr><td class="cuopt-opt-label">Method</td><td class="cuopt-opt-group" role="group" aria-label="Method">' +
      '<label class="cuopt-opt"><input type="radio" name="cuopt-method" value="pip" checked> pip</label>' +
      '<label class="cuopt-opt"><input type="radio" name="cuopt-method" value="conda"> Conda</label>' +
      '<label class="cuopt-opt"><input type="radio" name="cuopt-method" value="container"> Container</label>' +
      '<label class="cuopt-opt"><input type="radio" name="cuopt-method" value="maven"> Maven</label>' +
      '</td></tr>' +
      '<tr id="cuopt-component-row" style="display:none;"><td class="cuopt-opt-label">Component</td><td class="cuopt-opt-group" role="group" aria-label="Component">' +
      '<label class="cuopt-opt"><input type="radio" name="cuopt-component" value="full" checked> Full library</label>' +
      '<label class="cuopt-opt"><input type="radio" name="cuopt-component" value="client"> Client only</label>' +
      '<label class="cuopt-opt"><input type="radio" name="cuopt-component" value="mathopt"> Mathopt only</label>' +
      '<label class="cuopt-opt"><input type="radio" name="cuopt-component" value="routing"> Routing only</label>' +
      '</td></tr>' +
      '<tr id="cuopt-release-row"><td class="cuopt-opt-label">Release</td><td class="cuopt-opt-group" role="group" aria-label="Release">' +
      '<label class="cuopt-opt"><input type="radio" name="cuopt-release" value="stable" checked> Current release (' + V_CONDA + ')</label>' +
      '<label class="cuopt-opt"><input type="radio" name="cuopt-release" value="nightly"> Nightly (' + V_CONDA_NEXT + ')</label>' +
      '</td></tr>' +
      '<tr id="cuopt-cuda-row"><td class="cuopt-opt-label">CUDA</td><td class="cuopt-opt-group" role="group" aria-label="CUDA">' +
      '<label class="cuopt-opt"><input type="radio" name="cuopt-cuda" value="cu12" checked> 12.x</label>' +
      '<label class="cuopt-opt"><input type="radio" name="cuopt-cuda" value="cu13"> 13.x</label>' +
      '</td></tr>' +
      '<tr id="cuopt-variant-row" style="display:none;"><td class="cuopt-opt-label">Variant</td><td class="cuopt-opt-group" role="group" aria-label="Container variant">' +
      '<label class="cuopt-opt"><input type="radio" name="cuopt-variant" value="ubuntu" checked> Ubuntu</label>' +
      '<label class="cuopt-opt"><input type="radio" name="cuopt-variant" value="ubi10"> UBI10 (FIPS 140-3)</label>' +
      '</td></tr>' +
      '<tr id="cuopt-registry-row" style="display:none;"><td class="cuopt-opt-label">Registry</td><td class="cuopt-opt-group" role="group" aria-label="Container registry">' +
      '<label class="cuopt-opt"><input type="radio" name="cuopt-registry" value="hub" checked> Docker Hub</label>' +
      '<label class="cuopt-opt"><input type="radio" name="cuopt-registry" value="ngc"> NVIDIA NGC</label>' +
      '</td></tr>' +
      '<tr id="cuopt-arch-row" style="display:none;"><td class="cuopt-opt-label">Arch</td><td class="cuopt-opt-group" role="group" aria-label="Architecture">' +
      '<label class="cuopt-opt"><input type="radio" name="cuopt-arch" value="amd64" checked> amd64</label>' +
      '<label class="cuopt-opt"><input type="radio" name="cuopt-arch" value="arm64"> arm64</label>' +
      '</td></tr>' +
      "</table>" +
      '<div class="cuopt-install-output">' +
      '<textarea id="cuopt-cmd-out" class="cuopt-install-cmd-out" readonly rows="6" style="display:none;"></textarea>' +
      '<div class="cuopt-install-copy-wrap"><button type="button" id="cuopt-copy-btn" class="cuopt-install-copy-btn" style="display:none;">Copy command</button></div>' +
      "</div></div>";

    ["cuopt-iface", "cuopt-method", "cuopt-release", "cuopt-cuda", "cuopt-variant", "cuopt-registry", "cuopt-arch", "cuopt-component"].forEach(
      function (name) {
        var inputs = document.querySelectorAll('input[name="' + name + '"]');
        inputs.forEach(function (input) {
          input.addEventListener("change", updateVisibility);
        });
      }
    );
    document.getElementById("cuopt-copy-btn").addEventListener("click", copyToClipboard);
    updateVisibility();

    var defaultIface = root.getAttribute("data-default-iface");
    if (defaultIface && ["python", "c", "server", "cli", "java"].indexOf(defaultIface) !== -1) {
      var radio = document.querySelector('input[name="cuopt-iface"][value="' + defaultIface + '"]');
      if (radio) {
        radio.checked = true;
        updateVisibility();
      }
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", render);
  } else {
    render();
  }
})();
