/**
 * cuOpt install selector - generates install commands from user choices.
 * Stable version comes from window.CUOPT_INSTALL_VERSION (injected at build from cuopt.__version__).
 * Next = stable + 2 on minor (e.g. 26.04 -> 26.06). Update COMMANDS structure when commands change.
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
  var V_CONDA_NEXT = major + "." + (nextMinor < 10 ? "0" : "") + nextMinor;
  var V_NEXT = major + "." + nextMinor;

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
            ".* cuda-version=" +
            V_CONDA +
            ".*",
          cu13:
            "conda install -c rapidsai -c conda-forge -c nvidia cuopt=" +
            V_CONDA +
            ".* cuda-version=" +
            V_CONDA +
            ".*",
        },
        nightly: {
          cu12:
            "conda install -c rapidsai-nightly -c conda-forge -c nvidia cuopt=" +
            V_CONDA_NEXT +
            ".* cuda-version=" +
            V_CONDA_NEXT +
            ".*",
          cu13:
            "conda install -c rapidsai-nightly -c conda-forge -c nvidia cuopt=" +
            V_CONDA_NEXT +
            ".* cuda-version=" +
            V_CONDA_NEXT +
            ".*",
        },
      },
      container: {
        stable: {
          cu12: {
            default: "docker pull nvidia/cuopt:latest-cuda12.9-py3.13",
            run: "docker run --gpus all -it --rm nvidia/cuopt:latest-cuda12.9-py3.13 /bin/bash",
          },
          cu13: {
            default: "docker pull nvidia/cuopt:latest-cuda13.0-py3.13",
            run: "docker run --gpus all -it --rm nvidia/cuopt:latest-cuda13.0-py3.13 /bin/bash",
          },
        },
        nightly: {
          cu12: {
            default: "docker pull nvidia/cuopt:" + V_NEXT + ".0a-cuda12.9-py3.13",
            run: "docker run --gpus all -it --rm nvidia/cuopt:" + V_NEXT + ".0a-cuda12.9-py3.13 /bin/bash",
          },
          cu13: {
            default: "docker pull nvidia/cuopt:" + V_NEXT + ".0a-cuda13.0-py3.13",
            run: "docker run --gpus all -it --rm nvidia/cuopt:" + V_NEXT + ".0a-cuda13.0-py3.13 /bin/bash",
          },
        },
      },
    },
    c: {
      pip: {
        stable: {
          cu12:
            "pip uninstall -y cuopt-thin-client 2>/dev/null; pip install --extra-index-url=https://pypi.nvidia.com 'libcuopt-cu12==" +
            V +
            ".*'",
          cu13:
            "pip uninstall -y cuopt-thin-client 2>/dev/null; pip install --extra-index-url=https://pypi.nvidia.com 'libcuopt-cu13==" +
            V +
            ".*'",
        },
        nightly: {
          cu12:
            "pip uninstall -y cuopt-thin-client 2>/dev/null; pip install --pre --extra-index-url=https://pypi.nvidia.com --extra-index-url=https://pypi.anaconda.org/rapidsai-wheels-nightly/simple/ 'libcuopt-cu12==" +
            V_NEXT +
            ".*'",
          cu13:
            "pip uninstall -y cuopt-thin-client 2>/dev/null; pip install --pre --extra-index-url=https://pypi.nvidia.com --extra-index-url=https://pypi.anaconda.org/rapidsai-wheels-nightly/simple/ 'libcuopt-cu13==" +
            V_NEXT +
            ".*'",
        },
      },
      conda: {
        stable: {
          cu12:
            "conda remove cuopt-thin-client --yes 2>/dev/null; conda install -c rapidsai -c conda-forge -c nvidia libcuopt=" +
            V_CONDA +
            ".* cuda-version=" +
            V_CONDA +
            ".*",
          cu13:
            "conda remove cuopt-thin-client --yes 2>/dev/null; conda install -c rapidsai -c conda-forge -c nvidia libcuopt=" +
            V_CONDA +
            ".* cuda-version=" +
            V_CONDA +
            ".*",
        },
        nightly: {
          cu12:
            "conda install -c rapidsai-nightly -c conda-forge -c nvidia libcuopt=" +
            V_CONDA_NEXT +
            ".* cuda-version=" +
            V_CONDA_NEXT +
            ".*",
          cu13:
            "conda install -c rapidsai-nightly -c conda-forge -c nvidia libcuopt=" +
            V_CONDA_NEXT +
            ".* cuda-version=" +
            V_CONDA_NEXT +
            ".*",
        },
      },
      container: null,
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
            default: "docker pull nvidia/cuopt:latest-cuda12.9-py3.13",
            run: "docker run --gpus all -it --rm -p 8000:8000 -e CUOPT_SERVER_PORT=8000 nvidia/cuopt:latest-cuda12.9-py3.13",
          },
          cu13: {
            default: "docker pull nvidia/cuopt:latest-cuda13.0-py3.13",
            run: "docker run --gpus all -it --rm -p 8000:8000 -e CUOPT_SERVER_PORT=8000 nvidia/cuopt:latest-cuda13.0-py3.13",
          },
        },
        nightly: {
          cu12: {
            default: "docker pull nvidia/cuopt:" + V_NEXT + ".0a-cuda12.9-py3.13",
            run: "docker run --gpus all -it --rm -p 8000:8000 -e CUOPT_SERVER_PORT=8000 nvidia/cuopt:" + V_NEXT + ".0a-cuda12.9-py3.13",
          },
          cu13: {
            default: "docker pull nvidia/cuopt:" + V_NEXT + ".0a-cuda13.0-py3.13",
            run: "docker run --gpus all -it --rm -p 8000:8000 -e CUOPT_SERVER_PORT=8000 nvidia/cuopt:" + V_NEXT + ".0a-cuda13.0-py3.13",
          },
        },
      },
    },
    cli: {
      pip: null,
      conda: null,
      container: null,
      note: "cuopt_cli is installed with the C API (libcuopt). Install libcuopt via pip or Conda above, then use cuopt_cli.",
    },
  };

  function getSelectedValue(name) {
    var el = document.querySelector('input[name="' + name + '"]:checked');
    return el ? el.value : "";
  }

  function getCommand() {
    var iface = getSelectedValue("cuopt-iface");
    var method = getSelectedValue("cuopt-method");
    var release = getSelectedValue("cuopt-release");
    var cuda = getSelectedValue("cuopt-cuda");

    if (iface === "cli") {
      return COMMANDS.cli.note;
    }

    var data = COMMANDS[iface] && COMMANDS[iface][method];
    if (!data || !data[release]) return "";

    var cmd = "";
    if (method === "container") {
      var cudaKey = cuda || "cu12";
      var c = data[release][cudaKey] || data[release].cu12;
      cmd = c.default + "\n\n# Run the container:\n" + c.run;
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
    copyBtn.style.display = cmd && cmd !== COMMANDS.cli.note ? "inline-flex" : "none";
  }

  function updateVisibility() {
    var method = getSelectedValue("cuopt-method");
    var iface = getSelectedValue("cuopt-iface");
    var cudaRow = document.getElementById("cuopt-cuda-row");
    var releaseRow = document.getElementById("cuopt-release-row");

    var showCuda = (method === "pip" || method === "conda" || method === "container") && iface !== "cli";
    cudaRow.style.display = showCuda ? "table-row" : "none";
    releaseRow.style.display = iface !== "cli" ? "table-row" : "none";
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
      '</td></tr>' +
      '<tr><td class="cuopt-opt-label">Method</td><td class="cuopt-opt-group" role="group" aria-label="Method">' +
      '<label class="cuopt-opt"><input type="radio" name="cuopt-method" value="pip" checked> pip</label>' +
      '<label class="cuopt-opt"><input type="radio" name="cuopt-method" value="conda"> Conda</label>' +
      '<label class="cuopt-opt"><input type="radio" name="cuopt-method" value="container"> Container</label>' +
      '</td></tr>' +
      '<tr id="cuopt-release-row"><td class="cuopt-opt-label">Release</td><td class="cuopt-opt-group" role="group" aria-label="Release">' +
      '<label class="cuopt-opt"><input type="radio" name="cuopt-release" value="stable" checked> Current release (' + V_CONDA + ')</label>' +
      '<label class="cuopt-opt"><input type="radio" name="cuopt-release" value="nightly"> Nightly (' + V_CONDA_NEXT + ')</label>' +
      '</td></tr>' +
      '<tr id="cuopt-cuda-row"><td class="cuopt-opt-label">CUDA</td><td class="cuopt-opt-group" role="group" aria-label="CUDA">' +
      '<label class="cuopt-opt"><input type="radio" name="cuopt-cuda" value="cu12" checked> 12.x</label>' +
      '<label class="cuopt-opt"><input type="radio" name="cuopt-cuda" value="cu13"> 13.x</label>' +
      '</td></tr>' +
      "</table>" +
      '<div class="cuopt-install-output">' +
      '<textarea id="cuopt-cmd-out" class="cuopt-install-cmd-out" readonly rows="6" style="display:none;"></textarea>' +
      '<div class="cuopt-install-copy-wrap"><button type="button" id="cuopt-copy-btn" class="cuopt-install-copy-btn" style="display:none;">Copy command</button></div>' +
      "</div></div>";

    ["cuopt-iface", "cuopt-method", "cuopt-release", "cuopt-cuda"].forEach(
      function (name) {
        var inputs = document.querySelectorAll('input[name="' + name + '"]');
        inputs.forEach(function (input) {
          input.addEventListener("change", updateVisibility);
        });
      }
    );
    document.getElementById("cuopt-copy-btn").addEventListener("click", copyToClipboard);
    updateVisibility();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", render);
  } else {
    render();
  }
})();
