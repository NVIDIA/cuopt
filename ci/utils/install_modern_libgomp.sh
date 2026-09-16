#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Fetch a modern GNU libgomp from conda-forge (Rocky Linux 8's own is too old for OpenMP 5.0
# detached tasks) and leave it at <dest_dir>/libgomp.so.1.0.0 + libgomp.so.1 symlink. Fetched
# directly over HTTPS (.conda is just a zip) since this image has no conda/mamba CLI.
# See https://github.com/NVIDIA/cuopt/issues/1219

set -euo pipefail

dest_dir="${1:?Usage: install_modern_libgomp.sh <dest_dir>}"
mkdir -p "${dest_dir}"

case "$(arch)" in
    x86_64) subdir="linux-64" ;;
    aarch64) subdir="linux-aarch64" ;;
    *)
        echo "Unsupported architecture for modern libgomp fetch: $(arch)" >&2
        exit 1
        ;;
esac

# The floor lives in dependencies.yaml (also used for conda/recipes/libcuopt/recipe.yaml), read
# from there so it can't drift from the version resolved below.
deps_yaml="$(dirname "${BASH_SOURCE[0]}")/../../dependencies.yaml"
min_major="$(grep -oP '(?<=- libgomp >=)[0-9]+' "${deps_yaml}" | head -1)"
if [[ -z "${min_major}" ]]; then
    echo "Could not find a 'libgomp >=N' floor in ${deps_yaml}" >&2
    exit 1
fi

python3 -m pip install --quiet zstandard

# Resolve the newest libgomp build for this subdir/floor from conda-forge's own package metadata
# (not the download itself, so the sha256 check below is a real integrity check -- CWE-494).
read -r pkg sha256 <<< "$(python3 - "${subdir}" "${min_major}" <<'PYEOF'
import json
import sys
import urllib.request

subdir, min_major = sys.argv[1], int(sys.argv[2])
with urllib.request.urlopen("https://api.anaconda.org/package/conda-forge/libgomp", timeout=15) as resp:
    data = json.load(resp)

candidates = [
    f for f in data["files"]
    if f["attrs"].get("subdir") == subdir
    and f["basename"].endswith(".conda")
    and f["sha256"]
    and int(f["version"].split(".")[0]) >= min_major
]
if not candidates:
    sys.exit(f"No libgomp >={min_major} build found for {subdir}")

best = max(candidates, key=lambda f: (tuple(map(int, f["version"].split("."))), f["attrs"]["timestamp"]))
print(best["basename"], best["sha256"])
PYEOF
)"

url="https://conda.anaconda.org/conda-forge/${pkg}"
pkg_file="$(basename "${pkg}")"

workdir="$(mktemp -d)"
trap 'rm -rf "${workdir}"' EXIT

echo "Fetching ${url}"
curl -fsSL -o "${workdir}/${pkg_file}" "${url}"

echo "${sha256}  ${workdir}/${pkg_file}" | sha256sum -c -

python3 - "${workdir}/${pkg_file}" "${workdir}/extracted" <<'PYEOF'
import io
import sys
import tarfile
import zipfile

import zstandard

pkg_path, out_dir = sys.argv[1], sys.argv[2]
zf = zipfile.ZipFile(pkg_path)
pkg_name = next(n for n in zf.namelist() if n.startswith("pkg-"))
raw = zf.read(pkg_name)
tar_bytes = zstandard.ZstdDecompressor().decompress(raw, max_output_size=200 * 1024 * 1024)
tarfile.open(fileobj=io.BytesIO(tar_bytes)).extractall(out_dir)
PYEOF

libgomp_so="$(find "${workdir}/extracted" -name 'libgomp.so.1.0.0' | head -1)"
if [[ ! -f "${libgomp_so}" ]]; then
    echo "Could not find libgomp.so.1.0.0 in ${pkg_file}" >&2
    exit 1
fi

# Verify it actually exports what we need rather than trusting the download/version alone.
if ! nm -D "${libgomp_so}" 2>/dev/null | grep -qE ' T omp_fulfill_event(@|$)'; then
    echo "Fetched libgomp does not export omp_fulfill_event -- wrong package or bad extraction" >&2
    exit 1
fi

cp "${libgomp_so}" "${dest_dir}/libgomp.so.1.0.0"
ln -sf libgomp.so.1.0.0 "${dest_dir}/libgomp.so.1"

echo "Modern libgomp ready at ${dest_dir}/libgomp.so.1.0.0 (from ${pkg_file})"
