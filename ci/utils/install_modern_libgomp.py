#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Fetch a modern GNU libgomp from conda-forge for wheel builds.

Rocky Linux 8's own libgomp is too old for OpenMP 5.0 detached tasks
(omp_fulfill_event), so this resolves and fetches a newer build
directly from conda-forge's repodata; no conda/mamba CLI needed,
since .conda packages are just a zip containing a zstd-compressed tar.
See https://github.com/NVIDIA/cuopt/issues/1219
"""

import hashlib
import io
import json
import platform
import re
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

import zstandard

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPENDENCIES_YAML = REPO_ROOT / "dependencies.yaml"

ARCH_SUBDIRS = {"x86_64": "linux-64", "aarch64": "linux-aarch64"}


def min_libgomp_major():
    """Read the libgomp version floor from dependencies.yaml.

    Also used for conda/recipes/libcuopt/recipe.yaml, read from
    there so this can't drift from the version resolved below.
    """
    match = re.search(r"- libgomp >=(\d+)", DEPENDENCIES_YAML.read_text())
    if not match:
        sys.exit(
            f"Could not find a 'libgomp >=N' floor in {DEPENDENCIES_YAML}"
        )
    return int(match.group(1))


def curl(url, dest):
    """Download url to dest via curl.

    Not urllib: this pyenv-built Python's default SSL context fails
    TLS verification against conda.anaconda.org in the Rocky 8
    wheel-build image, even though curl (using the system's own
    trust store) resolves the same URL fine there.
    """
    print(f"Fetching {url}")
    subprocess.run(["curl", "-fsSL", "-o", str(dest), url], check=True)


def resolve_build(subdir, min_major, workdir):
    """Find the newest libgomp build satisfying the floor.

    Reads conda-forge's own repodata, the same host the package
    itself is downloaded from below.
    """
    repodata_path = workdir / "current_repodata.json"
    curl(
        f"https://conda.anaconda.org/conda-forge/{subdir}/current_repodata.json",
        repodata_path,
    )
    with open(repodata_path) as f:
        data = json.load(f)

    candidates = [
        (fn, v)
        for fn, v in data["packages.conda"].items()
        if v["name"] == "libgomp"
        and int(v["version"].split(".")[0]) >= min_major
    ]
    if not candidates:
        sys.exit(f"No libgomp >={min_major} build found for {subdir}")

    fn, v = max(
        candidates,
        key=lambda c: (
            tuple(map(int, c[1]["version"].split("."))),
            c[1]["timestamp"],
        ),
    )
    return fn, v["sha256"]


def download(subdir, pkg_file, sha256, dest):
    """Download pkg_file and verify it against conda-forge's own sha256.

    Verifying against the digest from repodata (not just the
    download itself) is a real integrity check (CWE-494).
    """
    curl(f"https://conda.anaconda.org/conda-forge/{subdir}/{pkg_file}", dest)

    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    if digest != sha256:
        sys.exit(
            f"sha256 mismatch for {pkg_file}: expected {sha256}, got {digest}"
        )


def extract_libgomp(pkg_path, out_dir):
    """Extract libgomp.so.1.0.0 out of a .conda package."""
    zf = zipfile.ZipFile(pkg_path)
    pkg_name = next(n for n in zf.namelist() if n.startswith("pkg-"))
    tar_bytes = zstandard.ZstdDecompressor().decompress(
        zf.read(pkg_name), max_output_size=200 * 1024 * 1024
    )
    tarfile.open(fileobj=io.BytesIO(tar_bytes)).extractall(out_dir)

    matches = list(Path(out_dir).rglob("libgomp.so.1.0.0"))
    if not matches:
        sys.exit(f"Could not find libgomp.so.1.0.0 in {pkg_path}")
    return matches[0]


def verify_symbol(libgomp_so):
    """Verify it exports what we need rather than trusting the pin alone."""
    result = subprocess.run(
        ["nm", "-D", str(libgomp_so)], capture_output=True, text=True
    )
    if not re.search(
        r" T omp_fulfill_event(@|$)", result.stdout, re.MULTILINE
    ):
        sys.exit(
            "Fetched libgomp does not export omp_fulfill_event: "
            "wrong package or bad extraction"
        )


def main():
    if len(sys.argv) != 2:
        sys.exit(f"Usage: {sys.argv[0]} <dest_dir>")
    dest_dir = Path(sys.argv[1])
    dest_dir.mkdir(parents=True, exist_ok=True)

    machine = platform.machine()
    subdir = ARCH_SUBDIRS.get(machine)
    if subdir is None:
        sys.exit(
            f"Unsupported architecture for modern libgomp fetch: {machine}"
        )

    # Ephemeral CI container, no need to clean this up ourselves.
    workdir = Path(tempfile.mkdtemp())

    min_major = min_libgomp_major()
    pkg_file, sha256 = resolve_build(subdir, min_major, workdir)

    pkg_path = workdir / pkg_file
    download(subdir, pkg_file, sha256, pkg_path)

    libgomp_so = extract_libgomp(pkg_path, workdir / "extracted")
    verify_symbol(libgomp_so)

    dest_file = dest_dir / "libgomp.so.1.0.0"
    dest_file.write_bytes(libgomp_so.read_bytes())
    (dest_dir / "libgomp.so.1").symlink_to("libgomp.so.1.0.0")

    print(f"Modern libgomp ready at {dest_file} (from {pkg_file})")


if __name__ == "__main__":
    main()
