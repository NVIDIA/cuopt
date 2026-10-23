#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

download() {
  local dir=$1
  local start=${SECONDS}
  pushd "${dir}/linear_programming" >/dev/null
  ./download_pdlp_test_dataset.sh >/dev/null
  popd >/dev/null
  local pdlp=$((SECONDS - start))
  start=${SECONDS}
  ./"${dir}"/mip/download_miplib_test_dataset.sh >/dev/null
  local miplib=$((SECONDS - start))
  start=${SECONDS}
  pushd "${dir}" >/dev/null
  ./get_test_data.sh >/dev/null
  popd >/dev/null
  local routing=$((SECONDS - start))
  echo "RESULT ${dir}: pdlp=${pdlp}s miplib=${miplib}s routing=${routing}s total=$((pdlp + miplib + routing))s"
}

rapids-logger "Prepare dataset directories"
cp -r datasets datasets_files
cp -r datasets datasets_tarball

rapids-logger "Download file by file"
export CUOPT_S3_NO_TARBALL=1
download datasets_files

rapids-logger "Download from tarballs"
unset CUOPT_S3_NO_TARBALL
download datasets_tarball

rapids-logger "Compare downloaded files"
for d in linear_programming mip acvrp cvrptw pdptw solomon; do
  [ -d "datasets_files/${d}" ] || [ -d "datasets_tarball/${d}" ] || continue
  diff -rq "datasets_files/${d}" "datasets_tarball/${d}"
done
