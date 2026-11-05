#!/bin/bash
# TEMP: WSL diagnostics; reuses wheel artifacts from run 41705297201 (b2bd214b) to skip builds.
set -euo pipefail
export GITHUB_RUN_ID=41705297201
DIAG_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"

source rapids-init-pip
bash ./ci/utils/install_openssl3_runtime.sh
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"
LIBCUOPT_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_cpp libcuopt cuopt --cuda "$RAPIDS_CUDA_VERSION")")
CUOPT_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_python cuopt cuopt --stable --cuda "$RAPIDS_CUDA_VERSION")")
CUOPT_SH_CLIENT_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_python cuopt-sh-client cuopt --pure --arch any)")
LIBCUOPT_CLIENT_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_cpp libcuopt_client cuopt)")
LIBCUOPT_MATHOPT_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_cpp libcuopt_mathopt cuopt --cuda "$RAPIDS_CUDA_VERSION")")
LIBCUOPT_ROUTING_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_cpp libcuopt_routing cuopt --cuda "$RAPIDS_CUDA_VERSION")")
cat > "${PIP_CONSTRAINT}" <<EOF
cuopt-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${CUOPT_WHEELHOUSE}"/cuopt_"${RAPIDS_PY_CUDA_SUFFIX}"-*.whl)
cuopt-sh-client @ file://$(echo "${CUOPT_SH_CLIENT_WHEELHOUSE}"/cuopt_sh_client-*.whl)
libcuopt-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${LIBCUOPT_WHEELHOUSE}"/libcuopt_"${RAPIDS_PY_CUDA_SUFFIX}"-*.whl)
libcuopt-client @ file://$(echo "${LIBCUOPT_CLIENT_WHEELHOUSE}"/libcuopt_client-*.whl)
libcuopt-mathopt-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${LIBCUOPT_MATHOPT_WHEELHOUSE}"/libcuopt_mathopt_*.whl)
libcuopt-routing-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${LIBCUOPT_ROUTING_WHEELHOUSE}"/libcuopt_routing_*.whl)
EOF
rapids-generate-pip-constraints test_python "${PIP_CONSTRAINT}"
rapids-pip-retry install \
    --prefer-binary \
    --constraint "${PIP_CONSTRAINT}" \
    "$(echo "${CUOPT_WHEELHOUSE}"/cuopt*.whl)[test]" \
    "${CUOPT_SH_CLIENT_WHEELHOUSE}"/cuopt_sh_client*.whl \
    "${LIBCUOPT_WHEELHOUSE}"/libcuopt*.whl

rapids-logger "System"
cat /proc/version
nproc
free -g
nvidia-smi
nvidia-smi -q | grep -iE -A3 "driver model|virtualization mode|performance state" || true

rapids-logger "Latency probes (cudaSetDeviceFlags 0=auto 1=spin 2=yield 4=blocking)"
for f in 0 1 2 4; do python -I "${DIAG_DIR}/latency.py" "$f" || true; done

RAPIDS_DATASET_ROOT_DIR="$(realpath datasets)"
export RAPIDS_DATASET_ROOT_DIR
cd python/cuopt/cuopt
for i in 1 2 3; do
  rapids-logger "test_model run ${i}"
  pytest -s -p no:cacheprovider tests/linear_programming/test_python_API.py::test_model --durations=0 2>&1 \
    | grep -vE '^(Objective|Constraint|Variable) ' | tail -60 || true
done
exit 0
