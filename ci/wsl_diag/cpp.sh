#!/bin/bash
# TEMP: WSL diagnostics; reuses conda artifacts from run 41705297201 (b2bd214b) to skip builds.
set -euo pipefail
export GITHUB_RUN_ID=41705297201
DIAG_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"

. /opt/conda/etc/profile.d/conda.sh
conda config --set channel_priority strict
CPP_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_cpp libcuopt cuopt --cuda "$RAPIDS_CUDA_VERSION")")
rapids-dependency-file-generator \
  --output conda \
  --file-key test_cpp \
  --prepend-channel "${CPP_CHANNEL}" \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch)" | tee env.yaml
rapids-mamba-retry env create --yes -f env.yaml -n test --channel "${CPP_CHANNEL}"
set +u
conda activate test
set -u

rapids-logger "System"
cat /proc/version
nproc
free -g
df -h /dev/shm
nvidia-smi
nvidia-smi -q | grep -iE -A3 "driver model|virtualization mode|performance state|clocks event reasons|^    Clocks$" || true

rapids-logger "Latency probe"
python -I "${DIAG_DIR}/latency.py" 0 || true
set +e

./datasets/mip/download_miplib_test_dataset.sh > /dev/null
./datasets/linear_programming/download_pdlp_test_dataset.sh > /dev/null
RAPIDS_DATASET_ROOT_DIR="$(realpath datasets)"
export RAPIDS_DATASET_ROOT_DIR
pushd "${RAPIDS_DATASET_ROOT_DIR}" > /dev/null
./get_test_data.sh > /dev/null 2>&1
popd > /dev/null

G="${CONDA_PREFIX}/bin/gtests/libcuopt"
run() {
  rapids-logger "$*"
  local t0=$SECONDS
  "$@" 2>&1 | grep -E '^\[ *(RUN|OK|FAILED|PASSED) *\]|Failure|Which is|actual|presolve time|scaling completed|thread budget|Time limit|NCCL (WARN|INFO)|ncclComm|CUDA|error' || true
  echo "elapsed: $((SECONDS - t0)) s"
}
run "${G}/NUMOPT_INTERNAL_TEST" --gtest_filter='termination_status.gf2*:termination_status.crossing*:cuts.test_cuts_1:Population.ExternalQueueDrain*'
run "${G}/ROUTING_UNIT_TEST" --gtest_filter='prize_collection.*'
NCCL_DEBUG=INFO run "${G}/C_API_TEST" --gtest_filter='c_api.multi_gpu*'
NCCL_DEBUG=WARN NCCL_CUMEM_ENABLE=0 run "${G}/C_API_TEST" --gtest_filter='c_api.multi_gpu*'
NCCL_DEBUG=WARN NCCL_CUMEM_ENABLE=0 NCCL_SHM_DISABLE=1 run "${G}/C_API_TEST" --gtest_filter='c_api.multi_gpu*'
NCCL_DEBUG=WARN NCCL_RUNTIME_CONNECT=0 run "${G}/C_API_TEST" --gtest_filter='c_api.multi_gpu*'
run "${G}/INCUMBENT_CALLBACK_TEST" --gtest_filter='mip_solve.incumbent_get_set_callback_test'
exit 0
