#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

# Download the cuopt built in the previous step
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"
CUOPT_MPS_PARSER_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="cuopt_mps_parser" rapids-download-wheels-from-github python)
CUOPT_SERVER_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="cuopt_server_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github python)
CUOPT_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="cuopt_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github python)
LIBCUOPT_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="libcuopt_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)

# echo to expand wildcard before adding `[extra]` requires for pip
rapids-pip-retry install \
    --extra-index-url=https://pypi.nvidia.com \
    "${CUOPT_MPS_PARSER_WHEELHOUSE}"/cuopt_mps_parser*.whl \
    "${CUOPT_WHEELHOUSE}"/cuopt*.whl \
    "${LIBCUOPT_WHEELHOUSE}"/libcuopt*.whl \
    "$(echo "${CUOPT_SERVER_WHEELHOUSE}"/cuopt_server*.whl)[test]"

check_message()
{
    local counter=$1
    local output=$2
    local expected=$3
    if [[ "$output" == *"$expected"* ]]
    then
            rapids-logger "CLI test $counter :PASSED"
    else
            rapids-logger "CLI test $counter :FAILED"
            echo "$output"
            echo "$expected"
            exit 1
    fi
}

counter=0
run_cli_test()
{
    local expected="$1"
    shift 1
    cli_test=$("$@")
    counter=$((counter+1))
    check_message $counter "${cli_test}" "$expected"
}

rapids-logger "Running cuOpt Server"

export CUOPT_SERVER_PORT=8000
CUOPT_DATA_DIR=$(mktemp -d)
CUOPT_RESULT_DIR=$(mktemp -d)
export CUOPT_DATA_DIR
export CUOPT_RESULT_DIR

trap 'rm -rf "$CUOPT_DATA_DIR" "$CUOPT_RESULT_DIR"' EXIT
# cuopt_problem_data and other small problems should be less than 1k
export CUOPT_MAX_RESULT=1
CERT_FOLDER=$(pwd)/python/cuopt_self_hosted/cuopt_sh_client/tests/utils/certs
export CUOPT_SSL_CERTFILE=${CERT_FOLDER}/server.crt
export CUOPT_SSL_KEYFILE=${CERT_FOLDER}/server.key
export CLIENT_CERT=${CERT_FOLDER}/ca.crt
python -m cuopt_server.cuopt_service &
export SERVER_PID=$!

DELAY=10

sleep $DELAY

server_status=$(curl -k -sL https://0.0.0.0:$CUOPT_SERVER_PORT/cuopt/health)

EXITCODE=0

trap "EXITCODE=1" ERR
set +e

doservertest=0
if [[ $server_status == *"RUNNING"* ]]
then
    rapids-logger "Server is ready to accept request."
    doservertest=1
else
    rapids-logger "Server failed to run, test failed."
    EXITCODE=1
fi

if [ "$doservertest" -eq 1 ]; then
    rapids-logger "Testing cuOpt self-hosted server"
    rapids-logger "Running self-hosted cli tests"
    pushd python/cuopt_self_hosted
    pip install .

    # Success, small problem, data and result over http
    run_cli_test "'status': 0" cuopt_sh -s -c "$CLIENT_CERT" -p $CUOPT_SERVER_PORT ../../datasets/cuopt_service_data/cuopt_problem_data.json

    # Success, small problem, read from data dir, result comes back over http
    cp ../../datasets/cuopt_service_data/cuopt_problem_data.json "$CUOPT_DATA_DIR"
    run_cli_test "'status': 0" cuopt_sh -s -c "$CLIENT_CERT" -p $CUOPT_SERVER_PORT -f cuopt_problem_data.json

    # Success, small LP problem with pure JSON
    run_cli_test "'status': 'Optimal'" cuopt_sh -s -c "$CLIENT_CERT" -p $CUOPT_SERVER_PORT -t LP ../../datasets/cuopt_service_data/good_lp.json

    # Success, small MILP problem with pure JSON which returns a solution with Optimal status
    run_cli_test "'status': 'Optimal'" cuopt_sh -s -c $CLIENT_CERT -p $CUOPT_SERVER_PORT -t LP ../../datasets/mixed_integer_programming/milp_data.json

    # Succes, small LP problem with mps. Data will be transformed to JSON
    run_cli_test "'status': 'Optimal'" cuopt_sh -s -c "$CLIENT_CERT" -p $CUOPT_SERVER_PORT -t LP ../../datasets/linear_programming/good-mps-1.mps

    # Succes, small Batch LP problem with mps. Data will be transformed to JSON
    run_cli_test "'status': 'Optimal'" cuopt_sh -s -c "$CLIENT_CERT" -p $CUOPT_SERVER_PORT -t LP ../../datasets/linear_programming/good-mps-1.mps ../../datasets/linear_programming/good-mps-1.mps

    # Error, local file mode is not allowed with mps
    run_cli_test "Cannot use local file mode with MPS data" cuopt_sh -s -c "$CLIENT_CERT" -p $CUOPT_SERVER_PORT -t LP -f good-mps-1.mps

    # Just run validator
    cp ../../datasets/cuopt_service_data/cuopt_problem_data.json "$CUOPT_DATA_DIR"
    run_cli_test "'msg': 'Input is Valid'" cuopt_sh -ov -s -c "$CLIENT_CERT" -p $CUOPT_SERVER_PORT -f cuopt_problem_data.json

    # Success, pre-packed data
    run_cli_test "'status': 0" cuopt_sh -s -c "$CLIENT_CERT" -p $CUOPT_SERVER_PORT ../../datasets/cuopt_service_data/cuopt_problem_data.msgpack

    # Success, pre-pickled data
    run_cli_test "'status': 0" cuopt_sh -s -c "$CLIENT_CERT" -p $CUOPT_SERVER_PORT ../../datasets/cuopt_service_data/cuopt_problem_data.pickle

    # Success, pre-packed data in local file mode
    cp ../../datasets/cuopt_service_data/cuopt_problem_data.msgpack "$CUOPT_DATA_DIR"
    run_cli_test "'status': 0" cuopt_sh -s -c "$CLIENT_CERT" -p $CUOPT_SERVER_PORT -f cuopt_problem_data.msgpack

    # Success, pre-pickled data in local file mode
    cp ../../datasets/cuopt_service_data/cuopt_problem_data.pickle "$CUOPT_DATA_DIR"
    run_cli_test "'status': 0" cuopt_sh -s -c "$CLIENT_CERT" -p $CUOPT_SERVER_PORT -f cuopt_problem_data.pickle

    # Success, zlib compressed data in local file mode
    cp ../../datasets/cuopt_service_data/cuopt_problem_data.zlib "$CUOPT_DATA_DIR"
    run_cli_test "'status': 0" cuopt_sh -s -c "$CLIENT_CERT" -p $CUOPT_SERVER_PORT -f cuopt_problem_data.zlib

    # Test repoll message
    run_cli_test "Check for status with the following command" cuopt_sh -s -c "$CLIENT_CERT" -p "$CUOPT_SERVER_PORT" -pt 0 ../../datasets/cuopt_service_data/cuopt_problem_data.json -k

    # Get the last line of output from the last command
    requestid=$(echo "$cli_test" | tail -1)

    # Get request id and remove single quotes and spaces
    requestid=$(echo "${requestid#cuopt_sh }" | sed "s/-p $CUOPT_SERVER_PORT//g" | tr -d "'" | tr -d " ")

    # Test repoll on requestid
    run_cli_test "'status': 0" cuopt_sh -s -c "$CLIENT_CERT" -p "$CUOPT_SERVER_PORT" "$requestid" -k

    # Test initial solutions
    run_cli_test "'status': 0" cuopt_sh -s -c "$CLIENT_CERT" -p "$CUOPT_SERVER_PORT" ../../datasets/cuopt_service_data/cuopt_problem_data.json -id "$requestid" "$requestid"

    # Test repoll message and pdlp warmstart
    run_cli_test "Check for status with the following command" cuopt_sh -s -c "$CLIENT_CERT" -p "$CUOPT_SERVER_PORT" -pt 0 ../../datasets/cuopt_service_data/good_lp.json -k
    requestid=$(echo "$cli_test" | tail -1)
    requestid=$(echo ${requestid#cuopt_sh } | sed "s/-p $CUOPT_SERVER_PORT//g" | tr -d "'" | tr -d " ")
    run_cli_test "'status': 'Optimal'" cuopt_sh -s -c $CLIENT_CERT -p $CUOPT_SERVER_PORT $requestid -k
    run_cli_test "'status': 'Optimal'" cuopt_sh -s -c $CLIENT_CERT -p $CUOPT_SERVER_PORT ../../datasets/cuopt_service_data/good_lp.json -wid $requestid

    # Success, larger problem, result comes back in results dir
    run_cli_test "'result_file': 'data.result'" cuopt_sh -s -c "$CLIENT_CERT" -p $CUOPT_SERVER_PORT ../../datasets/cuopt_service_data/service_data_200r.json -o data.result
    if [ ! -f "$CUOPT_RESULT_DIR/data.result" ]; then
        rapids-logger "CLI test $counter :FAILED"
        echo "Failed to write file to \"$CUOPT_RESULT_DIR/data.result\""
        echo ls "$CUOPT_RESULT_DIR"
        ls "$CUOPT_RESULT_DIR"
        exit 1
    fi

    # Success, larger problem, read from data dir, result comes back in results dir
    cp ../../datasets/cuopt_service_data/service_data_200r.json "$CUOPT_DATA_DIR"
    run_cli_test "'result_file': 'second.result'" cuopt_sh -s -c "$CLIENT_CERT" -p $CUOPT_SERVER_PORT -f service_data_200r.json -o second.result
    if [ ! -f "$CUOPT_RESULT_DIR/second.result" ]; then
        rapids-logger "CLI test $counter :FAILED"
        echo "Failed to write file to \"$CUOPT_RESULT_DIR/second.result\""
        echo ls "$CUOPT_RESULT_DIR"
        ls "$CUOPT_RESULT_DIR"
        exit 1
    fi

    # Success, larger problem, result file format is json as passed in result_type
    run_cli_test "'format': 'json'" cuopt_sh -s -c $CLIENT_CERT -p $CUOPT_SERVER_PORT ../../datasets/cuopt_service_data/service_data_200r.json -rt json -o data.result

    # Failure, larger problem, read from data dir, exception is still returned even though results dir is specified
    sed -i 's/cost_matrix_data/nothere/g' "$CUOPT_DATA_DIR/service_data_200r.json"
    run_cli_test 'cuOpt Error: Unprocessable Entity - 422: unable to validate optimization data file' cuopt_sh -s -c "$CLIENT_CERT" -p $CUOPT_SERVER_PORT -f service_data_200r.json -o data.result

    # Validation error
    run_cli_test 'cuOpt Error: Bad Request - 400: Cost matrix must be a square matrix' cuopt_sh -s -c "$CLIENT_CERT" -p $CUOPT_SERVER_PORT ../../datasets/cuopt_service_data/cuopt_problem_data_broken.json

    # Valid json but bad format error
    run_cli_test 'cuOpt Error: Unprocessable Entity - 422: unable to validate optimization data stream' cuopt_sh -s -c "$CLIENT_CERT" -p $CUOPT_SERVER_PORT ../../datasets/cuopt_service_data/cuopt_bad_format2.json

    # Unhandled exception with an int value that is too big
    run_cli_test 'cuOpt unhandled exception, please include this message in any error report: Python int too large to convert to C long' cuopt_sh -s -c "$CLIENT_CERT" -p $CUOPT_SERVER_PORT ../../datasets/cuopt_service_data/cuopt_unhandled_exception.json

    # Test for message on missing datafile
    run_cli_test "Specified path '$CUOPT_DATA_DIR/nada' does not exist" cuopt_sh -s -c "$CLIENT_CERT" -p $CUOPT_SERVER_PORT -f nada

    # Test for message on absolute path, missing datafile
    run_cli_test "Perhaps you did not intend" cuopt_sh -s -c "$CLIENT_CERT" -p $CUOPT_SERVER_PORT -f /nada

    # Test for message on absolute path, bad directory
    run_cli_test "Absolute path '/nohay' does not exist" cuopt_sh -s -c "$CLIENT_CERT" -p $CUOPT_SERVER_PORT -f /nohay/nada

    rapids-logger "Running cuopt_self_hosted Python tests"
    pytest tests

    popd
fi

# Kill server running on HTTPS
kill -s SIGTERM $SERVER_PID
wait $SERVER_PID

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
