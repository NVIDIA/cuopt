#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

# Download the cuopt built in the previous step
LIBCUOPT_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_cpp libcuopt cuopt --cuda "$RAPIDS_CUDA_VERSION")")
CUOPT_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_python cuopt cuopt --py "$RAPIDS_PY_VERSION" --cuda "$RAPIDS_CUDA_VERSION")")
CUOPT_SERVER_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_python cuopt-server cuopt --pure --arch any --cuda "$RAPIDS_CUDA_VERSION")")

# generate constraints (possibly pinning to oldest support versions of dependencies)
rapids-generate-pip-constraints test_python "${PIP_CONSTRAINT}"

# notes:
#
#   * echo to expand wildcard before adding `[test]` requires for pip
#   * just providing --constraint="${PIP_CONSTRAINT}" to be explicit, and because
#     that environment variable is ignored if any other --constraint are passed via the CLI
#
rapids-pip-retry install \
    --prefer-binary \
    --constraint "${PIP_CONSTRAINT}" \
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
export CUOPT_MAX_RESULT=2
CERT_FOLDER=$(pwd)/python/cuopt_self_hosted/cuopt_sh_client/tests/utils/certs
export CUOPT_SSL_CERTFILE=${CERT_FOLDER}/server.crt
export CUOPT_SSL_KEYFILE=${CERT_FOLDER}/server.key
export CLIENT_CERT=${CERT_FOLDER}/ca.crt
python -m cuopt_server.cuopt_service &
export SERVER_PID=$!

DELAY=10

sleep $DELAY

server_status=$(curl -k -sL https://0.0.0.0:$CUOPT_SERVER_PORT/cuopt/health) # NOSONAR — self-signed cert generated locally by this script for CI; not a real TLS endpoint.

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

    # Success, small LP problem with MPS. Data will be transformed to JSON
    run_cli_test "'status': 'Optimal'" cuopt_sh -s -c "$CLIENT_CERT" -p $CUOPT_SERVER_PORT -t LP ../../datasets/linear_programming/good-mps-1.mps

    # Success, small Batch LP problem with MPS. Data will be transformed to JSON
    run_cli_test "'status': 'Optimal'" cuopt_sh -s -c "$CLIENT_CERT" -p $CUOPT_SERVER_PORT -t LP ../../datasets/linear_programming/good-mps-1.mps ../../datasets/linear_programming/good-mps-1.mps

    # Success, small LP problem with LP format. Data will be transformed to JSON
    run_cli_test "'status': 'Optimal'" cuopt_sh -s -c "$CLIENT_CERT" -p $CUOPT_SERVER_PORT -t LP ../../datasets/linear_programming/good-mps-1.lp

    # Success, small Batch LP problem with LP format. Data will be transformed to JSON
    run_cli_test "'status': 'Optimal'" cuopt_sh -s -c "$CLIENT_CERT" -p $CUOPT_SERVER_PORT -t LP ../../datasets/linear_programming/good-mps-1.lp ../../datasets/linear_programming/good-mps-1.lp

    # Success, compressed LP inputs (.lp.gz / .lp.bz2) via Read dispatch
    run_cli_test "'status': 'Optimal'" cuopt_sh -s -c "$CLIENT_CERT" -p $CUOPT_SERVER_PORT -t LP ../../datasets/linear_programming/good-mps-1.lp.gz
    run_cli_test "'status': 'Optimal'" cuopt_sh -s -c "$CLIENT_CERT" -p $CUOPT_SERVER_PORT -t LP ../../datasets/linear_programming/good-mps-1.lp.bz2

    # Success, compressed MPS inputs (.mps.gz / .mps.bz2) for parity
    run_cli_test "'status': 'Optimal'" cuopt_sh -s -c "$CLIENT_CERT" -p $CUOPT_SERVER_PORT -t LP ../../datasets/linear_programming/good-mps-1.mps.gz
    run_cli_test "'status': 'Optimal'" cuopt_sh -s -c "$CLIENT_CERT" -p $CUOPT_SERVER_PORT -t LP ../../datasets/linear_programming/good-mps-1.mps.bz2

    # Error, local file mode is not allowed with MPS/LP file inputs
    run_cli_test "Cannot use local file mode with MPS/LP data" cuopt_sh -s -c "$CLIENT_CERT" -p $CUOPT_SERVER_PORT -t LP -f good-mps-1.mps
    run_cli_test "Cannot use local file mode with MPS/LP data" cuopt_sh -s -c "$CLIENT_CERT" -p $CUOPT_SERVER_PORT -t LP -f good-mps-1.lp

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

    # Set all current and deprecated solver_config values and make sure the service does not reject the dataset
    # This is a smoketest against parameter name misalignment
    run_cli_test "'status': 'Optimal'" cuopt_sh -s -c $CLIENT_CERT -p $CUOPT_SERVER_PORT ../../datasets/cuopt_service_data/lpmip_configs.json

    rapids-logger "Running cuopt_self_hosted Python tests"
    pytest tests

    popd
fi

# Kill server running on HTTPS
kill -s SIGTERM $SERVER_PID
wait $SERVER_PID

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
