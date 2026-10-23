#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Proxy-mode exit status for ci/docker/entrypoint.sh, without a container.
# Stub cuopt_grpc_server and python stand in for the real servers. grpc and
# legacy modes exec those programs, so their exit status is not decided here.

set -euo pipefail

entry="$(cd "$(dirname "$0")" && pwd)/entrypoint.sh"
command -v python3 >/dev/null || {
    echo "python3 is required to open the stub gRPC port" >&2
    exit 1
}

# Fake binaries live only in this directory. The EXIT trap removes it, so a
# passing run and a failing run both delete cuopt_grpc_server and python.
# INT and TERM exit the script, which runs that same trap.
workdir="$(mktemp -d)"
stubdir="${workdir}/bin"
pidfile="${workdir}/pids"
log="${workdir}/entry.log"
port=""
ep=""
code=0

cleanup() {
    local pid
    # Stop an entrypoint that is still running, then its stubs.
    if [[ -n "${ep}" ]] && kill -0 "${ep}" 2>/dev/null; then
        kill -TERM "${ep}" 2>/dev/null || true
        sleep 0.3
        kill -KILL "${ep}" 2>/dev/null || true
    fi
    if [[ -f "${pidfile}" ]]; then
        while read -r pid; do
            if [[ -n "${pid}" ]]; then
                kill -KILL "${pid}" 2>/dev/null || true
            fi
        done <"${pidfile}"
    fi
    rm -rf "${workdir}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

mkdir -p "${stubdir}"

# Stand-in for the gRPC server. entrypoint.sh runs: cuopt_grpc_server --port PORT
# The process records its pid, then follows STUB_GRPC_MODE:
#   serve — open the port and accept connections until SIGTERM or SIGINT
#   hang  — do not open the port, so a signal arrives during startup
#   exit  — leave immediately with STUB_GRPC_EXIT (startup failure)
cat >"${stubdir}/cuopt_grpc_server" <<'EOF'
#!/usr/bin/env bash
set -u
echo "$$" >>"${STUB_PIDFILE}"
port=5001
while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)
            port="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done
case "${STUB_GRPC_MODE:-serve}" in
    exit)
        exit "${STUB_GRPC_EXIT:-3}"
        ;;
    hang)
        trap 'exit 0' TERM INT
        while true; do
            read -r -t 60 || true
        done
        ;;
esac
exec python3 - "${port}" <<'PY'
import signal
import socket
import sys

port = int(sys.argv[1])
sock = socket.socket()
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(("127.0.0.1", port))
sock.listen(8)
sock.settimeout(0.5)

def _stop(_signum, _frame):
    raise SystemExit(0)

signal.signal(signal.SIGTERM, _stop)
signal.signal(signal.SIGINT, _stop)
while True:
    try:
        conn, _addr = sock.accept()
        conn.close()
    except socket.timeout:
        continue
    except OSError:
        break
PY
EOF

# Stand-in for the interpreter. entrypoint.sh runs:
#   python -m cuopt_server.cuopt_proxy
# The file is named python so that command runs this script. It accepts only
# that invocation. STUB_PROXY_MODE=exit sleeps, then exits STUB_PROXY_EXIT
# with no signal. Otherwise it waits until the entrypoint signals it.
cat >"${stubdir}/python" <<'EOF'
#!/usr/bin/env bash
set -u
if [[ "${1:-}" == "-m" && "${2:-}" == "cuopt_server.cuopt_proxy" ]]; then
    echo "$$" >>"${STUB_PIDFILE}"
    if [[ "${STUB_PROXY_MODE:-serve}" == "exit" ]]; then
        sleep 0.3
        exit "${STUB_PROXY_EXIT:-7}"
    fi
    trap 'exit 0' TERM INT
    while true; do
        read -r -t 60 || true
    done
fi
echo "python stub: unexpected invocation: $*" >&2
exit 99
EOF
chmod +x "${stubdir}/cuopt_grpc_server" "${stubdir}/python"

fail() {
    printf 'FAIL %s\n' "$*" >&2
    if [[ -f "${log}" ]]; then
        echo "----- entrypoint log -----" >&2
        cat "${log}" >&2
    fi
    exit 1
}

pass() {
    printf 'PASS %s\n' "$*"
}

# Free local port. Each case reuses it after the previous listener has exited.
port="$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')"

# Start entrypoint.sh with the stub directory first on PATH. The assignments
# apply only to that child. Inside the child, SIGINT and SIGTERM are set back
# to their default actions before exec: a background bash ignores SIGINT, and
# an ignored signal cannot be trapped. PID 1 does not start out ignoring it.
# This process's own handlers are unchanged.
start_entrypoint() {
    local grpc_mode="$1"
    local proxy_mode="$2"
    : >"${pidfile}"
    : >"${log}"
    ep=""
    PATH="${stubdir}:${PATH}" \
        CUOPT_GRPC_PORT="${port}" \
        STUB_PIDFILE="${pidfile}" \
        STUB_GRPC_MODE="${grpc_mode}" \
        STUB_PROXY_MODE="${proxy_mode}" \
        STUB_GRPC_EXIT="${STUB_GRPC_EXIT:-3}" \
        STUB_PROXY_EXIT="${STUB_PROXY_EXIT:-7}" \
        python3 -c 'import os, signal, sys; signal.signal(signal.SIGINT, signal.SIG_DFL); signal.signal(signal.SIGTERM, signal.SIG_DFL); os.execv("/bin/bash", ["bash", sys.argv[1]])' \
        "${entry}" >"${log}" 2>&1 &
    ep=$!
}

# Wait until the entrypoint exits, and store its status in code. Give up
# after 5 seconds so a missed signal fails the test instead of hanging.
reap() {
    local _i
    for _i in $(seq 1 50); do
        if ! kill -0 "${ep}" 2>/dev/null; then
            set +e
            wait "${ep}"
            code=$?
            set -e
            ep=""
            return 0
        fi
        sleep 0.1
    done
    fail "entrypoint still running after 5s"
}

# Both stubs append their pids when they start. None should still be running
# after the entrypoint has exited.
assert_stubs_dead() {
    local pid
    while read -r pid; do
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            fail "stub pid ${pid} still running"
        fi
    done <"${pidfile}"
}

# Startup cases signal as soon as the gRPC stub exists, before it listens.
wait_for_grpc_pid() {
    local _i
    for _i in $(seq 1 50); do
        if [[ -s "${pidfile}" ]]; then
            return 0
        fi
        if ! kill -0 "${ep}" 2>/dev/null; then
            return 1
        fi
        sleep 0.05
    done
    return 1
}

# Ready cases wait until the port accepts a connection and the proxy stub has
# recorded its pid. The pid file then has one line per child.
wait_for_proxy() {
    local _i lines
    for _i in $(seq 1 100); do
        if timeout 1 bash -c "echo >/dev/tcp/127.0.0.1/${port}" 2>/dev/null; then
            lines="$(wc -l <"${pidfile}" | tr -d ' ')"
            if [[ "${lines}" -ge 2 ]]; then
                return 0
            fi
        fi
        if ! kill -0 "${ep}" 2>/dev/null; then
            return 1
        fi
        sleep 0.05
    done
    return 1
}

# Send SIGINT or SIGTERM to the entrypoint. A handled signal exits 0.
run_signaled() {
    local when="$1"
    local sig="$2"
    local grpc_mode="$3"
    start_entrypoint "${grpc_mode}" serve
    if [[ "${when}" == "ready" ]]; then
        wait_for_proxy || fail "${when} ${sig}: proxy did not start"
    else
        wait_for_grpc_pid || fail "${when} ${sig}: gRPC stub did not start"
    fi
    kill -s "${sig}" "${ep}"
    reap
    [[ "${code}" -eq 0 ]] || fail "${when} ${sig}: exit ${code}, expected 0"
    assert_stubs_dead
    pass "proxy ${when} ${sig} exits 0"
}

# No signal. The proxy stub exits 7, and the entrypoint must return that 7.
run_proxy_status() {
    STUB_PROXY_EXIT=7
    start_entrypoint serve exit
    local _i
    for _i in $(seq 1 50); do
        if ! kill -0 "${ep}" 2>/dev/null; then
            break
        fi
        sleep 0.1
    done
    if kill -0 "${ep}" 2>/dev/null; then
        fail "child exit: entrypoint still running"
    fi
    reap
    [[ "${code}" -eq 7 ]] || fail "child exit: exit ${code}, expected 7"
    assert_stubs_dead
    pass "proxy child exit 7 is preserved"
}

# No signal. The gRPC stub exits before listening. The entrypoint exits 1,
# rather than forwarding the stub's status.
run_startup_failure() {
    STUB_GRPC_EXIT=3
    start_entrypoint exit serve
    local _i
    for _i in $(seq 1 50); do
        if ! kill -0 "${ep}" 2>/dev/null; then
            break
        fi
        sleep 0.1
    done
    if kill -0 "${ep}" 2>/dev/null; then
        fail "startup failure: entrypoint still running"
    fi
    reap
    [[ "${code}" -eq 1 ]] || fail "startup failure: exit ${code}, expected 1"
    assert_stubs_dead
    pass "proxy startup failure without a signal exits 1"
}

run_signaled ready INT serve
run_signaled ready TERM serve
run_signaled startup INT hang
run_signaled startup TERM hang
run_proxy_status
run_startup_failure

# The EXIT trap deletes the temp directory and both fake commands.
echo "entrypoint signal tests passed"
