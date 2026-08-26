#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Maven helpers shared by build.sh and scripts/test.sh.
#
# CI resolves plugins and dependencies from Maven Central with no warm local repository, so a
# rate-limited (429) response fails the build on the first artifact it needs. Maven's own
# retry settings are transport-specific — the wagon and resolver properties below only apply
# when that transport is the one in use — so they are set as a best effort and cuopt_mvn
# retries the invocation as the guarantee.

cuopt_maven_args() {
  # shellcheck disable=SC2034  # read by cuopt_mvn and by the sourcing script
  CUOPT_MVN_ARGS=(
    '-Dmaven.wagon.http.retryHandler.count=5'
    '-Daether.connector.http.retryHandler.count=5'
    '-Daether.connector.http.retryHandler.serviceUnavailable=429,500,502,503,504'
  )
}

# Runs mvn with those arguments, retrying with backoff when the failure looks like a transient
# artifact-resolution problem. A compile or test failure is returned immediately, so a genuinely
# broken build does not pay the retries.
cuopt_mvn() {
  local attempt=1
  local max="${CUOPT_MVN_RETRIES:-4}"
  local delay="${CUOPT_MVN_RETRY_DELAY:-15}"
  local log status
  log="$(mktemp)"

  while true; do
    # tee would otherwise report its own exit status rather than Maven's.
    mvn "${CUOPT_MVN_ARGS[@]}" "$@" 2>&1 | tee "${log}"
    status="${PIPESTATUS[0]}"
    if [[ "${status}" -eq 0 ]]; then
      rm -f "${log}"
      return 0
    fi
    if ! grep -qE 'Could not transfer artifact|Too Many Requests|Connection (timed out|reset)|could not be resolved' "${log}"; then
      echo "mvn failed for a non-transient reason; not retrying" >&2
      rm -f "${log}"
      return "${status}"
    fi
    if [[ "${attempt}" -ge "${max}" ]]; then
      echo "mvn still failing to resolve artifacts after ${max} attempts" >&2
      rm -f "${log}"
      return "${status}"
    fi
    echo "mvn attempt ${attempt}/${max} hit a transient resolution failure; retrying in ${delay}s" >&2
    sleep "${delay}"
    delay=$((delay * 2))
    attempt=$((attempt + 1))
  done
}
