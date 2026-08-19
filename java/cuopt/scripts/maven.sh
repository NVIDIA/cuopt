#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Populates CUOPT_MVN_ARGS with the resolver retry settings every mvn invocation should carry.
#
# CI resolves plugins and dependencies from Maven Central with no warm local repository, so a
# rate-limited (429) or briefly unavailable response fails the build outright. Maven does not
# retry those by default. The wagon and native-resolver properties cover both transports, since
# which one is in use depends on the Maven version conda resolves; the unused ones are ignored.
cuopt_maven_args() {
  # shellcheck disable=SC2034  # read by the script that sources this one
  CUOPT_MVN_ARGS=(
    '-Dmaven.wagon.http.retryHandler.count=5'
    '-Daether.connector.http.retryHandler.count=5'
    '-Daether.connector.http.retryHandler.serviceUnavailable=429,500,502,503,504'
  )
}
