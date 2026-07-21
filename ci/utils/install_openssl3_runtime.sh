#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Install OpenSSL 3 runtime libraries when the host distro doesn't ship them
# by default. cuopt wheels DT_NEEDED libssl.so.3 / libcrypto.so.3 and do NOT
# bundle them (see docs/openssl3-runtime-requirements.md), so the test image
# has to provide them at runtime.
#
# Coverage:
#   - Rocky/RHEL/Alma 8: needs openssl3 from EPEL (default OpenSSL is 1.0.x).
#   - Other distros (Ubuntu 22.04+, Debian 12+, RHEL/Rocky 9+, Fedora 36+):
#     already ship libssl.so.3 / libcrypto.so.3 — no-op.
#
# Safe to run unconditionally; only takes action where needed.

set -euo pipefail

if [ ! -f /etc/os-release ]; then
    exit 0
fi

. /etc/os-release

case "${ID:-}" in
  rocky|rhel|centos|almalinux)
    if [[ "${VERSION_ID%%.*}" == "8" ]]; then
        echo "==> Installing OpenSSL 3 runtime from EPEL (Rocky/RHEL/Alma 8)"
        if ! rpm -q epel-release >/dev/null 2>&1; then
            dnf install -y -q epel-release
        fi
        # 'openssl3' is the runtime package; it drops libssl.so.3 /
        # libcrypto.so.3 into /usr/lib64 and ldconfig picks them up
        # automatically. ('openssl3-devel' is what the build host uses.)
        dnf install -y -q openssl3
        ldconfig_out="$(ldconfig -p)"
        if ! grep -q "libssl\.so\.3" <<<"${ldconfig_out}" || \
           ! grep -q "libcrypto\.so\.3" <<<"${ldconfig_out}"; then
            echo "ERROR: libssl.so.3 / libcrypto.so.3 still not on linker path" >&2
            exit 1
        fi
    fi
    ;;
  *)
    # Ubuntu 22.04+, Debian 12+, etc. ship OpenSSL 3 by default; nothing to do.
    ;;
esac
