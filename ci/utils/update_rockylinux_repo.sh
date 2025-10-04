#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set -euxo pipefail

# Point all Rocky 8.10 repos to the vault so metadata never disappears
REPO_DIR="/etc/yum.repos.d"

# Disable existing Rocky repos
sed -i 's|^mirrorlist=|#mirrorlist=|g' ${REPO_DIR}/Rocky-*.repo || true
sed -i 's|^baseurl=http.*|#&|g' ${REPO_DIR}/Rocky-*.repo || true

# Write new repo definitions
cat <<'EOF' | sudo tee ${REPO_DIR}/Rocky-Vault.repo
[baseos]
name=Rocky Linux 8.10 - BaseOS (vault)
baseurl=https://dl.rockylinux.org/vault/rocky/8.10/BaseOS/$basearch/os/
enabled=1
gpgcheck=1
gpgkey=file:///etc/pki/rpm-gpg/RPM-GPG-KEY-rockyofficial

[appstream]
name=Rocky Linux 8.10 - AppStream (vault)
baseurl=https://dl.rockylinux.org/vault/rocky/8.10/AppStream/$basearch/os/
enabled=1
gpgcheck=1
gpgkey=file:///etc/pki/rpm-gpg/RPM-GPG-KEY-rockyofficial

[extras]
name=Rocky Linux 8.10 - Extras (vault)
baseurl=https://dl.rockylinux.org/vault/rocky/8.10/extras/$basearch/os/
enabled=1
gpgcheck=1
gpgkey=file:///etc/pki/rpm-gpg/RPM-GPG-KEY-rockyofficial
EOF

# Clean up caches and refresh
dnf clean all
dnf makecache
