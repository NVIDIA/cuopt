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

set -euo pipefail

REPO_DIR="/etc/yum.repos.d"

# Disable all existing Rocky repos to prevent mirror issues
for repo_file in ${REPO_DIR}/Rocky-*.repo; do
    if [ -f "$repo_file" ]; then
        sed -i 's/^enabled=1/enabled=0/g' "$repo_file"
    fi
done

# Overwrite the main Rocky repos with stable URLs
# Using ONLY direct dl.rockylinux.org URLs (no mirrorlist) to avoid mirror sync issues
# This prevents DNF from trying unreliable/out-of-sync mirrors
cat <<'EOF' > ${REPO_DIR}/Rocky-BaseOS.repo
[baseos]
name=Rocky Linux 8.10 - BaseOS
baseurl=https://dl.rockylinux.org/pub/rocky/8.10/BaseOS/$basearch/os/
enabled=1
gpgcheck=1
gpgkey=file:///etc/pki/rpm-gpg/RPM-GPG-KEY-rockyofficial
retries=10
timeout=30
skip_if_unavailable=False
metadata_expire=1h
EOF

cat <<'EOF' > ${REPO_DIR}/Rocky-AppStream.repo
[appstream]
name=Rocky Linux 8.10 - AppStream
baseurl=https://dl.rockylinux.org/pub/rocky/8.10/AppStream/$basearch/os/
enabled=1
gpgcheck=1
gpgkey=file:///etc/pki/rpm-gpg/RPM-GPG-KEY-rockyofficial
retries=10
timeout=30
skip_if_unavailable=False
metadata_expire=1h
EOF

cat <<'EOF' > ${REPO_DIR}/Rocky-Extras.repo
[extras]
name=Rocky Linux 8.10 - Extras
baseurl=https://dl.rockylinux.org/pub/rocky/8.10/extras/$basearch/os/
enabled=1
gpgcheck=1
gpgkey=file:///etc/pki/rpm-gpg/RPM-GPG-KEY-rockyofficial
retries=10
timeout=30
skip_if_unavailable=False
metadata_expire=1h
EOF

# Clean DNF cache and refresh repository metadata with retry logic
dnf clean all

max_attempts=3
attempt=1
while [ $attempt -le $max_attempts ]; do
    if dnf makecache --refresh; then
        break
    else
        if [ $attempt -lt $max_attempts ]; then
            sleep $((2 ** attempt))
        else
            echo "ERROR: Failed to refresh repository metadata after $max_attempts attempts"
            exit 1
        fi
    fi
    attempt=$((attempt + 1))
done
