#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

echo "[*] Backing up existing repo files..."
mkdir -p /etc/yum.repos.d/backup
mv /etc/yum.repos.d/Rocky-*.repo /etc/yum.repos.d/backup/ 2>/dev/null || true

echo "[*] Creating Rocky Linux 8.10 vault repo file..."
cat > /etc/yum.repos.d/rocky-vault.repo <<'EOF'
[baseos]
name=Rocky Linux 8.10 - BaseOS Vault
baseurl=https://dl.rockylinux.org/vault/rocky/8.10/BaseOS/$basearch/os/
enabled=1
gpgcheck=1
gpgkey=file:///etc/pki/rpm-gpg/RPM-GPG-KEY-rockyofficial

[appstream]
name=Rocky Linux 8.10 - AppStream Vault
baseurl=https://dl.rockylinux.org/vault/rocky/8.10/AppStream/$basearch/os/
enabled=1
gpgcheck=1
gpgkey=file:///etc/pki/rpm-gpg/RPM-GPG-KEY-rockyofficial

[extras]
name=Rocky Linux 8.10 - Extras Vault
baseurl=https://dl.rockylinux.org/vault/rocky/8.10/extras/$basearch/os/
enabled=1
gpgcheck=1
gpgkey=file:///etc/pki/rpm-gpg/RPM-GPG-KEY-rockyofficial
EOF

echo "[*] Cleaning old cache..."
dnf clean all

echo "[*] Rebuilding cache from vault..."
dnf makecache

echo "[✓] Done! Your system is now using Rocky Linux 8.10 vault repos."
