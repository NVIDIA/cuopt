#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euxo pipefail

cd "$(dirname "$(realpath "$1")")"

openssl req \
  -x509 \
  -new \
  -key ca.key \
  -out ca.crt \
  -not_before 20250502001153Z \
  -not_after 20350430001153Z \
  -sha256 \
  -subj '/C=US/ST=State/L=City/O=MyOrg/OU=Dev/CN=MyCustomCA' \
  -addext 'keyUsage=critical,keyCertSign,cRLSign' \
  -addext 'subjectKeyIdentifier=hash' \
  -addext 'authorityKeyIdentifier=keyid:always,issuer' \
  -addext 'basicConstraints=critical,CA:TRUE' \
  -set_serial 0x7a43f651644d3c65696152e474974669dd9fd7e2

openssl req \
  -x509 \
  -new \
  -key server.key \
  -out server.crt \
  -CA ca.crt \
  -CAkey ca.key \
  -not_before 20250502001540Z \
  -not_after 21250408001540Z \
  -sha256 \
  -subj '/C=US/ST=State/L=City/O=MyOrg/OU=Dev/CN=myserver.local' \
  -addext 'subjectAltName=DNS:myserver.local,DNS:localhost,IP:192.168.1.100,IP:0.0.0.0' \
  -addext 'subjectKeyIdentifier=hash' \
  -addext 'authorityKeyIdentifier=keyid:always,issuer' \
  -addext 'basicConstraints=critical,CA:FALSE' \
  -set_serial 0x1FFE9EC90D2765E3E1CE30F1804F2341C5E3A419
