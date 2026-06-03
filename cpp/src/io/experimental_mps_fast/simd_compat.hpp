// SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
// reserved. SPDX-License-Identifier: Apache-2.0

#pragma once

// Use SIMDe's explicit simde_* API. On x86 it can still lower to native
// intrinsics; on other targets it provides the portable implementation.
#include <simde/x86/aes.h>
#include <simde/x86/avx2.h>
#include <simde/x86/sse4.2.h>
