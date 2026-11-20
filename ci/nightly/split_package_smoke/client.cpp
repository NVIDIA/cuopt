// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuopt/status_codes.h>

#include <cstdio>

// The client's installed headers expose almost nothing to call, so this checks that the library
// and every shared library it links resolve when the binary starts.
int main()
{
  std::printf("CUOPT_SUCCESS=%d\n", CUOPT_SUCCESS);
  return CUOPT_SUCCESS;
}
