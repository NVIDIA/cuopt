// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// libcuopt-client only: no GPU, CUDA toolkit, rmm or raft. The client package ships a very small
// header set (the problem parsers are in the mathopt headers), so there is no exported API to
// call through them. This checks that the package is consumable: find_package(cuopt) and
// cuopt::client resolve, the headers compile, and libcuopt_client loads with every symbol
// resolved (RTLD_NOW) in an environment holding only this package.
#include <cuopt/mathematical_optimization/constants.h>

#include <dlfcn.h>
#include <gtest/gtest.h>

TEST(IsolatedClient, ConstantsHeaderIsUsable)
{
  EXPECT_EQ(CUOPT_MINIMIZE, 1);
  EXPECT_EQ(CUOPT_MAXIMIZE, -1);
}

TEST(IsolatedClient, LibraryLoadsWithAllSymbolsResolved)
{
  void* handle = dlopen(CUOPT_CLIENT_LIB, RTLD_NOW);
  EXPECT_NE(handle, nullptr) << dlerror();
  if (handle != nullptr) { dlclose(handle); }
}
