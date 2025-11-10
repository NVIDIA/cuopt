/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <string>

namespace cuopt {
namespace test {

// Define RAPIDS_DATASET_ROOT_DIR using a preprocessor variable to
// allow for a build to override the default. This is useful for
// having different builds for specific default dataset locations.
#ifndef RAPIDS_DATASET_ROOT_DIR
#define RAPIDS_DATASET_ROOT_DIR "./datasets"
#endif

inline const std::string get_rapids_dataset_root_dir()
{
  const char* envVar = std::getenv("RAPIDS_DATASET_ROOT_DIR");
  std::string rdrd   = (envVar != NULL) ? envVar : RAPIDS_DATASET_ROOT_DIR;
  return rdrd;
}

inline const std::string get_cuopt_home()
{
  std::string cuopt_home("");
  const char* env_var = std::getenv("CUOPT_HOME");
  cuopt_home          = (env_var != NULL) ? env_var : "";
  return cuopt_home;
}
}  // namespace test
}  // namespace cuopt
