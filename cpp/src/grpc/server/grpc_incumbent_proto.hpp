/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

// Codegen target: this file builds and parses cuopt::remote::Incumbent protobuf messages.
// A future version of cpp/codegen/generate_conversions.py can produce this from
// an incumbent section in field_registry.yaml.

#pragma once

#ifdef CUOPT_ENABLE_GRPC

#include <cstdint>
#include <limits>
#include <string>
#include <vector>
#include "cuopt_remote.pb.h"
#include "cuopt_remote_service.pb.h"

inline std::vector<uint8_t> build_incumbent_proto(const std::string& job_id,
                                                  double objective,
                                                  const std::vector<double>& assignment)
{
  cuopt::remote::Incumbent msg;
  msg.set_job_id(job_id);
  msg.set_objective(objective);
  for (double v : assignment) {
    msg.add_assignment(v);
  }
  auto size = msg.ByteSizeLong();
  if (size > static_cast<size_t>(std::numeric_limits<int>::max())) { return {}; }
  std::vector<uint8_t> buffer(size);
  if (!msg.SerializeToArray(buffer.data(), static_cast<int>(buffer.size()))) { return {}; }
  return buffer;
}

inline bool parse_incumbent_proto(const uint8_t* data,
                                  size_t size,
                                  std::string& job_id,
                                  double& objective,
                                  std::vector<double>& assignment)
{
  cuopt::remote::Incumbent incumbent_msg;
  if (!incumbent_msg.ParseFromArray(data, static_cast<int>(size))) { return false; }

  job_id    = incumbent_msg.job_id();
  objective = incumbent_msg.objective();
  assignment.clear();
  assignment.reserve(incumbent_msg.assignment_size());
  for (int i = 0; i < incumbent_msg.assignment_size(); ++i) {
    assignment.push_back(incumbent_msg.assignment(i));
  }
  return true;
}

#endif  // CUOPT_ENABLE_GRPC
