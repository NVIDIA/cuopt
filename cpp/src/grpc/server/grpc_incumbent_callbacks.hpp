/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuopt/mathematical_optimization/utilities/internals.hpp>

#include <algorithm>
#include <cstddef>
#include <mutex>
#include <utility>
#include <vector>

namespace cuopt::remote::detail {

class LastIncumbentState {
 public:
  LastIncumbentState(std::size_t num_variables, bool is_float)
    : num_variables_(num_variables), is_float_(is_float)
  {
  }

  void record_get_solution(const void* data, const void* objective_value)
  {
    std::vector<double> assignment(num_variables_);
    double objective = 0.0;
    if (is_float_) {
      const auto* float_data = static_cast<const float*>(data);
      for (std::size_t i = 0; i < num_variables_; ++i) {
        assignment[i] = static_cast<double>(float_data[i]);
      }
      objective = static_cast<double>(*static_cast<const float*>(objective_value));
    } else {
      const auto* double_data = static_cast<const double*>(data);
      std::copy(double_data, double_data + num_variables_, assignment.begin());
      objective = *static_cast<const double*>(objective_value);
    }

    std::lock_guard<std::mutex> lock(mutex_);
    assignment_ = std::move(assignment);
    objective_  = objective;
    has_value_  = true;
  }

  bool copy_last(void* data, void* objective_value) const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!has_value_ || num_variables_ == 0 || assignment_.size() != num_variables_) {
      return false;
    }
    if (is_float_) {
      auto* float_data = static_cast<float*>(data);
      for (std::size_t i = 0; i < num_variables_; ++i) {
        float_data[i] = static_cast<float>(assignment_[i]);
      }
      *static_cast<float*>(objective_value) = static_cast<float>(objective_);
    } else {
      auto* double_data = static_cast<double*>(data);
      std::copy(assignment_.begin(), assignment_.end(), double_data);
      *static_cast<double*>(objective_value) = objective_;
    }
    return true;
  }

 private:
  std::size_t num_variables_;
  bool is_float_;
  mutable std::mutex mutex_;
  std::vector<double> assignment_;
  double objective_ = 0.0;
  bool has_value_   = false;
};

// Echoes the last get-incumbent back into the MIP solver. Registering a
// set-solution callback preserves the legacy incumbent_set_solutions behavior.
class EchoSetSolutionCallback : public cuopt::internals::set_solution_callback_t {
 public:
  EchoSetSolutionCallback(LastIncumbentState* state, std::size_t num_variables, bool is_float)
    : state_(state)
  {
    n_variables = num_variables;
    isFloat     = is_float;
  }

  void set_solution(void* data,
                    void* objective_value,
                    void* /*solution_bound*/,
                    void* /*user_data*/) override
  {
    if (state_ != nullptr) { state_->copy_last(data, objective_value); }
  }

 private:
  LastIncumbentState* state_;
};

}  // namespace cuopt::remote::detail
