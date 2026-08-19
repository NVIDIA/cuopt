/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <mip_heuristics/early_heuristic.cuh>

#include <atomic>
#include <cstdint>
#include <memory>

namespace cuopt::mathematical_optimization::mip {

enum class structural_outcome_t : uint8_t { declined, constructed, budget_exhausted };

// A primal heuristic for one recognizable model structure.  Recognition and construction both live
// here: the pass owns everything from reading the model to producing a point, and the dispatcher
// owns the framework hook, the validation and the publication.
//
// Subclasses must be default constructible and cheap to construct, because the dispatcher builds
// one before it knows whether the model matches.  All real work belongs in recognize() and solve().
template <typename i_t, typename f_t>
class structural_heuristic_t {
 public:
  virtual ~structural_heuristic_t() = default;

  virtual const char* name() const = 0;

  // Necessary conditions on the model, cheap enough to run on the solve's thread before any GPU
  // work is committed.  Non-const so a subclass can keep the host view it built here for solve().
  // The tolerances decide every comparison the detector makes, so they belong here rather than only
  // in solve(): a view built against one set and read against another proves nothing.
  virtual bool recognize(
    const optimization_problem_t<i_t, f_t>& op_problem,
    const typename mip_solver_settings_t<i_t, f_t>::tolerances_t& tolerances) = 0;

  // The root position hands over the fully reduced problem instead.  Declining by default lets a
  // heuristic take one position without implementing the other.
  virtual bool recognize(const problem_t<i_t, f_t>&,
                         const typename mip_solver_settings_t<i_t, f_t>::tolerances_t&)
  {
    return false;
  }

  // Full detection and construction over the view recognize() kept, which is why the model is not a
  // parameter: it came from whichever source recognized it, and the assignment comes back indexed
  // in that source's columns.  A zero work_budget means unlimited.  The preemption flag is not
  // const because a sub-solver may need to bind it by reference; a subclass must only ever read it.
  virtual structural_outcome_t solve(
    const typename mip_solver_settings_t<i_t, f_t>::tolerances_t& tolerances,
    std::atomic<bool>& preemption,
    double work_budget,
    std::vector<f_t>& assignment) = 0;

  // How many threads the heuristic may use inside solve().  Only the position that launches it
  // knows what else is sharing the team, so it is told rather than asking.
  void set_lane_budget(int lanes) { lane_budget_ = lanes; }

  // Dense column permutations of the model, one per generator of its symmetry group, or null when
  // none are known.  Kept as plain permutations so a heuristic never depends on whatever computed
  // them.  The pointee must outlive solve().
  void set_column_symmetry(const std::vector<std::vector<i_t>>* generators)
  {
    column_symmetry_ = generators;
  }

 protected:
  int lane_budget() const { return lane_budget_; }
  const std::vector<std::vector<i_t>>* column_symmetry() const { return column_symmetry_; }

 private:
  int lane_budget_{1};
  const std::vector<std::vector<i_t>>* column_symmetry_{nullptr};
};

// Runs whichever structural heuristic recognizes the model, on one task during presolve.  Nothing
// is asked of the call site beyond construction: when no structure is recognized the object stays
// inert, having skipped the framework's problem copy entirely, and start() does nothing.
template <typename i_t, typename f_t>
class early_structural_t : public early_heuristic_t<i_t, f_t, early_structural_t<i_t, f_t>> {
 public:
  // op_problem must outlive this object: the publication gate reads its column count.
  early_structural_t(const optimization_problem_t<i_t, f_t>& op_problem,
                     const typename mip_solver_settings_t<i_t, f_t>::tolerances_t& tolerances,
                     early_incumbent_callback_t<f_t> incumbent_callback);

  ~early_structural_t();

  static constexpr const char* name() { return "Structural"; }

  // Name of the heuristic that recognized the model, or nullptr when none did.
  const char* recognized_name() const;

  void start();
  void stop();

 private:
  // Body of the task: solve, validate, publish.
  void run();

  // True when preprocessing left the column space of problem_ptr_ identical to op_problem's, which
  // is what lets an assignment built in op_problem space be published through post_process.
  bool preprocessing_is_identity() const;

  const optimization_problem_t<i_t, f_t>& op_problem_;
  typename mip_solver_settings_t<i_t, f_t>::tolerances_t tolerances_;
  std::unique_ptr<structural_heuristic_t<i_t, f_t>> active_;
  std::atomic<bool> preemption_flag_{false};
  bool task_launched_{false};
};

// The same heuristics run again from the root, on the fully reduced problem B&B itself solves.
// Nothing about the early positions carries over: there is no problem copy, no private handle and
// no preprocessing gate, because the point produced here is already in the space it is published
// in.  Its window is the root relaxation and the cut loop, during which B&B's worker pools do not
// yet exist, so the threads they will later claim are free to use.
template <typename i_t, typename f_t>
class root_structural_t {
 public:
  // problem, preemption and column_symmetry must outlive this object: run() reads them from inside
  // the task.  column_symmetry may be null when the solve found none.
  root_structural_t(problem_t<i_t, f_t>& problem,
                    const typename mip_solver_settings_t<i_t, f_t>::tolerances_t& tolerances,
                    std::atomic<bool>& preemption,
                    const std::vector<std::vector<i_t>>* column_symmetry,
                    int lane_budget);

  ~root_structural_t();

  // Name of the heuristic that recognized the model, or nullptr when none did.
  const char* recognized_name() const;

  bool recognized() const { return active_ != nullptr; }

  // Detect, construct, validate and hand the point to B&B.  Blocking: the caller supplies the task.
  void run();

 private:
  problem_t<i_t, f_t>& problem_;
  typename mip_solver_settings_t<i_t, f_t>::tolerances_t tolerances_;
  std::atomic<bool>& preemption_;
  std::unique_ptr<structural_heuristic_t<i_t, f_t>> active_;
};

}  // namespace cuopt::mathematical_optimization::mip
