/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <mip_heuristics/lns/cpufj.cuh>
#include <mip_heuristics/lns/repair_lns.cuh>
#include <mip_heuristics/lns/thread_budget.hpp>
#include <mip_heuristics/mip_constants.hpp>
#include <utilities/scope_guard.hpp>

#include <chrono>
#include <exception>
#include <memory>
#include <thread>

namespace cuopt::mathematical_optimization::mip {

// The two improvement tasks own private search state in the same representation as
// the presolve CPUFJ portfolio. All communication uses its shared host incumbent.
template <typename i_t, typename f_t>
class early_lns_t {
 public:
  using report_fn = std::function<void(f_t, const std::vector<f_t>&, const char*)>;

  early_lns_t(const fj_cpu_climber_t<i_t, f_t>& anchor,
              std::shared_ptr<fj_cpu_shared_incumbent_t<i_t, f_t>> shared,
              std::atomic<bool>& preemption,
              std::exception_ptr& task_exception,
              report_fn report,
              uint64_t seed)
    : shared_(std::move(shared)),
      preemption_(preemption),
      task_exception_(task_exception),
      report_(std::move(report))
  {
    fj_settings_t settings;
    settings.seed = seed;
    cpufj_        = init_fj_cpu_clone(anchor, preemption_, settings);
    // LNS improvements enter the shared portfolio only through submit.
    cpufj_->shared_incumbent.reset();
    cpufj_->log_prefix           = "[Early CPUFJ LNS] ";
    cpufj_->improvement_callback = [this](f_t, const std::vector<f_t>& x, double) {
      submit(x, "CPUFJ LNS");
    };
    repair_lns_ = std::make_unique<repair_lns_t<i_t, f_t>>(anchor, preemption_, seed);
    RAFT_CUDA_TRY(cudaGetDevice(&device_));
  }

  ~early_lns_t() { finish(); }

  void start()
  {
    if (started_) return;
    started_     = true;
    auto* worker = this;
    auto* cpu    = cpufj_.get();
    if (presolve_cpufj_lns_enabled) {
#pragma omp task firstprivate(worker, cpu) depend(out : *cpu) default(none) \
  priority(CUOPT_DEFAULT_TASK_PRIORITY)
      {
        try {
          worker->run_cpufj();
        } catch (...) {
          worker->fail(std::current_exception());
        }
      }
    }
    if (presolve_repair_lns_enabled) {
#pragma omp task firstprivate(worker) depend(out : *worker) default(none) \
  priority(CUOPT_DEFAULT_TASK_PRIORITY)
      {
        try {
          worker->run_repair_lns();
        } catch (...) {
          worker->fail(std::current_exception());
        }
      }
    }
    // Every posted task has reserved capacity. Do not enter a task scheduling point
    // until they are running on other team members: a taskwait for feasibility
    // lanes could otherwise execute a queued long-running task on the solve thread,
    // preventing that thread from ever reaching the LNS stop signal.
    while (workers_started_.load() <
           int(presolve_cpufj_lns_enabled) + int(presolve_repair_lns_enabled))
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  void request_stop()
  {
    stop_.store(true);
    cpufj_->halted      = true;
    repair_lns_->halted = true;
  }

  void finish()
  {
    request_stop();
    if (!started_) return;
    auto* worker = this;
    auto* cpu    = cpufj_.get();
#pragma omp taskwait depend(in : *cpu, *worker)
    started_ = false;
  }

 private:
  bool stopped() const { return stop_.load() || preemption_.load(); }

  void fail(std::exception_ptr error)
  {
    request_stop();
#pragma omp critical(cuopt_mip_task_exception)
    if (!task_exception_) task_exception_ = std::move(error);
  }

  bool snapshot(std::vector<f_t>& assignment)
  {
    assignment.resize(cpufj_->problem->n_variables);
    return shared_->adopt(std::numeric_limits<f_t>::infinity(), assignment);
  }

  void submit(const std::vector<f_t>& assignment, const char* origin)
  {
    if (stopped()) return;
    const f_t objective = repair_lns_->cost(assignment);
    // Publish before invoking callbacks, so polling only holds the short copy lock.
    shared_->publish(objective, cpufj_->get_user_objective(objective), assignment);
    report_(objective, assignment, origin);
  }

  void run_cpufj()
  {
    ++workers_started_;
    const int previous_max_threads = omp_get_max_threads();
    omp_set_num_threads(1);
    cuopt::scope_guard restore([&] { omp_set_num_threads(previous_max_threads); });
    run_cpufj_lns_ruin_repair<i_t, f_t>(
      cpufj_.get(),
      [this](auto& x) { return snapshot(x); },
      [this](const auto& x) { return repair_lns_->feasible(x); });
  }

  void run_repair_lns()
  {
    ++workers_started_;
    const int previous_max_threads = omp_get_max_threads();
    omp_set_num_threads(1);
    cuopt::scope_guard restore([&] { omp_set_num_threads(previous_max_threads); });
    RAFT_CUDA_TRY(cudaSetDevice(device_));
    repair_lns_->run(
      [this](auto& seeds) {
        std::vector<f_t> assignment;
        if (snapshot(assignment)) seeds.push_back(std::move(assignment));
      },
      [this](const auto& x, f_t) { submit(x, "Repair LNS"); });
  }

  std::shared_ptr<fj_cpu_shared_incumbent_t<i_t, f_t>> shared_;
  std::atomic<bool>& preemption_;
  std::exception_ptr& task_exception_;
  report_fn report_;
  std::unique_ptr<fj_cpu_climber_t<i_t, f_t>> cpufj_;
  std::unique_ptr<repair_lns_t<i_t, f_t>> repair_lns_;
  std::atomic<bool> stop_{false};
  std::atomic<int> workers_started_{0};
  bool started_{false};
  int device_{0};
};
}  // namespace cuopt::mathematical_optimization::mip
