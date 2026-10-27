/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file grpc_job_queue_test.cpp
 * @brief Checks that workers claim queued jobs in enqueue order.
 *
 * Header-only: no server, GPU or shared memory is needed.
 */

#include <gtest/gtest.h>

#include <grpc_job_queue.hpp>

#include <algorithm>
#include <thread>
#include <vector>

namespace {

void publish(JobQueueEntry& job, uint64_t seq)
{
  job.enqueue_seq.store(seq);
  job.ready.store(true);
  job.claimed.store(false);
}

}  // namespace

TEST(JobQueue, ClaimsLowestSequenceFirst)
{
  std::vector<JobQueueEntry> queue(4);
  publish(queue[0], 5);
  publish(queue[1], 2);
  publish(queue[2], 9);
  publish(queue[3], 3);

  EXPECT_EQ(claim_ready_job_slot(queue.data(), queue.size(), 0, 100), 1);
  EXPECT_EQ(claim_ready_job_slot(queue.data(), queue.size(), 0, 100), 3);
  EXPECT_EQ(claim_ready_job_slot(queue.data(), queue.size(), 0, 100), 0);
  EXPECT_EQ(claim_ready_job_slot(queue.data(), queue.size(), 0, 100), 2);
  EXPECT_EQ(claim_ready_job_slot(queue.data(), queue.size(), 0, 100), -1);
}

TEST(JobQueue, SkipsSlotsThatAreNotDispatchable)
{
  std::vector<JobQueueEntry> queue(4);
  publish(queue[0], 1);
  queue[0].claimed.store(true);  // already taken
  publish(queue[1], 2);
  queue[1].ready.store(false);  // not published
  publish(queue[2], 0);         // free slot
  publish(queue[3], 7);

  EXPECT_EQ(claim_ready_job_slot(queue.data(), queue.size(), 0, 100), 3);
  EXPECT_EQ(claim_ready_job_slot(queue.data(), queue.size(), 0, 100), -1);
}

TEST(JobQueue, StampsWorkerOnClaim)
{
  std::vector<JobQueueEntry> queue(1);
  publish(queue[0], 1);

  ASSERT_EQ(claim_ready_job_slot(queue.data(), queue.size(), 3, 4242), 0);
  EXPECT_TRUE(queue[0].claimed.load());
  EXPECT_EQ(queue[0].worker_index.load(), 3);
  EXPECT_EQ(queue[0].worker_pid.load(), 4242);
}

TEST(JobQueue, ResetMakesSlotUnclaimable)
{
  std::vector<JobQueueEntry> queue(1);
  publish(queue[0], 1);
  ASSERT_EQ(claim_ready_job_slot(queue.data(), queue.size(), 0, 100), 0);

  reset_job_slot(queue[0]);

  EXPECT_FALSE(queue[0].ready.load());
  EXPECT_FALSE(queue[0].claimed.load());
  EXPECT_EQ(queue[0].enqueue_seq.load(), 0u);
  EXPECT_EQ(queue[0].worker_index.load(), -1);
  EXPECT_EQ(claim_ready_job_slot(queue.data(), queue.size(), 0, 100), -1);
}

TEST(JobQueue, ConcurrentWorkersClaimEachJobOnce)
{
  constexpr int kJobs    = 64;
  constexpr int kWorkers = 8;
  std::vector<JobQueueEntry> queue(kJobs);
  for (int i = 0; i < kJobs; ++i) {
    publish(queue[i], static_cast<uint64_t>(i) + 1);
  }

  std::vector<std::vector<int>> claimed(kWorkers);
  std::vector<std::thread> workers;
  for (int w = 0; w < kWorkers; ++w) {
    workers.emplace_back([&, w] {
      for (int slot; (slot = claim_ready_job_slot(queue.data(), queue.size(), w, 100 + w)) >= 0;) {
        claimed[w].push_back(slot);
      }
    });
  }
  for (auto& t : workers) {
    t.join();
  }

  std::vector<int> all;
  for (int w = 0; w < kWorkers; ++w) {
    EXPECT_TRUE(std::is_sorted(claimed[w].begin(), claimed[w].end()));
    all.insert(all.end(), claimed[w].begin(), claimed[w].end());
  }
  std::sort(all.begin(), all.end());
  ASSERT_EQ(all.size(), static_cast<size_t>(kJobs));
  for (int i = 0; i < kJobs; ++i) {
    EXPECT_EQ(all[i], i);
  }
}
