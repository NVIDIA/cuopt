#pragma once

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <vector>

namespace cuopt::linear_programming::detail {

class managed_stream_pool {
 public:
  static constexpr std::size_t default_size{16};  ///< Default stream pool size

  /**
   * @brief Construct a new cuda stream pool object of the given non-zero size
   *
   * @throws logic_error if `pool_size` is zero
   * @param pool_size The number of streams in the pool
   */
  explicit managed_stream_pool(std::size_t pool_size = default_size) : streams_(pool_size)
  {
    RMM_EXPECTS(pool_size > 0, "Stream pool size must be greater than zero");
  }
  ~managed_stream_pool() = default;

  managed_stream_pool(managed_stream_pool&&)                 = delete;
  managed_stream_pool(managed_stream_pool const&)            = delete;
  managed_stream_pool& operator=(managed_stream_pool&&)      = delete;
  managed_stream_pool& operator=(managed_stream_pool const&) = delete;

  /**
   * @brief Get a `cuda_stream_view` of a stream in the pool.
   *
   * This function is thread safe with respect to other calls to the same function.
   *
   * @return rmm::cuda_stream_view
   */
  rmm::cuda_stream_view get_stream() const noexcept
  {
    int stream_id = (next_stream++) % streams_.size();
    // std::cout<<"get stream "<<stream_id<<"\n";
    end_unsycned = std::max(stream_id, end_unsycned);
    return streams_[stream_id].view();
  }

  /**
   * @brief Get the number of streams in the pool.
   *
   * This function is thread safe with respect to other calls to the same function.
   *
   * @return the number of streams in the pool
   */
  std::size_t get_pool_size() const noexcept { return streams_.size(); }

  void wait_issued_on_event(cudaEvent_t e)
  {
    for (int i = 0; i < end_unsycned + 1; ++i) {
      cudaStreamWaitEvent(streams_[i].view(), e, 0);
    }
  }

  int reset_issued() const noexcept
  {
    int max_streams = end_unsycned;
    end_unsycned    = -1;
    next_stream     = 0;
    return max_streams;
  }

  std::vector<cudaEvent_t> create_events_on_issued()
  {
    std::vector<cudaEvent_t> events(end_unsycned + 1);
    for (auto& e : events) {
      cudaEventCreate(&e);
    }
    for (int i = 0; i < end_unsycned + 1; ++i) {
      cudaEventRecord(events[i], streams_[i].view());
    }
    return events;
  }

  void sync_test_all_issued()
  {
    for (int i = 0; i < end_unsycned + 1; ++i) {
      streams_[i].synchronize();
      RAFT_CHECK_CUDA(streams_[i]);
    }
    end_unsycned = -1;
    next_stream  = 0;
  }

  void sync_all() const noexcept
  {
    for (size_t i = 0; i < streams_.size(); ++i) {
      streams_[i].synchronize();
    }
    end_unsycned = -1;
    next_stream  = 0;
  }

  void sync_all_issued() const noexcept
  {
    // std::cout<<"sync "<<end_unsycned + 1<<" threads\n";
    for (int i = 0; i < end_unsycned + 1; ++i) {
      streams_[i].synchronize();
    }
    end_unsycned = -1;
    next_stream  = 0;
  }

 private:
  std::vector<rmm::cuda_stream> streams_;
  mutable int next_stream{};
  mutable int end_unsycned{-1};
};

}  // namespace cuopt::linear_programming::detail
