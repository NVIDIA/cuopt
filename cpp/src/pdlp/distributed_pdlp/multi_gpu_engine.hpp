#pragma once

#include <pdlp/distributed_pdlp/shard.hpp>

#include <vector>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
struct multi_gpu_engine_t {
  std::vector<pdlp_shard_t<i_t, f_t>> shards;
};

}  // namespace cuopt::linear_programming::detail
