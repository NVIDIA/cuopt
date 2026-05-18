#pragma once
#include <pdlp/distributed_pdlp/rank_data.hpp>
#include <raft/core/handle.hpp>
#include <nccl.h>
#include <memory>
namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
class pdlp_solver_t;

template <typename i_t, typename f_t>
class pdlp_shard_t {
  // Declaration only, will be set as default in shard.cu . Needed to manage cyclic include of pdlp_solver_t.
  public: 
    ~pdlp_shard_t();
  pdlp_shard_t(int device_id,
    rank_data_t<i_t, f_t>&& rd,
    ncclComm_t comm
    /* ???????? */);

  pdlp_shard_t(const pdlp_shard_t&)            = delete;
  pdlp_shard_t& operator=(const pdlp_shard_t&) = delete;  // Specific multi-GPU data
  int device_id;
  raft::handle_t                            handle; 
  ncclComm_t                comm;
  rank_data_t<i_t, f_t>     rank_data;

  std::unique_ptr<pdlp_solver_t<i_t, f_t>> sub_pdlp;
};

}
