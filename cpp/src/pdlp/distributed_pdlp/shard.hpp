#pragma once
#include <pdlp/distributed_pdlp/rank_data.hpp>
#include <raft/core/handle.hpp>
#include <nccl.h>
#include <memory>
namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
class pdlp_solver_t;

struct nccl_comm_deleter_t {
  int device_id{-1};
  void operator()(ncclComm* comm) const noexcept
  {
    raft::device_setter guard(device_id);
    if (comm != nullptr) {
      ncclCommDestroy(comm);
    }
  }
};
using nccl_comm_unique_ptr_t = std::unique_ptr<ncclComm, nccl_comm_deleter_t>;

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
  pdlp_shard_t& operator=(const pdlp_shard_t&) = delete;  
  // Specific multi-GPU data
  int device_id;
  rmm::cuda_stream stream;
  raft::handle_t                            handle; 
  nccl_comm_unique_ptr_t comm; 
  rank_data_t<i_t, f_t>     rank_data;
  optimization_problem_t opt_problem;
  problem_t sub_problem;
  std::unique_ptr<pdlp_solver_t<i_t, f_t>> sub_pdlp;
};

}
