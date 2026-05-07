


template <typename i_t, typename f_t>
struct comm_planner_t {

    // The indices of the data we have to send to the others
    // Maybe could merge evrything if it gives a speedup but a bit harder to read
    std::vector<rmm::device_uvector<i_t>> send_indices_per_peer;
    std::vector<int> nb_elt_send_per_peer;
    std::vector<rmm::device_uvector<f_t>> send_buf_per_peer;

    // Where to start writing in full_local for each peer    
    std::vector<i_t> offset_per_peer;
    std::vector<i_t> nb_elt_recv_per_peer;
    rmm::device_uvector<f_t> full_local; // The full var/cstr vector containing all local data then all remote data
};

template <typename i_t, typename f_t>
struct pdlp_shard_t {

  // Local per-rank PDLP data
  raft::handle_t                   handle;          // owned: the actual handle for this shard's device/stream
  problem_t<i_t, f_t>              local_problem;   // owned: holds handle_ptr = &handle (back-ref)
  saddle_point_state_t<i_t, f_t>   saddle_point;    // owned: per-iter state, sized to local
  cusparse_view_t<i_t, f_t>        cusparse_view;   // owned: descriptors bound to local_problem + saddle_point

  // Specific multi-GPU data
  int device_id;
  ncclComm_t                comm;
  comm_planner_t<i_t, f_t> x_plan, y_plan;
};


