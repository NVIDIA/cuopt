


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

  // Specific multi-GPU data
  int device_id;
  ncclComm_t                comm;
  comm_planner_t<i_t, f_t> x_plan, y_plan;

  // Local per-rank PDLP data
  raft::handle_t                   handle;          // owned: the actual handle for this shard's device/stream
  problem_t<i_t, f_t>              local_problem;   // owned: holds handle_ptr = &handle (back-ref)
  saddle_point_state_t<i_t, f_t>   saddle_point;    // owned: per-iter state, sized to local
  cusparse_view_t<i_t, f_t>        cusparse_view;   // owned: descriptors bound to local_problem + saddle_point

  rmm::device_uvector<f_t>         tmp_primal;
  rmm::device_uvector<f_t>         tmp_dual;
  rmm::device_uvector<f_t>         potential_next_primal;
  rmm::device_uvector<f_t>         potential_next_dual;
  rmm::device_uvector<f_t>         dual_slack;
  rmm::device_uvector<f_t>         reflected_primal; // x, so it has primal_size + halo
  rmm::device_uvector<f_t>         reflected_dual; // y, so it has dual_size + halo

  rmm::device_scalar<f_t>          reusable_one;        // = 1.0
  rmm::device_scalar<f_t>          reusable_zero;       // = 0.0
  rmm::device_scalar<f_t>          reusable_neg_one;    // = -1.0

  // ===== Missing for cuPDLP+ Halpern update =====
  rmm::device_uvector<f_t>         initial_primal;      // snapshot at start of restart epoch
  rmm::device_uvector<f_t>         initial_dual;

  i_t                              primal_size_h;
  i_t                              dual_size_h;
  i_t                              primal_halo_size;
  i_t                              dual_halo_size;
  i_t                              full_primal_size_h;// = primal_size_h + primal_halo_size
  i_t                              full_dual_size_h;  // = dual_size_h + dual_halo_size
};


