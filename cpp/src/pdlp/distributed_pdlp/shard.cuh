

template <typename i_t, typename f_t>
struct pdlp_shard_t {
  size_t rank;
  comm_planner_t<i_t, f_t> x_plan;
  comm_planner_t<i_t, f_t> y_plan;
};


template <typename i_t, f_t>
struct comm_planner_t {

    // The indices of the data we have to send to the others
    // Maybe could merge evrything if it gives a speedup but a bit harder to read
    std::vector<std::vector<int>> send_indices_per_peer;
    std::vector<int> nb_elt_send_per_peer;
    std::vector<rmm::device_uvector<f_t>> send_buf_per_peer;

    // Where to start writing in full_local for each peer    
    std::vector<i_t> offset_per_peer;
    std::vector<i_t> nb_elt_recv_per_peer;
    rmm::device_uvector<f_t> full_local; // The full var/cstr vector containing all local data then all remote data
};