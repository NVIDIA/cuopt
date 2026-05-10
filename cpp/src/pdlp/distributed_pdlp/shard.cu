


void pre_SpMV_communication(bool is_A_x){
    // Prepare the send_buffers
    for (auto& shard: shards){
        comm_planner_t<i_t, f_t>& plan = is_A_x ? shard.x_plan : shard.y_plan;
        raft::device_setter guard(shard.device_id);
        for (size_t peer = 0; peer < partition.nb_parts; peer++){
            if (peer == shard.rank) continue;
            thrust::gather(
                shard.handle.get_thrust_policy(), // TODO what exactly do we put here
                plan.send_indices_per_peer[peer].begin(),
                plan.send_indices_per_peer[peer].end(),
                plan.full_local.begin(),
                plan.send_buf_per_peer[peer].begin());
        }
    }
    // Will merge them if it works
    ncclgroupstart();
    // Send all the data current shard has to send
    for (auto& shard: shards){
        comm_planner_t<i_t, f_t>& plan = is_A_x ? shard.x_plan : shard.y_plan;
        raft::device_setter guard(shard.device_id);
        for (size_t peer = 0; peer < partition.nb_parts; peer++){
            if (peer == shard.rank) continue;
            ncclSend(plan.send_buf_per_peer[peer].data(), plan.nb_elt_send_per_peer[peer], peer)
        }
    }
    // Receive all the data current shard has to receive
    for (auto& shard: shards){
        comm_planner_t<i_t, f_t>& plan = is_A_x ? shard.x_plan : shard.y_plan;
        raft::device_setter guard(shard.device_id);
        for (size_t peer = 0; peer < partition.nb_parts; peer++){
            if (peer == shard.rank) continue;
        f_t* recv_buff = &plan.full_local[offset_per_peer[peer]];
        ncclRecv(recv_buff, plan.nb_elt_recv_per_peer[peer], peer);
        }
    }
    ncclgroupend()
}