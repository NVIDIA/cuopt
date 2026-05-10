


namespace cuopt::linear_programming::detail {


template <typename i_t>
struct rank_data_t {
  // === Ownership ===
  std::vector<i_t> owned_var_indices;       // global indices of variables in S_r
  std::vector<i_t> owned_constr_indices;    // global indices of constraints in T_r
  // === Send plan: per peer, LOCAL positions to gather + send ===
  std::vector<std::vector<i_t>> y_send_per_peer;     // [peer] -> local positions in T_r to send
  std::vector<std::vector<i_t>> x_send_per_peer;   // [peer] -> local positions in S_r to send
  // === Recv plan: per peer, contiguous slot in halo region ===
  std::vector<int> y_recv_counts;        // [peer] -> count
  std::vector<int> y_recv_offsets;       // [peer] -> offset in dual halo region
  std::vector<int> x_recv_counts;
  std::vector<int> x_recv_offsets;
};


template <typename i_t, typename f_t>
class partition_t {
  public:
    // not sure, good luck hihi
    partition_t(std::vector<i_t> parts, std::vector<i_t> A_row_offsets, std::vector<i_t> A_indices, std::vector<i_t> A_t_row_offsets, std::vector<i_t> A_t_indices, );
    partition_t(const problem_t<i_t, f_t>& op_problem);
    void export_to_file(const std::string& partition_file) const;

  size_t nb_parts;
  
  std::vector<i_t> raw_parts;
  std::vector<i_t> cstr_parts;
  std::vector<i_t> var_parts;
  std::vector<rank_data_t<i_t>> rank_data; // [rank] -> partition data for this rank

  private:
  void fill_data();
  void validate() const;

};
} // namespace cuopt::linear_programming::detail