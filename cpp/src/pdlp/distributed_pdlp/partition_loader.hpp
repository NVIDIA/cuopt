

partition_loader_t {
    static std::vector<int> parse_distributed_pdlp_partition_file(std::string file);
    std::vector<rank_data_t> create_rank_data_from_parts(const std::vector<i_t>& parts,
        const std::vector<i_t>& A_row_offsets,
        const std::vector<i_t>& A_col_indices,
        const std::vector<f_t>& A_values,
        const std::vector<i_t>& A_t_row_offsets,
        const std::vector<i_t>& A_t_col_indices,
        const std::vector<f_t>& A_t_values,
        i_t nb_parts,
        i_t nb_cstr,
        i_t nb_vars,
        i_t nnz);
}