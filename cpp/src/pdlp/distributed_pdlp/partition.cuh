


namespace cuopt::linear_programming::detail {


template <typename i_t, typename f_t>
class partition_t {
    public:
    partition_t(const std::string& partition_file);
    partition_t(const problem_t<i_t, f_t>& op_problem);


  size_t nb_parts;
  
  std::vector<i_t> raw_parts;
  std::vector<i_t> cstr_parts;
  std::vector<i_t> var_parts;
  std::vector<std::vector<i_t>> owned_cstr_per_part;
  std::vector<std::vector<i_t>> owned_var_per_part;
  std::vector<std::unordered_set<i_t>> needed_cstr_per_part;
  std::vector<std::unordered_set<i_t>> needed_var_per_part;
  std::vector<std::vector<std::vector<i_t>>> sent_cstr_per_part;
  std::vector<std::vector<std::vector<i_t>>> sent_var_per_part;
  std::vector<std::vector<std::vector<i_t>>> received_cstr_per_part;
  std::vector<std::vector<std::vector<i_t>>> received_var_per_part;

  private:
  void fill_data();
  void validate() const;

};
} // namespace cuopt::linear_programming::detail