
namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
partition_t<i_t, f_t>::partition_t(const std::string& partition_file){
    
}

template <typename i_t, typename f_t>
partition_t<i_t, f_t>::partition_t(const problem_t<i_t, f_t>& op_problem)
{
  std::cout << "NOT IMPLEMENTED" << std::endl;
  return; // TODO: Implement
}

template <typename i_t, typename f_t>
void export_to_file(const std::string& partition_file) const{
    std::cout << "NOT IMPLEMENTED" << std::endl;
    return; // TODO: Implement
}



}