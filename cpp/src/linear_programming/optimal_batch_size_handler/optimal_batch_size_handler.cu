/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <linear_programming/cusparse_view.hpp>
#include <linear_programming/optimal_batch_size_handler/optimal_batch_size_handler.hpp>
#include <linear_programming/pdlp_constants.hpp>

#include <utilities/event_handler.cuh>

#include <raft/sparse/detail/cusparse_macros.h>

#include <mip/mip_constants.hpp>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
struct SpMM_benchmarks_context_t {
  SpMM_benchmarks_context_t(cusparse_sp_mat_descr_wrapper_t<i_t, f_t>& A,
                            cusparse_sp_mat_descr_wrapper_t<i_t, f_t>& A_T,
                            int primal_size,
                            int dual_size,
                            size_t current_batch_size,
                            raft::handle_t const* handle_ptr)
    : x(static_cast<size_t>(primal_size) * current_batch_size, handle_ptr->get_stream()),
      y(static_cast<size_t>(dual_size) * current_batch_size, handle_ptr->get_stream()),
      buffer_non_transpose_batch(0, handle_ptr->get_stream()),
      buffer_transpose_batch(0, handle_ptr->get_stream()),
      alpha(1, handle_ptr->get_stream()),
      beta(0, handle_ptr->get_stream()),
      A(A),
      A_T(A_T),
      handle_ptr(handle_ptr)
  {
    auto stream_view = handle_ptr->get_stream();

    int rows_primal = primal_size;
    int col_primal  = current_batch_size;
    int ld_primal   = current_batch_size;

    int rows_dual = dual_size;
    int col_dual  = current_batch_size;
    int ld_dual   = current_batch_size;

    x_descr.create(rows_primal, col_primal, ld_primal, x.data(), CUSPARSE_ORDER_ROW);
    y_descr.create(rows_dual, col_dual, ld_dual, y.data(), CUSPARSE_ORDER_ROW);

    // Init buffers for SpMMs
    size_t buffer_size_non_transpose_batch = 0;
    RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsespmm_bufferSize(
      handle_ptr->get_cusparse_handle(),
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      alpha.data(),
      A,
      x_descr,
      beta.data(),
      y_descr,
      (deterministic_batch_pdlp) ? CUSPARSE_SPMM_CSR_ALG3 : CUSPARSE_SPMM_CSR_ALG2,
      &buffer_size_non_transpose_batch,
      stream_view));

    size_t buffer_size_transpose_batch = 0;
    RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsespmm_bufferSize(
      handle_ptr->get_cusparse_handle(),
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      alpha.data(),
      A_T,
      y_descr,
      beta.data(),
      x_descr,
      (deterministic_batch_pdlp) ? CUSPARSE_SPMM_CSR_ALG3 : CUSPARSE_SPMM_CSR_ALG2,
      &buffer_size_transpose_batch,
      stream_view));

    buffer_transpose_batch     = rmm::device_buffer(buffer_size_transpose_batch, stream_view);
    buffer_non_transpose_batch = rmm::device_buffer(buffer_size_non_transpose_batch, stream_view);

#if CUDA_VER_12_4_UP
    // Preprocess buffers for SpMMs
    my_cusparsespmm_preprocess<f_t>(
      handle_ptr->get_cusparse_handle(),
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      alpha.data(),
      A_T,
      y_descr,
      beta.data(),
      x_descr,
      (deterministic_batch_pdlp) ? CUSPARSE_SPMM_CSR_ALG3 : CUSPARSE_SPMM_CSR_ALG2,
      buffer_transpose_batch.data(),
      stream_view);

    my_cusparsespmm_preprocess<f_t>(
      handle_ptr->get_cusparse_handle(),
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      alpha.data(),
      A,
      x_descr,
      beta.data(),
      y_descr,
      (deterministic_batch_pdlp) ? CUSPARSE_SPMM_CSR_ALG3 : CUSPARSE_SPMM_CSR_ALG2,
      buffer_non_transpose_batch.data(),
      stream_view);
#endif

    // First empty run for warm up
    // TODO batch mode: put back CUDA Graphs here once supported for SpMM
    this->launch();
  }

  void launch()
  {
    auto stream_view = handle_ptr->get_stream();
    RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsespmm(
      handle_ptr->get_cusparse_handle(),
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      alpha.data(),
      A,
      x_descr,
      beta.data(),
      y_descr,
      (deterministic_batch_pdlp) ? CUSPARSE_SPMM_CSR_ALG3 : CUSPARSE_SPMM_CSR_ALG2,
      (f_t*)buffer_non_transpose_batch.data(),
      stream_view));

    RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsespmm(
      handle_ptr->get_cusparse_handle(),
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      alpha.data(),
      A_T,
      y_descr,
      beta.data(),
      x_descr,
      (deterministic_batch_pdlp) ? CUSPARSE_SPMM_CSR_ALG3 : CUSPARSE_SPMM_CSR_ALG2,
      (f_t*)buffer_transpose_batch.data(),
      stream_view));
  }

  cusparse_dn_mat_descr_wrapper_t<f_t> x_descr;
  cusparse_dn_mat_descr_wrapper_t<f_t> y_descr;
  rmm::device_uvector<f_t> x;
  rmm::device_uvector<f_t> y;
  rmm::device_buffer buffer_non_transpose_batch;
  rmm::device_buffer buffer_transpose_batch;
  rmm::device_scalar<f_t> alpha;
  rmm::device_scalar<f_t> beta;
  cusparse_sp_mat_descr_wrapper_t<i_t, f_t>& A;
  cusparse_sp_mat_descr_wrapper_t<i_t, f_t>& A_T;
  raft::handle_t const* handle_ptr;
};

template <typename i_t, typename f_t>
static double evaluate_node(cusparse_sp_mat_descr_wrapper_t<i_t, f_t>& A,
                            cusparse_sp_mat_descr_wrapper_t<i_t, f_t>& A_T,
                            i_t primal_size,
                            i_t dual_size,
                            int current_batch_size,
                            int benchmark_runs,
                            raft::handle_t const* handle_ptr)
{
  cuopt_assert(current_batch_size > 0, "Current batch size must be greater than 0");

  rmm::cuda_stream_view stream_view = handle_ptr->get_stream();
  SpMM_benchmarks_context_t<i_t, f_t> spmm_benchmarks_context(
    A, A_T, primal_size, dual_size, current_batch_size, handle_ptr);

  event_handler_t start_event;
  event_handler_t end_event;
  double total_time = 0;
  for (int j = 0; j < benchmark_runs; ++j) {
    start_event.record(stream_view);
    spmm_benchmarks_context.launch();
    end_event.record(stream_view);
    end_event.synchronize();
    double elapsed_time = end_event.elapsed_time_since_ms(start_event);
    total_time += elapsed_time;
  }
  double average_time = total_time / benchmark_runs;
#ifdef BATCH_VERBOSE_MODE
  std::cout << "Average time for batch size " << current_batch_size << " is " << average_time
            << " ms" << std::endl;
  std::cout << "Ratio is " << average_time / current_batch_size << " ms/batch" << std::endl;
#endif
  return average_time / current_batch_size;
}

template <typename i_t, typename f_t>
int optimal_batch_size_handler(const optimization_problem_t<i_t, f_t>& op_problem,
                               int max_batch_size)
{
  cuopt_assert(max_batch_size > 0, "Max batch size must be greater than 0");
  if (max_batch_size == 1) return 1;

  // Try to quickly find what is the optimal batch size for the problem
  // We run the two most ran SpMMs for both A and A_T and compute "time / batch_size"
  // The one with the best ratio has the optimal batch size (since can solve most amount of work in
  // least time) To try to have something representative we run each SpMM 5 times and take the
  // average We do it for both A and A_T and take the sum since both will be run for each batch size

  // We start with batch size 128 (power two better for cuSparse) and try to improve by either
  // multitipling or dividing by 2 each time At max we take 5 steps of search

  constexpr int max_steps          = 4;  // 4 because we already do one step for direction
  constexpr int initial_batch_size = 128;
  constexpr int benchmark_runs     = 5;
  // Take the floor power of two
  // This ensures that we always start with a batch size that is a power of two or
  // initial_batch_size
  int current_batch_size =
    std::pow(2, std::floor(std::log2(std::min(initial_batch_size, max_batch_size))));
  int optimal_batch_size = current_batch_size;
  double best_ratio;
  rmm::cuda_stream_view stream_view = op_problem.get_handle_ptr()->get_stream();

  detail::problem_t<i_t, f_t> problem(op_problem);

  // Init cuSparse views
  cusparse_sp_mat_descr_wrapper_t<i_t, f_t> A;
  cusparse_sp_mat_descr_wrapper_t<i_t, f_t> A_T;
  i_t primal_size = problem.n_variables;
  i_t dual_size   = problem.n_constraints;

  A.create(problem.n_constraints,
           problem.n_variables,
           problem.nnz,
           problem.offsets.data(),
           problem.variables.data(),
           problem.coefficients.data());

  A_T.create(problem.n_variables,
             problem.n_constraints,
             problem.nnz,
             problem.reverse_offsets.data(),
             problem.reverse_constraints.data(),
             problem.reverse_coefficients.data());

  // Sync before starting anything to make sure everything is done
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream_view));

  // Evaluate current, left and right nodes to pick a direction

  const int left_node  = std::max(1, current_batch_size / 2);
  const int right_node = std::min(current_batch_size * 2, max_batch_size);
  double current_ratio = evaluate_node<i_t, f_t>(A,
                                                 A_T,
                                                 primal_size,
                                                 dual_size,
                                                 current_batch_size,
                                                 benchmark_runs,
                                                 op_problem.get_handle_ptr());
  double left_ratio    = evaluate_node<i_t, f_t>(
    A, A_T, primal_size, dual_size, left_node, benchmark_runs, op_problem.get_handle_ptr());
  double right_ratio = evaluate_node<i_t, f_t>(
    A, A_T, primal_size, dual_size, right_node, benchmark_runs, op_problem.get_handle_ptr());
  int current_step = 1;

#ifdef BATCH_VERBOSE_MODE
  std::cout << "Starting batch size: " << current_batch_size << " and ratio: " << current_ratio
            << std::endl;
  std::cout << "Left batch size: " << left_node << " and ratio: " << left_ratio << std::endl;
  std::cout << "Right batch size: " << right_node << " and ratio: " << right_ratio << std::endl;
#endif

  // Left is better, continue descreasing by dividing by 2 until we find worst
  // Then take middle and keep best found
  if (left_ratio < current_ratio) {
#ifdef BATCH_VERBOSE_MODE
    std::cout << "Left is better, continuing decreasing" << std::endl;
#endif
    if (left_node == 1) {
#ifdef BATCH_VERBOSE_MODE
      std::cout << "Left is already 1, returning 1" << std::endl;
#endif
      return 1;
    }
    current_batch_size = left_node;
    best_ratio         = left_ratio;
    optimal_batch_size = current_batch_size;
    do {
      current_batch_size = std::max(1, current_batch_size / 2);
#ifdef BATCH_VERBOSE_MODE
      std::cout << "Evaluating left node: " << current_batch_size << std::endl;
#endif
      left_ratio = evaluate_node<i_t, f_t>(A,
                                           A_T,
                                           primal_size,
                                           dual_size,
                                           current_batch_size,
                                           benchmark_runs,
                                           op_problem.get_handle_ptr());
#ifdef BATCH_VERBOSE_MODE
      std::cout << "Left node ratio: " << left_ratio << std::endl;
#endif
      if (left_ratio < best_ratio)  // Better found continue reducing
      {
        if (current_batch_size == 1) {
#ifdef BATCH_VERBOSE_MODE
          std::cout << "Left is now 1, returning 1" << std::endl;
#endif
          return 1;
        }
        ++current_step;
        best_ratio         = left_ratio;
        optimal_batch_size = current_batch_size;
      } else  // Worst found, stop reducing
      {
#ifdef BATCH_VERBOSE_MODE
        std::cout << "Left was worst, stopping decreasing" << std::endl;
#endif
        break;
      }
    } while (current_step < max_steps);
    // Testing one last time between the two
    const int middle_node = std::max(1, ((current_batch_size * 2) + current_batch_size) / 2);
    cuopt_assert(middle_node > 0, "Middle node should be greater than 0");
#ifdef BATCH_VERBOSE_MODE
    std::cout << "Testing one last time between the two at node: " << middle_node << std::endl;
#endif
    double middle_ratio = evaluate_node<i_t, f_t>(
      A, A_T, primal_size, dual_size, middle_node, benchmark_runs, op_problem.get_handle_ptr());
#ifdef BATCH_VERBOSE_MODE
    std::cout << "Middle node ratio: " << middle_ratio << std::endl;
#endif
    if (middle_ratio < best_ratio)  // Middle is better, returning better
    {
#ifdef BATCH_VERBOSE_MODE
      std::cout << "Middle is better, returning " << middle_node << std::endl;
#endif
      return middle_node;
    } else  // Middle was worst, keep previous best
    {
#ifdef BATCH_VERBOSE_MODE
      std::cout << "Middle was worst, keeping previous best " << optimal_batch_size << std::endl;
#endif
      return optimal_batch_size;
    }
  }
  // Right is better, continue increasing by multiplying by 2 until we find worst or reach max batch
  // size Then take middle and keep best found
  if (right_ratio < current_ratio) {
#ifdef BATCH_VERBOSE_MODE
    std::cout << "Right is better, continuing increasing" << std::endl;
#endif
    if (right_node == max_batch_size)  // Right as already reached max, returning it
    {
#ifdef BATCH_VERBOSE_MODE
      std::cout << "Right is already at max, returning " << right_node << std::endl;
#endif
      return right_node;
    }
    optimal_batch_size = right_node;
    current_batch_size = right_node;
#ifdef BATCH_VERBOSE_MODE
    std::cout << "Current batch size: " << current_batch_size << std::endl;
#endif
    best_ratio = right_ratio;
    do {
      current_batch_size = std::min(current_batch_size * 2, max_batch_size);
#ifdef BATCH_VERBOSE_MODE
      std::cout << "Evaluating right node: " << current_batch_size << std::endl;
#endif
      right_ratio = evaluate_node<i_t, f_t>(A,
                                            A_T,
                                            primal_size,
                                            dual_size,
                                            current_batch_size,
                                            benchmark_runs,
                                            op_problem.get_handle_ptr());
#ifdef BATCH_VERBOSE_MODE
      std::cout << "Right node ratio: " << right_ratio << std::endl;
#endif
      if (right_ratio < best_ratio)  // Better found continue increasing
      {
        if (current_batch_size == max_batch_size)  // Right as already reached max, returning it
        {
#ifdef BATCH_VERBOSE_MODE
          std::cout << "Right is now at max, returning " << current_batch_size << std::endl;
#endif
          return current_batch_size;
        }
        ++current_step;
        best_ratio         = right_ratio;
        optimal_batch_size = current_batch_size;
      } else  // Worst found, stop increasing
      {
#ifdef BATCH_VERBOSE_MODE
        std::cout << "Right was worst, stopping increasing" << std::endl;
#endif
        break;
      }
    } while (current_step < max_steps);
    // Testing one last time between the two
    int middle_node =
      std::min(std::max(1, ((current_batch_size / 2) + current_batch_size) / 2), max_batch_size);
#ifdef BATCH_VERBOSE_MODE
    std::cout << "Testing one last time between the two at node: " << middle_node << std::endl;
#endif
    double middle_ratio = evaluate_node<i_t, f_t>(
      A, A_T, primal_size, dual_size, middle_node, benchmark_runs, op_problem.get_handle_ptr());
#ifdef BATCH_VERBOSE_MODE
    std::cout << "Middle node ratio: " << middle_ratio << std::endl;
#endif
    if (middle_ratio < best_ratio)  // Middle is better, returning better
    {
#ifdef BATCH_VERBOSE_MODE
      std::cout << "Middle is better, returning " << middle_node << std::endl;
#endif
      return middle_node;
    } else  // Middle was worst, keep previous best
    {
#ifdef BATCH_VERBOSE_MODE
      std::cout << "Middle was worst, keeping previous best " << optimal_batch_size << std::endl;
#endif
      return optimal_batch_size;
    }
  }
  // Current is better -> directly return current and don't try to refine
  else {
#ifdef BATCH_VERBOSE_MODE
    std::cout << "Current is better" << std::endl;
#endif
    return current_batch_size;
  }

  cuopt_assert(false, "Should not be here");
  return 0;
}

#if MIP_INSTANTIATE_FLOAT
template int optimal_batch_size_handler<int, float>(
  const optimization_problem_t<int, float>& op_problem, int max_batch_size);
#endif
#if MIP_INSTANTIATE_DOUBLE
template int optimal_batch_size_handler<int, double>(
  const optimization_problem_t<int, double>& op_problem, int max_batch_size);
#endif

}  // namespace cuopt::linear_programming::detail
