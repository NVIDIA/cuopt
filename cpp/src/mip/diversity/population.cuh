/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "assignment_hash_map.cuh"
#include "population.cuh"

#include <mip/solution/solution.cuh>
#include <mip/solver.cuh>
#include <utilities/timer.hpp>

#include <mutex>
#include <random>
#include <string>
#include <vector>

namespace cuopt::linear_programming::detail {

// forward declare
template <typename i_t, typename f_t>
class diversity_manager_t;

enum class solution_origin_t { BRANCH_AND_BOUND, CPUFJ, EXTERNAL };

constexpr const char* solution_origin_to_string(solution_origin_t origin)
{
  switch (origin) {
    case solution_origin_t::BRANCH_AND_BOUND: return "B&B";
    case solution_origin_t::CPUFJ: return "CPUFJ";
    case solution_origin_t::EXTERNAL: return "injected";
    default: return "unknown";
  }
}

template <typename i_t, typename f_t>
class population_t {
 public:
  population_t(std::string const& name,
               mip_solver_context_t<i_t, f_t>& context,
               diversity_manager_t<i_t, f_t>& dm,
               int var_threshold_,
               size_t max_solutions_,
               f_t infeasibility_weight_);

  // keep the implementations of simple getters setters in the class
  // -------------------
  // functions without logic:
  // --------------------
  /*! \brief { Current number of solutions in the pool }*/
  size_t current_size() { return indices.size() - 1; }

  /*! \brief { Is feasible soution in the pool? } */
  bool is_feasible() { return solutions[0].first; }

  /*! \brief { Best feasible quality }*/
  f_t feasible_quality() { return indices[0].second; }

  /*! \brief { Best quality }*/
  f_t best_quality() { return indices[1].second; }
  /*! \brief { Best feasible solution. An empty solution may be returned - first test if feasible is
   * in the pool. }*/
  solution_t<i_t, f_t>& best_feasible() { return solutions[0].second; }
  /*! \brief { Best solution. If no solution is in the pool a Seg Fault may occur. }*/
  solution_t<i_t, f_t>& best() { return solutions[indices[1].first].second; }
  /*! \brief { Best solution. If no solution is in the pool a Seg Fault may occur. }*/
  solution_t<i_t, f_t>& solution_at_index(i_t idx) { return solutions[indices[idx].first].second; }
  // initializes the population lazily. after presolve and var removals
  void initialize_population();
  bool is_better_than_best_feasible(solution_t<i_t, f_t>& sol);
  void run_all_recombiners(solution_t<i_t, f_t>& sol);

  void allocate_solutions();

  void clear()
  {
    for (auto& a : solutions)
      a.first = false;
    indices[0].second = std::numeric_limits<f_t>::max();
    indices.erase(indices.begin() + 1, indices.end());
  }

  void clear_except_best_feasible()
  {
    for (auto& a : solutions) {
      a.first = false;
    }
    solutions[indices[0].first].first = true;
    indices.erase(indices.begin() + 1, indices.end());
  }

  // -------------------
  // functions with logic:
  // --------------------
  solution_t<i_t, f_t> get_random_solution(bool tournament);
  // return a pair of random solutions,
  std::pair<solution_t<i_t, f_t>, solution_t<i_t, f_t>> get_two_random(bool tournament);
  /*! \brief { Add a solution to population. Similar solutions may be ejected from the pool. }
   *  \return { -1 = not inserted , others = inserted index}
   */
  i_t add_solution(solution_t<i_t, f_t>&& sol);
  void add_external_solution(const std::vector<f_t>& solution,
                             f_t objective,
                             solution_origin_t origin);
  std::vector<solution_t<i_t, f_t>> get_external_solutions();
  size_t get_external_solution_size();
  void preempt_heuristic_solver();

  void add_solutions_from_vec(std::vector<solution_t<i_t, f_t>>&& solutions);

  // Updates the cstr weights according to the best solutions feasibility
  void compute_new_weights();
  /*! \brief Updates population weights according to the best solutions constraints
   */
  void update_weights();
  // updates qualities of each solution
  void update_qualities();
  // adjusts the threshold of the population
  void adjust_threshold(cuopt::timer_t timer);
  /*! \param sol { Input solution }
   *  \return { Index of the best solution similar to sol. If no similar is found we return
   * max_solutions. }*/
  size_t best_similar_index(solution_t<i_t, f_t>& sol);

  // normalizes the weights according to the cstr importance
  void normalize_weights();
  // returns whether solutions are similar according to the threshold
  bool check_sols_similar(solution_t<i_t, f_t>& sol1, solution_t<i_t, f_t>& sol2) const;
  /*! \brief { Selection sort. Insert index maintainig indices sorted }
   *  \param[in] { Index to solutions }
   */
  i_t insert_index(std::pair<i_t, f_t> to_insert);
  /*! \brief { Remove solutions similar to sol starting from index start_index}
   *  \param[in] start_index { Index from which we start eradicating similar. }
   */
  void eradicate_similar(size_t start_index, solution_t<i_t, f_t>& sol);
  /*! \brief { Find index in solution array to insert new solution. }
   *  \return { Free index or SIZE_MAX if nonexistant }
   */
  size_t find_free_solution_index();

  bool check_if_feasible_similar_exists(size_t start_index, solution_t<i_t, f_t>& sol);

  // finds the diversity of the population and sets the threshold
  void find_diversity(std::vector<solution_t<i_t, f_t>>& initial_sol_vector, bool avg);

  std::vector<solution_t<i_t, f_t>> population_to_vector();
  void halve_the_population();

  void run_solution_callbacks(solution_t<i_t, f_t>& sol);

  void adjust_weights_according_to_best_feasible();

  void start_threshold_adjustment();

  // does some consistency tests
  bool test_invariant();

  void print();

  std::string name;
  mip_solver_context_t<i_t, f_t>& context;
  problem_t<i_t, f_t>* problem_ptr;
  diversity_manager_t<i_t, f_t>& dm;
  i_t var_threshold;
  i_t initial_threshold;
  double population_start_time;
  // the normalization target for the infeasibility
  // this is used to cover the importance of the weights
  f_t infeasibility_importance = 100.;
  size_t max_solutions;
  weight_t<i_t, f_t> weights;
  std::vector<std::pair<size_t, f_t>> indices;
  std::vector<std::pair<bool, solution_t<i_t, f_t>>> solutions;

  struct external_solution_t {
    external_solution_t() = default;
    external_solution_t(const std::vector<f_t>& solution, f_t objective, solution_origin_t origin)
      : solution(solution), objective(objective), origin(origin)
    {
    }
    std::vector<f_t> solution;
    f_t objective;
    solution_origin_t origin;
  };

  std::vector<external_solution_t> external_solution_queue;
  std::vector<external_solution_t> external_solution_queue_cpufj;
  std::mt19937 rng;
  i_t update_iter = 0;
  std::mutex solution_mutex;
  std::atomic<bool> early_exit_primal_generation = false;
  std::atomic<bool> preempt_heuristic_solver_    = false;
  f_t best_feasible_objective                    = std::numeric_limits<f_t>::max();
  assignment_hash_map_t<i_t, f_t> population_hash_map;
  cuopt::timer_t timer;
};

}  // namespace cuopt::linear_programming::detail
