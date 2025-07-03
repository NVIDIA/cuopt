/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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

#include "diversity_config.hpp"
#include "helpers.hpp"
#include "population.hpp"

#include <utilities/seed_generator.cuh>
#include <utilities/timer.hpp>
#include "../crossovers/dispose.hpp"
#include "../crossovers/eax_recombiner.hpp"
#include "../crossovers/inversion_recombiner.hpp"
#include "../crossovers/ox_recombiner.cuh"
#include "../crossovers/srex_recombiner.hpp"
#include "../solution/pool_allocator.cuh"
#include "../utilities/env_utils.hpp"

#include <cuda_profiler_api.h>

#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

namespace cuopt::routing {

enum class recombiner_t : int { SREX, OX, EAX, AEAX, IX, DISPOSE, SIZE };

constexpr int max_sol_per_population          = 32;
constexpr int default_reserve_population_size = 32;
constexpr int max_perturbation_reserve        = 10;
constexpr int min_perturbation_reserve        = 0;

/*! \brief { The flow:
 *              1) Generate 3 solutions for each of 10 initial islands (
 * solutions may be infeasible ) 2) Solve the problem for each of 10 islands
 * separately by means of procedure improve(population) = ( get two random
 * solutions -> recombine -> reinsert to population -> if (criterion) lower
 * diversity ) 2')  As an option incorporate inter-island search 3) Populate
 * best solution from each of the islands to reserve population. 4) while (
 * !exit_criterion )
 *                 {
 *                      - copy 5 random soltions from reserve_population to
 * working population
 *                      - improve ( working_population )
 *                      - propagate best solution to reserve
 *                      - adjust weights of working_population
 *                  }
 *
 * } */

struct recombine_stats {
  int attempts;
  int success;
  int better_than_one;
  int better_than_both;

  void reset()
  {
    attempts         = 0;
    success          = 0;
    better_than_one  = 0;
    better_than_both = 0;
  }

  void add_success() { ++success; }

  void update_improve_stats(double cost_new, double cost_first, double cost_second)
  {
    if (cost_new < (std::min(cost_first, cost_second) - MOVE_EPSILON)) ++better_than_both;
    if (cost_new < (std::max(cost_first, cost_second) - MOVE_EPSILON)) ++better_than_one;
  }

  void add_attempt() { ++attempts; }

  void print(const char* recombiner_name, file_buffer_t& f)
  {
    fprintf(f.file_ptr,
            "%s : (better_than_one: %d better_than_both: %d success: %d "
            "attempts: %d)\t",
            recombiner_name,
            better_than_one,
            better_than_both,
            success,
            attempts);
  }
};

struct all_recombine_stats {
  static constexpr size_t recombiner_count      = static_cast<int>(recombiner_t::SIZE);
  static constexpr std::array recombiner_labels = {"SREX", "OX", "EAX", "AEAX", "IX", "DISPOSE"};

  std::array<recombine_stats, recombiner_count> stats;

  static_assert(recombiner_labels.size() == (size_t)recombiner_t::SIZE,
                "Mismatch between names and enums");

  // enum of the last attempted recombiner
  std::optional<recombiner_t> last_attempt;

  void reset()
  {
    for (size_t i = 0; i < recombiner_count; ++i) {
      stats[i].reset();
    }
    last_attempt.reset();
  }

  void add_attempt(recombiner_t r)
  {
    last_attempt = r;
    stats[static_cast<int>(r)].add_attempt();
  }

  void add_success() { stats[static_cast<int>(last_attempt.value())].add_success(); }

  void update_improve_stats(double cost_new, double cost_first, double cost_second)
  {
    stats[static_cast<int>(last_attempt.value())].update_improve_stats(
      cost_new, cost_first, cost_second);
  }

  void print([[maybe_unused]] file_buffer_t& f)
  {
    benchmark_call(fprintf(f.file_ptr, "Recombiner stats: "));
    for (size_t i = 0; i < recombiner_count; ++i) {
      benchmark_call(stats[i].print(recombiner_labels[i], f));
    }
    benchmark_call(fprintf(f.file_ptr, "\n"));
    benchmark_call(fflush(f.file_ptr));
  }
};

template <class allocator, class solution, class problem, class generator, class local_modifier>
struct solve {
  const problem* p;

  // solution resource pool
  allocator& pool_allocator;

  //! Working weights that might change through optimization
  costs weights;

  // weight adjustment coefficient
  double adjust_coeff_weights;

  //! Weights that measure the final result. These remain unchanged throught
  //! optimization process.
  costs final_weights;

  int target_vehicles_{-1};

  std::vector<int> target_vehicle_ids_;

  //! Vector of islands: for each of islands problem is solved separately. If
  //! time is left islands are propadated to reserve.
  std::vector<population<allocator, solution, problem>> initial_islands;

  //! Diversified population storing best solutions found so far (clearing
  //! radius and weights for this population do not change)
  population<allocator, solution, problem> reserve_population;

  //! Working population: here mechanisms for weight management and clearing
  //! radius adjustment are applied
  population<allocator, solution, problem> working_population;
  // This vector is used to gather solutions from various sources
  std::vector<solution> working_vector;

  // Optimization tools:
  srex<solution> sr;
  dispose<solution> d;
  a_eax<solution> eax;
  detail::OX<solution> ox;
  inversion<solution> inversion;
  local_modifier lm;
  generator g;
  injection_info_t<allocator, solution, problem> injection_info;

  all_recombine_stats recombine_stats;

  //! Diversity levels (clearing radius) used in the improvement phase
  std::vector<double> diversity_levels;
  //! Number of steps without improvement considered stagnation for each
  //! clearing radius
  std::vector<int> step_lengths;

  bool feasible_only{false};
  bool ox_called{false};

  timer_t timer;
  timer_t improvement_timer;
  double reserve_start_time;
  double initial_reserve_threshold;

  int perturbation_count;

  // 100 and less ,<200,<300,<400, ...,800,<1000, 1000 and more
  std::array<int, 11> min_single_island_generation_time = {30 /*<100*/,
                                                           40 /*<200*/,
                                                           50 /*<300*/,
                                                           80 /*<400*/,
                                                           90 /*<500*/,
                                                           100 /*<600*/,
                                                           120 /*<700*/,
                                                           180 /*<800*/,
                                                           240 /*<900*/,
                                                           260 /*<1000*/,
                                                           300 /*>1000*/};

  // temporary solution keeps the allocations alive
  std::pair<solution, solution> temp_pair;
  // buffer changer struct in raii style
  file_buffer_t f;
  // random number generator
  std::mt19937 rng;
  // control variable to limit GES time for prize collection
  double ges_time_fraction = 1.0;

  solve(const problem* p_,
        costs& final_weights_,
        allocator& pool_allocator_,
        std::string file_name,
        const timer_t& timer_)
    : pool_allocator(pool_allocator_),
      p(p_),
      weights(final_weights_),
      final_weights(final_weights_),
      adjust_coeff_weights(10.),
      reserve_population(0.8, max_sol_per_population, final_weights, p, pool_allocator),
      working_population(0.85, max_sol_per_population, final_weights, p, pool_allocator),
      sr(p->get_num_orders()),
      eax(p->order_info.get_num_depot_excluded_orders() + 1),
      ox(p->get_num_orders(), final_weights, pool_allocator_.sol_handles[0]->get_stream()),
      inversion(p->get_num_orders()),
      lm(pool_allocator_),
      g(*p, pool_allocator_),
      injection_info(p, pool_allocator),
      temp_pair(solution{p_, pool_allocator_.sol_handles[0].get()},
                solution{p_, pool_allocator_.sol_handles[0].get()}),
      f(file_name),
      rng(seed_generator::get_seed()),
      timer(timer_),
      improvement_timer(timer_),
      perturbation_count(0)
  {
    raft::common::nvtx::range fun_scope("solve ctr");

    std::uniform_int_distribution<uint64_t> uniform(0, UINT64_MAX);
    s_xoshiro[0]                       = uniform(rng);
    s_xoshiro[1]                       = uniform(rng);
    s_xoshiro[2]                       = uniform(rng);
    s_xoshiro[3]                       = uniform(rng);
    constexpr int max_number_of_levels = 8;

    int max = std::max<int>((int)p->get_num_orders() - 2, 1);

    // Special parametrization of mini problems
    if (max <= 40) {
      diversity_levels.push_back(0.8);
      step_lengths.push_back(20);
    } else {
      int levels_number   = max_number_of_levels;
      int step_number     = std::max<int>(100, std::min<int>((int)p->get_num_orders() / 2, 200));
      double nodes_number = (int)p->get_num_orders();

      while (levels_number > 0 && max > 1) {
        diversity_levels.push_back((double)max / nodes_number);
        step_lengths.push_back(step_number);
        if (step_number > 149) step_number -= 50;
        if (diversity_levels.back() <= 0.55) break;
        auto delta = p->is_cvrp() ? 8 : 32;
        max -= delta;
        levels_number--;
      }
    }
    working_population.threshold = diversity_levels.back();
    recombine_stats.reset();

    if (injection_info.has_info()) { injection_info.load_solutions(); }
    // When we have prize collection, don't spend too much time on ges
    if (p->has_prize_collection()) { ges_time_fraction = 0.1; }
  }

  void benchmark_print([[maybe_unused]] const char* format, ...)
  {
    benchmark_call(va_list argptr);
    benchmark_call(va_start(argptr, format));
    benchmark_call(vfprintf(f.file_ptr, format, argptr));
    benchmark_call(va_end(argptr));
    benchmark_call(fflush(f.file_ptr));
  }

  /*! \brief { Checks if the current weights are equal to final ones } */
  bool weights_equal()
  {
    for (size_t i = 0; i < weights.size(); i++) {
      if (weights[i] != final_weights[i]) return false;
    }
    return true;
  }

  void run_ls_and_exit()
  {
    std::string sol_dir = "";
    int routes_number   = 0;
    // read instance path
    detail::set_if_env_set(sol_dir, "CUOPT_LS_INPUT_PATH");
    // read number of routes for instance
    detail::set_if_env_set(routes_number, "CUOPT_LS_ROUTES");
    if (sol_dir == "" || routes_number == 0) {
      // silently return to normal execution
      return;
    }
    // run LS and print the costs before and after
    while (true) {
      solution sol         = load_solution(sol_dir, routes_number);
      double original_cost = sol.get_cost(final_weights);
      // cost before LS
      printf("BEFORE: feasible %d cost %f\n", sol.is_feasible(), sol.get_cost(final_weights));
      lm.perturbate(sol, final_weights, 1);
      printf("COST AFTER PERTURBATION: feasible %d cost %f\n",
             sol.is_feasible(),
             sol.get_cost(final_weights));
      lm.improve(sol, final_weights, std::numeric_limits<double>::max());
      printf("AFTER: feasible %d cost %f\n", sol.is_feasible(), sol.get_cost(final_weights));
      if (sol.get_cost(final_weights) < original_cost) {
        output_sol(sol);
        exit(0);
      }
    }
  }

  void start_reserve_threshold_adjustment()
  {
    reserve_start_time        = timer.elapsed_time();
    initial_reserve_threshold = reserve_population.threshold;
  }

  void adjust_reserve_threshold()
  {
    const double max_diversity_threshold = 0.99;
    double reserve_time_ratio =
      (timer.elapsed_time() - reserve_start_time) / (timer.get_time_limit() - reserve_start_time);
    reserve_time_ratio *= reserve_time_ratio;
    reserve_population.threshold =
      initial_reserve_threshold +
      reserve_time_ratio * (max_diversity_threshold - initial_reserve_threshold);
  }

  void generate_from_scratch()
  {
    reserve_population.max_solutions = default_reserve_population_size;
    working_population.max_solutions = default_reserve_population_size;
    generate_initial(target_vehicles_);

    auto from_islands = load_sols_from_islands();
    if (initial_islands.size() > 1) {
      int threshold_index = p->is_cvrp() ? std::max(3, find_initial_diversity(from_islands, true))
                                         : find_initial_diversity(from_islands, true);
      reserve_population.threshold = std::max<double>(0.8, diversity_levels[threshold_index]);
      benchmark_print("Resetting the reserve diversity level %d reserve treshold %f: \n",
                      threshold_index,
                      reserve_population.threshold);
    } else {
      reserve_population.threshold = 0.99;
      benchmark_print(
        "Generated only one island, so updating reserve "
        "diversity threshold to %f. We should just "
        "add few more solutions and compute\n",
        reserve_population.threshold);
    }

    // Populate best to reserve_population
    for (auto& a : initial_islands) {
      if (a.is_feasible()) {
        temp_pair.first = a.best_feasible();
        reserve_population.add_solution(timer.elapsed_time(), temp_pair.first);
      } else if (a.current_size() > 0) {
        temp_pair.first = a.best();
        reserve_population.add_solution(timer.elapsed_time(), temp_pair.first);
      }
    }
    // In case of reserve degeneration refill:
    if (reserve_population.current_size() < 10) {
      benchmark_print("Refilling reserve, reserve size: %d\n", reserve_population.current_size());
      refill_reserve(target_vehicle_ids_);
      benchmark_print("Reserve size after refill: %d\n", reserve_population.current_size());
    }
  }

  void generate_from_dir(std::string path)
  {
    auto solutions                   = load_all_solutions_from_dir(target_vehicles_, path);
    reserve_population.threshold     = 0.85;
    reserve_population.max_solutions = default_reserve_population_size;
    working_population.max_solutions = default_reserve_population_size;
    reserve_population.clear();

    for (auto& sol : solutions) {
      reserve_population.add_solution(timer.elapsed_time(), sol);
    }
    benchmark_print("%d solutions loaded to reserve with %f diversity\n",
                    (int)reserve_population.current_size(),
                    reserve_population.threshold);
    const int n_best_solutions    = std::min(5, (int)reserve_population.current_size());
    const int n_sampled_solutions = std::min(27, (int)reserve_population.current_size());
    auto best_solutions           = reserve_population.get_n_best(n_best_solutions);
    auto sampled_solutions        = reserve_population.get_n_random(n_sampled_solutions, false);
    benchmark_print("%d solutions selected\n", n_sampled_solutions);
    reserve_population.clear();
    reserve_population.threshold     = 0.85;
    reserve_population.max_solutions = default_reserve_population_size;
    for (auto& sol : best_solutions) {
      reserve_population.add_solution(timer.elapsed_time(), sol);
    }
    for (auto& sol : sampled_solutions) {
      reserve_population.add_solution(timer.elapsed_time(), sol);
    }
    if (reserve_population.current_size() < 5) { refill_reserve(target_vehicle_ids_); }
  }

  bool check_reserve_degenerated_or_time_reached()
  {
    return reserve_population.current_size() < 2 || timer.check_time_limit();
  }

  void print_population_best(population<allocator, solution, problem>& population)
  {
    if (!population.best().is_feasible() && population.is_feasible()) {
      benchmark_call(output_sol(population.best_feasible()));
    } else {
      benchmark_call(output_sol(population.best()));
    }
  }

  void populate_random_solutions(int number_to_fill)
  {
    auto ret = reserve_population.get_n_random(number_to_fill, true);
    for (size_t i = 0; i < ret.size(); ++i) {
      working_vector.push_back(ret[i]);
    }
  }

  void populate_best_solutions(int number_to_fill)
  {
    auto ret = reserve_population.get_n_best(number_to_fill);
    for (size_t i = 0; i < ret.size(); ++i) {
      working_vector.push_back(ret[i]);
    }
  }

  void populate_eax_based_solutions(int number_to_fill)
  {
    constexpr bool tournament = false;
    for (int i = 0; i < number_to_fill; ++i) {
      reserve_population.get_two_random(temp_pair, tournament);
      bool use_perfect_edges_limit = false;
      bool success =
        eax.recombine(temp_pair.first, temp_pair.second, false, use_perfect_edges_limit);
      if (success) {
        success = lm.add_cycles_request(temp_pair.first, eax.cycles, weights);
        success = success && lm.make_cluster_order_feasible_request(temp_pair.first, weights);
        if (success) { lm.improve(temp_pair.first, weights, timer.remaining_time()); }
        working_vector.push_back(temp_pair.first);
      } else {
        --i;
      }
    }
  }

  void populate_ox_based_solutions(int number_to_fill)
  {
    const auto sol = reserve_population.get_random_solution(true);
    ox.set_weight(weights);
    for (int i = 0; i < number_to_fill; ++i) {
      temp_pair.first  = sol;
      temp_pair.second = reserve_population.get_random_solution(false);
      bool success     = eax.recombine(temp_pair.first, temp_pair.second);
      if (success) {
        success = lm.add_cycles_request(temp_pair.first, eax.cycles, weights);
        success = success && lm.make_cluster_order_feasible_request(temp_pair.first, weights);
        if (success) { lm.improve(temp_pair.first, weights, timer.remaining_time()); }
        working_vector.push_back(temp_pair.first);
      } else {
        --i;
      }
    }
  }

  void populate_ges_generated_solutions(int number_to_fill)
  {
    for (int i = 0; i < number_to_fill; ++i) {
      double gen_time = 2.;
      g.generate_solution(temp_pair.first, target_vehicle_ids_, gen_time, final_weights, timer);
      working_vector.push_back(temp_pair.first);
    }
  }

  void populate_perturbated_solutions(int number_to_fill)
  {
    // this might be inefficient. avoid copy ctr later
    constexpr bool tournament = true;
    auto ret                  = reserve_population.get_n_random(number_to_fill, tournament);
    for (size_t i = 0; i < ret.size(); ++i) {
      lm.perturbate(ret[i], final_weights, 3);
      working_vector.push_back(ret[i]);
    }
  }

  void populate_working_vector()
  {
    working_vector.clear();
    // auto ret = reserve_population.get_n_random(32, false);
    // for (size_t i = 0; i < ret.size(); ++i) {
    //   working_vector.push_back(ret[i]);
    // }

    // Improve population containing 8 random solutions from reserve
    int number_to_fill = std::max<int>(2, std::min<int>(5, reserve_population.current_size() / 2));
    bool include_only_best = timer.elapsed_time() > 0.7 * timer.get_time_limit();
    if (include_only_best) {
      populate_best_solutions(number_to_fill);
    } else {
      populate_random_solutions(number_to_fill);
    }
    // add best from reserve
    // working_vector.push_back(reserve_population.best());

    // reserve_population.get_two_random(temp_pair, true);
    // working_vector.push_back(temp_pair.first);
    // working_vector.push_back(temp_pair.second);

    // const int ox_sol_count = 10;
    // populate_ox_based_solutions(ox_sol_count);

    // const int eax_sol_count = 10;
    // populate_eax_based_solutions(eax_sol_count);

    // const int ges_sol_count = 3;
    // populate_ges_generated_solutions(ges_sol_count);

    // const int perturbation_sol_count = 3;
    // populate_perturbated_solutions(perturbation_sol_count);
  }

  void populate_working_population()
  {
    for (size_t i = 0; i < working_vector.size(); i++) {
      working_population.add_solution(timer.elapsed_time(), working_vector[i]);
    }
  }

  void print_working_weights()
  {
    benchmark_print(" Working weights: \n ");
    for (auto& a : weights) {
      benchmark_print(" %f ", a);
    }
    benchmark_print("\n");
  }

  void add_working_to_reserve()
  {
    if (!weights_equal()) {
      for (auto idx : working_population.indices) {
        if (idx.first && !working_population.solutions[idx.first].second.is_feasible()) {
          lm.improve(
            working_population.solutions[idx.first].second, final_weights, timer.remaining_time());
        }
      }
    }
    working_population.add_solutions_to_island(timer.elapsed_time(), reserve_population);
  }

  void run_make_feasible()
  {
    temp_pair.first = reserve_population.get_random_solution_tournament();
    benchmark_print("Cost before make_feasible : %f \n", temp_pair.first.get_cost(final_weights));

    bool feasibilized =
      g.make_feasible(temp_pair.first, timer.clamp_remaining_time(60), final_weights, false);
    benchmark_print("Cost after make_feasible : %f After make_feasible : %d \n ",
                    temp_pair.first.get_cost(final_weights),
                    feasibilized);

    lm.improve(temp_pair.first, final_weights, timer.remaining_time());
    benchmark_print("Cost after improve : %f \n ", temp_pair.first.get_cost(final_weights));

    reserve_population.add_solution(timer.elapsed_time(), temp_pair.first);
  }

  void run_working_loop()
  {
    improvement_timer = timer;
    while (!timer.check_time_limit()) {
      recombine_stats.reset();
      benchmark_print("time elapsed: %f \n", timer.elapsed_time());
      adjust_reserve_threshold();

      populate_working_vector();
      constexpr bool use_average = false;
      int threshold_index = p->is_cvrp() ? 1 : find_initial_diversity(working_vector, use_average);
      working_population.threshold = diversity_levels[threshold_index];
      if (!p->is_cvrp()) { threshold_index = std::min(4, std::max(2, threshold_index)); }
      populate_working_population();

      // for pure cvrp problems, we add more solutions to the reserve population
      // this is because it is quicker to find feasible solutions in case of
      // cvrp, so we can afford to add more solutions to the reserve population
      // We probably should generalize this for all easy problems
      if (p->is_cvrp()) {
        double time_left       = timer.remaining_time();
        double single_gen_time = std::min(time_left * 0.05, 20.) * ges_time_fraction;
        // add five or so solutions to reserve
        for (int i = 0; i < 5; ++i) {
          g.generate_solution(
            temp_pair.first, target_vehicle_ids_, single_gen_time, final_weights, timer);
          lm.improve(temp_pair.first, final_weights, timer.remaining_time());
          reserve_population.add_solution(timer.elapsed_time(), temp_pair.first);
          working_population.add_solution(timer.elapsed_time(), temp_pair.first);
        }
      }

      if (working_population.current_size() == 0) { continue; }
      double best_before_improvement =
        working_population.is_feasible()
          ? working_population.best_feasible().get_cost(final_weights)
          : std::numeric_limits<double>::max();
      benchmark_call(display_pool(working_population, "Working before: \n"));

      const bool island_generation_mode = false;
      improve_population(working_population, island_generation_mode, threshold_index);

      benchmark_call(display_pool(working_population, "Working after: \n"));

      auto best_found = working_population.best();
      if (!best_found.is_feasible()) {
        lm.perturbate(best_found, final_weights, perturbation_count + 1);
      }

      // adjust working weights
      adjust_weights(best_before_improvement);
      print_working_weights();
      add_working_to_reserve();

      if (!reserve_population.is_feasible()) { run_make_feasible(); }

      print_population_best(working_population);

      working_population.clear();

      if (!best_found.is_feasible()) {
        working_population.add_solution(timer.elapsed_time(), best_found);
      }

      // Adjust working population weights:
      working_population.change_weights(weights);

      benchmark_call(display_pool(reserve_population, "Updated reserve: \n"));
      if (reserve_population.current_size() < 5) { refill_reserve(target_vehicle_ids_); }
      recombine_stats.print(f);
    }
  }

  /*! \brief { Solve the problem with respect to final_weights. The best
   * solution is stored in the reserve. } */
  void perform_search(int routes_number,
                      bool feasible_only_,
                      const std::string& path = "./",
                      bool from_dir           = false)
  {
    raft::common::nvtx::range fun_scope("perform_search");
    feasible_only    = feasible_only_;
    target_vehicles_ = routes_number;

    if (target_vehicles_ > 0) {
      target_vehicle_ids_.resize(target_vehicles_);
      std::iota(target_vehicle_ids_.begin(), target_vehicle_ids_.end(), 0);
    }
    run_ls_and_exit();
    if (!from_dir) {
      generate_from_scratch();
    } else {
      generate_from_dir(path);
    }
    benchmark_call(display_pool(reserve_population));

    if (check_reserve_degenerated_or_time_reached()) {
      print_population_best(reserve_population);
      return;
    }

    start_reserve_threshold_adjustment();
    run_working_loop();
    print_population_best(reserve_population);
  }

  /*! \brief { Is there soltution in the population? }*/
  bool is_best() { return (reserve_population.current_size() > 0); }

  /*! \brief { Get the best solution so far }*/
  solution get_best() { return reserve_population.best(); }

 private:
  /*! \brief { If reserve become to small (possibly due to finding solutions
   * similar to all other) refill with generated ones } */
  void refill_reserve(const std::vector<int>& vehicle_ids, int sols_num = 5)
  {
    raft::common::nvtx::range fun_scope("refill_reserve");

    if (timer.check_time_limit()) return;

    // TODO check refill reserve times
    double time_left       = timer.remaining_time();
    double single_gen_time = std::min(time_left * 0.05, 20.) * ges_time_fraction;
    sols_num               = std::min<int>(
      sols_num, (int)reserve_population.max_solutions - (int)reserve_population.current_size());
    for (int i = 0; i < sols_num; ++i) {
      g.generate_solution(temp_pair.first, vehicle_ids, single_gen_time, final_weights, timer);
      lm.improve(temp_pair.first, final_weights, timer.remaining_time());
      reserve_population.add_solution(timer.elapsed_time(), temp_pair.first);
    }
  }

  /*! \brief { Based on average diversity between island population we fin the
   * best initial diversity level. } */
  int find_initial_diversity(std::vector<solution>& sols, bool avg)
  {
    raft::common::nvtx::range fun_scope("find_initial_diversity");
    int threshold_index = 0;
    double average      = 0.0;
    double max          = 0.0;
    int sum             = 0;
    for (size_t i = 0; i < sols.size(); i++)
      for (size_t j = i + 1; j < sols.size(); j++) {
        sum++;
        double similarity = sols[i].calculate_similarity_radius(sols[j]);
        average += similarity;
        max = std::max(max, similarity);
      }

    if (sum > 0) {
      if (avg)
        average /= (double)sum;
      else
        average = max;

      average         = std::min(average, 0.99);
      threshold_index = diversity_levels.size() - 1;
      for (size_t i = 0; i < diversity_levels.size(); i++) {
        if (diversity_levels[i] > average)
          threshold_index = i;
        else
          break;
      }
      return threshold_index;
    }
    return threshold_index;
  }

  /*! \brief { Generate initial population of solutions. We perform at most 10
   * generations to achieve pool size of at least 3. } */
  void generate_initial(int routes_number, int islands_size = -1)
  {
    raft::common::nvtx::range fun_scope("generate_initial");
    bool first_gen       = true;
    size_t start_index   = std::min<size_t>(3, diversity_levels.size() - 1);
    auto next_injection  = 0;
    auto injection_state = false;
    auto min_island_size = p->is_cvrp()
                             ? diversity_config_t<int>::min_island_size<config_t::CVRP>()
                             : diversity_config_t<int>::min_island_size<config_t::DEFAULT>();

    // if island size is not given
    if (islands_size == -1) {
      size_t generation_time_index =
        std::min<size_t>(min_single_island_generation_time.size() - 1, p->get_num_orders() / 100);
      if (p->is_cvrp()) {
        islands_size = diversity_config_t<int>::island_size<config_t::CVRP>();
      } else {
        islands_size = std::max<int>(
          min_island_size,
          std::min<int>(5,
                        timer.get_time_limit() /
                          (3 * min_single_island_generation_time[generation_time_index])));
      }
    }
    /*
     * 1. Allocate 30% of time for route minimization that happens for the first
     * time
     * 2. Allocate maximum of 5% for fixed route loops. Once we do the route
     * minimization, the next fixed route loops should be fast, so even 5%
     * should be enough
     * 3. Maximum island time is 40%, but this won't be reached in most cases.
     * In the toughest case, 100% might be used. 30 % (for fist gen) + 14 * 5% =
     * 100%. This is because in total we will have at most 15 solutions ( max of
     * 5 islands x 3 per island), and if all of these take the entire allocated
     * time
     */
    auto first_sol_gen_time   = std::min(timer.get_time_limit() * 0.3,
                                       300.);  // Upper limit of 5 mins (targetted for 15 min runs)
    auto sol_gen_time         = ges_time_fraction * std::min(timer.get_time_limit() * 0.05, 60.);
    auto const n_islands_size = islands_size;
    double max_island_generation_time;
    auto pop_size = p->is_tsp      ? diversity_config_t<int>::population_size<config_t::TSP>()
                    : p->is_cvrp() ? diversity_config_t<int>::population_size<config_t::CVRP>()
                                   : diversity_config_t<int>::population_size<config_t::DEFAULT>();
    while (islands_size > 0) {
      if (islands_size == n_islands_size) {
        max_island_generation_time = std::min(timer.get_time_limit(), 2000.);
      } else {
        auto time_left             = (0.6 * timer.get_time_limit()) - timer.elapsed_time();
        max_island_generation_time = std::min(std::max(0.0, time_left * 0.4), 2000.);
      }
      // We should at least generate one solution before exiting. When the time
      // limit is too small, we are returning before generating single solution,
      // especially in debug or assert mode
      if (timer.elapsed_time() >= 0.6 * timer.get_time_limit() && !first_gen) { break; }

      if (first_gen && timer.check_time_limit() && p->solver_settings_ptr->enable_verbose_mode_) {
        std::cout << "Warning:: Time limit is too small to generate a "
                     "solution. Exceeding the time "
                     "limit to generate one solution."
                  << std::endl;
      }
      islands_size--;
      // Set small diversity is introduced inside the islands
      double threshold = diversity_levels[std::min<size_t>(3, diversity_levels.size() - 1)];
      initial_islands.push_back(population<allocator, solution, problem>(
        threshold, pop_size, final_weights, p, pool_allocator));

      auto& a = initial_islands.back();

      timer_t island_creation_timer(max_island_generation_time);
      for (int i = 0; i < min_island_size; ++i) {
        if (first_gen && routes_number == -1) {
          auto time_limit = timer.clamp_remaining_time(first_sol_gen_time);
          // We don't know what vehicles will be used until we generate first
          // time
          std::vector<int> desired_vehicle_ids;
          g.generate_solution(
            temp_pair.first, desired_vehicle_ids, time_limit, final_weights, timer);
          first_gen           = false;
          target_vehicles_    = temp_pair.first.sol.get_n_routes();
          target_vehicle_ids_ = temp_pair.first.sol.get_used_vehicle_ids();
        } else if (injection_info.has_info() && next_injection < injection_info.n_sol) {
          injection_state = true;
          temp_pair.first = injection_info.solutions[next_injection];
          if (!p->has_vehicle_fixed_costs()) {
            auto injection_it = next_injection;
            while (injection_it < injection_info.n_sol &&
                   temp_pair.first.sol.get_n_routes() > target_vehicles_) {
              temp_pair.first = injection_info.solutions[injection_it++];
            }
            next_injection = injection_it;
          }
        } else {
          auto time_limit = timer.clamp_remaining_time(sol_gen_time);
          g.generate_solution(
            temp_pair.first, target_vehicle_ids_, time_limit, final_weights, timer);
          first_gen = false;
        }
        // increase the solution generation time if we didn't find
        if (!temp_pair.first.is_feasible() && !injection_state) { sol_gen_time += 20; }

        bool is_feasible_before_improve = temp_pair.first.is_feasible();

        lm.improve(temp_pair.first, final_weights, timer.remaining_time());

        // If LS is making the initial feasible solutions infeasible,
        // it means that the infeasible weights are not sufficient
        if (is_feasible_before_improve && !temp_pair.first.is_feasible()) {
          auto& inf_cost = temp_pair.first.infeasibility_cost;
          for (size_t inf_dim = 0; inf_dim < inf_cost.size(); ++inf_dim) {
            if (inf_cost[inf_dim] > 0.) { final_weights[inf_dim] *= 10.0; }
          }
        }

        if (injection_info.has_info() && injection_state) {
          auto ret        = a.add_solution(timer.elapsed_time(), temp_pair.first);
          injection_state = false;
          auto accepted   = ret < 0 ? 0 : 1;
          injection_info.accepted[next_injection++] = accepted;
        } else {
          a.add_solution(timer.elapsed_time(), temp_pair.first);
        }
        reserve_population.add_solution(timer.elapsed_time(), temp_pair.first);
        if (island_creation_timer.check_time_limit()) { break; }
      }

      benchmark_call(display_pool(
        a, std::string("Island: ") + std::to_string(initial_islands.size()) + std::string(" \n")));

      // Give all the island generation time as some problems might consume
      // all the time improving the first threshold.
      double improve_time_limit =
        std::max(0.0, max_island_generation_time - island_creation_timer.elapsed_time());

      improvement_timer = timer_t(improve_time_limit);
      benchmark_print(
        "Time limit for improvement %f, elapsed time before "
        "improvement = %f \n",
        improve_time_limit,
        timer.elapsed_time());

      const bool island_generation_mode = true;
      improve_population(a, island_generation_mode, start_index);
      if (timer.check_time_limit()) return;

      benchmark_call(display_pool(a,
                                  std::string(" After initial improvement Island: ") +
                                    std::to_string(initial_islands.size()) + std::string(" \n")));
    }
  }

  /*! \brief { Adjust search weights based on feasibility of each dimension. }
   */
  void adjust_weights(double best_before_improvement)
  {
    raft::common::nvtx::range fun_scope("adjust_weights");

    const auto& best_found       = working_population.best();
    double cost_of_best_feasible = working_population.is_feasible()
                                     ? working_population.best_feasible().get_cost(final_weights)
                                     : std::numeric_limits<double>::max();
    bool is_new_feasible_better  = (cost_of_best_feasible + MOVE_EPSILON) < best_before_improvement;
    double adjust_coeff_tmp      = adjust_coeff_weights;
    // if it was feasible before the improvement
    if (best_before_improvement != std::numeric_limits<double>::max() &&
        !best_found.is_feasible() && is_new_feasible_better) {
      std::uniform_real_distribution<> uni_dis(0.99, 1.01);
      adjust_coeff_tmp = uni_dis(rng);
    }
    // if it was infeasible before the improvement
    else {
      std::uniform_real_distribution<> uni_dis(-0.04, 0.04);
      adjust_coeff_tmp += uni_dis(rng);
      if (adjust_coeff_weights < 1.1) {
        adjust_coeff_weights = 1.06;
      } else {
        adjust_coeff_weights *= 0.8 + uni_dis(rng);
      }
    }

    for (size_t i = 0; i < best_found.infeasibility_cost.size(); i++) {
      if (best_found.infeasibility_cost[i] == 0.0) {
        if (weights[i] > MACHINE_EPSILON) { weights[i] *= ((1. / adjust_coeff_tmp)); }
      } else if (weights[i] < final_weights[i]) {
        weights[i] *= (adjust_coeff_tmp);
      }
    }
  }

  /*! \brief { Improve population by gradually decresing the clearing radius
   * (threshold) and searching with a fixed radius. } */
  void improve_population(population<allocator, solution, problem>& p,
                          bool island_generation_mode,
                          int start_threshold_index,
                          bool consider_expensive_recombiners = true)
  {
    raft::common::nvtx::range fun_scope("improve_population");
    if (p.current_size() < 2) return;

    while (start_threshold_index >= 0) {
      // Lowering the threshold does not require updating the population
      int valid_start_threshold_index =
        std::min(start_threshold_index, (int)step_lengths.size() - 1);
      p.threshold = diversity_levels[valid_start_threshold_index];
      benchmark_print("time elapsed: %f \n", timer.elapsed_time());
      benchmark_print("Improvement steps: %d\n", step_lengths[valid_start_threshold_index]);
      p.add_solutions_to_island(timer.elapsed_time(), reserve_population);
      if (p.best().is_feasible()) {
        p.best_feasible().reset_viable_of_problem();
      } else {
        p.best().reset_viable_of_problem();
      }
      improve_population_fixed_threshold(p,
                                         step_lengths[valid_start_threshold_index],
                                         start_threshold_index,
                                         consider_expensive_recombiners);

      // last iteration, it might be a duplicate ls with the working_loop search
      // but the random nature of LS makes it okay to search twice
      if (!p.best().is_feasible()) {
        temp_pair.first = p.best();
        lm.improve(temp_pair.first, final_weights, timer.remaining_time(), true);
        p.add_solution(timer.elapsed_time(), temp_pair.first);
      }

      start_threshold_index--;
      if (island_generation_mode) {
        // In island generation, and if the first threshold is completed, we can
        // stop improving after 60% of the time limit is used.
        double improve_time_limit =
          std::max(0.0, (0.6 * timer.get_time_limit()) - timer.elapsed_time());
        benchmark_print("Spent time on island generation: %f \n", improvement_timer.elapsed_time());
        benchmark_print("Improvement time remaining on island generation: %f \n",
                        improve_time_limit);
        improvement_timer = timer_t(improve_time_limit);
        if (improvement_timer.check_time_limit()) { return; }
      }
      if (timer.check_time_limit()) return;
    }
    benchmark_print("time elapsed: %f \n", timer.elapsed_time());
  }

  /*! \brief { Improve input population p without changing threshold.If
   * max_iterations_without_improvement tries do not improve best in the pool
   * return } */
  void improve_population_fixed_threshold(population<allocator, solution, problem>& p,
                                          int max_iterations_without_improvement = 100,
                                          int start_threshold_index              = 0,
                                          bool consider_expensive_recombiners    = true)
  {
    // std::cout << "Improve population\n";
    raft::common::nvtx::range fun_scope("improve_population_fixed_threshold");
    if (p.current_size() < 2) return;
    bool improved = true;

    while (improved) {
      int k                 = max_iterations_without_improvement;
      improved              = false;
      double quality_before = p.best_quality();
      while (k-- > 0) {
        fflush(f.file_ptr);
        if (improvement_timer.check_time_limit()) return;

        if (p.current_size() < 2) {
          benchmark_print("Population degenerated \n");

          return;
        }
        constexpr bool tournament = true;
        p.get_two_random(temp_pair, tournament);
        // for inversion make it less equal than 8
        // for eax make it less equal than 5
        bool run_expensive_recombiners =
          consider_expensive_recombiners && start_threshold_index <= 4;
        bool run_cycle_finder = start_threshold_index <= 1;
        if (solution::request_type == request_t::VRP) {
          run_expensive_recombiners = true;
          run_cycle_finder          = true;
        }
        double cost_first  = temp_pair.first.get_cost(weights);
        double cost_second = temp_pair.second.get_cost(weights);
        bool guiding       = false;
        // reset the routes to search before hand so that we can mark the routes
        // that can be searched
        temp_pair.first.unset_routes_to_search();
        temp_pair.second.unset_routes_to_search();
        int working_insertion_index = -1;
        if (recombine(temp_pair.first, temp_pair.second, guiding, run_expensive_recombiners)) {
          auto& offspring = guiding == false ? temp_pair.first : temp_pair.second;
          if (!feasible_only || offspring.is_feasible()) {
            lm.improve(offspring, weights, improvement_timer.remaining_time(), run_cycle_finder);
            recombine_stats.update_improve_stats(
              offspring.get_cost(weights), cost_first, cost_second);
            working_insertion_index = p.add_solution(timer.elapsed_time(), offspring);
          }
        }
        temp_pair.first.set_routes_to_search();
        temp_pair.second.set_routes_to_search();
        // if we have inserted to the first 2 positions
        if (working_insertion_index != -1 && working_insertion_index <= 3) {
          improved = true;
          break;
        }
      }
    }
  }

  void print_route_sizes(solution& a, const char* prefix)
  {
    fprintf(f.file_ptr, "%s : ", prefix);
    for (size_t i = 0; i < a.routes.size(); ++i) {
      fprintf(f.file_ptr,
              "%d , %f : ",
              a.routes[i].length,
              a.sol.get_route(i).get_weighted_cost(detail::get_cuopt_cost(weights)));
    }
    fprintf(f.file_ptr, "\n");
    fflush(f.file_ptr);
  }

  /*! \brief { Recombining two solutions choosing between EAX and SREX at
   * probablity 1/2 (a learning policy could be applied here). }*/
  bool recombine(solution& a,
                 solution& b,
                 bool& guiding,
                 bool consider_expensive_recombiners = false)
  {
    raft::common::nvtx::range fun_scope("recombine");

    guiding      = false;
    bool success = false;
    std::set<recombiner_t> recombine_options;

    if (p->is_cvrp()) {
      recombine_options = {recombiner_t::SREX, recombiner_t::IX};
    } else {
      recombine_options.insert(recombiner_t::OX);
      bool single_route = !(a.get_routes().size() > 1 && b.get_routes().size() > 1);
      if (!single_route) {
        recombine_options.insert(recombiner_t::DISPOSE);
        recombine_options.insert(recombiner_t::SREX);
      }
      // single route PDP
      else if (solution::request_type == request_t::PDP) {
        recombine_options.insert(recombiner_t::EAX);
        recombine_options.insert(recombiner_t::AEAX);
      }
      // True for VRP
      if (consider_expensive_recombiners) {
        recombine_options.insert(recombiner_t::EAX);
        recombine_options.insert(recombiner_t::AEAX);
        if (!a.problem->is_tsp) { recombine_options.insert(recombiner_t::IX); }
      }
      if (recombine_options.size() == 0) { return false; }
    }
    std::uniform_int_distribution<int> dist(0, recombine_options.size() - 1);

    const auto& dimensions_info = a.problem->dimensions_info;

    // Pick a random element from set
    auto recombiner_it = std::begin(recombine_options);
    std::advance(recombiner_it, dist(rng));
    auto recombiner = *recombiner_it;
    recombine_stats.add_attempt(recombiner);

    switch (recombiner) {
      case recombiner_t::DISPOSE: {
        guiding         = next_random() % 2;
        auto& offspring = guiding ? b : a;
        success         = d.recombine(offspring);
        if (success) {
          recombine_stats.add_success();
          lm.add_selected_unserviced_requests(offspring, d.removed_nodes, weights);
        }
        break;
      }

      case recombiner_t::SREX: {
        success         = sr.recombine(a, b, guiding);
        auto& offspring = guiding == false ? a : b;
        if (success) {
          recombine_stats.add_success();
          if (!p->has_prize_collection()) { lm.add_unserviced_request(offspring, weights); }
        }
        break;
      }

      case recombiner_t::IX: {
        success = inversion.recombine(a, b);
        if (success) recombine_stats.add_success();
        break;
      }

      case recombiner_t::AEAX:
      case recombiner_t::EAX: {
        guiding = a.sol.get_n_routes() > b.sol.get_n_routes();
        lm.equalize_routes_and_nodes(a, b, weights);
        bool asymmetric = (recombiner == recombiner_t::AEAX);
        success = guiding ? eax.recombine(b, a, asymmetric) : eax.recombine(a, b, asymmetric);
        if (success) {
          recombine_stats.add_success();
          auto& offspring = guiding ? b : a;
          success         = lm.add_cycles_request(offspring, eax.cycles, weights);
          success         = success && lm.make_cluster_order_feasible_request(offspring, weights);
        }
        break;
      }

      case recombiner_t::OX: {
        ox.set_weight(weights);
        success = ox.recombine(a, b);
        if (success) {
          recombine_stats.add_success();
          success = lm.make_cluster_order_feasible_request(a, weights);
        }
        break;
      }

      case recombiner_t::SIZE: {
        break;
      }
    }

    if (success && p->has_vehicle_breaks()) {
      auto& offspring = guiding ? b : a;
      lm.squeeze_breaks(offspring, weights);
    }

    return success;
  }

  /*! \brief {Copy all solutions from the islands to a vector. Only the best
   * from each island is taken} */
  std::vector<solution> load_sols_from_islands()
  {
    std::vector<solution> ret;
    for (size_t i = 0; i < initial_islands.size(); i++) {
      if (initial_islands[i].current_size()) ret.push_back(initial_islands[i].best());
    }
    return ret;
  }

  solution load_solution(const std::string& path, int routes_number)
  {
    std::cout << path << std::endl;
    std::ifstream instance(path);
    if (!instance.is_open()) {
      std::cout << "Path error\n";
      throw std::invalid_argument("Path error");
    }

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(instance, line))
      lines.push_back(line);

    if (lines.size() <= 1) {
      benchmark_print("Empty file!\n");
      throw std::invalid_argument("Empty file!");
    }
    std::string s;
    std::stringstream ss(lines[0]);

    std::vector<std::pair<int, std::vector<detail::NodeInfo<>>>> inst_data;
    std::set<int> added_node_ids;
    // Note that the BKS search is currently supported only for homogenous case,
    // so we will use first n vehicles. For heterogenous case, we have to read
    // it from the file as well
    std::vector<int> desired_vehicle_ids(lines.size());
    std::iota(desired_vehicle_ids.begin(), desired_vehicle_ids.end(), 0);

    for (size_t j = 0; j < lines.size(); j++) {
      // inst_data.push_back(std::vector<detail::NodeInfo<>>());
      int vehicle_id = desired_vehicle_ids[j];
      std::vector<detail::NodeInfo<>> curr_route;
      std::string s;
      std::stringstream ss(lines[j]);
      int i = 0;
      while (getline(ss, s, ' ')) {
        if (i > 2) {
          int node_id = atoi(s.c_str());
          if (added_node_ids.count(node_id) != 0) {
            benchmark_print("Error, double node ids!\n");
            break;
          }
          auto node = p->get_node_info_of_node(node_id);
          curr_route.push_back(node);
          added_node_ids.insert(node_id);
        }
        i++;
      }
      if (curr_route.size() == 0) {
        benchmark_print("Empty route!\n");
        throw std::invalid_argument("Empty route!");
      }
      inst_data.push_back({vehicle_id, curr_route});
    }
    if (inst_data.size() != (size_t)routes_number) {
      benchmark_print("Error too many routes given!\n");
      throw std::invalid_argument("Error too many routes given!");
    }

    if ((p->get_num_orders() - 1) != (int)added_node_ids.size()) {
      benchmark_print("Rejecting solution! Missing nodes!\n");
      throw std::invalid_argument("Rejecting solution! Missing nodes!");
    }

    if ((p->get_num_orders() - 1) != *std::prev(added_node_ids.end())) {
      benchmark_print("Rejecting solution! Inconsistent ids!\n");
      throw std::invalid_argument("Rejecting solution! Inconsistent ids!");
    }
    if (1 != *added_node_ids.begin()) {
      benchmark_print("Rejecting solution! Inconsistent ids!\n");
      throw std::invalid_argument("Rejecting solution! Inconsistent ids!");
    }

    solution S(p, pool_allocator.sol_handles[0].get(), desired_vehicle_ids);
    std::vector<int> sequence(inst_data.size());
    std::iota(sequence.begin(), sequence.end(), 0);
    S.remove_routes(sequence);
    S.add_new_routes(inst_data);
    return S;
  }

  std::vector<solution> load_all_solutions_from_dir(int routes_number,
                                                    const std::string& input_dir = "./")
  {
    std::vector<solution> solutions;
    std::string sol_dir = input_dir;
    detail::set_if_env_set(sol_dir, "CUOPT_SOL_INPUT_DIR");
    printf("Env to read CUOPT_SOL_INPUT_DIR Reading solutions from dir: %s\n", sol_dir.c_str());
    for (const auto& entry : std::filesystem::directory_iterator(sol_dir)) {
      try {
        solution sol = load_solution(entry.path(), routes_number);
        solutions.emplace_back(std::move(sol));
      } catch (const std::invalid_argument& e) {
        printf("skipping file\n");
        continue;
      }
    }
    // TODO read this with an env var or something like that
    bool load_bks = true;
    if (load_bks) {
      detail::set_if_env_set(sol_dir, "CUOPT_BKS_INPUT_DIR");
      printf("Env to read CUOPT_BKS_INPUT_DIR . Reading BKS from dir: %s\n", sol_dir.c_str());
      for (const auto& entry : std::filesystem::directory_iterator(sol_dir)) {
        try {
          solution sol = load_solution(entry.path(), routes_number);
          solutions.emplace_back(std::move(sol));
        } catch (const std::invalid_argument& e) {
          printf("error loading BKS file\n");
          continue;
        }
      }
    }
    return solutions;
  }

  void display_pool(const population<allocator, solution, problem>& p,
                    const std::string& preambule = " Pool: \n")
  {
    int i = 0;
    fprintf(f.file_ptr, "%s", preambule.c_str());
    fprintf(f.file_ptr, "threshold: %f\n", p.threshold);
    for (auto& index : p.indices) {
      if (index.first == 0 && p.solutions[0].first) {
        fprintf(f.file_ptr, " Best feasible: %f\n", index.second);
      }
      fprintf(f.file_ptr, "%d :  %f\n", i, index.second);
      i++;
    }
    fprintf(f.file_ptr, " -------------- \n");
    fflush(f.file_ptr);
  }

  void output_sol(const solution& S)
  {
    for (size_t i = 0; i < S.get_routes().size(); i++) {
      auto start = S.routes[i].start;
      fprintf(f.file_ptr, "Route %d : ", (int)i + 1);
      while (start.node_type() != node_type_t::DEPOT) {
        if (S.succ[start.node()].node_type() != node_type_t::DEPOT)
          fprintf(f.file_ptr, "%d ", start.node());
        else
          fprintf(f.file_ptr, "%d", start.node());
        start = S.succ[start.node()];
      }
      fprintf(f.file_ptr, "\n");
      fflush(f.file_ptr);
    }
  }
};  // namespace cuopt::routing

}  // namespace cuopt::routing
