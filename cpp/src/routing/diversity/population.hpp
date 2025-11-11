/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include "../solution/solution_handle.cuh"
#include "injection_info.hpp"

#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>
#include "helpers.hpp"
#include "macros.hpp"

namespace cuopt::routing {

/*! \brief {The class maintains the population of solutions. We follow the
 * forced distancing strategy: each two solutions in the population must be
 * sufficiently different in terms of a normalized [0,1] measure. }*/
template <class allocator, class solution, class problem>
struct population {
  //! A clearing radius: The forced difference between solutions. For any two
  //! solutions A,B in population we have dist(A,B) < threshold. Value 1.0
  //! indicates elitary pool as dist(A,B) <= 1
  double threshold;
  //! Max number of solutions in the population (incl. feasible)
  size_t max_solutions;
  //! Weights that allow quality measurement
  costs weights;
  //! A vector with indices to solutions. Vector will be maintained sorted (to
  //! quickly reject weak solutions). Indices [0] stores the objective of best
  //! feasibke solution.
  std::vector<std::pair<size_t, double>> indices;
  //! Physical solution storage (boolean value indicates if the place is
  //! occupied). The place [0] is reserved for the best feasible.
  std::vector<std::pair<bool, solution>> solutions;
  std::ofstream result_file_;
  const problem* problem_ptr;
  int last_stamp{0};

  std::vector<int> helper;
  //! \param threshold { Has to be in range [0.0,1.0]. Value 0.5 is usually
  //! considered very low. }
  population(double threshold_,
             size_t max_solutions_,
             costs& weights_,
             const problem* P_,
             allocator& pool_allocator)
    : max_solutions(max_solutions_), weights(weights_), problem_ptr(P_)
  {
    raft::common::nvtx::range fun_scope("population ctr");
    solutions.reserve(max_solutions_);
    for (size_t i = 0; i < max_solutions_; ++i) {
      bool occupied = false;
      solutions.emplace_back(occupied, solution(P_, pool_allocator.sol_handles[i].get()));
    }
    indices.reserve(max_solutions);
    helper.reserve(max_solutions);
    // indices[0] always points to solutions[0] - a special place for feasible
    // solution
    indices.emplace_back(0, DBL_MAX);
    // if (P_->solver_settings_ptr->dump_best_results_) {
    //   auto curr     = P_->solver_settings_ptr->best_result_file_name_;
    //   auto filename = curr.substr(0, curr.size() - 4) + "_" + name + ".csv";
    //   result_file_.open(filename);
    //   result_file_ <<
    //   "elapsed_time,best_feasible,best_unfeasible_feasible\n";
    // }

    threshold = threshold_;
    RUNTIME_TEST(test_invariant());

    // test_invariant();
  }

  std::vector<solution> get_n_best(int number)
  {
    raft::common::nvtx::range fun_scope("get_n_best");
    number = std::min<int>(number, indices.size() - 1);

    size_t add = (size_t)(!solutions[0].first || solutions[indices[1].first].second.is_feasible());
    helper.resize(indices.size() - add, 0);
    std::iota(helper.begin(), helper.end(), add);
    std::vector<solution> ret;

    while (number > 0 && !helper.empty()) {
      size_t reserve_index = indices[helper[0]].first;
      ret.push_back(solutions[reserve_index].second);
      helper.erase(helper.begin());
      number--;
    }
    return ret;
  }

  /*! \brief { Get at max number of random solutions. If number is unavailible
   * return all availible solutions. } */
  std::vector<solution> get_n_random(int number, bool tournament = false, int ignore_first_n = 0)
  {
    raft::common::nvtx::range fun_scope("get_n_random");
    number = std::min<int>(number, indices.size() - 1 - ignore_first_n);

    size_t add = (size_t)(!solutions[0].first || solutions[indices[1].first].second.is_feasible());
    add += ignore_first_n;
    helper.resize(indices.size() - add, 0);
    std::iota(helper.begin(), helper.end(), add);
    std::vector<solution> ret;

    while (number > 0 && !helper.empty()) {
      auto index = next_random() % helper.size();
      if (tournament) {
        auto index1 = next_random() % helper.size();
        index       = std::min<size_t>(index, index1);
      }
      size_t reserve_index = indices[helper[index]].first;
      ret.push_back(solutions[reserve_index].second);
      helper.erase(helper.begin() + index);
      number--;
    }
    return ret;
  }

  void add_solutions_to_island(int elapsed_time, population& popupation_to_add)
  {
    for (size_t i = 0; i < indices.size(); ++i) {
      size_t index = indices[i].first;
      if (solutions[index].first) {
        popupation_to_add.add_solution(elapsed_time, solutions[index].second);
      }
    }
  }

  /*! \brief { Current number of solutions in the pool }*/
  size_t current_size() { return indices.size() - 1; }

  /*! \brief { Is feasible soution in the pool? } */
  bool is_feasible() { return solutions[0].first; }

  /*! \brief { Best feasible quality }*/
  double feasible_quality() { return indices[0].second; }

  /*! \brief { Best quality }*/
  double best_quality() { return indices[1].second; }

  void dump_results([[maybe_unused]] int elapsed_time)
  {
    // if (problem_ptr->solver_settings_ptr->dump_best_results_ &&
    //     (elapsed_time - last_stamp) >=
    //     problem_ptr->solver_settings_ptr->dump_interval_) {
    //   last_stamp = elapsed_time;
    //   result_file_ << elapsed_time << "," << feasible_quality() << "," <<
    //   best_quality() << "\n"; result_file_.flush();
    // }
  }

  /*! \brief { Best feasible solution. An empty solution may be returned - first
   * test if feasible is in the pool. }*/
  solution best_feasible() { return solutions[0].second; }

  /*! \brief { Best solution. If no solution is in the pool a Seg Fault may
   * occur. }*/
  solution best() { return solutions[indices[1].first].second; }

  /*! \brief { Completely clear population } */
  void clear()
  {
    for (auto& a : solutions)
      a.first = false;
    indices[0].second = DBL_MAX;
    indices.erase(indices.begin() + 1, indices.end());
  }

  /*! \brief { Get random solution. } */
  solution get_random_solution(bool tournament)
  {
    raft::common::nvtx::range fun_scope("get_random_solution");
    // Assert size > 1
    size_t add = (size_t)(!solutions[0].first || solutions[indices[1].first].second.is_feasible());
    size_t i   = add + next_random() % (indices.size() - 1);
    if (tournament) {
      size_t j = add + next_random() % (indices.size() - 1);
      i        = std::min<size_t>(i, j);
    }
    return solutions[indices[i].first].second;
  }

  /*! \brief { Get random solution with strong tournament selection. } */
  solution get_random_solution_tournament() const
  {
    raft::common::nvtx::range fun_scope("get_random_solution_tournament");
    // Assert size > 1
    size_t i1 = 1 + next_random() % (indices.size() - 1);
    for (int i = 0; i < 14; i++) {
      size_t i2 = 1 + next_random() % (indices.size() - 1);
      i1        = std::min<size_t>(i1, i2);
    }
    return solutions[indices[i1].first].second;
  }
  /*! \brief { Get two random solutions. } */
  void get_two_random(std::pair<solution, solution>& random_pair, bool tournament)
  {
    raft::common::nvtx::range fun_scope("get_two_random");
    // Assert size > 2
    size_t add = (size_t)(!solutions[0].first || solutions[indices[1].first].second.is_feasible());
    size_t i   = add + next_random() % (indices.size() - 1);
    size_t j   = add + next_random() % (indices.size() - 2);
    if (tournament) {
      size_t i1 = add + next_random() % (indices.size() - 1);
      size_t j1 = add + next_random() % (indices.size() - 2);
      i         = std::min<size_t>(i, i1);
      j         = std::min<size_t>(j, j1);
    }
    if (j >= i) j++;
    random_pair.first  = solutions[indices[i].first].second;
    random_pair.second = solutions[indices[j].first].second;
  }

  void inject_solutions(int elapsed_time,
                        injection_info_t<allocator, solution, problem>& injection_info)
  {
    if (!injection_info.has_info()) { return; }

    for (int i = 0; i < injection_info.n_sol; ++i) {
      if (injection_info.accepted[i] < 0) {
        auto ret                   = add_solution(elapsed_time, injection_info.solutions[i]);
        auto accepted              = ret < 0 ? 0 : 1;
        injection_info.accepted[i] = accepted;
      }
    }
  }

  /*! \brief { Add a solution to population. Similar solutions may be ejected
   * from the pool. } \return { -1 = not inserted , others = inserted index}
   */
  int add_solution(int elapsed_time, solution& sol)
  {
    raft::common::nvtx::range fun_scope("add_solution");
    double sol_cost = sol.get_cost(weights);

    cuopt_func_call(sol.sol.check_cost_coherence(detail::default_weights));

    // We store the best feasible found so far at index 0.
    if (sol.is_feasible() &&
        (solutions[0].first == false || sol_cost + MOVE_EPSILON < indices[0].second)) {
      solutions[0].first  = true;
      solutions[0].second = sol;
      indices[0].second   = sol_cost;
    }

    // Fast reject
    if (indices.size() == max_solutions && indices.back().second <= sol_cost + MOVE_EPSILON) {
      dump_results(elapsed_time);
      return -1;
    }

    // Find index best solution similar to sol (within the threshold radius) in
    // the indices array
    size_t index = best_similar_index(sol);

    // No similar was found and added solution is better then worse in
    // population (if the population is full)
    if (index == max_solutions) {
      // Place in the solutions vector:
      int hint = -1;
      // If the population is full eject the worse solution
      if (indices.size() == max_solutions) {
        hint = (int)indices.back().first;
        indices.pop_back();
        solutions[hint].first = false;
      }

      // ASSERT ( there is some free place )
      if (hint == -1) hint = find_free_solution_index();

      solutions[hint].first  = true;
      solutions[hint].second = sol;

      int inserted_pos = insert_index(std::pair<size_t, double>((size_t)hint, sol_cost));
      RUNTIME_TEST(test_invariant());
      dump_results(elapsed_time);
      return inserted_pos;

    } else if (sol_cost + MOVE_EPSILON < indices[index].second) {
      eradicate_similar(index, sol);

      size_t free = find_free_solution_index();

      solutions[free].first  = true;
      solutions[free].second = sol;

      // ASSERT ( there is some free place )
      int inserted_pos = insert_index(std::pair<size_t, double>((size_t)free, sol_cost));
      RUNTIME_TEST(test_invariant());
      dump_results(elapsed_time);
      return inserted_pos;
    }

    dump_results(elapsed_time);
    return -1;
  }

  /*! \brief { Change weights applied when measuring solution quality. This
   * reevaluationg of all solutions quality and re sorting the indices. }
   */
  void change_weights(costs& weights_)
  {
    raft::common::nvtx::range fun_scope("change_weights");
    weights = weights_;
    if (indices.size() == 1) return;
    using pr = std::pair<size_t, double>;
    for (size_t i = 1; i < indices.size(); i++)
      indices[i].second = solutions[indices[i].first].second.get_cost(weights);

    std::sort(indices.begin() + 1, indices.end(), [](const pr& a, const pr& b) {
      return a.second < b.second;
    });
    RUNTIME_TEST(test_invariant());
  }

 private:
  /*! \param sol { Input solution }
   *  \return { Index of the best solution similar to sol. If no similar is
   * found we return max_solutions. }*/
  size_t best_similar_index(const solution& sol)
  {
    raft::common::nvtx::range fun_scope("best_similar_index");
    if (indices.size() == 1) return max_solutions;
    for (size_t i = 1; i < indices.size(); i++) {
      if (sol.calculate_similarity_radius(solutions[indices[i].first].second) > threshold) {
        return i;
      }
    }

    RUNTIME_TEST(test_invariant());
    return max_solutions;
  }

  /*! \brief { Selection sort. Insert index maintainig indices sorted }
   *  \param[in] { Index to solutions }
   */
  int insert_index(std::pair<int, double> to_insert)
  {
    raft::common::nvtx::range fun_scope("insert_index");
    // Assert free index is availible
    indices.emplace_back(0, 0.0);
    size_t start = indices.size() - 1;
    while (start > 1 && indices[start - 1].second > to_insert.second) {
      indices[start] = indices[start - 1];
      start--;
    }
    indices[start] = to_insert;
    RUNTIME_TEST(test_invariant());
    return start;
  }

  /*! \brief { Remove solutions similar to sol starting from index start_index}
   *  \param[in] start_index { Index from which we start eradicating similar. }
   */
  void eradicate_similar(size_t start_index, solution& sol)
  {
    raft::common::nvtx::range fun_scope("eradicate_similar");
    for (size_t i = start_index; i < indices.size(); i++) {
      if (sol.calculate_similarity_radius(solutions[indices[i].first].second) > threshold) {
        solutions[indices[i].first].first = false;     // mark place as availible
        indices[i].first                  = SIZE_MAX;  // mark as deleted in indices
      }
    }

    // Copy all element == SIZE_MAX to the right part of the indices array
    size_t count = start_index;
    for (size_t i = start_index; i < indices.size(); i++)
      if (indices[i].first != SIZE_MAX) indices[count++] = indices[i];  // here count is incremented

    indices.erase(indices.begin() + count, indices.end());
    RUNTIME_TEST(test_invariant());
  }

  /*! \brief { Find index in solution array to insert new solution. }
   *  \return { Free index or SIZE_MAX if nonexistant }
   */
  size_t find_free_solution_index()
  {
    raft::common::nvtx::range fun_scope("find_free_solution_index");
    // ASSERT such index exists
    for (size_t i = 1; i < solutions.size(); i++)
      if (solutions[i].first == false) return i;

    RUNTIME_TEST(test_invariant());
    return SIZE_MAX;
  }

  bool test_invariant()
  {
    // Indices size >= 1
    for (size_t i = 1; i < indices.size(); i++) {
      // Every index should point valid solution. Number should match each other
      if (solutions[indices[i].first].first == false) {
        printf("Solution %d empty\n", (int)i);
        return false;
      }
      // Quality in index should match the quality of solution
      if (std::fabs(solutions[indices[i].first].second.get_cost(weights) - indices[i].second) >
          MACHINE_EPSILON) {
        printf("Solution %d quality does not match: %f %f \n",
               (int)i,
               solutions[indices[i].first].second.get_cost(weights),
               indices[i].second);
        return false;
      }
      // Indices should be sorted
      if (i + 1 < indices.size() && indices[i].second > indices[i + 1].second) {
        printf("Indices not sorted: %d \n", (int)i);
        return false;
      }
      // Each two solutions radius should be lower then threshold
      for (size_t j = i + 1; j < indices.size(); j++) {
        if (solutions[indices[i].first].second.calculate_similarity_radius(
              solutions[indices[j].first].second) > threshold) {
          printf("Solutions radius greater then threshold: %d %d\n",
                 (int)indices[i].first,
                 (int)indices[j].first);
          return false;
        }
      }
    }

    // solutions[0] should be feasible
    if (solutions[0].first && !solutions[0].second.is_feasible()) {
      printf(" Non feasible marked as feasible: \n");
      return false;
    }
    if (indices.size() > max_solutions) {
      printf(" Size excess \n");
      return false;
    }

    return true;
  }
};

}  // namespace cuopt::routing
