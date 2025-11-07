/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <utilities/common_utils.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <fstream>
#include <limits>
#include <unordered_set>
#include <vector>

namespace cuopt {
namespace routing {
namespace test {

struct Point {
  double x;
  double y;
};

constexpr double euc_2d(Point a, Point b)
{
  return std::hypot(double(a.x - b.x), double(a.y - b.y));
  // return std::sqrt(std::pow(a.x - b.x, 2) + std::pow(a.y - b.y, 2));
}

template <typename f_t>
void build_dense_matrix(f_t* mat, const std::vector<f_t>& x, const std::vector<f_t>& y)
{
  auto nodes = x.size();
  for (auto i = 0u; i < nodes; ++i) {
    for (auto j = 0u; j < nodes; ++j) {
      Point p1{x[i], y[i]};
      Point p2{x[j], y[j]};
      auto dist          = euc_2d(p1, p2);
      mat[i * nodes + j] = dist;
    }
  }
}

template <typename i_t, typename f_t>
struct Route {
  std::vector<i_t> cities;
  std::vector<i_t> demand_h;
  std::vector<i_t> capacity_h;
  std::vector<i_t> earliest_time_h;
  std::vector<i_t> latest_time_h;
  std::vector<i_t> service_time_h;
  std::vector<i_t> pickup_indices_h;
  std::vector<i_t> delivery_indices_h;
  std::vector<f_t> x_h;
  std::vector<f_t> y_h;
  std::vector<f_t> cost_matrix_h;
  i_t n_vehicles;
  i_t n_locations;
};

template <typename i_t, typename f_t>
void load_tsp(std::string const& fname, Route<i_t, f_t>& input)
{
  std::fstream fs;
  fs.open(fname);
  std::string line;
  std::vector<std::string> tokens;
  input.n_vehicles = 1;
  while (std::getline(fs, line) && line.find(':') != std::string::npos) {
    tokens           = cuopt::test::split(line, ':');
    auto strip_token = cuopt::test::split(tokens[0], ' ')[0];
    if (strip_token == "DIMENSION") input.n_locations = std::stoi(tokens[1]);
  }

  i_t dump, city_id = 0;
  i_t cities_size = input.cities.size();
  while (cities_size < input.n_locations) {
    f_t x, y;

    fs >> dump;
    fs >> x;
    fs >> y;

    input.cities.push_back(city_id);
    input.x_h.push_back(x);
    input.y_h.push_back(y);
    input.earliest_time_h.push_back(0);
    input.latest_time_h.push_back(std::numeric_limits<short>::max());
    input.service_time_h.push_back(0);
    input.demand_h.push_back(0);
    input.capacity_h.push_back(1);
    city_id++;
  }
  fs.close();
}

template <typename i_t, typename f_t>
void load_cvrp(const std::string& fname, Route<i_t, f_t>& input)
{
  std::fstream fs;
  fs.open(fname);
  cuopt_assert(fs.is_open(), "File cannot be opened.");

  std::string line;
  std::vector<std::string> tokens;
  i_t capacity{};
  while (std::getline(fs, line) && line.find(':') != std::string::npos) {
    tokens           = cuopt::test::split(line, ':');
    auto strip_token = cuopt::test::split(tokens[0], ' ')[0];
    if (strip_token == "CAPACITY") capacity = std::stoi(tokens[1]);
    if (strip_token == "DIMENSION") input.n_locations = std::stoi(tokens[1]);
  }

  // add 5 vehicles as a margin of error(we should be within 7 with most cases)
  input.n_vehicles = input.n_locations / 5;
  input.capacity_h.assign(input.n_vehicles, capacity);

  i_t dump, city_id = 0;
  i_t cities_size = input.cities.size();
  while (cities_size < input.n_locations) {
    f_t x, y;

    fs >> dump;
    fs >> x;
    fs >> y;

    input.cities.push_back(city_id);
    input.x_h.push_back(x);
    input.y_h.push_back(y);
    input.earliest_time_h.push_back(0);
    input.latest_time_h.push_back(std::numeric_limits<short>::max());
    input.service_time_h.push_back(0);
    city_id++;
  }

  std::string tmp;
  fs >> tmp;
  i_t demand_size = input.demand_h.size();
  while (demand_size < input.n_locations) {
    i_t city_id, demand;
    fs >> city_id;
    fs >> demand;
    input.demand_h.push_back(demand);
  }
  fs.close();
}

template <typename i_t, typename f_t>
void load_acvrp(const std::string& fname, Route<i_t, f_t>& input)
{
  std::fstream fs;
  fs.open(fname);
  std::string line;
  std::vector<std::string> tokens;
  i_t capacity{};
  while (std::getline(fs, line) && line.find(':') != std::string::npos) {
    tokens           = cuopt::test::split(line, ':');
    auto strip_token = cuopt::test::split(tokens[0], ' ')[0];
    if (strip_token == "CAPACITY") capacity = std::stoi(tokens[1]);
    if (strip_token == "DIMENSION") input.n_locations = std::stoi(tokens[1]);
    if (strip_token == "VEHICLES") input.n_vehicles = std::stoi(tokens[1]);
  }
  // add 5 vehicles as a margin of error(we should be within 7 with most cases)
  input.n_vehicles += 7;
  input.capacity_h.assign(input.n_vehicles, capacity);

  i_t matrix_size = input.n_locations * input.n_locations;
  for (i_t i = 0; i < matrix_size; i++) {
    f_t curr_edge;
    fs >> curr_edge;
    input.cost_matrix_h.push_back(curr_edge);
  }

  for (i_t city = 0; city < input.n_locations; city++) {
    input.cities.push_back(city);
    input.x_h.push_back(0.f);
    input.y_h.push_back(0.f);
    input.earliest_time_h.push_back(0);
    input.latest_time_h.push_back(std::numeric_limits<short>::max());
    input.service_time_h.push_back(0);
  }

  std::string tmp;
  fs >> tmp;
  i_t demand_size = input.demand_h.size();
  while (demand_size < input.n_locations) {
    i_t city_id, demand;
    fs >> city_id;
    fs >> demand;
    input.demand_h.push_back(demand);
  }
  std::swap(input.demand_h[0], input.demand_h[input.n_locations - 1]);
  fs.close();
}

template <typename i_t, typename f_t>
void load_cvrptw(const std::string& fileName, Route<i_t, f_t>& route, i_t limit)
{
  std::ifstream infile(fileName.c_str());
  cuopt_assert(infile.is_open(), "File cannot be opened.");

  std::string str;
  long dump;

  getline(infile, str);
  getline(infile, str);
  getline(infile, str);
  getline(infile, str);

  infile >> route.n_vehicles;

  i_t capacity{};
  infile >> capacity;
  route.capacity_h.assign(route.n_vehicles, capacity);

  getline(infile, str);
  getline(infile, str);
  getline(infile, str);
  getline(infile, str);

  i_t curr_node_id = 0;
  while (curr_node_id < limit && infile >> dump) {
    f_t cx, cy;
    i_t demand, ready, due, service;
    infile >> cx;
    infile >> cy;
    infile >> demand;
    infile >> ready;
    infile >> due;
    infile >> service;

    route.x_h.push_back(cx);
    route.y_h.push_back(cy);
    route.cities.push_back(curr_node_id);
    route.demand_h.push_back(demand);
    route.earliest_time_h.push_back(ready);
    route.latest_time_h.push_back(due);
    route.service_time_h.push_back(service);
    curr_node_id++;
  }
  route.n_locations = route.x_h.size();
}

template <typename i_t, typename f_t>
void load_pickup(const std::string& fileName, Route<i_t, f_t>& route)
{
  std::ifstream infile(fileName.c_str());
  cuopt_assert(infile.is_open(), "File cannot be opened.");

  std::string str;
  long dump;

  infile >> route.n_vehicles;

  i_t capacity{};
  infile >> capacity;
  route.capacity_h.assign(route.n_vehicles, capacity);
  infile >> dump;
  i_t curr_node_id = 0;
  std::unordered_set<int> parsed_orders;
  while (infile >> dump) {
    f_t cx, cy;
    i_t demand, ready, due, service, pickup_index, delivery_index;
    infile >> cx;
    infile >> cy;
    infile >> demand;
    infile >> ready;
    infile >> due;
    infile >> service;
    infile >> pickup_index;
    infile >> delivery_index;

    route.x_h.push_back(cx);
    route.y_h.push_back(cy);
    route.cities.push_back(curr_node_id);
    route.demand_h.push_back(demand);
    route.earliest_time_h.push_back(ready);
    route.latest_time_h.push_back(due);
    route.service_time_h.push_back(service);
    if (pickup_index != 0) {
      delivery_index = curr_node_id;
    } else if (delivery_index != 0) {
      pickup_index = curr_node_id;
    }
    if (pickup_index != 0 || delivery_index != 0) {
      if (!parsed_orders.count(pickup_index)) {
        route.pickup_indices_h.push_back(pickup_index);
        route.delivery_indices_h.push_back(delivery_index);
        parsed_orders.insert(pickup_index);
        parsed_orders.insert(delivery_index);
      }
    }
    curr_node_id++;
  }
  route.n_locations = route.x_h.size();
}
}  // namespace test
}  // namespace routing
}  // namespace cuopt
