/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

/**
 * Standalone VRP gRPC test driver.
 *
 * Reads a cuOpt service JSON file, builds a cpu_routing_problem_t, submits a
 * SolveVRPRequest to cuopt_grpc_server, and prints the RoutingSolution.
 *
 * Usage:
 *   GRPC_VRP_TEST_DRIVER --server localhost:5001 \
 *     --json datasets/cuopt_service_data/cuopt_problem_data.json
 */

#include "routing/grpc_routing_problem_mapper.hpp"

#include <cuopt/routing/cpu_routing_problem.hpp>

#include <cuopt_remote.pb.h>
#include <cuopt_remote_service.grpc.pb.h>
#include <cuopt_routing.pb.h>
#include <cuopt_routing_solution.pb.h>

#include <grpcpp/grpcpp.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

using json = nlohmann::json;

namespace {

void print_usage(char const* argv0)
{
  std::cerr << "Usage: " << argv0
            << " --server HOST:PORT --json PATH [--time-limit SEC] [--verbose]\n";
}

cuopt::routing::cpu_routing_problem_t load_cuopt_json(std::string const& path)
{
  std::ifstream in(path);
  if (!in) { throw std::runtime_error("Failed to open JSON: " + path); }
  json root;
  in >> root;

  cuopt::routing::cpu_routing_problem_t p;

  // Cost matrix (vehicle type "0" by default)
  auto const& cm_data = root.at("cost_matrix_data").at("data");
  for (auto it = cm_data.begin(); it != cm_data.end(); ++it) {
    cuopt::routing::cpu_cost_matrix_t cm;
    cm.vehicle_type = static_cast<uint8_t>(std::stoul(it.key()));
    auto const& mat = it.value();
    p.num_locations = static_cast<int32_t>(mat.size());
    cm.matrix.reserve(static_cast<size_t>(p.num_locations) * p.num_locations);
    for (auto const& row : mat) {
      for (auto const& v : row) {
        cm.matrix.push_back(v.get<float>());
      }
    }
    p.cost_matrices.push_back(std::move(cm));
  }

  auto const& fleet = root.at("fleet_data");
  auto const& locs  = fleet.at("vehicle_locations");
  p.fleet_size      = static_cast<int32_t>(locs.size());
  for (auto const& pair : locs) {
    p.vehicle_start_locations.push_back(pair.at(0).get<int32_t>());
    p.vehicle_return_locations.push_back(pair.at(1).get<int32_t>());
  }

  if (fleet.contains("vehicle_time_windows") && !fleet["vehicle_time_windows"].is_null()) {
    for (auto const& tw : fleet.at("vehicle_time_windows")) {
      p.vehicle_tw_earliest.push_back(tw.at(0).get<int32_t>());
      p.vehicle_tw_latest.push_back(tw.at(1).get<int32_t>());
    }
  }
  if (fleet.contains("skip_first_trips") && !fleet["skip_first_trips"].is_null()) {
    for (auto const& v : fleet.at("skip_first_trips")) {
      p.skip_first_trips.push_back(v.get<bool>() ? 1 : 0);
    }
  }
  if (fleet.contains("drop_return_trips") && !fleet["drop_return_trips"].is_null()) {
    for (auto const& v : fleet.at("drop_return_trips")) {
      p.drop_return_trips.push_back(v.get<bool>() ? 1 : 0);
    }
  }
  if (fleet.contains("vehicle_max_costs") && !fleet["vehicle_max_costs"].is_null()) {
    for (auto const& v : fleet.at("vehicle_max_costs")) {
      p.vehicle_max_costs.push_back(v.get<float>());
    }
  }

  auto const& tasks = root.at("task_data");
  if (tasks.contains("task_locations") && !tasks["task_locations"].is_null()) {
    for (auto const& loc : tasks.at("task_locations")) {
      p.order_locations.push_back(loc.get<int32_t>());
    }
    p.num_orders = static_cast<int32_t>(p.order_locations.size());
  } else {
    p.num_orders = p.num_locations;
  }

  // Capacities: fleet_data.capacities is [dim][vehicle]; task_data.demand is [dim][order]
  if (fleet.contains("capacities") && !fleet["capacities"].is_null() && tasks.contains("demand") &&
      !tasks["demand"].is_null()) {
    auto const& caps   = fleet.at("capacities");
    auto const& demand = tasks.at("demand");
    size_t n_dims      = caps.size();
    for (size_t d = 0; d < n_dims; ++d) {
      cuopt::routing::cpu_capacity_dimension_t dim;
      dim.name = "dim_" + std::to_string(d);
      for (auto const& c : caps.at(d)) {
        dim.capacity.push_back(c.get<int32_t>());
      }
      for (auto const& dem : demand.at(d)) {
        dim.demand.push_back(dem.get<int32_t>());
      }
      p.capacity_dimensions.push_back(std::move(dim));
    }
  }

  if (tasks.contains("task_time_windows") && !tasks["task_time_windows"].is_null()) {
    for (auto const& tw : tasks.at("task_time_windows")) {
      p.order_tw_earliest.push_back(tw.at(0).get<int32_t>());
      p.order_tw_latest.push_back(tw.at(1).get<int32_t>());
    }
  }
  if (tasks.contains("service_times") && !tasks["service_times"].is_null()) {
    std::vector<int32_t> times;
    for (auto const& t : tasks.at("service_times")) {
      times.push_back(t.get<int32_t>());
    }
    p.order_service_times[-1] = std::move(times);
  }

  return p;
}

void print_solution(cuopt::remote::RoutingSolution const& sol)
{
  std::cout << "status=" << static_cast<int>(sol.status()) << " (" << sol.status_message() << ")\n";
  if (!sol.error_message().empty()) { std::cout << "error: " << sol.error_message() << "\n"; }
  std::cout << "vehicle_count=" << sol.vehicle_count()
            << " total_objective=" << sol.total_objective_value() << "\n";
  std::cout << "route (" << sol.route_size() << "):";
  for (int i = 0; i < sol.route_size(); ++i) {
    std::cout << " " << sol.route(i);
  }
  std::cout << "\ntruck_id:";
  for (int i = 0; i < sol.truck_id_size(); ++i) {
    std::cout << " " << sol.truck_id(i);
  }
  std::cout << "\nlocations:";
  for (int i = 0; i < sol.locations_size(); ++i) {
    std::cout << " " << sol.locations(i);
  }
  std::cout << "\nunserviced:";
  for (int i = 0; i < sol.unserviced_nodes_size(); ++i) {
    std::cout << " " << sol.unserviced_nodes(i);
  }
  std::cout << "\n";
}

}  // namespace

int main(int argc, char** argv)
{
  std::string server;
  std::string json_path;
  float time_limit = 10.0f;
  bool verbose     = false;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--server" && i + 1 < argc) {
      server = argv[++i];
    } else if (arg == "--json" && i + 1 < argc) {
      json_path = argv[++i];
    } else if (arg == "--time-limit" && i + 1 < argc) {
      time_limit = std::stof(argv[++i]);
    } else if (arg == "--verbose") {
      verbose = true;
    } else if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      return 0;
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      print_usage(argv[0]);
      return 1;
    }
  }

  if (server.empty() || json_path.empty()) {
    print_usage(argv[0]);
    return 1;
  }

  try {
    std::cout << "Loading cuOpt JSON: " << json_path << "\n";
    auto problem = load_cuopt_json(json_path);
    std::cout << "Problem: " << problem.num_locations << " locations, " << problem.fleet_size
              << " vehicles, "
              << (problem.num_orders < 0 ? problem.num_locations : problem.num_orders)
              << " orders, " << problem.capacity_dimensions.size() << " capacity dims, "
              << problem.cost_matrices.size() << " cost matrices\n";

    cuopt::remote::SubmitJobRequest submit_req;
    auto* vrp = submit_req.mutable_vrp_request();
    vrp->mutable_header()->set_version(1);
    vrp->mutable_header()->set_problem_category(cuopt::remote::VRP);
    cuopt::routing::map_routing_problem_to_proto(problem, vrp->mutable_problem());
    vrp->mutable_settings()->set_time_limit(time_limit);
    vrp->mutable_settings()->set_verbose(verbose);

    std::cout << "Request proto size: " << submit_req.ByteSizeLong() << " bytes\n";
    std::cout << "Connecting to " << server << "...\n";

    auto channel = grpc::CreateChannel(server, grpc::InsecureChannelCredentials());
    auto stub    = cuopt::remote::CuOptRemoteService::NewStub(channel);

    cuopt::remote::SubmitJobResponse submit_resp;
    {
      grpc::ClientContext ctx;
      auto status = stub->SubmitJob(&ctx, submit_req, &submit_resp);
      if (!status.ok()) {
        std::cerr << "SubmitJob failed: " << status.error_message() << "\n";
        return 1;
      }
    }
    std::string job_id = submit_resp.job_id();
    std::cout << "Submitted job_id=" << job_id << "\n";

    // Poll until complete
    for (;;) {
      cuopt::remote::StatusRequest status_req;
      status_req.set_job_id(job_id);
      cuopt::remote::StatusResponse status_resp;
      grpc::ClientContext ctx;
      auto status = stub->CheckStatus(&ctx, status_req, &status_resp);
      if (!status.ok()) {
        std::cerr << "CheckStatus failed: " << status.error_message() << "\n";
        return 1;
      }
      auto js = status_resp.job_status();
      if (js == cuopt::remote::COMPLETED || js == cuopt::remote::FAILED ||
          js == cuopt::remote::CANCELLED) {
        std::cout << "Job finished with status=" << static_cast<int>(js) << "\n";
        if (js != cuopt::remote::COMPLETED) {
          std::cerr << "message: " << status_resp.message() << "\n";
          return 1;
        }
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    cuopt::remote::GetResultRequest result_req;
    result_req.set_job_id(job_id);
    cuopt::remote::ResultResponse result_resp;
    {
      grpc::ClientContext ctx;
      auto status = stub->GetResult(&ctx, result_req, &result_resp);
      if (!status.ok()) {
        std::cerr << "GetResult failed: " << status.error_message() << "\n";
        return 1;
      }
    }

    if (result_resp.status() != cuopt::remote::SUCCESS) {
      std::cerr << "Result error: " << result_resp.error_message() << "\n";
      return 1;
    }
    if (!result_resp.has_routing_solution()) {
      std::cerr << "Result missing routing_solution\n";
      return 1;
    }

    // routing_solution is a structured message now (no manual parse).
    print_solution(result_resp.routing_solution());
    return 0;
  } catch (std::exception const& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
